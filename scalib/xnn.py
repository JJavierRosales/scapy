# Libraries used for hinting
from __future__ import annotations
from typing import Union

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
from torch.nn.utils.rnn import pack_padded_sequence, pad_packed_sequence

from sklearn.model_selection import train_test_split

import os
import numpy as np
import pandas as pd
from matplotlib import pyplot as plt
import warnings

from . import utils
from .event import DatasetEventDataset
from .event import ConjunctionEvent as CE
from .event import ConjunctionEventsDataset as CED
from .cdm import ConjunctionDataMessage as CDM

#%% FUNCTION: MAPELoss
def MAPELoss(output:torch.Tensor, target:torch.Tensor, 
                epsilon:float=1e-8) -> torch.Tensor:
    """Mean Absolute Percentage Error (MAPE)

    Args:
        output (torch.Tensor): Predicted values.
        target (torch.Tensor): True values.
        epsilon (float, optional): Infinitesimal delta to skip 
        singularities caused by target=0. Defaults to 1e-8.

    Returns:
        torch.Tensor: _description_
    """
    
    return float(torch.mean(torch.abs((target - output) / \
                                torch.add(target, epsilon))*100))
#%% FUNCTION: POCID
def POCID(output:torch.Tensor, target:torch.Tensor) -> float:
    """Prediction Of Change In Direction (POCID)

    Args:
        output (torch.Tensor): Predicted value.
        target (torch.Tensor): True value.

    Returns:
        float: Percentage of predictions whose direction matches those of the 
        true values.
    """

    curr_output, prev_output = output[:, 1:], output[:, :-1]
    curr_target, prev_target = target[:, 1:], target[:, :-1]

    d = 0
    for i, cout in enumerate(curr_output):
        pout = prev_output[i]
        ctar, ptar = curr_target[i], prev_target[i]

        # Get boolean tensor
        pocid = (cout-pout)*(ctar-ptar)>0

        # Count number of conditions matching True
        d += int(torch.sum(pocid))

    return float(d/torch.numel(curr_output)*100)


#%% CLASS: Dataset
class Dataset(TensorDataset):
  def __init__(self, X, y):
    self.inputs = X
    self.outputs = y

    self.input_size = X.size(1)
    self.output_size = y.size(1)

    self.len = self.inputs.shape[0]

  def split(self, split_size:float=0.8, shuffle:bool=True, random_state:int=42):

    Xi, Xj, yi, yj = train_test_split(self.inputs, self.outputs, 
                                      train_size = split_size, 
                                      random_state = random_state, 
                                      shuffle = shuffle)
    
    return Dataset(Xi, yi), Dataset(Xj, yj)
  
  def __getitem__(self, index):
    return self.inputs[index], self.outputs[index]
  
  def __len__(self):
    return self.len
  
  def __repr__(self) -> str:
     description = f'TensorDataset(Inputs: {self.inputs.size(1)} ' + \
                   f'| Outputs: {self.outputs.size(1)} ' + \
                   f'| Entries: {len(self.inputs)})'
     return description

#%% CLASS: TensorDatasetFromDataFrame
class TensorDatasetFromDataFrame():
    def __init__(self, df:pd.DataFrame, output_features:list, input_features:list, 
                 normalize_inputs:bool=True):
        
        # Remove empty columns
        df.dropna(inplace=True, axis=1, how='all')

        # Initialise list with the categorical features and numerical features
        self.cat_features = []
        self.num_features = []
        undefined_features = []

        for f in input_features:
            if not f in df.columns: continue

            if df[f].dtype=='category':
                self.cat_features.append(f)
            elif f in df._get_numeric_data().columns:
                self.num_features.append(f)
            elif not f in output_features:
                undefined_features.append(f)

        if len(undefined_features)>0:
            unsupported_dtypes = []
            for f in undefined_features:
                unsupported_dtypes.append(df[f].dtype)

            raise ValueError(f'Some features ({undefined_features}) have ' + \
                             f'unsupported dtypes ({set(unsupported_dtypes)}).')
        
        # Check that all output features are part of the input dataframe.
        for o in output_features:
           if not o in df.columns:
              undefined_features.append(o)
        
        self.input_features = self.cat_features + self.num_features
        self.output_features = output_features

        self.df_outputs = df[self.output_features]
        self.df_inputs = df[self.input_features]

        # Get Torch for output features
        self.outputs = torch.tensor(self.df_outputs.to_numpy(), 
                        dtype=torch.float, 
                        requires_grad=True) 

        # Get Torch for all continuous features
        X_num = torch.tensor(self.df_inputs[self.num_features].to_numpy(), 
                            dtype=torch.float, 
                            requires_grad=True)
        X_num = torch.nan_to_num(X_num)

        # Normalise continuous variables
        if normalize_inputs:
            bn_nums = nn.BatchNorm1d(len(self.num_features))
            X_num = bn_nums(X_num)

        # Get Torch for all categorical features
        cats = np.stack([self.df_inputs[f].cat.codes.values for f in self.cat_features], 1)
        cats = torch.tensor(cats, dtype=torch.int)

        # This will set embedding sizes for the categorical columns:
        # an embedding size is the length of the array into which every category
        # is converted
        cat_szs = [len(self.df_inputs[f].cat.categories) for f in self.cat_features]
        emb_szs = [(size, min(50, (size+1)//2)) for size in cat_szs]

        # Initialize list of embedding operations.
        #  - nuv = Number of unique vectors
        #  - nvc = Number of vector componets
        embeddings = nn.ModuleList([nn.Embedding(nuv, nvc) for nuv, nvc in emb_szs])

        # Initialize embeddings list
        embeddings_tensors = []

        # Apply embedding operation to every column of categorical tensor
        for i, embedding in enumerate(embeddings):
            #print(cats[:,i])
            embeddings_tensors.append(embedding(cats[:,i]))

        # Concatenate embedding sections into 1
        X_cat = torch.cat(embeddings_tensors, 1)
    
        # Concatenate embeddings with continuous variables into one torch
        self.inputs = torch.cat([X_cat, X_num], 1)

        # Get final dataset with inputs and targets
        self.data = Dataset(self.inputs, self.outputs)

        self.len = self.inputs.shape[0]
        self.input_size = self.data.input_size
        self.output_size = self.data.output_size

    def __repr__(self) -> str:
       description = f'TensorDatasetFromDataFrame(' + \
                     f'Inputs: {self.input_size} ' + \
                     f'| Outputs: {self.output_size} ' + \
                     f'| Entries: {len(self.inputs)})'
       return description
    
    def __getitem__(self, index):
      return self.inputs[index], self.outputs[index]
    
    def __len__(self):
      return self.len
  
#%%CLASS: CosineWarmupScheduler
# https://github.com/pytorch/pytorch/blob/main/torch/optim/lr_scheduler.py
class CosineWarmupScheduler(optim.lr_scheduler._LRScheduler):

    def __init__(self, optimizer:optim.Optimizer, warmup_epochs:int, 
                 max_epochs:int) -> None:
        """Initialize scheduler object.

        Args:
            optimizer (optim.Optimizer): Optimization algorithm (SGD, Adam, ...).
            warmup_epochs (int): Number of epochs to linearly increment base 
            learning rate provided.
            max_epochs (int): Number of iterations at which the learning rate 
            becomes 0. Learning rate adaptation takes place between warmup and 
            max_iters. 
        """
        self.warmup_epochs = warmup_epochs
        self.max_epochs = max_epochs
        super().__init__(optimizer)

    def get_lr(self) -> list:
        """Recompute new base learning rates using the new learning rate factor.

        Returns:
            list: New base learning rates list.
        """

        # Get new learning rate factor from the last computed learning rate by 
        # current scheduler.
        lr_factor = self.get_lr_factor(epoch=self.last_epoch)

        # Return new list of learning rates using the new learning factor and 
        # the initial learning rate.
        return [base_lr * lr_factor for base_lr in self.base_lrs]

    def get_lr_factor(self, epoch:int) -> float:

        # Get learning factor using cosine trigonometric function.
        lr_factor = 0.5 * (1 + np.cos(np.pi * epoch / self.max_epochs))

        # Implement warmup adaptation to the cosine evolution of the learning 
        # rate. This adaptation is gradual from 0 to 100% the cosine function 
        # function when epoch = warmup_epochs.
        if epoch <= self.warmup_epochs:
            lr_factor *= epoch / self.warmup_epochs

        return lr_factor
    
    @staticmethod
    def plot_lr_factor(warmup_epochs:int, max_epochs:int, 
                       figsize:tuple = (8, 3), figtitle:str = None, 
                       filepath:str = None) -> None:
        
        # Initialze lr scheduler with any optimizer and learnable parameter.
        p = nn.Parameter(torch.empty(4,4))
        optimizer = optim.Adam([p], lr=1e-3)

        lr_scheduler = CosineWarmupScheduler(optimizer = optimizer, 
                                             warmup_epochs=warmup_epochs, 
                                             max_epochs = max_epochs)

        # Plotting
        epochs = list(range(max_epochs))

        fig, ax = plt.subplots(figsize=figsize)

        ax.plot(epochs, [lr_scheduler.get_lr_factor(e) for e in epochs])
        ax.set_ylabel("Learning rate factor")
        ax.set_xlabel("Epochs")
        if figtitle is not None:
            ax.set_title(figtitle)
        else:
            ax.set_title("Learning rate evolution")

        ax.grid(True, linestyle='--', color='gray')
        ax.set_xlim(0, max_epochs)
        ax.set_ylim(0, 1)
        ax.set_xticks(np.arange(max_epochs))

        # Save figure if filepath is provided.
        if filepath is not None:
            print('Plotting to file: {}'.format(filepath))
            fig.savefig(filepath)


#%% CLASS: ConjunctionEventForecaster
# Define Feature Forecaster module
class ConjunctionEventForecaster(nn.Module):
    def __init__(self, network:list, features:Union[list, str] = None) -> None:
        super(ConjunctionEventForecaster, self).__init__()
        if features is None:
            features = ['__CREATION_DATE',
                        '__TCA',
                        'MISS_DISTANCE',
                        'RELATIVE_SPEED',
                        'RELATIVE_POSITION_R',
                        'RELATIVE_POSITION_T',
                        'RELATIVE_POSITION_N',
                        'RELATIVE_VELOCITY_R',
                        'RELATIVE_VELOCITY_T',
                        'RELATIVE_VELOCITY_N',
                        'OBJECT1_X',
                        'OBJECT1_Y',
                        'OBJECT1_Z',
                        'OBJECT1_X_DOT',
                        'OBJECT1_Y_DOT',
                        'OBJECT1_Z_DOT',
                        'OBJECT1_CR_R',
                        'OBJECT1_CT_R',
                        'OBJECT1_CT_T',
                        'OBJECT1_CN_R',
                        'OBJECT1_CN_T',
                        'OBJECT1_CN_N',
                        'OBJECT1_CRDOT_R',
                        'OBJECT1_CRDOT_T',
                        'OBJECT1_CRDOT_N',
                        'OBJECT1_CRDOT_RDOT',
                        'OBJECT1_CTDOT_R',
                        'OBJECT1_CTDOT_T',
                        'OBJECT1_CTDOT_N',
                        'OBJECT1_CTDOT_RDOT',
                        'OBJECT1_CTDOT_TDOT',
                        'OBJECT1_CNDOT_R',
                        'OBJECT1_CNDOT_T',
                        'OBJECT1_CNDOT_N',
                        'OBJECT1_CNDOT_RDOT',
                        'OBJECT1_CNDOT_TDOT',
                        'OBJECT1_CNDOT_NDOT',
                        'OBJECT2_X',
                        'OBJECT2_Y',
                        'OBJECT2_Z',
                        'OBJECT2_X_DOT',
                        'OBJECT2_Y_DOT',
                        'OBJECT2_Z_DOT',
                        'OBJECT2_CR_R',
                        'OBJECT2_CT_R',
                        'OBJECT2_CT_T',
                        'OBJECT2_CN_R',
                        'OBJECT2_CN_T',
                        'OBJECT2_CN_N',
                        'OBJECT2_CRDOT_R',
                        'OBJECT2_CRDOT_T',
                        'OBJECT2_CRDOT_N',
                        'OBJECT2_CRDOT_RDOT',
                        'OBJECT2_CTDOT_R',
                        'OBJECT2_CTDOT_T',
                        'OBJECT2_CTDOT_N',
                        'OBJECT2_CTDOT_RDOT',
                        'OBJECT2_CTDOT_TDOT',
                        'OBJECT2_CNDOT_R',
                        'OBJECT2_CNDOT_T',
                        'OBJECT2_CNDOT_N',
                        'OBJECT2_CNDOT_RDOT',
                        'OBJECT2_CNDOT_TDOT',
                        'OBJECT2_CNDOT_NDOT']

        self.input_size = len(features)
        self._batched = None

        # Set model using the modules parameter
        self.model = network

        self._features = features
        self._features_stats = None

        self._learn_results = {'total_iterations':[],
                               'validation_loss':[],
                               'training_loss':[],
                               'epoch':[],
                               'learning_rate':[],
                               'batch_size':[]}


    def plot_loss(self, filepath:str = None, figsize:tuple = (6, 3), 
                  log_scale:bool = False, validation_only:bool=False, 
                  plot_lr:bool=False, label:str = None,
                  ax:plt.Axes = None, return_ax:bool = False) -> None:
        """Plot RNN loss in the training set (orange) and validation set (blue) 
        vs number of iterations during model training.

        Args:
            filepath (str, optional): Path where the plot is saved. Defaults to 
            None.
            figsize (tuple, optional): Size of the plot. Defaults to (6 ,3).
            log_scale (bool, optional): Flag to plot Loss using logarithmic 
            scale. Defaults to False.
            plot_lr (bool, optional): Include learning rate evolution during
            training. Defaults to False.
        """

        # Apply logarithmic transformation if log_scale is set to True. This 
        # helps to see the evolution when variations between iterations are 
        # small.
        iterations = self._learn_results['total_iterations']

        
        # Create axes instance if not passed as a parameter.
        if ax is None: fig, ax = plt.subplots(figsize=figsize)

        # Plot loss vs iterations.
        colors = ['tab:orange', 'tab:blue']

        for p, process in enumerate(['training', 'validation']):

            if process=='training' and validation_only: continue

            loss = self._learn_results[f'{process}_loss']
            loss = pd.Series(np.log(loss) if log_scale else loss, 
                                iterations).drop_duplicates(keep='first')

            ax.plot(loss, color = colors[p], 
                    label = process.capitalize() if label is None else label)

            # Set X-axis limits.
            if process=='validation': ax.set_xlim(0, loss.index.max())
        

        # Plot learning rate if required
        if plot_lr:

            # Get learning rate
            lr = pd.Series(self._learn_results['learning_rate'], 
                           iterations)
            
            # Add right axis to plot learning rate.
            ax_lr = ax.twinx()

            ax_lr.set_ylabel('Learning rate', color = 'tab:green')  
            ax_lr.set_ylim(0, lr.max()*1.25)
            ax_lr.plot(lr, label='Learning rate', color = 'tab:green')
        


        # Set axes labels.
        ax.set_xlabel('Number of iterations')
        ax.set_ylabel('MSE Loss')

        # Set legend and grid for better visualization.
        ax.legend(fontsize=8)
        ax.grid(True, linestyle='--')

        # Save figure if filepath is provided.
        if filepath is not None:
            print('Plotting to file: {}'.format(filepath))
            fig.savefig(filepath)

        if return_ax:
            return ax

    def learn(self, event_set:list, epochs:int = 2, lr:float = 1e-3, 
              batch_size:int = 8, device:str = None, 
              valid_proportion:float = 0.15, num_workers:int = 4, 
              event_samples_for_stats:int = 250, filepath:str = None,
              epoch_step_checkpoint:int = None, **kwargs) -> None:
        """Train RNN model.

        Args:
            event_set (list): List of Conjunction Event objects to use for 
            training (including validationd data).
            epochs (int, optional): Number of epochs used for training. Defaults 
            to 2.
            lr (float, optional): Learning rate. Defaults to 1e-3.
            batch_size (int, optional): Batch size. Defaults to 8.
            device (str, optional): Device where torchs are allocated. Defaults 
            to 'cpu'.
            valid_proportion (float, optional): Proportion of all data used for 
            validation (value must be between 0 and 1). Defaults to 0.15.
            num_workers (int, optional): _description_. Defaults to 4.
            event_samples_for_stats (int, optional): Number of events considered 
            to compute the mean and standard deviation used for normalization. 
            Defaults to 250.
            filepath (str, optional): Filepath for the model 
            to be saved. If None, model is not saved. Defaults to None.
            epoch_step_checkpoint (int, optional): Number of epochs to process 
            before saving a new checkpoint. Only applicable if filepath is
            not None. Defaults to None.

        Raises:
            ValueError: valid_proportion is not in the range (0, 1).
            RuntimeError: Validation set does not contain any event as a result 
            of valid_proportion being too low.
        """


        
        # Define the device on which the torch will be allocated:
        device = torch.device('cpu') if device is None else torch.device(device)

        self._device = device
        self.to(device)

        # Get number of parameters in the model.
        num_params = sum(p.numel() for p in self.parameters())
        print(f'Number of learnable parameters of the model: {num_params:,}')

        # Ensure number of events considered to compute statistics measures
        # (mean and standard deviation) is lower or equal to the total number 
        # of events available.
        if event_samples_for_stats > len(event_set):
            event_samples_for_stats = len(event_set)

        # If mean and standard deviations per feature are not provided, get 
        # them using the DatasetEventDataset class to normalize features in the
        # training and validation datasets.
        if self._features_stats is None:
            self._features_stats = \
                DatasetEventDataset(event_set[:event_samples_for_stats], 
                                    self._features)._features_stats
        
        # Consider only events that contain at least 2 CDM objects.
        event_set = event_set.filter(lambda event: len(event) > 1)

        # Check valid_proportion is between 0.0 and 1.0
        if valid_proportion<0 or valid_proportion>1.0:
            raise ValueError('Parameter valid_proportion ({})'+ \
                             ' must be greater than 0 and lower than 1' \
                             .format(valid_proportion))

        # Compute the size of the validation set from valid_proportion.
        valid_set_size = int(len(event_set) * valid_proportion)
        
        # Check size of validation set is greater than 0.
        if valid_set_size == 0:
            raise RuntimeError('Validation set size is 0 for the given' + \
                               ' valid_proportion ({}) and number of ' + \
                               'events ({})' \
                               .format(valid_proportion, len(event_set)))
        
        # Get training set size.
        train_set_size = len(event_set) - valid_set_size

        # Get training and validation datasets with normalized features using
        # the stats metrics. 
        train_set = DatasetEventDataset(event_set[:train_set_size], 
                                        self._features, self._features_stats)
        valid_set = DatasetEventDataset(event_set[train_set_size:], 
                                        self._features, self._features_stats)

        # Get loader objects for training and validation sets. The DataLoader 
        # class works by creating an iterable dataset object and iterating over 
        # it in batches, which are then fed into the model for processing. The 
        # object it creates has the shape (batches, items) where items are the 
        # a number of elements n = int(len(Dataset)/batch_size) taken from the 
        # Dataset passed into the class.
        train_loader = DataLoader(train_set, batch_size = batch_size, 
                                  shuffle = True, num_workers = num_workers)
        valid_loader = DataLoader(valid_set, batch_size = len(valid_set), 
                                  shuffle = True, num_workers = num_workers)



        # Set-up optimizer and criterion.
        self.optimizer = optim.Adam(self.parameters(), lr = lr)
        self.criterion = nn.MSELoss()

        # Initialize lr_scheduler
        # lr_scheduler = CosineWarmupScheduler(optimizer = self.optimizer, 
        #                                      warmup_epochs = 2, 
        #                                      max_epochs = 20)

        # Check if the same model already exists. If it does, load model 
        # parameters.
        if filepath is not None and os.path.exists(filepath):
            self.load(filepath)
            self.optimizer.param_groups[0]['lr'] = lr
            print('\nModel parameters loaded from {}\n'
                  ' - Total epochs       = {}\n'
                  ' - Total iterations   = {}\n'
                  ' - Validation loss    = {:6.4e}\n'
                  ' - Last learning rate = {:6.4e}\n'
                  ''.format(filepath,
                            self._learn_results['epoch'][-1],
                            self._learn_results['total_iterations'][-1],
                            self._learn_results['validation_loss'][-1],
                            self._learn_results['learning_rate'][-1]))

        # Set training mode ON to inform layers such as Dropout and BatchNorm, 
        # which are designed to behave differently during training and 
        # evaluation. For instance, in training mode, BatchNorm updates a moving 
        # average on each new batch; whereas, for evaluation mode, these updates 
        # are frozen.
        self.train()
        
        if len(self._learn_results['total_iterations']) == 0:
            total_iters = 0
            last_epoch = 0
        else:
            total_iters = self._learn_results['total_iterations'][-1]
            last_epoch = self._learn_results['epoch'][-1]

        # Initialize progress bar to show training progress.
        n_batches = len(train_loader)
        pb_epochs = utils.ProgressBar(iterations=range(epochs*n_batches),
                                      title = 'TRAINING FORECASTING MODEL:')

        for epoch in range(epochs):
            with torch.no_grad():
                for _, (events, event_lengths) in enumerate(valid_loader):

                    # Allocate events and event_lengths tensors to the device
                    # defined by the model.
                    events = events.to(device)
                    event_lengths = event_lengths.to(device)

                    # Get batch_size from event_lengths as it can be smaller for
                    # the last minibatch of an epoch.
                    batch_size = event_lengths.nelement()

                    # For every event object, take all CDMs except the last one 
                    # as inputs.
                    inputs = events[:, :-1]

                    # For every event object, shift CDM object list to set as 
                    # targets from 2nd CDM to last one. 
                    target = events[:, 1:]
                    event_lengths -= 1

                    # Initialize LSTM hidden state (h0) and cell state (c0).
                    self.reset(batch_size)

                    # Forecast next CDMs of the mini-batch using the inputs. The 
                    # model also requires a second parameter with the number of 
                    # CDMs per event object in order to pack padded sequences to 
                    # optimize computation.
                    output = self.forward(inputs, event_lengths)

                    # Compute loss using the criterion and add it to the array.
                    self._loss = self.criterion(output, target)
                    valid_loss = float(self._loss)

            # Iterate over all batches containes in the training loader. Every 
            # batch (t_batch) contains an equal number of events which in 
            # turn may contain a different number of CDM objects.
            for t_batch, (events, event_lengths) in enumerate(train_loader):
                total_iters += 1
                relative_iters = (epoch*n_batches + t_batch + 1)

                # Allocate events and event_lengths tensors to the device 
                # defined by the model.
                events = events.to(device)
                event_lengths = event_lengths.to(device)

                # Get batch_size from event_lengths as it can be smaller for the 
                # last minibatch of an epoch.
                batch_size = event_lengths.nelement() 

                # Set as inputs all the CDM objects but the last one from all 
                # events in the mini-batch. For every event contained in the
                # inputs tensor the shape is (n_cdms-1, n_features)
                inputs = events[:, :-1, :]

                # Set as targets the next CDM object following every CDM in the 
                # inputs tensor. The objective is to train the model using only 
                # one CDM to forecast the next one. For every event contained in 
                # the targets tensor the shape is (n_cdms-1, n_features)
                target = events[:, 1:, :]
                event_lengths -= 1

                # Initialize LSTM hidden state (h0) and cell state (c0).
                self.reset(batch_size)

                # Clear all the gradients of all the parameters of the model.
                self.optimizer.zero_grad()
                output = self.forward(inputs, event_lengths)

                # Compute MSE loss using criterion and store it in an array.
                self._loss = self.criterion(output, target)
                train_loss = float(self._loss)

                # Get learning rate used
                lr = self.optimizer.param_groups[0]['lr']

                # Backpropagate MSE loss.
                self._loss.backward()

                # Update model hyperparameters taking into account the loss.
                self.optimizer.step()
                # lr_scheduler.step()

                # Update progress bar.
                description = f'E({epoch+1}/{epochs}) ' + \
                    f'B({t_batch+1}/{n_batches}) | ' + \
                    f'Loss > T({train_loss:6.4e}) ' + \
                    f'V({valid_loss:6.4e})'

                pb_epochs.refresh(i = relative_iters, 
                                  description = description,
                                  nested_progress = True)
                

                # Save training information.
                self._learn_results['total_iterations'].append(total_iters)
                self._learn_results['validation_loss'].append(valid_loss)
                self._learn_results['training_loss'].append(train_loss)
                self._learn_results['epoch'].append(last_epoch + epoch + 1)
                self._learn_results['learning_rate'].append(lr)
                self._learn_results['batch_size'].append(batch_size)

                if epoch_step_checkpoint is not None and \
                   ((epoch+1) % epoch_step_checkpoint) == 0 and \
                    (epoch+1) < epochs:
                    pb_epochs.refresh(i = relative_iters, 
                                  description = 'Saving checkpoint...',
                                  nested_progress = True)
                    self.save(filepath = filepath)
                    pb_epochs.refresh(i = relative_iters, 
                                      description = description,
                                      nested_progress = True)
                    

        # Print message at the end of the mini batch.
        pb_epochs.refresh(i = relative_iters, description = description)

        if filepath is not None:
            print(f'\nSaving model parameters ...', end='\r')
            self.save(filepath = filepath)
            print(f'Saving model parameters ... Done.')
            
    def save(self, filepath:str, only_parameters:bool=True):
        """Save model to an external file.

        Args:
            filepath (str): Path where the model is saved.
            only_parameters (bool, optional): Save only models parameters or the 
            entire model. Defaults to True.
        """
        if only_parameters:
            torch.save({
                'num_params': sum(p.numel() for p in self.parameters()),
                'epochs': self._learn_results['epoch'][-1],
                'model': self.state_dict() if only_parameters else self,
                'optimizer': self.optimizer.state_dict() if only_parameters \
                    else self.optimizer,
                'loss': self._loss,
                'learn_results': self._learn_results,
                'features_stats': self._features_stats
            },  filepath)
        else:
            torch.save(self, filepath)

    def load(self, filepath:str):
        """Load model instance or model parameters.

        Args:
            filepath (str): Path of the model source file.
        """

        if 'parameters' in filepath:

            # Get the checkpoint file
            torch_file = torch.load(filepath)

            self.load_state_dict(torch_file['model'])
            if not hasattr(self, 'optimizer'):
                self.optimizer = optim.Adam(self.parameters(), lr = 1e-3)
            self.optimizer.load_state_dict(torch_file['optimizer'])
            self._loss = torch_file['loss']
            self._learn_results = torch_file['learn_results']
            self._features_stats = torch_file['features_stats']
        else:
            self = torch.load(filepath)
    
    def test(self, events_test:CED, test_batch_size:int, 
             num_workers:int = 4) -> np.ndarray:
        """Compute MSE loss on a test set using the trained model.

        Args:
            events_test (CED): Conjunction Events Dataset.
            test_batch_size (int): Batch size.
            num_workers (int, optional): Number of workers for the DataLoader
            class. Defaults to 4.

        Returns:
            np.ndarray: NumPy array containing the MSE loss values per batch.
        """



        # Get test dataset with normalized features using the stats metrics. 
        test_set = DatasetEventDataset(event_set = events_test, 
                                       features = self._features, 
                                       features_stats = self._features_stats)
        
        if test_batch_size > len(events_test):
            test_batch_size = len(events_test)
            warnings.warn(f'\nParameter test_batch_size ({test_batch_size})'
                          f'can only be less or equal to the number of '
                          f'items on events_set input ({len(events_test)})'
                          f'\nSetting new value for test_batch_size='
                          f'{len(events_test)}.')
        
        # Get loader objects for test set. The DataLoader class works by 
        # creating an iterable dataset object and iterating over it in batches, 
        # which are then fed into the model for processing. The object it 
        # creates has the shape (batches, items) where items are the a number of 
        # elements n = int(len(Dataset)/batch_size) taken from the Dataset 
        # passed into the class.
        test_loader = DataLoader(dataset = test_set, 
                                 batch_size = test_batch_size, 
                                 shuffle = True, 
                                 num_workers = num_workers)


        # Set-up device and criterion.
        device = list(self.parameters())[0].device

        # Get number of parameters
        k = sum(p.numel() for p in self.parameters())

        # Define all criterions required for the regression metrics.
        mae_criterion = nn.L1Loss()
        mse_criterion = nn.MSELoss(reduction='mean')
        sse_criterion = nn.MSELoss(reduction='sum')
        mape_criterion = MAPELoss
        pocid_criterion = POCID

        # Initialize dictionary with the different regression metrics.
        results = {'sse':np.zeros((len(test_loader))),
                   'mse':np.zeros((len(test_loader))),
                   'mae':np.zeros((len(test_loader))),
                   'mape':np.zeros((len(test_loader))),
                   'bic':np.zeros((len(test_loader))),
                   'pocid': np.zeros((len(test_loader)))}

        # Iterate over all items in the test_loader
        with torch.no_grad():
            for t, (events, event_lengths) in enumerate(test_loader):

                # Allocate events and event_lengths tensors to the device
                # defined by the model.
                events = events.to(device)
                event_lengths = event_lengths.to(device)

                # Get batch_size from event_lengths as it can be smaller for
                # the last minibatch of an epoch.
                batch_size = event_lengths.nelement()

                # For every event object, take all CDMs except the last one 
                # as inputs.
                inputs = events[:, :-1]

                # For every event object, shift CDM object list to set as 
                # targets from 2nd CDM to last one. 
                target = events[:, 1:]
                event_lengths -= 1

                # Initialize LSTM hidden state (h0) and cell state (c0).
                self.reset(batch_size)

                # Forecast next CDMs of the mini-batch using the inputs. The 
                # model also requires a second parameter with the number of 
                # CDMs per event object in order to pack padded sequences to 
                # optimize computation.
                output = self.forward(inputs, event_lengths)

                # Get test size
                t_sz = len(output)

                # Compute regression metrics using the criterion and add it to 
                # the array.
                results['sse'][t] = float(sse_criterion(output, target))
                results['mse'][t] = float(mse_criterion(output, target))
                results['mae'][t] = float(mae_criterion(output, target))
                results['mape'][t] = float(mape_criterion(output, target))
                results['bic'][t] = t_sz*np.log(results['mse'][t])+k*np.log(t_sz)
                results['pocid'][t] = pocid_criterion(output, target)

        return results

    def predict(self, event: CE) -> CDM:
        """Predict next CDM object from a given ConjunctionEvent object.

        Args:
            event (ConjunctionEvent): Conjunction Event object containing CDM(s) 
            object(s).

        Raises:
            RuntimeError: _description_

        Returns:
            ConjunctionDataMessage: CDM object.
        """

        ds = DatasetEventDataset(CED(events=[event]), 
                                 features = self._features, 
                                 features_stats = self._features_stats)
        
        # Get CDM objects and number of CDMs contained in the dataset taking the 
        # event passed as a parameter.
        inputs, inputs_length = ds[0]

        # Allocate torch to the 
        inputs = inputs.to(self._device)
        inputs_length = inputs_length.to(self._device)

        self.train()

        # Initialize LSTM hidden state (h) and cell state (c) assuming 
        # batch_size = 1.
        self.reset(1)

        # Forecast next CDM content. Use unsqueeze function to add one extra 
        # dimension to the tensor to simulate batch_size=1; from shape 
        # (seq_length, n_features) to (1, seq_length, n_features).
        output = self.forward(inputs.unsqueeze(0), 
                              inputs_length.unsqueeze(0)).squeeze()

        if utils.has_nan_or_inf(output):
            raise RuntimeError(f'Network output has nan or inf: {output}\n')

        output_last = output if output.ndim == 1 else output[-1]
 
        # Get creation date from first CDM object contained in the Conjunction 
        # Event.
        date0 = event[0]['CREATION_DATE']

        # Initialize new CDM object to store de-normalized values resulting from 
        # the RNN model.
        cdm = CDM()

        # Iterate over all the featues
        for i in range(len(self._features)):

            # Get feature name, mean and standard deviation.
            feature = self._features[i]
            feature_mean = self._features_stats['mean'][i]
            feature_stddev = self._features_stats['stddev'][i]

            # De-normalize the values for the feature using its assciated mean 
            # and standard deviation.
            value = feature_mean + feature_stddev * float(output_last[i].item()) 

            if feature == '__CREATION_DATE':
                # CDM creation date shall be equal or greater than the creation
                # date of the last CDM contained in the event. Otherwise, set 
                # the creation date equal to the previous CDM creation date.
                if value < event[-1]['__CREATION_DATE']:
                    value = event[-1]['__CREATION_DATE']
                cdm['CREATION_DATE'] = utils.add_days_to_date_str(date0, value)
            elif feature == '__TCA':
                cdm['TCA'] = utils.add_days_to_date_str(date0, value)
            else:
                cdm[feature] = value

        return cdm

    def predict_event_step(self, event:CE, num_samples:int = 1) \
        -> Union[CE, CED]:
        """Predict next CDM n-times for a given event object.

        Args:
            event (ConjunctionEvent): Conjunction Event object from which the 
            CDM is forecasted.
            num_samples (int, optional): Number of predictions. Defaults to 1.

        Returns:
            Union[ConjunctionEvent, ConjunctionEventsDataset]: Two possible 
            outputs are returned depending on the parameter num_samples:
             - If num_samples = 1: Returns one ConjunctionEvent object with all 
                CDMs forecasted. 
             - If num_samples > 1: Returns ConjunctionEventsDataset object 
                containing all possible evolutions of the event (combinations of
                CDMs).
        """

        # Initialize empty list of events
        events = []
        for i in range(num_samples):

            # Create a copy of the event to avoid modifying it.
            i_event = event.copy()

            # Predict next CDM of the event.
            cdm = self.predict(i_event)

            # Add CDM object to the ConjunctionEvent object.
            i_event.add(cdm)

            # Append event to the Conjunction Events list.
            events.append(i_event)

        # Return Event object or ConjunctionEventsDataset objects.
        return events[0] if num_samples == 1 \
            else CED(events=events)
            

    def predict_event(self, event:CE, num_samples:int = 1, 
        max_length:int = None) -> Union[CE, CED]:
        """Forecast the evolution of a given Conjunction Event by predicting 
        upcoming CDMs until TCA.

        Args:
            event (ConjunctionEvent): Conjunction Event to forecast.
            num_samples (int, optional): Number of possible CDMs considered in 
            every forecasting step. Defaults to 1.
            max_length (int, optional): Maximum number of CDM objects contained 
            in the event object. Defaults to None.

        Returns:
            Union[ConjunctionEvent, ConjunctionEventsDataset]: Two possible 
            outputs are returned depending on the parameter num_samples:
             - If num_samples = 1: Returns one ConjunctionEvent object with all 
                CDMs forecasted. 
             - If num_samples > 1: Returns ConjunctionEventsDataset object 
                containing all possible evolutions of the event (combinations of 
                CDMs).
        """

        if max_length is not None and max_length<len(event):
            max_length = len(event)+1

        # Initialize list to store Conjunction Events.
        events = []

        # Iterate over all sequences
        # pb_samples = utils.ProgressBar(iterations = range(num_samples),  
        #     description='Forecasting event evolution ...')

        for i in range(num_samples):

            # Update progress bar.
            # pb_samples.refresh(i = i+1)

            # Create a deep copy of the input event to avoid modifying the 
            # original object
            i_event = event.copy()

            # Run loop to forecast new CDMs until one of the following 
            # conditions are reached:
            #  - CDM creation date is later than TCA.
            #  - TCA is later than 7 days.
            #  - The conjunction event contains same number of CDMs equal to 
            #       max_length.
            while True:

                # Predict new CDM from i_event and append object to i_event.
                cdm = self.predict(i_event)

                i_event.add(cdm)

                # Check predicted CDM time to TCA is lower than previous CDM.
                if cdm['__TCA'] >= i_event[-2]['__TCA']: 
                    i_event = i_event[:-1]
                    continue

                # Stop loop if one of the conditions is met.
                if (cdm['__CREATION_DATE'] > cdm['__TCA']) or \
                   (cdm['__TCA'] > 7) or \
                   (len(i_event) > max_length): break

            # Append i_event to the final list of events.       
            events.append(i_event)

        # Update progress bar.
        # pb_samples.refresh(i = i+1, description = 'Event forecasted.')

        return events[0] if num_samples==1 \
            else CED(events = events)

    def reset(self, batch_size:int):
        """Initialize hidden state (h) and cell state (c) for all the LSTM 
        layers in the Sequencial object.
        """
        # Initialize dictionary for hidden states tuple
        self.hidden = {}

        device = list(self.parameters())[0].device

        for module_name, module in self.model.items():
            if 'lstm' in module_name:

                h = torch.zeros(module.num_layers, batch_size, module.hidden_size)
                c = torch.zeros(module.num_layers, batch_size, module.hidden_size)

                h = h.to(device)
                c = c.to(device)
                
                self.hidden[module_name] = (h.squeeze(0), c.squeeze(0))
            elif ('gru' in module_name) or ('mgu' in module_name):

                h = torch.zeros(module.num_layers, batch_size, module.hidden_size)
                h = h.to(device)
                
                self.hidden[module_name] = h.squeeze(0)



    def forward(self, x:torch.Tensor, x_lengths:torch.IntTensor) -> torch.Tensor:
        """Predict new CDM containing normalized values.

        Args:
            x (torch.FloatTensor): Tensor with shape (n_events, 
            max_event_length, features) containing the input values for RNN 
            processing.
            x_lengths (torch.IntTensor): Tensor with shape (n_events, 1) 
            containing the number of CDMs contained in every event. This tensor 
            is used to unpack padded torch.

        Returns:
            torch.Tensor: Tensor containing normalized values of the new CDM.
        """

        # Iterate over all modules to perform the forward operation.
        for module_name, module in self.model.items():

            if ('lstm' in module_name) or \
                ('gru' in module_name) or \
                ('mgu' in module_name):

                # Get size of inputs tensor.
                # batch_size, x_length_max, n_features = x.size()

                # All events are padded with zeros in order to get the same 
                # number of tensors, and therefore same event_length. This makes 
                # batch processing 
                # easier.

                # To optimize training and avoid computing empty CDMs, PyTorch 
                # allows packing a padded sequence by producing a PackedSequence 
                # object. A PackedSequence object is a tuple of two lists. One 
                # contains the elements of sequences interleaved by time steps 
                # and other contains the the batch size at each step. 

                # Pack Tensor containing padded sequences of variable length.
                # x = pack_padded_sequence(input = x, 
                #                          lengths = x_lengths, 
                #                          batch_first = module.batch_first, 
                #                          enforce_sorted = False)
                
                x, self.hidden[module_name] = module(x, self.hidden[module_name])
                
                # # Pads a packed batch of variable length sequences from LSTM layer.
                # x, _ = pad_packed_sequence(sequence = x, 
                #                            batch_first = module.batch_first, 
                #                            total_length = x_length_max)
                
            else:
                x = module(x)

        return x
    
    def __hash__(self):
        return hash(self.model)


#%% CLASS: CollisionRiskProbabilityEstimator
# Define Collision Risk Probability Estimator
class CollisionRiskProbabilityEstimator(nn.Module):

    def __init__(self, input_size:int, output_size:int, layers:list, 
                 bias:Union[bool, list] = True, act_functions = nn.ReLU(), 
                 dropout_probs:Union[float, list] = 0.2, 
                 classification:bool=False, 
                 class_weights:torch.Tensor=torch.tensor([0.5, 0.5])):
        
        # Inherit attributes from nn.Module class
        super().__init__()

        # Initialise input and output sizes
        self.input_size = input_size
        self.output_size = output_size
        self.layers = layers
        self.classification = classification
        self._class_weights = class_weights

        if isinstance(act_functions,list):
            if len(act_functions)< len(layers):
                raise ValueError(f'Length of act_functions parameter shall match the length of layers.')
            
        if isinstance(dropout_probs,list):
            if len(act_functions)< len(layers):
                raise ValueError(f'Length of act_functions parameter shall match the length of layers.')


        # Set model using the modules parameter
        self.model = nn.ModuleList()
        input_neurons = self.input_size
        for l, hidden_neurons in enumerate(self.layers):
            # On layer l, which contains n_neurons, perform the following 
            # operations:
            # 1. Apply Linear neural network model regression (fully connected 
            # network -> z = Sum(wi*xi+bi))
            b = bias[l] if isinstance(bias, list) else bias
            self.model.append(nn.Linear(input_neurons, hidden_neurons, bias = b))
            
            # 2. Apply ReLU activation function (al(z))
            af = act_functions[l] if isinstance(act_functions, list) else act_functions
            if af is not None: self.model.append(af)
            
            # 3. Normalize data using the n_neurons
            self.model.append(nn.BatchNorm1d(hidden_neurons))
            
            # 4. Cancel out a random proportion p of the neurons to avoid 
            # overfitting
            p = dropout_probs[l] if isinstance(dropout_probs, list) else dropout_probs
            if p is not None:
                self.model.append(nn.Dropout(p))
            
            # 5. Set new number of input features n_in for the next layer l+1.
            input_neurons = hidden_neurons

        # Set the last layer of the list which corresponds to the final output
        self.model.append(nn.Linear(self.layers[-1], self.output_size))

        # Initialize dictionary to store training results
        self._learn_results = {'total_iterations':[],
                               'validation_loss':[],
                               'training_loss':[],
                               'epoch':[],
                               'learning_rate':[],
                               'batch_size':[]}
                               
    def plot_loss(self, filepath:str = None, figsize:tuple = (6, 3), 
                  log_scale:bool = False, validation_only:bool=False, 
                  plot_lr:bool=False, label:str = None,
                  ax:plt.Axes = None, return_ax:bool = False) -> None:
        """Plot ANN loss in the training set (orange) and validation set (blue) 
        vs number of iterations during model training.

        Args:
            filepath (str, optional): Path where the plot is saved. Defaults to 
            None.
            figsize (tuple, optional): Size of the plot. Defaults to (6 ,3).
            log_scale (bool, optional): Flag to plot Loss using logarithmic 
            scale. Defaults to False.
            plot_lr (bool, optional): Include learning rate evolution during
            training. Defaults to False.
        """

        # Apply logarithmic transformation if log_scale is set to True. This 
        # helps to see the evolution when variations between iterations are 
        # small.
        iterations = self._learn_results['total_iterations']

        
        # Create axes instance if not passed as a parameter.
        if ax is None: fig, ax = plt.subplots(figsize=figsize)

        # Plot loss vs iterations.
        colors = ['tab:orange', 'tab:blue']

        for p, process in enumerate(['training', 'validation']):

            if process=='training' and validation_only: continue

            loss = self._learn_results[f'{process}_loss']
            loss = pd.Series(np.log(loss) if log_scale else loss, 
                                iterations).drop_duplicates(keep='first')

            ax.plot(loss, color = colors[p], 
                    label = process.capitalize() if label is None else label)

            # Set X-axis limits.
            if process=='validation': ax.set_xlim(0, loss.index.max())
        

        # Plot learning rate if required
        if plot_lr:

            # Get learning rate
            lr = pd.Series(self._learn_results['learning_rate'], 
                           iterations)
            
            # Add right axis to plot learning rate.
            ax_lr = ax.twinx()

            ax_lr.set_ylabel('Learning rate', color = 'tab:green')  
            ax_lr.set_ylim(0, lr.max()*1.25)
            ax_lr.plot(lr, label='Learning rate', color = 'tab:green')

        # Set axes labels.
        ax.set_xlabel('Number of iterations')
        ax.set_ylabel('MSE Loss')

        # Set legend and grid for better visualization.
        ax.legend(fontsize=8)
        ax.grid(True, linestyle='--')

        # Save figure if filepath is provided.
        if filepath is not None:
            print('Plotting to file: {}'.format(filepath))
            fig.savefig(filepath)

        if return_ax:
            return ax
        

    def learn(self, data:TensorDataset, epochs:int = 10, lr:float = 1e-3, 
              batch_size:int = 8, device:str = None, 
              valid_proportion:float = 0.15, filepath:str = None,
              epoch_step_checkpoint:int = None, 
              **kwargs) -> None:
        """Train ANN model.

        Args:
            data (TensorDataset): List of Conjunction Event objects to use for 
            training (including validationd data).
            epochs (int, optional): Number of epochs used for training. Defaults 
            to 10.
            lr (float, optional): Learning rate. Defaults to 1e-3.
            batch_size (int, optional): Batch size. Defaults to 8.
            device (str, optional): Device where torchs are allocated. Defaults 
            to 'cpu'.
            valid_proportion (float, optional): Proportion of all data used for 
            validation (value must be between 0 and 1). Defaults to 0.15.
            filepath (str, optional): Filepath for the model 
            to be saved. If None, model is not saved. Defaults to None.
            epoch_step_checkpoint (int, optional): Number of epochs to process 
            before saving a new checkpoint. Only applicable if filepath is
            not None. Defaults to None.

        Raises:
            ValueError: valid_proportion is not in the range (0, 1).
            RuntimeError: Validation set does not contain any event as a result 
            of valid_proportion being too low.
        """

        
        # Define the device on which the torch will be allocated:
        device = torch.device('cpu') if device is None else torch.device(device)

        self._device = device
        self.to(device)

        # Get number of parameters in the model.
        num_params = sum(p.numel() for p in self.parameters())
        print(f'Number of learnable parameters of the model: {num_params:,}')

        # Check valid_proportion is between 0.0 and 1.0
        if valid_proportion < 0 or valid_proportion > 1.0:
            raise ValueError('Parameter valid_proportion ({})'+ \
                             ' must be greater than 0 and lower than 1' \
                             .format(valid_proportion))

        # Compute the size of the validation set from valid_proportion.
        data_train, data_valid = data.split(split_size = 1-valid_proportion)
        
        # Check size of validation set is greater than 0.
        if len(data_valid) == 0:
            raise RuntimeError('Validation set size is 0 for the given' + \
                               ' valid_proportion ({}) and number of ' + \
                               'events ({})' \
                               .format(valid_proportion, len(data)))

        # Get loader objects for training and validation sets. The DataLoader 
        # class works by creating an iterable dataset object and iterating over 
        # it in batches, which are then fed into the model for processing. The 
        # object it creates has the shape (batches, items) where items are the 
        # a number of elements n = int(len(Dataset)/batch_size) taken from the 
        # Dataset passed into the class.
        train_loader = DataLoader(data_train, batch_size = batch_size, 
                                  shuffle = True)
        valid_loader = DataLoader(data_valid, batch_size = len(data_valid), 
                                  shuffle = True)
        

        # Set-up optimizer and criterion.
        self.optimizer = optim.Adam(self.parameters(), lr = lr)

        if self.classification:
            self.criterion = nn.CrossEntropyLoss(weight = self._class_weights)
        else:
            self.criterion = nn.MSELoss()


        # Check if the same model already exists. If it does, load model 
        # parameters.
        if filepath is not None and os.path.exists(filepath):
            self.load(filepath)
            self.optimizer.param_groups[0]['lr'] = lr
            print('\nModel parameters loaded from {}\n'
                  ' - Total epochs       = {}\n'
                  ' - Total iterations   = {}\n'
                  ' - Validation loss    = {:6.4e}\n'
                  ' - Last learning rate = {:6.4e}\n'
                  ''.format(filepath,
                            self._learn_results['epoch'][-1],
                            self._learn_results['total_iterations'][-1],
                            self._learn_results['validation_loss'][-1],
                            self._learn_results['learning_rate'][-1]))

        # Set training mode ON to inform layers such as Dropout and BatchNorm, 
        # which are designed to behave differently during training and 
        # evaluation. For instance, in training mode, BatchNorm updates a moving 
        # average on each new batch; whereas, for evaluation mode, these updates 
        # are frozen.
        self.train()
        
        if len(self._learn_results['total_iterations']) == 0:
            total_iters = 0
            last_epoch = 0
        else:
            total_iters = self._learn_results['total_iterations'][-1]
            last_epoch = self._learn_results['epoch'][-1]


        # Initialize progress bar to show training progress.
        n_batches = len(train_loader)
        pb_epochs = utils.ProgressBar(iterations = range(epochs*n_batches),
                title = 'TRAINING COLLISION RISK PROBABILITY ESTIMATOR MODEL:')

        for epoch in range(epochs):

            with torch.no_grad():
                for _, (inputs, targets) in enumerate(valid_loader):

                    # Allocate events and event_lengths tensors to the device
                    # defined by the model.
                    inputs = inputs.to(device)
                    targets = targets.to(device)

                    # Get batch_size from event_lengths as it can be smaller for
                    # the last minibatch of an epoch.
                    batch_size = targets.nelement()

                    # Forecast next CDMs of the mini-batch using the inputs. The 
                    # model also requires a second parameter with the number of 
                    # CDMs per event object in order to pack padded sequences to 
                    # optimize computation.
                    outputs = self.forward(inputs)

                    # Compute loss using the criterion and add it to the array.
                    self._loss = self.criterion(outputs, targets)
                    valid_loss = float(self._loss)

            # Iterate over all batches containes in the training loader. Every 
            # batch (t_batch) contains an equal number of events which in 
            # turn may contain a different number of CDM objects.
            for t_batch, (inputs, targets) in enumerate(train_loader):
                total_iters += 1
                relative_iters = (epoch*n_batches + t_batch + 1)

                # Allocate events and event_lengths tensors to the device 
                # defined by the model.
                inputs = inputs.to(device)
                targets = targets.to(device)

                # Get batch_size from event_lengths as it can be smaller for the 
                # last minibatch of an epoch.
                batch_size = targets.nelement() 


                # Clear all the gradients of all the parameters of the model.
                self.optimizer.zero_grad()
                outputs = self.forward(inputs)

                # Compute MSE loss using criterion and store it in an array.
                self._loss = self.criterion(outputs, targets)
                train_loss = float(self._loss)

                # Get learning rate used
                lr = self.optimizer.param_groups[0]['lr']

                # Backpropagate MSE loss.
                self._loss.backward(retain_graph=True)

                # Update model hyperparameters taking into account the loss.
                self.optimizer.step()
                # lr_scheduler.step()

                # Update progress bar.
                description = f'E({epoch+1}/{epochs}) ' + \
                    f'B({t_batch+1}/{n_batches}) | ' + \
                    f'Loss > T({train_loss:6.4e}) ' + \
                    f'V({valid_loss:6.4e})'

                pb_epochs.refresh(i = relative_iters, 
                                  description = description,
                                  nested_progress = True)
                

                # Save training information.
                self._learn_results['total_iterations'].append(total_iters)
                self._learn_results['validation_loss'].append(valid_loss)
                self._learn_results['training_loss'].append(train_loss)
                self._learn_results['epoch'].append(last_epoch + epoch + 1)
                self._learn_results['learning_rate'].append(lr)
                self._learn_results['batch_size'].append(batch_size)

                if epoch_step_checkpoint is not None and \
                   ((epoch+1) % epoch_step_checkpoint) == 0 and \
                    (epoch+1) < epochs:
                    pb_epochs.refresh(i = relative_iters, 
                                  description = 'Saving checkpoint...',
                                  nested_progress = True)
                    self.save(filepath = filepath)
                    pb_epochs.refresh(i = relative_iters, 
                                      description = description,
                                      nested_progress = True)
                    

        # Print message at the end of the mini batch.
        pb_epochs.refresh(i = relative_iters, description = description)

        if filepath is not None:
            print(f'\nSaving model parameters ...', end='\r')
            self.save(filepath = filepath)
            print(f'Saving model parameters ... Done.')

    def save(self, filepath:str, only_parameters:bool=True):
        """Save model to an external file.

        Args:
            filepath (str): Path where the model is saved.
            only_parameters (bool, optional): Save only models parameters or the 
            entire model. Defaults to True.
        """
        if only_parameters:
            torch.save({
                'classification': self.classification,
                'class_weights':self._class_weights,
                'num_params': sum(p.numel() for p in self.parameters()),
                'epochs': self._learn_results['epoch'][-1],
                'model': self.state_dict() if only_parameters else self,
                'optimizer': self.optimizer.state_dict() if only_parameters \
                    else self.optimizer,
                'loss': self._loss,
                'learn_results': self._learn_results,
            },  filepath)
        else:
            torch.save(self, filepath)

    def load(self, filepath:str):
        """Load model instance or model parameters.

        Args:
            filepath (str): Path of the model source file.
        """

        if 'parameters' in filepath:

            # Get the checkpoint file
            torch_file = torch.load(filepath)

            self.load_state_dict(torch_file['model'])
            if not hasattr(self, 'optimizer'):
                self.optimizer = optim.Adam(self.parameters(), lr = 1e-3)
            self.optimizer.load_state_dict(torch_file['optimizer'])
            self._loss = torch_file['loss']
            self._learn_results = torch_file['learn_results']
            if 'classification' in torch_file.keys():
                self.classification = torch_file['classification']
                self._class_weights = torch_file['class_weights']
            else:
                self.classification = False
                self._class_weights = torch.Tensor([0.5, 0.5])
        else:
            self = torch.load(filepath)

    def test(self, data_test:TensorDataset, test_batch_size:int) -> np.ndarray:
        """Compute MSE loss on a test set using the trained model.

        Args:
            data_test (TensorDataset): Conjunction Data Messages dataset.
            test_batch_size (int): Batch size.

        Returns:
            np.ndarray: NumPy array containing the MSE loss values per batch.
        """

        def MAPELoss(output:torch.Tensor, target:torch.Tensor, 
                     epsilon:float=1e-8):

            return torch.mean(torch.abs((target - output) / (target+epsilon)))

        
        if test_batch_size > len(data_test):
            test_batch_size = len(data_test)
            warnings.warn(f'\nParameter test_batch_size ({test_batch_size})'
                          f'can only be less or equal to the number of '
                          f'items on events_set input ({len(data_test)})'
                          f'\nSetting new value for test_batch_size='
                          f'{len(data_test)}.')
        
        # Get loader objects for test set. The DataLoader class works by 
        # creating an iterable dataset object and iterating over it in batches, 
        # which are then fed into the model for processing. The object it 
        # creates has the shape (batches, items) where items are the a number of 
        # elements n = int(len(Dataset)/batch_size) taken from the Dataset 
        # passed into the class.
        test_loader = DataLoader(dataset = data_test, 
                                 batch_size = test_batch_size, 
                                 shuffle = True)


        # Set-up device and criterion.
        device = list(self.parameters())[0].device

        # Get number of parameters
        k = sum(p.numel() for p in self.parameters())

        # Define all criterions required for the regression metrics.
        mae_criterion = nn.L1Loss()
        mse_criterion = nn.MSELoss(reduction='mean')
        sse_criterion = nn.MSELoss(reduction='sum')
        mape_criterion = MAPELoss
        pocid_criterion = POCID

        # Initialize dictionary with the different regression metrics.
        results = {'sse':np.zeros((len(test_loader))),
                   'mse':np.zeros((len(test_loader))),
                   'mae':np.zeros((len(test_loader))),
                   'mape':np.zeros((len(test_loader))),
                   'bic':np.zeros((len(test_loader))),
                   'pocid': np.zeros((len(test_loader)))}

        # Iterate over all items in the test_loader
        with torch.no_grad():
            for t, (inputs, targets) in enumerate(test_loader):

                # Allocate events and event_lengths tensors to the device
                # defined by the model.
                inputs = inputs.to(device)
                targets = targets.to(device)

                # Estimate collision probability risk of the mini-batch using 
                # the inputs.
                outputs = self.forward(inputs)

                # Get test size
                t_sz = len(outputs)

                # Compute regression metrics using the criterion and add it to 
                # the array.
                results['sse'][t] = float(sse_criterion(outputs, targets))
                results['mse'][t] = float(mse_criterion(outputs, targets))
                results['mae'][t] = float(mae_criterion(outputs, targets))
                results['mape'][t] = float(mape_criterion(outputs, targets))
                results['bic'][t] = t_sz*np.log(results['mse'][t])+k*np.log(t_sz)
                results['pocid'][t] = pocid_criterion(outputs, targets)

        return results

    def forward(self, x):

        # Iterate over all modules to perform the forward operation.
        for layer in self.model:
            x = layer(x)
        
        return x
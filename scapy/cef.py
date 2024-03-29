# Libraries used for hinting
from __future__ import annotations
from typing import Union

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
from torch.nn.utils.rnn import pack_padded_sequence, pad_packed_sequence



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
        """Get learning rates.

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
        """Get learning rate factor.

        Args:
            epoch (int): Epoch of training.

        Returns:
            float: Learning rate factor.
        """

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
        """Plot learning rate factor.

        Args:
            warmup_epochs (int): Number of epochs for warm-up.
            max_epochs (int): Maximum number of epochs for LR adaptation.
            figsize (tuple, optional): Figure size. Defaults to (8, 3).
            figtitle (str, optional): Figure title. Defaults to None.
            filepath (str, optional): File path where plot is saved. Defaults to 
            None.
        """
        
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
            fig.savefig(filepath, bbox_inches='tight')


#%% CLASS: ConjunctionEventForecaster
# Define Feature Forecaster module
class ConjunctionEventForecaster(nn.Module):
    """Model instanciator for Conjunction Event forecasting.

    Args:
        nn (class): Base class for all neural networks in Pytorch.
    """

    def __init__(self, network:list, features:Union[list, str] = None) -> None:
        """Initialises the model instanciator.

        Args:
            network (list): List of objects composing the RNN network.
            features (Union[list, str], optional): List of feature names to be
            trained on. Defaults to compulsory CDM features as per defined in
            the CCSDS Recommended standards.
        """
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
        
        """Plot training and/or validation loss.

        Args:
            filepath (str, optional): Path where the plot is saved. Defaults to 
            None.
            figsize (tuple, optional): Size of the plot. Defaults to (6 ,3).
            log_scale (bool, optional): Use logarithmic scale. Defaults to 
            False.
            validation_only (bool, optional): Plot validation loss only. 
            Defaults to False.
            plot_lr (bool, optional): Plot learning rate. Defaults to False.
            label (str, optional): Label of feature plotted ('validation' or 
            'training'). Defaults to None.
            ax (plt.Axes, optional): Axis object. Defaults to None.
            return_ax (bool, optional): Return axis object. Defaults to False.

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

        max_loss = max(max(self._learn_results[f'training_loss']),
                       max(self._learn_results[f'validation_loss']))
        
        ax.set_ylim(0, max_loss*1.05)

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
        if not validation_only: ax.legend(fontsize=8)
        ax.grid(True, linestyle='--')

        # Save figure if filepath is provided.
        if filepath is not None:
            print('Plotting to file: {}'.format(filepath))
            fig.savefig(filepath)

        if return_ax:
            return ax

    def learn(self, event_set:list, epochs:int = 2, lr:float = 1e-3, 
              batch_size:int = 8, device:torch.device = torch.device('cpu'), 
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
            device (torch.device, optional): Device where tensors are allocated. 
            Defaults to 'cpu'.
            valid_proportion (float, optional): Proportion of data used for 
            validation (value must be between 0 and 1). Defaults to 0.15.
            num_workers (int, optional): Parallel processes for data loading. 
            Defaults to 4.
            event_samples_for_stats (int, optional): Number of events considered 
            to compute the mean and standard deviation used for normalisation. 
            Defaults to 250.
            filepath (str, optional): Filepath where the model parameters shall
            be saved. If None, parameters are not saved. Defaults to None.
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

        # Print message at the end of the epochs training.
        pb_epochs.refresh(i = relative_iters, 
                          nested_progress = False,
                          description = f'Validation loss = {valid_loss:6.4e}')

        if filepath is not None:
            print(f'\nSaving model parameters ...', end='\r')
            self.save(filepath = filepath)
            print(f'Saving model parameters ... Done.')

    @staticmethod
    def _get_filepath(filepath:str) -> str:
        """Get filepath from file. If only filename is provided, parameters
        are saved by default in ../scapy/models/parameters/cef folder.

        Args:
            filepath (str): File path or name.

        Returns:
            str: Absolute filepath.
        """

        # Check if filepath is actually only the name of the file
        if len(filepath.split('/'))==1:
            folderpath = os.path.join(utils.cwd, 'models', 'parameters', 'cef')

            # Create local directory if it does not exist already.
            utils.mkdirtree(folderpath)

            filepath =  os.path.join(folderpath, filepath)


        return filepath
            
    def save(self, filepath:str, only_parameters:bool=True):
        """Save model to an external file.

        Args:
            filepath (str): Path where the model is saved.
            only_parameters (bool, optional): Save only models parameters or the 
            entire model. Defaults to True.
        """

        filepath = self._get_filepath(filepath)

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

        filepath = self._get_filepath(filepath)

        # Check if filepath exists.
        if not os.path.exists(filepath):
            print(f'Model not found in {os.path.relpath(filepath)}')
        else:
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
            
            if not hasattr(self, '_device'):
                self._device = torch.device('cpu')

    
    def test(self, event_set:CED, test_batch_size:int, 
             num_workers:int = 4, as_dataframe:bool=False) -> Union[dict, pd.DataFrame]:
        """Compute loss on a test set.

        Args:
            event_set (ConjunctionEventsDataset): Conjunction Events Dataset
            object.
            test_batch_size (int): Batch size.
            num_workers (int, optional): Parallel processes for data loading. 
            Defaults to 4.
            as_dataframe (bool, optional): Return results as pandas Dataframe 
            (True) or as a dictionary (False). Defaults to False.

        Returns:
            Union[dict, pd.DataFrame]: Dictionary or DataFrame containing the 
            test results for the performance metrics in the format key:results. 
            Every performance metric (key) contains an array of size equal to 
            test_batch_size with the results.
        """

        # Get test dataset with normalized features using the stats metrics. 
        test_set = DatasetEventDataset(event_set = event_set, 
                                       features = self._features, 
                                       features_stats = self._features_stats)
        
        if test_batch_size > len(event_set):
            test_batch_size = len(event_set)
            warnings.warn(f'\nParameter test_batch_size ({test_batch_size})'
                          f'can only be less or equal to the number of '
                          f'items on events_set input ({len(event_set)})'
                          f'\nSetting new value for test_batch_size='
                          f'{len(event_set)}.')
        
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
        criterion = {'mae': nn.L1Loss(),
                    'mse': nn.MSELoss(reduction='mean'),
                    'sse': nn.MSELoss(reduction='sum'),
                    'mape': utils.mape,
                    'pocid': utils.pocid}

        # Initialize list of labels with the different regression metrics.
        metrics = ['sse', 'mse', 'mae', 'mape', 'bic', 'pocid']

        # Initialise arrays to store the results
        results = {m:np.zeros((len(test_loader))) for m in metrics}

        # Initialize progress bar to show training progress.
        n_batches = len(test_loader)
        pb_batches = utils.ProgressBar(iterations=range(n_batches),
                                      title = 'TESTING FORECASTING MODEL:')

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
                for m, function in criterion.items():
                    results[m][t] = float(function(output, target))

                results['bic'][t] = t_sz*np.log(results['mse'][t]) + \
                                    k*np.log(t_sz)
                
                # Update progress bar.
                mean_loss = np.mean(results["mse"][:t+1])
                description =  f'MSE Loss = {mean_loss:6.4e} (mean)'

                pb_batches.refresh(i = t+1, 
                                  description = description,
                                  nested_progress = True)
                
        if as_dataframe:
            return pd.DataFrame.from_dict(data = results, orient='columns')
        else:
            return results

    def predict(self, event: CE) -> CDM:
        """Predict next CDM object from a given event.

        Args:
            event (ConjunctionEvent): Conjunction Event object containing CDM(s) 
            object(s).

        Raises:
            RuntimeError: Invalid output from the model.

        Returns:
            ConjunctionDataMessage: Conjunction Data Message object.
        """

        ds = DatasetEventDataset(CED(events=[event]), 
                                 features = self._features, 
                                 features_stats = self._features_stats)
        
        # Get CDM objects and number of CDMs contained in the dataset taking the 
        # event passed as a parameter.
        inputs, inputs_length = ds[0]

        # Allocate tensors to the device
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
        """Predict remaining CDMs of a given Conjunction Event.

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
        """Reset hidden and cell states for all the RNN layers.

        Args:
            batch_size (int): Batch size.
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
        """Predict new CDM containing normalised values.

        Args:
            x (torch.FloatTensor): Tensor with shape (n_events, 
            max_event_length, features) containing the input values for RNN 
            processing.
            x_lengths (torch.IntTensor): Tensor with shape (n_events, 1) 
            containing the number of CDMs contained in every event. This tensor 
            is used to unpack padded tensor.

        Returns:
            torch.Tensor: Tensor with shape (n_events, max_event_length, 
            features) containing normalised values of the new CDM.
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
                
                x, self.hidden[module_name]= module(x, self.hidden[module_name])
                
                # # Pads a packed batch of variable length sequences from LSTM 
                # # layer.
                # x, _ = pad_packed_sequence(sequence = x, 
                #                            batch_first = module.batch_first, 
                #                            total_length = x_length_max)
                
            else:
                x = module(x)

        return x
    

#%% Initialise model object as a shortcut
model = ConjunctionEventForecaster

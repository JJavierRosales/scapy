
# Libraries used for hinting
from __future__ import annotations
from typing import Union

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset

import os
import numpy as np
import pandas as pd
from matplotlib import pyplot as plt
import warnings

from . import utils


#%% CLASS: CollisionRiskEvaluator
class CollisionRiskEvaluator(nn.Module):
    """Model instanciator for Conjunction Event forecasting.

    Args:
        nn (class): Base class for all neural networks in Pytorch.
    """

    def __init__(self, input_size:int, output_size:int, layers:list, 
                 bias:Union[bool, list] = True, 
                 act_functions:torch.nn = nn.ReLU(), 
                 dropout_probs:Union[float, list] = 0.2, 
                 classification:bool=False, 
                 class_weights:torch.Tensor = torch.tensor([0.5, 0.5])):
        """Initialises the model instanciator.

        Args:
            input_size (int): Number of input features.
            output_size (int): Number of output features.
            layers (list): List of neurons per hidden layer.
            bias (Union[bool, list], optional): Include bias in hidden layer(s).
            If a list is passed, it needs to have the same length as layers 
            parameter. Defaults to True.
            act_functions (torch.nn, optional): Activation function per hidden 
            layer. If a list is passed, it needs to have the same length as 
            layers parameter. Defaults to nn.ReLU().
            dropout_probs (Union[float, list], optional): Probability dropout. 
            If a list is passed, it needs to have the same length as layers 
            parameter. Defaults to 0.2.
            classification (bool, optional): Classification mode: True for risk 
            classification, False for probability risk regression. Defaults to 
            False.
            class_weights (torch.Tensor, optional): Classification weights to 
            compensate skewness of dataset no risk events vs risk. Defaults to 
            torch.tensor([0.5, 0.5]) (equal relevance).

        Raises:
            ValueError: act_functions passed as a list but does not contain the 
            same number of items as layers parameters.
            ValueError: dropout_probs passed as a list but does not contain the 
            same number of items as layers parameters.
        """
        
        # Inherit attributes from nn.Module class
        super().__init__()

        # Initialise input and output sizes
        self.input_size = input_size
        self.output_size = output_size
        self.layers = layers
        self.classification = classification
        self._class_weights = class_weights

        if isinstance(act_functions,list):
            if len(act_functions) < len(layers):
                raise ValueError(f'Length of act_functions parameter shall ' + \
                                 f'match the length of layers.')
            
        if isinstance(dropout_probs,list):
            if len(dropout_probs) < len(layers):
                raise ValueError(f'Length of dropout_probs parameter shall ' + \
                                 f'match the length of layers.')

        self._act_functions = act_functions
        self._dropout_probs = dropout_probs

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
            af = act_functions[l] if isinstance(act_functions, list) \
                                  else act_functions
            if af is not None: self.model.append(af)
            
            # 3. Normalize data using the n_neurons
            self.model.append(nn.BatchNorm1d(hidden_neurons))
            
            # 4. Cancel out a random proportion p of the neurons to avoid 
            # overfitting
            p = dropout_probs[l] if isinstance(dropout_probs, list) \
                                 else dropout_probs
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
        """Plot loss in the training set (orange) and validation set (blue) 
        versus the number of iterations during model training.

        Args:
            filepath (str, optional): Path where the plot is saved. Defaults to 
            None.
            figsize (tuple, optional): Size of the plot. Defaults to (6 ,3).
            log_scale (bool, optional): Use logarithmic scale. Defaults to 
            False.
            validation_only (bool, optional): Plot validation loss only. 
            Defaults to False.
            plot_lr (bool, optional): Plot learning rate. Defaults to False.
            label (str, optional): Label of feature plotted. Defaults to None.
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
        
        if not self.classification:
            ax.set_ylabel('MSE Loss')
        else:
            ax.set_ylabel('Cross Entropy Loss')

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
              batch_size:int = 8, device:torch.device = torch.device('cpu'), 
              valid_proportion:float = 0.15, filepath:str = None,
              epoch_step_checkpoint:int = None, 
              **kwargs) -> None:
        """Train ANN model.

        Args:
            data (TensorDataset): Dataset of tensors with inputs and targets 
            (including the validation dataset).
            epochs (int, optional): Number of epochs used for training. Defaults 
            to 10.
            lr (float, optional): Learning rate. Defaults to 1e-3.
            batch_size (int, optional): Batch size. Defaults to 8.
            device (torch.device, optional): Device where tensors are allocated. 
            Defaults to 'cpu'.
            valid_proportion (float, optional): Proportion of data used for 
            validation (value must be between 0 and 1). Defaults to 0.15.
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
        """Compute loss on a test set of events using the trained model.

        Args:
            data_test (TensorDataset): Tensor dataset for testing with inputs 
            and targets.
            test_batch_size (int): Batch size.

        Returns:
            dict: Dictionary containing the test results for the performance
            metrics in the format key:results. Every performance metric (key) 
            contains an array of size equal to test_batch_size with the results.
        """

        
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
        
        # Get criterion depending on the type of ANN model
        if self.classification:
            # Initialise confusion matrix function
            bcm_criterion = utils.binary_confusion_matrix
            
            # Initialize list of labels with the different classification metrics.
            metrics = ['tp', 'fn', 'fp', 'tn', 'accuracy', 'precision', 
                      'recall', 'f1', 'aucroc']
            
        else:
            # Define all criterions required for the regression metrics.
            reg_criterion = {'mae': nn.L1Loss(),
                             'mse': nn.MSELoss(reduction='mean'),
                             'sse': nn.MSELoss(reduction='sum'),
                             'mape': utils.mape,
                             'pocid': utils.pocid}

            # Initialize list of labels with the different regression metrics.
            metrics = ['sse', 'mse', 'mae', 'mape', 'bic', 'pocid']

        # Initialise arrays to store the results
        results = {m:np.zeros((len(test_loader))) for m in metrics}

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

                if self.classification:

                    # Compute confusion matrix results
                    bcm_results = bcm_criterion(outputs, targets)

                    # Add all results from the criterion
                    for m, r in bcm_results.items():
                        results[m][t] = r

                    results['aucroc'][t]= utils.binary_auc_roc(outputs, targets)

                else:
                    # Compute regression metrics using the criterion and add it 
                    # to the array.
                    for m, criterion in reg_criterion.items():
                        results[m][t] = float(criterion(outputs, targets))

                    results['bic'][t] = t_sz*np.log(results['mse'][t]) + \
                                        k*np.log(t_sz)

        return results

    def forward(self, x:torch.Tensor) -> torch.Tensor:
        """Produce risk evaluation: conunction classification or collision risk 
        probability estimation.

        Args:
            x (torch.FloatTensor): Tensor with shape (batch_size, features) 
            containing the input values for ANN processing.

        Returns:
            torch.Tensor: Tensor with shape (batch_size, features) containing 
            the predicted risk classification or risk probability.
        """

        # Iterate over all modules to perform the forward operation.
        for layer in self.model:
            x = layer(x)
        
        return x

#%% Initialise model object as a shortcut    
model = CollisionRiskEvaluator
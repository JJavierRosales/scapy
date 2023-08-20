# Libraries used for hinting
from __future__ import annotations
from typing import Union

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from torch.nn.utils.rnn import pack_padded_sequence, pad_packed_sequence

import numpy as np
from matplotlib import pyplot as plt
import warnings

from . import utils
from .event import ConjunctionEvent as CE
from .event import ConjunctionEventsDataset as CED
from .cdm import ConjunctionDataMessage as CDM

#%% CLASS: LSTMLayer
class LSTMLayer(nn.Module):
    """Layer constructor for LSTM based RNN architecture.
    """

    def __init__(self, cell, input_size:int, hidden_size:int, **cell_args:dict):
        """Initialize LSTM cell.

        Args:
            cell (constructor): LSTM cell constructor.
            input_size (int): Number of inputs.
            hidden_size (int): Number of hidden cells (outputs of the cell).
            *cell_args (dict, optional): Dictionary containing optional 
            arguments required for the LSTM cell constructor (i.e. SLIMX 
            constructor receives the additional parameter 'version').
        """
    
        super(LSTMLayer, self).__init__()

        # Initialize inputs and hidden sizes
        self.input_size = input_size
        self.hidden_size = hidden_size
        
        # Initialize cell attribute in class witht the LSTM cell object 
        # initialized.
        
        self.cell = cell(input_size = input_size, 
                         hidden_size = hidden_size, 
                         **cell_args)

    def forward(self, input: torch.TensorFloat, state:tuple) -> tuple:
        """Forward operation through all time steps of a given input.

        Args:
            input (torch.TensorFloat): Tensor containing the values at every 
            time step of a sequence.
            state (tuple): Tuple of tensors containing previous hidden state and
            cell state (at time t-1) required to produce the next output.

        Returns:
            tuple: Tuple with two tensors:
                - outputs (torch.TensorFloat): Forecasted values of X for the 
                    next time step (ht ~ Xt+1) 
                - states (tuple): Tuple containing  two tensors: one with the 
                    last hidden state (predicted Xt+1 value) and cell state 
                    (cell context) required to produce the next prediction.
        """

        
    
        # Remove tensor dimension from inputs.
        inputs = input.unbind(0)
        outputs = []

        # Iterate over all the different time steps of a given sequence (inputs).
        for x_t in inputs:

            # Forecast Xt value t+1 (hidden state - ht) and the cell state 
            # holding the cell "configuration parameters" for the next time step.
            out, state = self.cell(x_t, state)
            outputs += [out]

        # Return the list of outputs produced by the LSTM cell and the last 
        # hidden states at time t (hidden_state, cell_state).

        # Return predicted values at every time step (ht ~ Xt+1) and the last 
        # hidden state (cell configuration/context) from the sequence. Only the 
        # last hidden states are returned because they are the most relevant in 
        # terms of learning (learnt from the entire sequence of inputs).
        return torch.stack(outputs), state
      
        
#%% CLASS: LSTM
class LSTM(nn.Module):
    """Adapted nn.LSTM class that allows the use of custom LSTM cells.
    """
    def __init__(self, input_size:int, hidden_size:int, cell, 
        batch_first:bool=True, num_layers:int=1, 
        dropout:float=None, **cell_args:dict) -> None:
        """Initialize adapted LSTM class.

        Args:
            input_size (int): Number of input features.
            hidden_size (int): Number of hidden neurons (outputs of the LSTM).
            cell (constructor): LSTM cell constructor.
            num_layers (int, optional): Number of stacked LSTM layers (LSTM 
            depth). Defaults to 1.
            dropout (float, optional): Dropout probability to use 
            between consecutive LSTM layers. Only applicable if num_layers is 
            greater than 1. Defaults to None.
        """
    
        super(LSTM, self).__init__()
        
        # Get all LSTM layers in a list using the LSTMLayer constructor.
        layers = [LSTMLayer(cell = cell, 
                            input_size = input_size, 
                            hidden_size = hidden_size, 
                            **cell_args)] + \
                 [LSTMLayer(cell = cell, 
                            input_size = hidden_size, 
                            hidden_size = hidden_size, 
                            **cell_args) for _ in range(num_layers - 1)]
        
        # Set network dimensions
        self.input_size = input_size
        self.hidden_size = hidden_size
        
        # Set batch_first parameter
        self.batch_first = batch_first
        self._batched = None
        
        # Convert list of LSTM layers to list of nn.Modules.
        self.layers = nn.ModuleList(layers)
        
        # Introduces a Dropout layer on the outputs of each LSTM layer except
        # the last layer.
        self.num_layers = num_layers

        # If number of LSTM layers is 1 and the dropout_probability provided is
        # not None, print warning to the user.
        if num_layers == 1 and dropout > 0:
            warnings.warn(
                "Dropout parameter in LSTM class adds dropout layers after " 
                "all but last recurrent layer. It expects num_layers greater "
                "> 1, but got num_layers = 1."
            )
        
        # If dropout_probability is provided initialize Dropout Module. 
        if not dropout is None and num_layers > 1:
            self.dropout_layer = nn.Dropout(p = dropout)
        else:
            self.dropout_layer = None


    def forward(self, input: torch.Tensor, states: torch.Tensor) -> tuple:
        """Forward operation through all time steps of a given input and all 
        LSTM layers.

        Args:
            input (torch.Tensor): Tensor of shape (seq_length, hidden_size) 
            if unbatched, (batch_size, seq_length, hidden_size) if batched and 
            batch_first = True, or (seq_length, batch_size, hidden_size) if 
            batched and batch_first = False. It containins the values at 
            every time step of a sequence.
            states (torch.Tensor): Tensor of shape (num_layers, hidden_size) 
            if input is unbatched, or (num_layers, batch_size, hidden_size) if 
            batched. It contains tuples of tensors of every LSTM layer. Every 
            tuple contains two tensors for the previous hidden state and cell 
            state (at time t-1) required to produce the next output for the 
            layer.

        Returns:
            tuple: Tuple with two values:
                - outputs (torch.Tensor): Tensor of shape (seq_length, 
                hidden_size) for unbatched input, (batch_size, seq_length, 
                hidden_size) for batched input if batch_first = True, or 
                (seq_length, batch_size, hidden_size) for batched input if 
                batch_first = False. It contains the forecasted values of X for 
                the next time step (ht ~ Xt+1).
                - states (tuple): Tuple with two tensors: one with the last 
                hidden state (predicted Xt+1 value) and cell state (cell 
                context) required to produce the next prediction of the layer.
        """

        if (self._batched is None) and \
            (isinstance(input, torch.nn.utils.rnn.PackedSequence) or \
            (isinstance(input, torch.Tensor) and len(input.size())==3)):
            self._batched = True
    
        if self._batched:
            # Input tensor is batched
            if self.batch_first:
                batch_size, seq_length, _ = input.size()
            else:
                seq_length, batch_size, _ = input.size()
        else:
            # Input tensor is unbatched.
            seq_length, _ = input.size()

        

        # Initialize list to store a tuple per layer containing the hidden and 
        # cell states required for the next output prediction
        output_hstates = []
        output_cstates = []

        # Initialize input layer tensor (tensor to be passed first to the layer
        # whose input_size is the input number of features)
        input_layer = input

        for i, layer in enumerate(self.layers):

            # Get the hidden states tensor at layer i:
            #  - batched=True -> (num_layers, batch_size, hidden_size)
            #  - batched=False -> (num_layers, hidden_size)
            #state = states[i]
            hstate = states[0][i]
            cstate = states[1][i]
            
            # If inputs are batched, iterate over all batches.
            if self._batched:

                # Initialize output_layer with the dimensions of the outputs 
                # from the layer.
                if self.batch_first:
                    output_layer = torch.zeros((batch_size, 
                                                seq_length, 
                                                layer.hidden_size))
                else:
                    output_layer = torch.zeros((seq_length,
                                                batch_size,  
                                                layer.hidden_size))

                # Initialize tensor to keep the states returned at the end of 
                # the sequence processing in every batch b.
                out_hstate = torch.zeros((batch_size, layer.hidden_size))
                out_cstate = torch.zeros((batch_size, layer.hidden_size))

                # Iterate through all batches- to get the output and states per 
                # batch.
                for b in range(batch_size):

                    # Get the batch input tensor b for the layer i depending on 
                    # the shape of the input tensor.
                    batch_input = input_layer[b, :, :] if self.batch_first \
                        else torch.squeeze(input_layer[:, b, :], dim = 1)
                    
                    # Get the batch states for the layer i
                    batch_state = (torch.squeeze(hstate[b]), 
                                   torch.squeeze(cstate[b]))

                    # Get the output and states of the layer for the batch b
                    batch_output, batch_state = layer.forward(batch_input, 
                                                              batch_state)

                    # Save batch output depending on the inputs shape.              
                    if self.batch_first:
                        output_layer[b, :, :] = batch_output
                    else:
                        output_layer[:, b, :] = batch_output

                    # Keep last cell state of the sequence for batch b and i 
                    # layer.
                    out_hstate[b] = batch_state[0]
                    out_cstate[b] = batch_state[1]

            else:

                # Keep last cell state of sequence
                output_layer, (out_hstate, out_cstate) = layer(input_layer, 
                                                               (hstate, cstate))

                
            # Apply the dropout layer except the last layer
            if (i < self.num_layers - 1) and not (self.dropout_layer is None):
                output_layer = self.dropout_layer(output_layer)

            # Redefine input_layer tensor from the output of previous layer.
            input_layer = output_layer

            # Add last states from squences (and from all batches if input is 
            # batched) to final list.    
            output_hstates += [out_hstate]
            output_cstates += [out_cstate]

        # Convert lists to tensors 
        hstates = torch.stack(output_hstates) if self.num_layers > 1 \
                                else output_hstates[0]
        cstates = torch.stack(output_cstates) if self.num_layers > 1 \
                                else output_cstates[0]
        
        # Return the outputs of the LSTM and the hidden states for every LSTM 
        # layer.
        return output_layer, (hstates, cstates)
        
        
#%% CLASS: DatasetEventDataset
class DatasetEventDataset(Dataset):
    def __init__(self, event_set:list, features:list, 
                 features_stats:dict = None) -> None:
                 
        # Initialize the list of events.
        self._event_set = event_set
        
        # Get the maximum number of CDMs stored in a single conjunction event. 
        # This internal variable will be used to pad the torch of every event 
        # with zeros, that is, empty CDM objects will be added to Events with 
        # less CDMs that max_event_length.
        self._max_event_length = max(map(len, self._event_set))
        self._features = features
        self._features_length = len(features)
        
        # Compute statistics for every feature if not already passed
        # into the class
        if features_stats is None:
            # Cast list of events objects to pandas DataFrame.
            df = event_set.to_dataframe()
            
            # Get list of features containing any missing value.
            null_features = df.columns[df.isnull().any()]
            for feature in features:
                if feature in null_features:
                    raise RuntimeError('Feature {} is not present in the ' + \
                                       'dataset'.format(feature))
                    
            # Convert feature data to numpy array to compute statistics.
            features_numpy = df[features].to_numpy() 
            self._features_stats = {'mean': features_numpy.mean(0), 
                                    'stddev': features_numpy.std(0)}
        else:
            self._features_stats = features_stats

    def __len__(self) -> int:
        return len(self._event_set)

    def __getitem__(self, i:int) -> tuple:
        """Get item from Events Dataset. 

        Args:
            i (int): Index of item to retrieve (CDM index).

        Returns:
            tuple: Tuple containing two tensors:
             - First item contains the feature values normalized. Dimensions 
                vary depending on the method used to retrieve the data:
                  + Single item (from Dataset): (max_event_length, features)
                  + Batch of items (DataLoader): (batch_size, max_event_length, 
                  features)
             - Second item stores the number of the CDM objects the event 
                contains. This item is required for packing padded torch and 
                optimize computing. Dimensions  vary depending on the method 
                used to retrieve the data:
                  + Single item (from Dataset): (max_event_length, 1)
                  + Batch of items (DataLoader): (batch_size, max_event_length, 1)
        """
        
        # Get event object from the set.
        event = self._event_set[i]
        
        # Initialize torch with zeros and shape (max_event_length, n_features).
        # This torch forces all events to have the same number of CDMs by using
        # the internal variable _max_event_length. It basically creates a padded
        # torch, which helps to do batch processing with the DataLoader class.
        x = torch.zeros(self._max_event_length, self._features_length)
        
        # Iterate over all CDM objects in the event i and apply standard 
        # normalization (x - mean)/(std + epsilon). Note: A constant epsilon
        # is introduced to remove value errors caused by division by zero.
        epsilon = 1e-8
        for i, cdm in enumerate(event):

            # Initialize list to store normalized values for every feature in a
            # given CDM.
            norm_values = []

            for j, feature in enumerate(self._features):
                # Get mean and standard deviation per feature.
                feature_mean = self._features_stats['mean'][j]
                feature_stddev = self._features_stats['stddev'][j]
                
                # Append normalized values from feature to the list.
                norm_values += [(cdm[feature] - feature_mean)/
                                (feature_stddev + epsilon)]
                
            # Add normalized values from all features to the output tensor.
            x[i] = torch.tensor(norm_values)

        return x, torch.tensor(len(event))
    
#%% CLASS: ConjunctionEventForecaster
# Define Feature Forecaster module
class ConjunctionEventForecaster(nn.Module):
    def __init__(self, layers:list, features:Union[list, str] = None) -> None:
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
        self.model = layers

        self._features = features
        self._features_stats = None
        self._hist_train_loss = []
        self._hist_train_loss_iters = []
        self._hist_valid_loss = []
        self._hist_valid_loss_iters = []

    def plot_loss(self, filepath:str = None, figsize:tuple = (6, 3), 
                  log_scale:bool = False) -> None:
        """Plot RNN loss in the training set (orange) and validation set (blue) 
        vs number of iterations during model training.

        Args:
            filepath (str, optional): Path where the plot is saved. Defaults to 
            None.
            figsize (tuple, optional): Size of the plot. Defaults to (6 ,3).
            log_scale (bool, optional): Flag to plot Loss using logarithmic 
            scale. Defaults to False.
        """

        # Apply logarithmic transformation if log_scale is set to True. This 
        # helps to see the evolution when variations between iterations are 
        # small.
        train_loss = np.log(self._hist_train_loss) if log_scale \
                     else self._hist_train_loss
        valid_loss = np.log(self._hist_valid_loss) if log_scale \
                     else self._hist_valid_loss
        
        # Initialize plot object.
        fig, ax = plt.subplots(figsize = figsize)

        # Plot training loss vs iterations.
        ax.plot(self._hist_train_loss_iters, train_loss, 
                label='Training', color='tab:orange')
        
        # Plot validation loss vs iterations.
        ax.plot(self._hist_valid_loss_iters, valid_loss, 
                label='Validation', color='tab:blue')
        
        # Set X-axis limits.
        ax.set_xlim(0, max(self._hist_train_loss_iters))

        # Set axes labels.
        ax.set_xlabel('Number of iterations')
        ax.set_ylabel('MSE Loss')

        # Set legend and grid for better visualization.
        ax.legend(fontsize=8)
        ax.grid(True, linestyle='--')

        # Save figure if filepath is provided.
        if filepath is not None:
            print('Plotting to file: {}'.format(filepath))
            plt.savefig(filepath)

    def learn(self, event_set:list, epochs:int = 2, lr:float = 1e-3, 
              batch_size:int = 8, device:str = None, 
              valid_proportion:float = 0.15, num_workers:int = 4, 
              event_samples_for_stats:int = 250, filename_prefix:str = None, 
              **kwargs) -> None:
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
            filename_prefix (str, optional): Prefix of the training check point 
            (model saved at the end of every epoch). If None, no checkpoint is 
            created. Defaults to None.

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
        print('LSTM predictor with params: {:,}'.format(num_params))

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
        optimizer = optim.Adam(self.parameters(), lr = lr)
        criterion = nn.MSELoss()

        # Set training mode ON to inform layers such as Dropout and BatchNorm, 
        # which are designed to behave differently during training and 
        # evaluation. For instance, in training mode, BatchNorm updates a moving 
        # average on each new batch; whereas, for evaluation mode, these updates 
        # are frozen.
        self.train()
        
        if len(self._hist_train_loss_iters) == 0:
            total_iters = 0
        else:
            total_iters = self._hist_train_loss_iters[-1]

        # Initialize progress bar to show training progress.
        n_batches = len(train_loader)
        pb_epochs = utils.ProgressBar(iterations=range(epochs*n_batches),
                                      title = 'FORECASTING MODEL TRAINING:')

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
                    loss = criterion(output, target)
                    valid_loss = float(loss)
                    self._hist_valid_loss.append(valid_loss)
                    self._hist_valid_loss_iters.append(total_iters)
                    
            # Iterate over all batches containes in the training loader. Every 
            # batch (i_minibatch) contains an equal number of events which in 
            # turn may contain a different number of CDM objects.
            for i_minibatch, (events, event_lengths) in enumerate(train_loader):
                total_iters += 1
                relative_iters = (epoch*n_batches + i_minibatch + 1)

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
                optimizer.zero_grad()
                output = self.forward(inputs, event_lengths)

                # Compute MSE loss using criterion and store it in an array.
                loss = criterion(output, target)
                train_loss = float(loss)
                self._hist_train_loss.append(train_loss)

                # Backpropagate MSE loss.
                loss.backward()

                # Update model hyperparameters taking into account the loss.
                optimizer.step()

                # Convert loss from the training dataset to numpy and store it.
                self._hist_train_loss_iters.append(total_iters)

                # Update progress bar.
                description = f'Minibatch {i_minibatch+1}/{n_batches} | ' + \
                    f'Loss -> Train = {train_loss:6.4e} : ' + \
                    f'Valid = {valid_loss:6.4e}'

                pb_epochs.refresh(i = relative_iters, 
                                  description = description,
                                  nested_progress = True)
                
            # If filename_prefix is provided, save check point at the end of the
            # epoch.
            if filename_prefix is not None:
                filename = filename_prefix + '_epoch_{}'.format(epoch+1)
                pb_epochs.refresh(i = relative_iters, 
                                  description = f'Saving model checkpoint' + \
                                                f' to file: {filename} ...',
                                  nested_progress = True)
                self.save(filename)

        # Print message at the end of the mini batch.
        pb_epochs.refresh(i = relative_iters, description = description)

    @staticmethod      
    def _get_lr(optimizer) -> float:
        for param_group in optimizer.param_groups:
            return param_group['lr']

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
        max_length:int = 22) -> Union[CE, CED]:
        """Forecast the evolution of a given Conjunction Event by predicting 
        upcoming CDMs until TCA.

        Args:
            event (ConjunctionEvent): Conjunction Event to forecast.
            num_samples (int, optional): Number of possible CDMs considered in 
            every forecasting step. Defaults to 1.
            max_length (int, optional): Maximum number of CDM objects contained 
            in the event object. Defaults to 22.

        Returns:
            Union[ConjunctionEvent, ConjunctionEventsDataset]: Two possible 
            outputs are returned depending on the parameter num_samples:
             - If num_samples = 1: Returns one ConjunctionEvent object with all 
                CDMs forecasted. 
             - If num_samples > 1: Returns ConjunctionEventsDataset object 
                containing all possible evolutions of the event (combinations of 
                CDMs).
        """

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

    def save(self, filepath:str) -> None:
        """Save model to an external file.

        Args:
            filepath (str): Path where the model is saved.
        """
        print('Saving LSTM predictor to file: {}'.format(filepath))
        torch.save(self, filepath)

    @staticmethod
    def load(filepath:str):
        print('Loading LSTM predictor from file: {}'.format(filepath))
        return torch.load(filepath)

    def reset(self, batch_size:int):
        """Initialize hidden state (h) and cell state (c) for all the LSTM 
        layers in the Sequencial object.
        """
        # Initialize dictionary for hidden states tuple
        self.hidden = {}

        for module_name, module in self.model.items():
            if not 'lstm' in module_name: continue
            
            h = torch.zeros(module.num_layers, batch_size, module.hidden_size)
            c = torch.zeros(module.num_layers, batch_size, module.hidden_size)

            h = h.to(self._device)
            c = c.to(self._device)

            self.hidden[module_name] = (h, c)

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
            
            if 'lstm' in module_name:

                # Get size of inputs tensor.
                batch_size, x_length_max, n_features = x.size()

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


#%% CLASS: CollisionRiskProbabilityEstimator
# Define Collision Risk Probability Estimator
class CollisionRisk(nn.Module):

    def __init__(self, emb_szs, n_cont, out_sz, layers, p=0.5):
        
        # Inherit attributes from nn.Module class
        super().__init__()
        
        ########################################################################
        # Instanciate functions to use on the forward operation:
        
        # self.embeds: Creates a list of pre-configured Embedding operations (it 
        # is configured by passing the number of categories ni and the length of
        #  the embedding nf)
        self.embeds = nn.ModuleList([nn.Embedding(ni, nf) for ni,nf in emb_szs])
        
        # self.emb_drop: Cancels a proportion p of the embeddings.
        self.emb_drop = nn.Dropout(p)
        
        # self.bn_cont = Normalizes continuous features. This function is 
        # configured by passing the number of continuous features to normalize.
        self.bn_cont = nn.BatchNorm1d(n_cont)
        
        ########################################################################
        # Count total number of embeddings (Total number of vector components 
        # for every feature)
        n_emb = sum((nf for ni,nf in emb_szs))
        
        # Compute total number of inputs to pass to the initial layer (data 
        # point = Nb. of embeddings + Nb. of continuous variables)
        n_in = n_emb + n_cont
        
        # Run through every layer to set up the operations to perform per layer.
        # (i.e. layers=[100, 50, 200])
        layerlist = []
        for l, n_neurons in enumerate(layers):
            # On layer l, which contains n_neurons, perform the following 
            # operations:
            # 1. Apply Linear neural network model regression (fully connected 
            # network -> z = Sum(wi*xi+bi))
            layerlist.append(nn.Linear(n_in,n_neurons))
            
            # 2. Apply ReLU activation function (al(z))
            layerlist.append(nn.ReLU(inplace=True))
            
            # 3. Normalize data using the n_neurons
            layerlist.append(nn.BatchNorm1d(n_neurons))
            
            # 4. Cancel out a random proportion p of the neurons to avoid 
            # overfitting
            layerlist.append(nn.Dropout(p))
            
            # 5. Set new number of input features n_in for the next layer l+1.
            n_in = n_neurons
        
        # Set the last layer of the list which corresponds to the final output
        layerlist.append(nn.Linear(layers[-1],out_sz))
        
        # Instantiate layers as a Neural Network sequential task
        self.layers = nn.Sequential(*layerlist)
    
    def forward(self, x_cat, x_cont):
        # Initialize embeddings list
        embeddings = []
        
        # Apply embedding function e from self.embeds to the category i
        # in x_cat array
        for i,e in enumerate(self.embeds):
            embeddings.append(e(x_cat[:,i]))
        
        # Concatenate embedding sections into 1
        x = torch.cat(embeddings, 1)
        
        # Apply dropout function to the embeddings torch
        x = self.emb_drop(x)
        
        # Normalize continuous variables
        x_cont = self.bn_cont(x_cont)
        
        # Concatenate embeddings with continuous variables into one torch
        x = torch.cat([x, x_cont], 1)
        
        # Process all data points with the layers functions (sequential of 
        # operations)
        x = self.layers(x)
        
        return x
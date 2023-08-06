# Libraries used for hinting
from __future__ import annotations
from typing import Type, Union

import os
import numpy as np
import pandas as pd
import random

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
import matplotlib.pyplot as plt
import sys

from . import utils
from .cdm import ConjunctionDataMessage as CDM
from .event import ConjunctionEvent, ConjunctionEventsDataset


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

    # 
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
            for j, feature in enumerate(self._features):
                # Get mean and standard deviation per feature.
                feature_mean = self._features_stats['mean'][j]
                feature_stddev = self._features_stats['stddev'][j]
                
                # Add normalzied feature to the tensor.
                x[i] = torch.tensor([(cdm[feature]-feature_mean)/\
                                     (feature_stddev+epsilon)])

        return x, torch.tensor(len(event))


class ConjunctionEventForecaster(nn.Module):
    def __init__(self, lstm_size:int = 256, lstm_depth:int = 2, 
                 dropout:float = 0.2, features:Union[list, str] = None):
        super().__init__()
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


        self.lstm_size = lstm_size
        self.lstm_depth = lstm_depth
        self.dropout = dropout

        self.lstm = nn.LSTM(input_size=self.input_size, 
                            hidden_size=self.lstm_size, 
                            num_layers=lstm_depth, 
                            batch_first=True, 
                            dropout=dropout if dropout else 0)
        
        self.fc1 = nn.Linear(in_features = lstm_size, 
                             out_features = self.input_size)
        
        if dropout is not None:
            self.dropout1 = nn.Dropout(p=dropout)



        self._features = features
        self._features_stats = None
        self._hist_train_loss = []
        self._hist_train_loss_iters = []
        self._hist_valid_loss = []
        self._hist_valid_loss_iters = []

    def plot_loss(self, filepath:str = None, figsize:tuple = (6,3)) -> None:
        """Plot RNN loss in the training set (orange) and validation set (blue) 
        vs number of iterations during model training.

        Args:
            filepath (str, optional): Path where the plot is saved. Defaults to 
            None.
            figsize (tuple, optional): Size of the plot. Defaults to (6 ,3).
        """
        fig, ax = plt.subplots(figsize = figsize)
        ax.plot(self._hist_train_loss_iters, self._hist_train_loss, 
                label='Training', color='tab:orange')
        ax.plot(self._hist_valid_loss_iters, self._hist_valid_loss, 
                label='Validation', color='tab:blue')
        ax.set_xlabel('Iterations')
        ax.set_ylabel('Loss')
        ax.legend()
        if filepath is not None:
            print('Plotting to file: {}'.format(filepath))
            plt.savefig(filepath)

    def learn(self, event_set:list, epochs:int = 2, lr:float = 1e-3, 
              batch_size:int = 8, device:str = 'cpu', 
              valid_proportion:float = 0.15, num_workers:int = 4, 
              event_samples_for_stats:int = 250, filename_prefix:str = None) -> None:
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
            filename_prefix (str, optional): _description_. Defaults to None.

        Raises:
            ValueError: valid_proportion is not in the range (0, 1).
            RuntimeError: Validation set does not contain any event as a result 
            of valid_proportion being too low.
        """

        
        # Define the device on which the torch will be allocated:
        if device is None: device = torch.device('cpu')
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

        # If mean and standard deviations per feature are not provided, get 
        # them using the DatasetEventDataset class to normalize features in the
        # training and validation datasets.
        if self._features_stats is None:
            print('Computing normalization statistics')
            self._features_stats = \
                DatasetEventDataset(event_set[:event_samples_for_stats], 
                                    self._features)._features_stats
        
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

        pb_epochs = utils.ProgressBar(iterations=range(epochs), 
            description = 'Training Feature Forecaster model...', 
            desc_loc='right')
            
        for epoch in pb_epochs.iterations:
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
                    output = self(inputs, event_lengths)

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
                output = self(inputs, event_lengths)

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

                description = f'Iterations {total_iters} | ' + \
                    f'Minibatch {i_minibatch+1}/{len(train_loader)} | ' + \
                    f'Training loss {train_loss:.4e} | ' + \
                    f'Validation loss {valid_loss:.4e}'
                
                pb_epochs.refresh(i = epoch+1, description = description, 
                    nested_progress = True)


            if filename_prefix is not None:
                filename = filename_prefix + '_epoch_{}'.format(epoch+1)
                description = f'Saving model checkpoint to file {filename}'
                pb_epochs.refresh(i = epoch, description = description, 
                    nested_progress = True)
                self.save(filename)

    def predict(self, event: ConjunctionEvent) -> ConjunctionDataMessage:
        """Predict next CDM object from a given ConjunctionEvent object.

        Args:
            event (ConjunctionEvent): Conjunction Event object containing CDM(s) 
            object(s).

        Raises:
            RuntimeError: _description_

        Returns:
            ConjunctionDataMessage: CDM object.
        """

        ds = DatasetEventDataset(ConjunctionEventsDataset(events=[event]), 
                                 features = self._features, 
                                 features_stats = self._features_stats)
        
        # Get CDM objects and number of CDMs contained in the dataset taking the 
        # event passed as a parameter.
        inputs, inputs_length = ds[0]

        # Allocate torch to the 
        inputs = inputs.to(self._device)
        input_length = inputs_length.to(self._device)

        self.train()

        # Initialize LSTM hidden state (h) and cell state (c) assuming 
        # batch_size = 1.
        self.reset(1)

        # Forecast next CDM content
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
        cdm = ConjunctionDataMessage()

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

    def predict_event_step(self, event:ConjunctionEvent, num_samples:int = 1) \
        -> Union[ConjunctionEvent, ConjunctionEventsDataset]:
        """Predict next CDM n-times for a given event object.

        Args:
            event (ConjunctionEvent): Conjunction Event object from which the 
            CDM is forecasted.
            num_samples (int, optional): Number of predictions. Defaults to 1.

        Returns:
            Union[ConjunctionEvent, ConjunctionEventsDataset]: Two possible outputs are returned
            depending on the parameter num_samples:
             - If num_samples = 1: Returns one ConjunctionEvent object with all CDMs 
                forecasted. 
             - If num_samples > 1: Returns ConjunctionEventsDataset object containing all 
                possible evolutions of the event (combinations of CDMs).
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
        return es[0] if num_samples == 1 \
            else ConjunctionEventsDataset(events=events)
            

    def predict_event(self, event:ConjunctionEvent, num_samples:int = 1, 
                      max_length:int = 22) -> Union[ConjunctionEvent, ConjunctionEventsDataset]:
        """Forecast the evolution of a given Conjunction Event by predicting 
        upcoming CDMs until TCA.

        Args:
            event (ConjunctionEvent): Conjunction Event to forecast.
            num_samples (int, optional): Number of possible CDMs considered in 
            every forecasting step. Defaults to 1.
            max_length (int, optional): Maximum number of CDM objects contained 
            in the event object. Defaults to 22.

        Returns:
            Union[ConjunctionEvent, ConjunctionEventsDataset]: Two possible outputs are returned
            depending on the parameter num_samples:
             - If num_samples = 1: Returns one ConjunctionEvent object with all CDMs 
                forecasted. 
             - If num_samples > 1: Returns ConjunctionEventsDataset object containing all 
                possible evolutions of the event (combinations of CDMs).
        """

        # Initialize list to store Conjunction Events.
        events = []

        # Iterate over all sequences
        pb_samples = utils.progressbar(iterations = range(num_samples),  
            description='> Forecasting Conjunction Event evolution ...',
            desc_loc='right')

        for i in pb_samples.iterations:

            # Update progress bar.
            pb_features.refresh(i = i+1)

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

                # Stop loop if one of the conditions is met.
                if (cdm['__CREATION_DATE'] > cdm['__TCA']) or \
                   (cdm['__TCA'] > 7) or \
                   (len(i_event) > max_length): break

            # Append i_event to the final list of events.       
            events.append(i_event)

        # Update progress bar.
        pb_features.refresh(i = i+1, 
            description = '> Conjunction Event evolution forecasted.')

        return events[0] if num_samples==1 \
            else ConjunctionEventsDataset(events = events)

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
        """Initialize LSTM hidden state (h) and cell state (c).

        Args:
            batch_size (int): Batch size.
        """
        h = torch.zeros(self.lstm_depth, batch_size, self.lstm_size)
        c = torch.zeros(self.lstm_depth, batch_size, self.lstm_size)

        h = h.to(self._device)
        c = c.to(self._device)

        self.hidden = (h, c)

    def forward(self, x:torch.FloatTensor, x_lengths: torch.IntTensor) \
        -> torch.FloatTensor:
        """Predict new CDM containing normalized values.

        Args:
            x (torch.FloatTensor): Tensor with shape (n_events, 
            max_event_length, features) containing the input values for RNN 
            processing.
            x_lengths (torch.IntTensor): Tensor with shape (n_events, 1) 
            containing the number of CDMs contained in every event. This tensor 
            is used to unpack padded torch.

        Returns:
            torch.FloatTensor: Tensor containing normalized values of the new CDM.
        """

        # Get the size from the inputs. In this case, batch_size is the number 
        # of events contained in a minibatch coming from the DataLoader class.
        batch_size, x_length_max, n_features = x.size()

        # In the __getitem__() method of DatasetEventDataset class, all events
        # are padded with zeros in order to get the same number of tensors, and 
        # therefore same event_length. This makes batch processing easier.
        #
        # To optimize training and avoid computing empty CDMs, PyTorch allows 
        # to packing a padded sequence by producing a PackedSequence object. 
        # A PackedSequence object is a tuple of two lists. One contains the 
        # elements of sequences interleaved by time steps and other contains the 
        # the batch size at each step. 
        # Pack Tensor containing padded sequences of variable length.
        x = nn.utils.rnn.pack_padded_sequence(input = x, 
                                              lengths = x_lengths, 
                                              batch_first=True, 
                                              enforce_sorted=False)

        # Get outputs from the LSTM layer.
        x, self.hidden = self.lstm(x, self.hidden)

        # Pads a packed batch of variable length sequences from LSTM layer.
        x, _ = nn.utils.rnn.pad_packed_sequence(sequence = x, 
                                                batch_first=True, 
                                                total_length = x_length_max)
        
        # Implement a dropout layer to avoid overfitting.
        if self.dropout: x = self.dropout1(x)

        # Set Rectified Linear Unit activation function
        x = torch.relu(x)

        # Apply linear regression to fully connected layer.
        x = self.fc1(x)

        return x


#%%
def get_lr(optimizer):
    for param_group in optimizer.param_groups:
        return param_group['lr']
#%%
def event_ts_sets(full_seq:np.ndarray, window_size:int, 
                  events_to_forecast:int=1) -> list:  
    """Get all possible Time-Series subsets (sequence->target) from a complete 
    Time-Series set associated to an event.

    Args:
        full_sequence (np.ndarray): Array containing the full sequence 
        of data for a given event.
        window_size (int): Window size of events.
        events_to_forecast (int): Number of events to forecast. Defaults to 1.

    Returns:
        list: List of tuples with sequences and labels.
    """

    # Get number of TS sets to extract from the full sequence. 
    n = len(full_seq) - (window_size + events_to_forecast)+1

    # Initialize Time-Series sets list containing tuples with sequence-target
    # for a given event.
    ts_sets = []

    # Create the list of Time-Series sets using a loop.
    for i in range(n):

        # Get sequence and target value for element i
        seq_i       = full_seq[i:i+window_size]
        target_i    = full_seq[i+window_size:i+window_size+events_to_forecast]

        # Add tuple to the output list
        ts_sets.append((seq_i, target_i))

    return ts_sets

#%%
def get_tsf_tensors(df:pd.DataFrame, features:list, 
                    seq_length:int, events_to_forecast:int, 
                    rootpath:str, shuffle:bool = True,
                    overwrite_features:bool = False,
                    save_new_features:bool = True) -> dict:
    """Extract list of tuples in the format (sequence, target) from a DataFrame.

    Args:
        df (pd.DataFrame): Data where the mini batch of time series and targets
        are exracted from.
        features (list): List of features to extract the data.
        seq_length (int): Length of the Time-Series sequence (batch).
        events_to_forecast (int): Number of time steps to forecast.
        rootpath (str): Root path where tensors shall be downloaded or saved to.
        shuffle (bool, optional): Flag to shuffle list of tensors. Defaults to 
        True.
        overwrite_features (bool, optional): Flag to determine wether features
        shall be saved regardless of tensor file availability. Defaults to 
        False.
        save_new_features (bool, optional): Flag to determine wether the tensors
        from new features shall be saved or not. Defaults to True.

    Returns:
        dict: Dictionary containing the list of tuples per feature.
    """

    # Count number of CDMs per event
    ts_events  = df[['event_id', features[0]]].groupby(['event_id']).count() \
                    .rename(columns={features[0]:'nb_cdms'})

    # Define window size and number of events to forecast
    min_cdms = seq_length + events_to_forecast

    # Get events that have a minimum number of CDMs equal to the window_size + 
    # events_to_forecast
    events_filter = list(ts_events[ts_events['nb_cdms']>=min_cdms].index.values)

    # Initialize output variable
    tsf_tensors = {}

    # Iterate over all features to get the time series subsets
    pb_features = utils.ProgressBar(iterations = range(len(features)), 
                                    desc_loc='right')
    for f in pb_features.iterations:

        # Initialize list of tensors for feature f
        feature = features[f]
        
        # Get filename and filepath and check if it already exists
        filename = f'ts_{seq_length}-{events_to_forecast}-{feature}.pt'
        filepath = os.path.join(rootpath, filename)

        if os.path.exists(filepath) and not overwrite_features:
            # If tensors already exists, load them.
            description = f'> Loading tensors from feature {feature} ...'
            pb_features.refresh(i = f+1, description = description)
            
            tsf_tensors[feature] = torch.load(filepath)
        else:

            # If tensor do not exist for a feature, initialize and get all 
            # time-series
            tsf_tensors[feature] = []

            # Get full sequence from dataset and convert it to a tensor.
            feature_dtype = str(df[feature].dtype).lower()
            data = df[feature].to_numpy(dtype=feature_dtype)

            for e, event_id in enumerate(events_filter):

                # Print progress bar
                subprogress = (e+1)/len(events_filter)*100
                description = f'> Extracting sequences of time-series from' + \
                            f' feature {feature:<30s} ' + \
                            f'(Progress: {subprogress:5.1f}%)'
                pb_features.refresh(i = f+1, description = description, 
                                    last_iteration = False)

                # Get full sequence from dataset and convert it to a tensor.
                event_filter = (df['event_id']==event_id)
                full_seq = torch.nan_to_num(torch.FloatTensor(data[event_filter]))

                # Add Time-Series subsets from full sequence tensor and add it 
                # to the list for the feature f
                tsf_tensors[feature] += event_ts_sets(full_seq, seq_length)
            if shuffle:
                # Set random seed for reproducibility.
                random.seed(0)
                # Shuffle list to avoid biased learning from model.
                random.shuffle(tsf_tensors[feature])

            # Save tensors containing all Time-Series subsets for training 
            # organised by feature.
            if not save_new_features: continue

            description = f'Saving tensors with sequences of time-series' + \
                          f' into external file {"."*(len(description)-64)}'
            torch.save(tsf_tensors[feature], os.path.join(rootpath, filename))
            pb_features.refresh(i = f+1, description = description, 
                                nested_progress = True)

    # Print final message
    pb_features.refresh(i = f+1, 
                        description = 'Time series extracted successfully.')

    return tsf_tensors
# %%

#%%
def tsf_iotensors(tsf_tensors:dict, features:list, seq_length:int, 
                  filepath:str = None, batch_first=True) -> dict:
    """Convert input dictionary with the shape:
     {'feature_1': [([time_series_1], [forecast_1]), ... ([time_series_n], 
     [forecast_n])],
      'feature_2': [([time_series_1], [forecast_1]), ... ([time_series_n], 
      [forecast_n])], 
      ...,
      'feature_m': [([time_series_1], [forecast1]), ... ([time_series_n], 
      [forecast_n])]}
    to a dictionary with shape {'inputs': tensos with shape (seq_length, 
    batch_size, input_size), 'outputs': tensos with shape (batch_size, 
    input_size)}.

    Args:
        tsf_tensors (dict): Dictionary containing tensors. Every key contains a 
        list of tensors in the tuple format (input, forecast).
        features (list): List of features to extract the inputs/outputs from.
        filepath (str, optional): Directory path where the tensors are stored. 
        Defaults to None.
        batch_first (bool, optional): Shape of inputs tensor. Defaults to True.

    Returns:
        dict: Dictionary containing a torch with inputs in the format 
        (batch_size, seq_length, input_size) and a torh with outputs in the 
        format (batch_size, input_size) to train the model.
    """

    # Get features available in tsf_tensors dictionary
    features_available = list(tsf_tensors.keys())

    # Check all features requested by user are actually available in the 
    # tsf_tensors dictionary.
    for feature in features:
        if not feature in features_available: 
            print(f'Tensors from {feature} not available.')
            return None

    # Get length of sequence and number of sequences to process from tensor file
    seq_length = tsf_tensors[features[0]][0][0].size(0)
    batch_size = len(tsf_tensors[features[0]])
    input_size = len(features)

    # Initialize inputs and outputs arrays to contain sequences to process
    # (batch_size, input_size)
    outputs = torch.empty((batch_size, input_size), dtype=torch.float32) 
    if batch_first:
        # (batch_size, seq_length, input_size)
        inputs =  torch.empty((batch_size, seq_length, input_size), 
                              dtype=torch.float32)    
    else:
        # (seq_length, batch_size, input_size)
        inputs =  torch.empty((seq_length, batch_size, input_size), 
                              dtype=torch.float32)    

    # Iterate over all sequences
    pb_sequences = utils.ProgressBar(iterations = range(batch_size), 
                        desc_loc='right', 
                        description='> Getting training and target tensors ...')
    for b in pb_sequences.iterations:

        # Initialize list for sequence s
        inputs_s    = torch.empty((input_size, seq_length), dtype=torch.float32)
        outputs_s   = torch.empty((input_size, 1), dtype=torch.float32)

        # Get sequence s from all features
        for f, feature in enumerate(features):
            
            # Get sequence s (input and output) from feature f
            # - inputs_s  = [[f1_t1, f1_t2, ..., f1_tm], 
            #                [f2_t1, f2_t2, ..., f2_tm], 
            #                ...,
            #                [fn_t1, fn_t2, ..., fn_tm]]
            #
            # - outputs_s = [[f1_tm+1], 
            #                [f2_tm+1], 
            #                ...,
            #                [fn_tm+1]]
            #
            # Where n = input_size, and m = seq_length.
            inputs_s[f], outputs_s[f] = tsf_tensors[feature][b]


        # Reshape torch to be in the shape (seq_length, input_size) and 
        # (1, input_size)
        inputs_s = torch.transpose(inputs_s,0,1).reshape(seq_length, input_size)

        if batch_first:

            # Get sequence s (input and output) from feature f in the shape 
            # (batch_size, seq_length, input_size) 
            # 
            # - inputs  = [[[f1_s1_t1, f2_s1_t1, ..., fn_s1_t1], 
            #               [f1_s1_t2, f2_s1_t2, ..., fn_s1_t2], 
            #                ..., 
            #               [f1_s1_tm, f2_s1_tm, ..., fn_s1_tm]],
            #              ...,
            #              [[f1_sb_t1, f2_sb_t1, ..., fn_sb_t1], 
            #               [f1_sb_t2, f2_sb_t2, ..., fn_sb_t2], 
            #                ..., 
            #               [f1_sb_tm, f2_sb_tm, ..., fn_sb_tm]]]
            #
            #
            # - outputs = [[f1_s1_tm+1, f2_s1_tm+1, ..., fn_s1_tm+1],
            #               ...,
            #              [f1_sb_tm+1, f2_sb_tm+1, ..., fn_sb_tm+1]]
            #
            # Where n = input_size, m = seq_length, and b = batch_size.

            inputs[b] = inputs_s
        else:
            for s in range(seq_length): inputs[s,b] = inputs_s[s]

        outputs[b] = outputs_s.reshape(1, input_size)
        

        # Update progress bar
        pb_sequences.refresh(i=b+1)

    io_tensors = {"inputs":inputs, "outputs":outputs}

    # Save the model trained parameters (weights and biases)
    if filepath!=None:
        print('Saving training data...', end='\r')
        torch.save(io_tensors, filepath)
        print('Saving training data... Done.\n')

    return io_tensors

#%%

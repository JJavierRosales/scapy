import torch
import os
import torch.nn as nn
import numpy as np
import pandas as pd
from tqdm import tqdm, trange
import random

from . import utils

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
    pb_features = utils.progressbar(iterations = range(len(features)), 
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
                                last_iteration = False)

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
    pb_sequences = utils.progressbar(iterations = range(batch_size), 
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

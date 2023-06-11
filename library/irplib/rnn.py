import torch
import torch.nn as nn
import numpy as np
import pandas as pd
from tqdm import tqdm, trange

from . import utils

#%%
def get_lr(optimizer):
    for param_group in optimizer.param_groups:
        return param_group['lr']
#%%
def event_ts_sets(full_seq:np.ndarray, window_size:int, events_to_forecast:int=1) -> list:  
    """Get all possible Time-Series subsets (sequence->target) from a complete Time-Series set 
    associated to an event.

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
def tsf_iotensors(tsf_tensors:dict, features:list, seq_length:int, filepath:str = None) -> dict:
    """Convert list of tensors from the shape {'feature1': [([time_series1], [forecast1]), ... ([time_seriesN], [forecastN])]}
    to a dictionary with shape {'inputs': tensos with shape (seq_length, batch_size, input_size), 'outputs': tensos with shape (batch_size, input_size)}.

    Args:
        tsf_tensors (dict): Dictionary containing tensors. Every key contains a list of tensors in the tuple format (input, forecast).
        features (list): List of features to extract the inputs/outputs from.
        filepath (str, optional): Directory path where the tensors are stored. Defaults to None.

    Returns:
        dict: Dictionary containing a torch with inputs in the format (batch_size, seq_length, input_size) and a torh with outputs in the 
        format (batch_size, input_size) to train the model.
    """

    # Get features available in tsf_tensors dictionary
    features_available = list(tsf_tensors.keys())

    # Check all features requested by user are actually available in the tsf_tensors dictionary.
    for feature in features:
        if not feature in features_available: 
            print(f'Feature {feature} has not been processed yet. Please extract tensors from this feature before proceeding.')
            return None

    # Get length of sequence and number of sequences to process from tensor file
    seq_length = tsf_tensors[features[0]][0][0].size(0)
    batch_size = len(tsf_tensors[features[0]])
    input_size = len(features)

    # Initialize inputs and outputs arrays to contain sequences to process
    inputs =  torch.empty((seq_length, batch_size, input_size), dtype=torch.float32)    # (seq_length, batch_size, input_size)
    outputs = torch.empty((batch_size, input_size), dtype=torch.float32)             # (batch_size, input_size)

    # Iterate over all sequences
    pb_sequences = utils.progressbar(iterations = range(batch_size), desc_loc='right', description='> Getting training and target tensors ...')
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

        # Reshape torch to be in the shape (seq_length, input_size) and (1, input_size)
        inputs_s = torch.transpose(inputs_s,0,1).reshape(seq_length, input_size)
        outputs_s = outputs_s.reshape(1, input_size)

        # Get sequence s (input and output) from feature f in the shape (seq_length, batch_size, input_size) 
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

        for s in range(seq_length): inputs[s,b] = inputs_s[s]
        outputs[b] = outputs_s

        # inputs[s,:] =  torch.transpose(inputs_s, 0, 1)
        # outputs[s,:] = torch.transpose(outputs_s, 0, 1)

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

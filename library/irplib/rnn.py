import torch
import torch.nn as nn
import numpy as np
import pandas as pd

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

# Define Multivariate LSTM network class
class EventPropagation(nn.Module):
    def __init__(self,input_size, hidden_size, output_size, seq_length, num_layers=1):
        super(EventPropagation, self).__init__()
        self.input_size = input_size    # Number of input features
        self.hidden_size = hidden_size  # Number of hidden neurons
        self.output_size = output_size  # Number of outputs
        self.num_layers = num_layers    # Number of recurrent (stacked) layers
        self.seq_length = seq_length
    
        self.lstm = nn.LSTM(input_size = self.input_size, 
                            hidden_size = self.hidden_size,
                            num_layers = self.num_layers,
                            batch_first = True)
        # according to pytorch docs LSTM output is 
        # (batch_size,seq_len, num_directions * hidden_size)
        # when considering batch_first = True
        self.linear = nn.Linear(self.hidden_size*self.seq_length, 
                                self.output_size)
        
    
    def init_hidden(self, n_sequences):
        # Initialize states. Even with batch_first = True this remains same as docs
        h_state = torch.zeros(self.num_layers, n_sequences, self.hidden_size) # Hidden state
        c_state = torch.zeros(self.num_layers, n_sequences, self.hidden_size) # Cell state
        self.hidden = (h_state, c_state)
    
    
    def forward(self, inputs):        
        n_sequences, seq_length, n_features = inputs.size()
        
        lstm_out, self.hidden = self.lstm(inputs, self.hidden)
        # lstm_out(with batch_first = True) is 
        # (batch_size,seq_len,num_directions * hidden_size)
        # for following linear layer we want to keep batch_size dimension and merge rest       
        # .contiguous() -> solves tensor compatibility error
        inputs = lstm_out.contiguous().view(n_sequences,-1)
        outputs = self.linear(inputs)
        
        return outputs
import torch
import torch.nn as nn
import numpy as np
import pandas as pd

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
#%%
# Define Feature Forecaster module
class FeatureForecaster(nn.Module):

    def __init__(self, input_size, output_size, layers, p=0.5):
        
        # Inherit attributes from nn.Module class
        super().__init__()
        
        ########################################################################
        # Instanciate functions to use on the forward operation:
        
        # self.bn_cont = Normalizes continuous features. This function is 
        # configured by passing the number of continuous features to normalize.
        self.BatchNorm = nn.BatchNorm1d(input_size)
        
        ########################################################################
        # Compute total number of inputs to pass to the initial layer (data 
        # point = Nb. of embeddings + Nb. of continuous variables)
        n_in = input_size
        
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
        layerlist.append(nn.Linear(layers[-1],output_size))
        
        # Instantiate layers as a Neural Network sequential task
        self.layers = nn.Sequential(*layerlist)
    
    def forward(self, x):
        
        # Normalize continuous variables
        x = self.BatchNorm(x)
        
        # Process all data points with the layers functions (sequential of 
        # operations)
        x = self.layers(x)
        
        return x
#%%
def event_ts_sets(feature_seq:np.ndarray, time_seq:np.ndarray):  
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
    n = len(feature_seq) - 2

    # Initialize Time-Series sets list containing tuples with sequence-target
    # for a given event.
    ts_sets = []
    seq_tensor = torch.empty((n, 3), dtype=torch.float32)
    target_tensor = torch.empty((n, 2), dtype=torch.float32)

    # Create the list of Time-Series sets using a loop.
    for i in range(n):

        # Get sequence and target value for element i
        seq_tensor[i] = torch.from_numpy(np.append(feature_seq[i:i+2],
                           np.abs(time_seq[i+1]-time_seq[i])))

        target_tensor[i] = torch.from_numpy(np.append(feature_seq[i+2],
                            np.abs(time_seq[i+2]-time_seq[i+1])))


        # seq_i  = np.append(feature_seq[i:i+2],
        #                    np.abs(time_seq[i+1]-time_seq[i]))
        # target_i  = np.append(feature_seq[i+2],
        #                       np.abs(time_seq[i+2]-time_seq[i+1]))

        # Add tuple to the output list
        # ts_sets.append((seq_i, target_i))

    return seq_tensor, target_tensor
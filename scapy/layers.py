# Libraries used for hinting
from __future__ import annotations
from typing import Union

import torch
import torch.nn as nn

import warnings


#%% CLASS: SelfAttentionLayer
# Link: https://www.analyticsvidhya.com/blog/2023/06/time-series-forecasting-using-attention-mechanism/
# https://stats.stackexchange.com/questions/421935/what-exactly-are-keys-queries-and-values-in-attention-mechanisms

# Self-Attention Mechanism
# The self-attention mechanism calculates attention weights by comparing the 
# similarities between all pairs of time steps in the sequence. Let’s denote the 
# encoded hidden states as H = [H1, H2, …, H_T]. Given an encoded hidden state Hi 
# and the previous decoder hidden state (prev_dec_hidden = Hd_T-1), the attention 
# mechanism calculates a score for each encoded hidden state:

# Score(t) = V * tanh(W1 * HT + W2 * prev_dec_hidden)

# Here, W1 and W2 are learnable weight matrices, and V is a learnable vector. 
# The tanh function applies non-linearity to the weighted sum of the encoded 
# hidden state and the previous decoder hidden state.

# The scores are then passed through a softmax function to obtain attention 
# weights (alpha1, alpha2, …, alphaT). The softmax function ensures that the 
# attention weights sum up to 1, making them interpretable as probabilities. The 
# softmax function is defined as:

# softmax(x) = exp(x) / sum(exp(x))   ->    alpha_T = softmax(Score_T)

# Where x represents the input vector.

# The context vector (context) is computed by taking the weighted sum of 
# the encoded hidden states:

# context = alpha1 * H1 + alpha2 * H2 + … + alpha_T * H_T

# The context vector represents the attended representation of the input 
# sequence, highlighting the relevant information for making 
# predictions. By utilizing self-attention, the model can efficiently 
# capture dependencies between different time steps, allowing for more 
# accurate forecasts by considering the relevant information across the 
# entire sequence.
class SelfAttentionLayer(nn.Module):
    """Self-Attention layer instanciator.

    Args:
        nn (torch.nn): Pytorch base module for any neural network development.
    """
    def __init__(self, input_size:int, batch_first:bool=True, 
                 num_heads:int=1) -> None:
        """Initialise self-attention layer constructor.

        Args:
            input_size (int): Number of inputs to the layer.
            batch_first (bool, optional): Batch first. Defaults to True.
            num_heads (int, optional): Number of parallel heads used for the 
            attention mechanism. Defaults to 1.
        """

        super(SelfAttentionLayer, self).__init__()

        # Initialize input and output sizes
        self.input_size = input_size
        self.attention = nn.MultiheadAttention(embed_dim = input_size, 
                                               num_heads = num_heads,
                                               batch_first = batch_first)

    def forward(self, x:torch.Tensor) -> torch.Tensor:
        """Process inputs through the layer.

        Args:
            x (torch.Tensor): Inputs.

        Returns:
            torch.Tensor: Outputs of the layer (same shape as inputs).
        """

        attn_output = self.attention(query = x, 
                                     key = x, 
                                     value = x, 
                                     need_weights = False)

        return attn_output[0]
    
#%% CLASS: RNNLayer
class RNNLayer(nn.Module):
    """Layer constructor for LSTM, GRU and MGU based RNN architecture.
    """

    def __init__(self, cell, input_size:int, hidden_size:int, **cell_args:dict):
        """Initialize RNN layer constructor.

        Args:
            cell (constructor): RNN cell constructor.
            input_size (int): Number of inputs.
            hidden_size (int): Number of hidden cells (outputs of the cell).
            *cell_args (dict, optional): Dictionary containing optional 
            arguments required for the RNN cell constructor (i.e. SLIMX 
            constructor receives the additional parameter 'version').
        """
    
        super(RNNLayer, self).__init__()
        
        # Initialize cell attribute in class witht the RNN cell object 
        # initialized.
        
        self.cell = cell(input_size = input_size, 
                         hidden_size = hidden_size, 
                         **cell_args)
        
        # Initialize inputs and hidden sizes
        self.input_size = self.cell.input_size
        self.hidden_size = self.cell.hidden_size
        

    def forward(self, input: torch.Tensor, state:Union[tuple,torch.Tensor]) -> tuple:
        """Process inputs through all layers.

        Args:
            input (torch.Tensor): Tensor containing the values at every 
            time step of a sequence.
            state (tuple): Tuple of tensors containing previous hidden state and
            cell state (at time t-1) required to produce the next output.

        Returns:
            tuple: Tuple with two tensors:
                - outputs (torch.Tensor): Forecasted values of X for the 
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

        # Return the list of outputs produced by the RNN cell and the last 
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
        dropout:float=0.0, **cell_args:dict) -> None:
        """Initialize LSTM custom layer constructor.

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
        layers = [RNNLayer(cell = cell, 
                            input_size = input_size, 
                            hidden_size = hidden_size, 
                            **cell_args)] + \
                 [RNNLayer(cell = cell, 
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
                "\nDropout parameter in LSTM class adds dropout layers after " 
                "all but last recurrent layer. \nIt expects num_layers greater "
                "> 1, but got num_layers = 1."
            )
        
        # If dropout_probability is provided initialize Dropout Module. 
        if dropout > 0 and num_layers > 1:
            self.dropout_layer = nn.Dropout(p = dropout)
        else:
            self.dropout_layer = None


    def forward(self, input: torch.Tensor, states: torch.Tensor) -> tuple:
        """Process inputs through the layer.

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
                - states (tuple): Tuple with two tensors with shape: 
                    + output: Tensor with shape (num_layers, hidden_size) for 
                    unbatched input or (num_layers, batch_size, hidden_size) for
                    batched output. It contains the last hidden state 
                    (predicted Xt+1 value) 
                    + (h_t, c_t): Tensors with shape (num_layers, hidden_size) 
                    for unbatched input or (num_layers, batch_size, hidden_size) 
                    for batched output. It contains the cell states (cell 
                    context) required to produce the next prediction of the 
                    layer.
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

            hstate = states[0][i] if self.num_layers > 1 else states[0]
            cstate = states[1][i] if self.num_layers > 1 else states[1]
            
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

    
#%% CLASS: GU
class GU(nn.Module):
    """Adapted nn.GRU and nn.MGU class that allows the use of custom cells.
    """
    def __init__(self, input_size:int, hidden_size:int, cell, 
        batch_first:bool=True, num_layers:int=1, 
        dropout:float=0.0, **cell_args:dict) -> None:
        """Initialise custom constructor for GRU and MGU cells.

        Args:
            input_size (int): Number of input features.
            hidden_size (int): Number of hidden neurons (outputs of the RNN).
            cell (constructor): RNN cell constructor.
            num_layers (int, optional): Number of stacked layers. Defaults to 1.
            dropout (float, optional): Dropout probability to use 
            between consecutive layers. Only applicable if num_layers is 
            greater than 1. Defaults to None.
        """
    
        super(GU, self).__init__()
        
        # Get all GU layers in a list using the RNNLayer constructor.
        layers = [RNNLayer(cell = cell, 
                           input_size = input_size, 
                           hidden_size = hidden_size, 
                           **cell_args)]
        
        layers += [RNNLayer(cell = cell, 
                           input_size = layers[0].hidden_size, 
                           hidden_size = layers[0].hidden_size, 
                           **cell_args) for _ in range(num_layers - 1)]
        
        # Set network dimensions
        self.input_size = layers[0].input_size
        self.hidden_size = layers[0].hidden_size
        
        # Set batch_first parameter
        self.batch_first = batch_first
        self._batched = None
        
        # Convert list of LSTM layers to list of nn.Modules.
        self.layers = nn.ModuleList(layers)
        
        # Introduces a Dropout layer on the outputs of each layer except
        # the last layer.
        self.num_layers = num_layers

        # If number of layers is 1 and the dropout_probability provided is
        # not None, print warning to the user.
        if num_layers == 1 and dropout > 0:
            warnings.warn(
                "\nDropout parameter in GU class adds dropout layers after " 
                "all but last recurrent layer. \nIt expects num_layers greater "
                "> 1, but got num_layers = 1."
            )
        
        # If dropout_probability is provided initialize Dropout Module. 
        if dropout > 0 and num_layers > 1:
            self.dropout_layer = nn.Dropout(p = dropout)
        else:
            self.dropout_layer = None


    def forward(self, input: torch.Tensor, states: torch.Tensor) -> tuple:
        """Process inputs through the layer.

        Args:
            input (torch.Tensor): Tensor of shape (seq_length, hidden_size) 
            if unbatched, (batch_size, seq_length, hidden_size) if batched and 
            batch_first = True, or (seq_length, batch_size, hidden_size) if 
            batched and batch_first = False. It containins the values at 
            every time step of a sequence.
            states (torch.Tensor): Tensor of shape (num_layers, hidden_size) 
            if input is unbatched, or (num_layers, batch_size, hidden_size) if 
            batched. It contains tensors of every GRU layer with the previous 
            hidden state (at time t-1) required to produce the next output for 
            the layer.

        Returns:
            tuple: Tuple with two values:
                - outputs (torch.Tensor): Tensor of shape (seq_length, 
                hidden_size) for unbatched input, (batch_size, seq_length, 
                hidden_size) for batched input if batch_first = True, or 
                (seq_length, batch_size, hidden_size) for batched input if 
                batch_first = False. It contains the forecasted values of X for 
                the next time step (ht ~ Xt+1).
                - states (torch.Tensor): Tensor with shape (num_layers, 
                hidden_size) for unbatched input or (num_layers, batch_size, 
                hidden_size) for batched output. It contains the hidden state 
                required to produce the next prediction of the layer.
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

        # Initialize list to store the hidden state required for the next output
        # prediction
        output_hstates = []

        # Initialize input layer tensor (tensor to be passed first to the layer
        # whose input_size is the input number of features)
        input_layer = input
        
        for i, layer in enumerate(self.layers):
            
            # Get the hidden state tensor at layer i:
            #  - batched = True -> (num_layers, batch_size, hidden_size)
            #  - batched = False -> (num_layers, hidden_size)
            hstate = states[i] if self.num_layers > 1 else states

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


                # Iterate through all batches- to get the output and states per 
                # batch.
                for b in range(batch_size):

                    # Get the batch input tensor b for the layer i depending on 
                    # the shape of the input tensor.
                    batch_input = input_layer[b, :, :] if self.batch_first \
                        else torch.squeeze(input_layer[:, b, :], dim = 1)
                    
                    # Get the batch states for the layer i
                    batch_state = torch.squeeze(hstate[b])
                    
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
                    out_hstate[b] = batch_state

            else:

                # Keep last cell state of sequence
                output_layer, out_hstate = layer(input_layer, hstate)

                
            # Apply the dropout layer except the last layer
            if (i < self.num_layers - 1) and not (self.dropout_layer is None):
                output_layer = self.dropout_layer(output_layer)

            # Redefine input_layer tensor from the output of previous layer.
            input_layer = output_layer

            # Add last states from squences (and from all batches if input is 
            # batched) to final list.    
            output_hstates += [out_hstate]

        # Convert lists to tensors 
        hstates = torch.stack(output_hstates) if self.num_layers > 1 \
                                else output_hstates[0]
        
        # Return the outputs and hidden states for every GU layer.
        return output_layer, hstates
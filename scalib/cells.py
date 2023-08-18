# Libraries used for type hinting
from __future__ import annotations
from typing import Union

import torch
from torch import nn


# AI Summer link: https://theaisummer.com/understanding-lstm/
# Github Pytorch: https://github.com/pytorch/examples/tree/main/time_sequence_prediction


#%% BIAS LAYER
class BiasLayer(nn.Module):
    """
    A learnable bias layer for gates with no weight parameters in LSTM cell 
    architectures.
    """
    def __init__(self) -> None:
        super().__init__()
        bias_value = torch.randn((1))
        self.bias_layer = torch.nn.Parameter(bias_value)
    
    def forward(self, x:torch.TensorFloat) -> torch.TensorFloat:
        return x + self.bias_layer
#%% LSTM VANILLA CELL ARCHITECTURE
class LSTM_Vanilla(nn.Module):
    """
    Vanilla LSTM cell with input, forget, output and cell gates and associated 
    activation functions.
    """
    def __init__(self, input_size:int, hidden_size:int) -> None:
        """Initialize LSTM vanilla cell class.

        Args:
            input_size (int): Number of input features.
            hidden_size (int): Number of hidden neurons (outputs of the LSTM).
        """
    
        super(LSTM_Vanilla, self).__init__()
        
        # Initialize inputs and hidden sizes
        self.input_size = input_size
        self.hidden_size = hidden_size
        
        # Initialize gate components (weight and bias) for every gate:
        for gate in ['forget','input','cell','output']:
        
            # Initialize gate weights (Wxg) that will process inputs at time t.
            # Gate component computation = Wxg * xt
            setattr(self,'gate_{}_x'.format(gate),
                nn.Linear(in_features = self.input_size, 
                          out_features = self.hidden_size, 
                          bias = False))
            
            # Initialize gate weights (Whg) that will process hidden states at 
            # time t-1, including bias associated to the gate.
            # Gate component computation = h[t-1]*Whg + bg
            setattr(self,'gate_{}_h'.format(gate),
                nn.Linear(in_features = self.hidden_size, 
                          out_features = self.hidden_size, 
                          bias = True))
        
        
    def _forward_gate(self, gate:str, x:torch.TensorFloat, h:torch.TensorFloat, 
            input_gate:torch.TensorFloat = None,
            forget_gate:torch.TensorFloat = None,
            c_prev:torch.TensorFloat = None) -> torch.TensorFloat:
        """Forward operation through nominated gate.

        Args:
            gate (str): Name of the gate upon which the forward operation is 
            performed.
            x (torch.TensorFloat): Input tensor at time t.
            h (torch.TensorFloat): Previous hidden states (at time t-1).
            input_gate (torch.TensorFloat, optional): Tensor with the outputs 
            from the input gate. Only required when gate is set to 'cell'. 
            Defaults to None.
            forget_gate (torch.TensorFloat, optional): Tensor with the outputs 
            from the forget gate. Only required when gate is set to 'cell'. 
            Defaults to None.
            c_prev (torch.TensorFloat, optional): Tensor with the previous cell 
            state (at time t-1). Only required when gate is set to 'cell'. 
            Defaults to None.

        Raises:
            ValueError: Gate is not defined.
            ValueError: One of the parameters required when cell is set to gate 
            is not provided.

        Returns:
            torch.TensorFloat: Tensor with the output of the gate.
        """
    
        # Check gate requested is within LSTM cell arquitecture.
        if not gate in ['forget','input','cell','output']:
            raise ValueError('Gate () not defined.'.format(gate))
        
        # Get gate components for the input x and hidden states h.
        x = getattr(self,'gate_{}_x'.format(gate))(x)
        h = getattr(self,'gate_{}_h'.format(gate))(h)
        
        if gate=='cell':
            
            # Check input_gate, forget_gate and cell state from t-1 is provided.
            if input_gate is None or forget_gate is None or c_prev is None:
                raise ValueError('Parameter(s) input_gate, forget_gate, ' + \
                                 'and/or c_prev missing.')
            
            # New information part that will be injected in the new context 
            # (candidate cell)
            g = nn.Tanh(x + h) * input_gate

            # Apply forget gate to forget old context/cell information.
            c = forget_gate * c_prev
            
            # Add/learn new context/cell information.
            c_next = g + c
            
            return c_next
        else:
        
            # Apply sigmoid function to gate.
            return nn.Sigmoid(x + h)
   

    def forward(self, x:torch.TensorFloat, hidden_states:tuple) -> tuple:
        """Forward for the cell.

        Args:
            x (torch.TensorFloat): Inputs to process through the cell.
            hidden_states (tuple): Tuple containing two tensors with the hidden 
            state and cell state respectively at time t-1.

        Returns:
            tuple: Tuple containing two tensors with the hidden state and cell 
            state values at time t.
        """
    
        # Get hidden states from t-1.
        (h_prev, c_prev) = hidden_states
        
        # Get outputs from input gate (to know what to learn).
        i = self._forward_gate(gate = 'input', x = x, h = h_prev)
        
        # Get outputs from forget gate (to know what to forget).
        f = self._forward_gate(gate = 'forget', x = x, h = h_prev)
        
        # User input and forget gates to forget old context and learn new 
        # context (cell information).
        c_next = self._forward_gate(gate = 'cell', x = x, h = h_prev, 
                    input_gate = i, forget_gate = f, c_prev = c_prev)
                    
        # Get outputs from the main output gate.
        o = self._forward_gate(gate = 'output', x = x, h = h_prev)
        
        # Produce next hidden output
        h_next = o * nn.Tanh(c_next)
        
        return h_next, c_next

#%% LSTM SLIMX (X = 1, 2 OR 3) CELL ARCHITECTURE
class LSTM_SLIMX(nn.Module):
    """
    SLIM LSTM cell architecture: input, forget and output gates only relying on 
    incoming hidden states. There are three different versions for the SLIM 
    architecture where the input, forget, and output gates perform different 
    calculations:
        SLIM1 - Gates contain hidden states components and bias (Wh + b).
        SLIM2 - Gates contain only weights for hidden states (Wh).
        SLIM3 - Gates contain only learnable bias (b).
    """
    def __init__(self, input_size:int, hidden_size:int, slim_version:int = 1) -> None:
        """Initialize LSTM SLIM cell class.

        Args:
            input_size (int): Number of inputs.
            hidden_size (int): Number of hidden cells (outputs of the LSTM).
            slim_version (int, optional): Version of the cell architecture to 
            use. Three versions are available:
                1 - Gates contain hidden states components and bias (Wh + b).
                2 - Gates contain only weights for hidden states (Wh).
                3 - Gates contain only learnable bias (b).
            Defaults to 1.
        """
        
        super(LSTM_SLIMX, self).__init__()
        
        # Initialize inputs and hidden sizes
        self.input_size = input_size
        self.hidden_size = hidden_size
        self._version = slim_version
        
        # Initialize gate components (weight and bias) for every gate:
        for gate in ['forget','input','cell','output']:
        
            # In the SLIM architecture, only the cell gate received the values 
            # from the inputs. Therefore weights (Wxg) are only initialized for 
            # the cell gate.
            if gate=='cell':
                # Initialize gate weights (Wxc) that will process inputs at t.
                # Gate component computation = Wxc * xt + bc
                setattr(self,'gate_{}_x'.format(gate),
                    nn.Linear(in_features = self.input_size, 
                              out_features = self.hidden_size, 
                              bias = True))
                              
                # Initialize gate weights (Whc) that will process hidden states  
                # at time t-1, including bias associated to the gate.
                # Gate component computation = h[t-1]*Whc.
                setattr(self,'gate_{}_h'.format(gate),
                    nn.Linear(in_features = self.hidden_size, 
                              out_features = self.hidden_size, 
                              bias = False))
            else:
        
                # If SLIM1 or SLIM2, initialize gate weight components.
                if slim_version<=2:
                    # Initialize gate components that will process hidden 
                    # states at time t-1, including bias associated to the gate 
                    # if SLIM1.
                    # Gate component computation = h[t-1]*Whg + <bg>
                    setattr(self,'gate_{}_h'.format(gate),
                        nn.Linear(in_features = self.hidden_size, 
                                  out_features = self.hidden_size, 
                                  bias = True if slim_version == 1 else False))
                else:
                                  
                    # Initialize gate bias (bg) that will process hidden states 
                    # at t-1.
                    setattr(self, 'gate_{}_h'.format(gate), BiasLayer())
                

    def _forward_gate(self, gate:str, x:torch.TensorFloat, h:torch.TensorFloat, 
            input_gate:torch.TensorFloat = None,
            forget_gate:torch.TensorFloat = None,
            c_prev:torch.TensorFloat = None) -> torch.TensorFloat:
        """Forward operation through nominated gate.

        Args:
            gate (str): Name of the gate upon which the forward operation is 
            performed.
            x (torch.TensorFloat):  Input tensor at time t.
            h (torch.TensorFloat): Previous hidden states (at time t-1).
            input_gate (torch.TensorFloat, optional): Tensor with the outputs 
            from the input gate. Only required when gate is set to 'cell'. 
            Defaults to None. 
            forget_gate (torch.TensorFloat, optional): Tensor with the outputs 
            from the forget gate. Only required when gate is set to 'cell'. 
            Defaults to None.
            c_prev (torch.TensorFloat, optional): Tensor with the previous cell 
            state (at time t-1). Only required when gate is set to 'cell'. 
            Defaults to None.

        Raises:
            ValueError: Gate is not defined.
            ValueError: One of the parameters required when cell is set to gate 
            is not provided.

        Returns:
            torch.TensorFloat: Tensor with the output of the gate.
        """
    
        # Check gate requested is within LSTM cell arquitecture.
        if not gate in ['forget','input','cell','output']:
            raise ValueError('Gate () not defined.'.format(gate))
            
        # Get gate components for the hidden states at t-1.
        h = getattr(self,'gate_{}_h'.format(gate))(h)
        
        
        if gate=='cell':
        
            # Check input_gate, forget_gate and cell state from t-1 is provided.
            if input_gate is None or forget_gate is None or c_prev is None:
                raise ValueError('Parameter input_gate cannot be None.')
                
            # In the SLIM architecture, only the cell gate received the values 
            # from the inputs. Therefore weights are only initialized for this 
            # gate.
            x = getattr(self,'gate_{}_x'.format(gate))(x)
            
            # New information part that will be injected in the new context 
            g = nn.Tanh(x + h) * input_gate

            # Apply forget gate to forget old context/cell information.
            c = forget_gate * c_prev
            
            # Add/learn new context/cell information.
            c_next = g + c
            
            return c_next
        else:
        
            return nn.Sigmoid(h)
   

    def forward(self, x:torch.TensorFloat, hidden_states:tuple) -> tuple:
        """Forward for the cell.

        Args:
            x (torch.TensorFloat): Inputs to process through the cell.
            hidden_states (tuple): Tuple containing two tensors with the hidden 
            state and cell state respectively at time t-1.

        Returns:
            tuple: Tuple containing two tensors with the hidden state and cell 
            state values at time t.
        """
    
        # Get hidden states from t-1.
        (h_prev, c_prev) = hidden_states
        
        # Get outputs from input gate (to know what to learn).
        i = self._forward_gate(gate = 'input', x = None, h = h_prev)
        
        # Get outputs from forget gate (to know what to forget).
        f = self._forward_gate(gate = 'forget', x = None, h = h_prev)
        
        # User input and forget gates to forget old context and learn new 
        # context (cell information).
        c_next = self._forward_gate(gate = 'cell', x = x, h = h_prev, 
                    input_gate = i, forget_gate = f, c_prev = c_prev)
                    
        # Get outputs from the main output gate.
        o = self._forward_gate(gate = 'output', x = None, h = h_prev)
        
        # Produce next hidden output
        h_next = o * nn.Tanh(c_next)
        
        
        return h_next, c_next

#%% LSTM NO X-GATE CELL ARCHITECTURE (X = INPUT, FORGET, OR OUTPUT)
class LSTM_NXG(nn.Module):
    """
    LSTM cell with either the input, forget, or output gates cancelled out.
    """
    def __init__(self, input_size:int, hidden_size:int, drop_gate:str) -> None:
    
        super(LSTM_NXG, self).__init__()
        
        # Initialize inputs and hidden sizes
        self.input_size = input_size
        self.hidden_size = hidden_size
        
        if not drop_gate in ['input', 'forget', 'output']:
            raise ValueError('Parameter drop_gate ({}) not valid.' \
                             .format(drop_gate))
        
        self.drop_gate = drop_gate
        
        # Initialize gate components (weight and bias) for every gate:
        for gate in ['forget','input','cell','output']:
        
            # Skip initialization for the gate to drop.
            if drop_gate == gate: continue
        
            # Initialize gate weights (Wxg) that will process inputs at time t.
            # Gate component computation = Wxg * xt
            setattr(self,'gate_{}_x'.format(gate),
                nn.Linear(in_features = self.input_size, 
                          out_features = self.hidden_size, 
                          bias = False))
            
            # Initialize gate weights (Whg) that will process hidden states at 
            # time t-1, including bias associated to the gate.
            # Gate component computation = h[t-1]*Whg + bg
            setattr(self,'gate_{}_h'.format(gate),
                nn.Linear(in_features = self.hidden_size, 
                          out_features = self.hidden_size, 
                          bias = True))
        
        
    def _forward_gate(self, gate:str, x:torch.TensorFloat, h:torch.TensorFloat, 
            input_gate:torch.TensorFloat = None,
            forget_gate:torch.TensorFloat = None,
            c_prev:torch.TensorFloat = None) -> torch.TensorFloat:
        
        # Cancel the effect of the gate by making it 1.
        if self.drop_gate == gate: return 1
    
        # Check gate requested is within LSTM cell arquitecture.
        if not gate in ['forget','input','cell','output']:
            raise ValueError('Gate () not defined.'.format(gate))
        
        # Get gate components for the input x and hidden states h.
        x = getattr(self,'gate_{}_x'.format(gate))(x)
        h = getattr(self,'gate_{}_h'.format(gate))(h)
        
        if gate=='cell':
            
            # Check input_gate, forget_gate and cell state from t-1 is provided.
            if input_gate is None or forget_gate is None or c_prev is None:
                raise ValueError('Parameter(s) input_gate, forget_gate, ' + \
                                 'and/or c_prev missing.')
            
            # New information part that will be injected in the new context 
            g = nn.Tanh(x + h) * input_gate

            # Apply forget gate to forget old context/cell information.
            c = forget_gate * c_prev
            
            # Add/learn new context/cell information.
            c_next = g + c
            
            return c_next
        else:
        
            # Apply sigmoid function to gate.
            return nn.Sigmoid(x + h)
   

    def forward(self, x:torch.TensorFloat, hidden_states:tuple) -> tuple:
    
        # Get hidden states from t-1.
        (h_prev, c_prev) = hidden_states
        
        # Get outputs from input gate (to know what to learn).
        i = self._forward_gate(gate = 'input', x = x, h = h_prev)
        
        # Get outputs from forget gate (to know what to forget).
        f = self._forward_gate(gate = 'forget', x = x, h = h_prev)
        
        # User input and forget gates to forget old context and learn new 
        # context (cell information).
        c_next = self._forward_gate(gate = 'cell', x = x, h = h_prev, 
                    input_gate = i, forget_gate = f, c_prev = c_prev)
                    
        # Get outputs from the main output gate.
        o = self._forward_gate(gate = 'output', x = x, h = h_prev)
        
        # Produce next hidden output
        h_next = o * nn.Tanh(c_next)
        
        
        return h_next, c_next

 
#%% LSTM NO X-GATE ACTIVATION FUNCTION CELL ARCHITECTURE
class LSTM_NXGAF(nn.Module):
    """
    LSTM cell with the no activation function in either the inputs, forget, 
    outputs or cell gates.
    """
    def __init__(self, input_size:int, hidden_size:int, naf_gate:str) -> None:
    
        super(LSTM_NXGAF, self).__init__()
        
        # Initialize inputs and hidden sizes
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.naf_gate = naf_gate
        
        # Initialize gate components (weight and bias) for every gate:
        for gate in ['forget','input','cell','output']:
        
            # Initialize gate weights (Wxg) that will process inputs at time t.
            # Gate component computation = Wxg * xt
            setattr(self,'gate_{}_x'.format(gate),
                nn.Linear(in_features = self.input_size, 
                          out_features = self.hidden_size, 
                          bias = False))
            
            # Initialize gate weights (Whg) that will process hidden states at 
            # time t-1, including bias associated to the gate.
            # Gate component computation = h[t-1]*Whg + bg
            setattr(self,'gate_{}_h'.format(gate),
                nn.Linear(in_features = self.hidden_size, 
                          out_features = self.hidden_size, 
                          bias = True))
        
        
    def _forward_gate(self, gate:str, x:torch.TensorFloat, h:torch.TensorFloat, 
            input_gate:torch.TensorFloat = None,
            forget_gate:torch.TensorFloat = None,
            c_prev:torch.TensorFloat = None) -> torch.TensorFloat:
    
        # Check gate requested is within LSTM cell arquitecture.
        if not gate in ['forget','input','cell','output']:
            raise ValueError('Gate () not defined.'.format(gate))
        
        # Get gate components for the input x and hidden states h.
        x = getattr(self,'gate_{}_x'.format(gate))(x)
        h = getattr(self,'gate_{}_h'.format(gate))(h)
        
        if gate=='cell':
            
            # Check input_gate, forget_gate and cell state from t-1 is provided.
            if input_gate is None or forget_gate is None or c_prev is None:
                raise ValueError('Parameter(s) input_gate, forget_gate, ' + \
                                 'and/or c_prev missing.')
            
            # New information part that will be injected in the new context
            if self.naf_gate == 'cell':
                g = nn.Tanh(x + h) * input_gate
            else:
                g = (x + h) * input_gate

            # Apply forget gate to forget old context/cell information.
            c = forget_gate * c_prev
            
            # Add/learn new context/cell information.
            c_next = g + c
            
            return c_next
        else:
        
            # Apply sigmoid function to gate if applicable.
            return (x + h) if self.naf_gate == 'cell' else nn.Sigmoid(x + h)
   

    def forward(self, x:torch.TensorFloat, hidden_states:tuple) -> tuple:
    
        # Get hidden states from t-1.
        (h_prev, c_prev) = hidden_states
        
        # Get outputs from input gate (to know what to learn).
        i = self._forward_gate(gate = 'input', x = x, h = h_prev)
        
        # Get outputs from forget gate (to know what to forget).
        f = self._forward_gate(gate = 'forget', x = x, h = h_prev)
        
        # User input and forget gates to forget old context and learn new 
        # context (cell information).
        c_next = self._forward_gate(gate = 'cell', x = x, h = h_prev, 
                    input_gate = i, forget_gate = f, c_prev = c_prev)
                    
        # Get outputs from the main output gate.
        o = self._forward_gate(gate = 'output', x = x, h = h_prev)
        
        # Produce next hidden output
        h_next = o * nn.Tanh(c_next)
        
        
        return h_next, c_next

#%% GRU VANILLA CELL ARCHITECTURE
class GRU_Vanilla(nn.Module):
    """
    Vanilla Gate Recurrent Unit (GRU) cell with update, reset gates and 
    associated activation functions.
    """
    def __init__(self, input_size:int, hidden_size:int) -> None:

        super(GRU_Vanilla, self).__init__()
        
        # Initialize inputs and hidden sizes
        self.input_size = input_size
        self.hidden_size = hidden_size
        
        # Initialize gate components (weight and bias) for every gate:
        for gate in ['update','reset', 'hidden']:
        
            # Initialize gate weights (Wxg) that will process inputs at time t.
            # Gate component computation = Wxg * xt
            setattr(self,'gate_{}_x'.format(gate),
                nn.Linear(in_features = self.input_size, 
                          out_features = self.hidden_size, 
                          bias = False))
            
            # Initialize gate weights (Whg) that will process hidden states at 
            # time t-1, including bias associated to the gate.
            # Gate component computation = h[t-1]*Whg + bg
            setattr(self,'gate_{}_h'.format(gate),
                nn.Linear(in_features = self.hidden_size, 
                          out_features = self.hidden_size, 
                          bias = True))
            

    def _forward_gate(self, gate:str, x:torch.TensorFloat, h:torch.TensorFloat, 
            reset_gate:torch.TensorFloat = None) -> torch.TensorFloat:
    
        # Check gate requested is within LSTM cell arquitecture.
        if not gate in ['update','reset','hidden']:
            raise ValueError('Gate () not defined.'.format(gate))
        
        # Get gate components for the input x and hidden states h.
        x = getattr(self,'gate_{}_x'.format(gate))(x)
        
        if gate=='hidden':
            
            # Check input_gate, forget_gate and cell state from t-1 is provided.
            if reset_gate is None or h is None:
                raise ValueError('Parameter(s) input_gate, forget_gate, ' + \
                                 'and/or c_prev missing.')

            # New information part that will be injected in the new context 
            h_candidate = nn.Tanh(x + \
                getattr(self,'gate_{}_h'.format(gate))(reset_gate * h))

            return h_candidate
        else:

            h = getattr(self,'gate_{}_h'.format(gate))(h)

            # Apply sigmoid function to gate.
            return nn.Sigmoid(x + h)
        
    def forward(self, x:torch.TensorFloat, h_prev:torch.TensorFloat) -> tuple:
            
        # Get outputs from update gate.
        u = self._forward_gate(gate = 'update', x = x, h = h_prev)
        
        # Get outputs from reset gate.
        r = self._forward_gate(gate = 'reset', x = x, h = h_prev)
        
        # Get candidate hidden state.
        h_candidate = self._forward_gate(gate = 'hidden', x = x, h = h_prev, 
                                    relevance_gate = r)
        
        # Produce next hidden output
        h_next = (1 - u) * h_prev + u * h_candidate
        
        return h_next

#%% GRU SLIMX (X = 1, 2 OR 3) CELL ARCHITECTURE
class GRU_SLIMX(nn.Module):
    """
    SLIM GRU cell architecture: update and reset gates only relying on 
    incoming hidden states. There are three different versions for the SLIM 
    architecture where the update, and reset gates perform different 
    calculations:
        SLIM1 - Gates contain hidden states components and bias (Wh + b).
        SLIM2 - Gates contain only weights for hidden states (Wh).
        SLIM3 - Gates contain only learnable bias (b).
    """
    def __init__(self, input_size:int, hidden_size:int, slim_version:int = 1) -> None:

        super(GRU_SLIMX, self).__init__()
        
        # Initialize inputs and hidden sizes
        self.input_size = input_size
        self.hidden_size = hidden_size
        self._version = slim_version
        
        # Initialize gate components (weight and bias) for every gate:
        for gate in ['update','reset', 'hidden']:
        
            if gate=='hidden':
                # Initialize gate weights (Wxg) that will process inputs at time t.
                # Gate component computation = Wxg * xt
                setattr(self,'gate_{}_x'.format(gate),
                    nn.Linear(in_features = self.input_size, 
                            out_features = self.hidden_size, 
                            bias = False))
                
                # Initialize gate weights (Whg) that will process hidden states at 
                # time t-1, including bias associated to the gate.
                # Gate component computation = h[t-1]*Whg + bg
                setattr(self,'gate_{}_h'.format(gate),
                    nn.Linear(in_features = self.hidden_size, 
                            out_features = self.hidden_size, 
                            bias = True))
                
            else:
        
                # If SLIM1 or SLIM2, initialize gate weight components.
                if slim_version<=2:
                    # Initialize gate components that will process hidden 
                    # states at time t-1, including bias associated to the gate 
                    # if SLIM1.
                    # Gate component computation = h[t-1]*Whg + <bg>
                    setattr(self,'gate_{}_h'.format(gate),
                        nn.Linear(in_features = self.hidden_size, 
                                  out_features = self.hidden_size, 
                                  bias = True if slim_version == 1 else False))
                else:
                                  
                    # Initialize gate bias (bg) that will process hidden states 
                    # at t-1.
                    setattr(self, 'gate_{}_h'.format(gate), BiasLayer())
            

    def _forward_gate(self, gate:str, x:torch.TensorFloat, h:torch.TensorFloat, 
            reset_gate:torch.TensorFloat = None) -> torch.TensorFloat:
    
        # Check gate requested is within LSTM cell arquitecture.
        if not gate in ['update','reset','hidden']:
            raise ValueError('Gate () not defined.'.format(gate))
        
        # Get gate components for the input x and hidden states h.
        x = getattr(self,'gate_{}_x'.format(gate))(x)
        
        if gate=='hidden':
            
            # Check input_gate, forget_gate and cell state from t-1 is provided.
            if reset_gate is None or h is None:
                raise ValueError('Parameter(s) input_gate, forget_gate, ' + \
                                 'and/or c_prev missing.')

            # New information part that will be injected in the new context 
            h_candidate = nn.Tanh(x + \
                getattr(self,'gate_{}_h'.format(gate))(reset_gate * h))

            return h_candidate
        else:

            h = getattr(self,'gate_{}_h'.format(gate))(h)

            # Apply sigmoid function to gate.
            return nn.Sigmoid(x + h)
        
    def forward(self, x:torch.TensorFloat, h_prev:torch.TensorFloat) -> tuple:
            
        # Get outputs from update gate.
        u = self._forward_gate(gate = 'update', x = x, h = h_prev)
        
        # Get outputs from reset gate.
        r = self._forward_gate(gate = 'reset', x = x, h = h_prev)
        
        # Get candidate hidden state.
        h_candidate = self._forward_gate(gate = 'hidden', x = x, h = h_prev, 
                                    relevance_gate = r)
        
        # Produce next hidden output
        h_next = (1 - u) * h_prev + u * h_candidate
        
        return h_next


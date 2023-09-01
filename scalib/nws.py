
# Import custom libraries from local folder.
from importlib import reload
import os
import sys
sys.path.append("..")

import torch
import torch.nn as nn

import scalib.layers as layers
import scalib.cells as cell


networks = nn.ModuleDict({})

input_size = 66
output_size = 66
dropout = 0.2

#%% Kessler (LSTM Vanilla)
networks.update({'kessler':
                 nn.ModuleDict({'lstm': nn.LSTM(input_size = input_size,
                                        batch_first = True,
                                        hidden_size = 264,
                                        num_layers = 2,
                                        dropout = dropout),
                                'dropout': nn.Dropout(p = dropout),
                                'relu': nn.ReLU(),
                                'linear': nn.Linear(264, output_size)
                                })
                })

#%% LSTM layer with Vanilla cell architecture
networks.update({'lstm_vanilla':
                 nn.ModuleDict({'lstm': layers.LSTM(input_size = input_size,
                                         batch_first = True, 
                                         hidden_size = 264, 
                                         num_layers = 2,
                                         dropout = dropout,
                                         cell = cell.LSTM_Vanilla),
                                'dropout': nn.Dropout(p = dropout),
                                'relu': nn.ReLU(),
                                'linear': nn.Linear(264, output_size)
                                })
                })

#%% LSTM layer with SLIMx (x = 1, 2 & 3) cell architecture
h_szs = [313, 313, 897]
for v in [1, 2, 3]:
    networks.update({f'lstm_slim{v}':
                     nn.ModuleDict({'lstm': layers.LSTM(input_size = input_size, 
                                            hidden_size = h_szs[v-1],
                                            cell = cell.LSTM_SLIMX, 
                                            num_layers = 2 if v < 3 else 1,
                                            dropout = dropout,
                                            **dict(version = v)),
                                    'dropout': nn.Dropout(p = dropout),
                                    'relu': nn.ReLU(),
                                    'linear': nn.Linear(h_szs[v-1], output_size)
                                    })
                    })
    
#%% LSTM layer with NXG (X = Input, Forget & Output) cell architecture
h_szs = [306, 306, 306] 
for g, gate in enumerate(['input', 'output', 'forget']):
    networks.update({f'lstm_n{gate[0]}g':
                     nn.ModuleDict({'lstm': layers.LSTM(input_size = input_size, 
                                            hidden_size = h_szs[g],
                                            cell = cell.LSTM_NXG, 
                                            num_layers = 2,
                                            dropout = dropout,
                                            **dict(drop_gate = gate)),
                                    'dropout': nn.Dropout(p = dropout),
                                    'relu': nn.ReLU(),
                                    'linear': nn.Linear(h_szs[g], output_size)
                                    })
                    })
    
#%% LSTM layer with NXGAF (X = Input, Forget & Output) cell architecture    
for gate in ['input', 'output', 'forget']:
    networks.update({f'lstm_n{gate[0]}gaf':
                     nn.ModuleDict({'lstm': layers.LSTM(input_size = input_size, 
                                            hidden_size = 264,
                                            cell = cell.LSTM_NXGAF, 
                                            num_layers = 2,
                                            dropout = dropout,
                                            **dict(naf_gate = gate)),
                                    'dropout': nn.Dropout(p = dropout),
                                    'relu': nn.ReLU(),
                                    'linear': nn.Linear(264, output_size)
                                    }),
                    
                    })
    
#%% Self-Attention layer with LSTM Vanilla layers as encoders
networks.update({f'lstm_attn-vanilla':
                 nn.ModuleDict({'lstm_encoder': 
                                    layers.LSTM(
                                        input_size = input_size,
                                        hidden_size = 230,
                                        cell = cell.LSTM_Vanilla),
                                'attention': 
                                    layers.SelfAttentionLayer(
                                        input_size = 230),
                                'lstm_decoder': 
                                    layers.LSTM(
                                        input_size = 230, 
                                        hidden_size = 230,
                                        cell = cell.LSTM_Vanilla),
                                'dropout': nn.Dropout(p = dropout),
                                'relu': nn.ReLU(),
                                'linear': nn.Linear(230, output_size)
                                })
                })

#%% GRU layer with Vanilla cell architecture
networks.update({'gru_vanilla':
                 nn.ModuleDict({'gru': layers.GRU(input_size = input_size,
                                         batch_first = True, 
                                         hidden_size = 306, 
                                         num_layers = 2,
                                         dropout = dropout,
                                         cell = cell.GRU_Vanilla),
                                'dropout': nn.Dropout(p = dropout),
                                'relu': nn.ReLU(),
                                'linear': nn.Linear(306, output_size)
                                })
                })

#%% LSTM layer with SLIMx (x = 1, 2 & 3) cell architecture
h_szs = [354, 354, 897]
for v in [1, 2, 3]:
    networks.update({f'gru_slim{v}':
                     nn.ModuleDict({'gru': layers.GRU(input_size = input_size, 
                                            hidden_size = h_szs[v-1],
                                            cell = cell.GRU_SLIMX, 
                                            num_layers = 2 if v < 3 else 1,
                                            dropout = dropout,
                                            **dict(version = v)),
                                    'dropout': nn.Dropout(p = dropout),
                                    'relu': nn.ReLU(),
                                    'linear': nn.Linear(h_szs[v-1], output_size)
                                    })
                    })
    
#%% GRU layer with MUT3 cell architecture
networks.update({f'gru_mut3':
                    nn.ModuleDict({'gru': layers.GRU(input_size = input_size, 
                                        hidden_size = 306,
                                        cell = cell.GRU_MUTX, 
                                        num_layers = 2,
                                        dropout = dropout,
                                        **dict(version = 3)),
                                    'dropout': nn.Dropout(p = dropout),
                                    'relu': nn.ReLU(),
                                    'linear': nn.Linear(306, output_size)
                                })
                })

#%% Self-Attention layer with LSTM Vanilla layers as encoders
networks.update({f'gru_attn-vanilla':
                 nn.ModuleDict({'gru_encoder': 
                                    layers.GRU(
                                        input_size = input_size,
                                        hidden_size = 256,
                                        cell = cell.GRU_Vanilla),
                                'attention': 
                                    layers.SelfAttentionLayer(
                                        input_size = 256),
                                'gru_decoder': 
                                    layers.GRU(
                                        input_size = 256, 
                                        hidden_size = 256,
                                        cell = cell.GRU_Vanilla),
                                'dropout': nn.Dropout(p = dropout),
                                'relu': nn.ReLU(),
                                'linear': nn.Linear(256, output_size)
                                })
                })
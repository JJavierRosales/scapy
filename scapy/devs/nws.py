
# Import custom libraries from local folder.
from importlib import reload
import os


import torch
import torch.nn as nn

import sys
sys.path.append("..")

import scapy.layers
import scapy.cells as cell

networks = nn.ModuleDict({})

input_size = 66
output_size = 66
dropout = 0.2

#%% Kessler (LSTM Vanilla)
# networks.update({'kessler':
#                  nn.ModuleDict({'lstm': nn.LSTM(input_size = input_size,
#                                         batch_first = True,
#                                         hidden_size = 264,
#                                         num_layers = 2,
#                                         dropout = dropout),
#                                 'dropout': nn.Dropout(p = dropout),
#                                 'relu': nn.ReLU(),
#                                 'linear': nn.Linear(264, output_size)
#                                 })
#                 })

#%% LSTM layer with Vanilla cell architecture
h_sz = 264
networks.update({'lstm_vanilla':
                 nn.ModuleDict({'lstm': layers.LSTM(input_size = input_size,
                                         batch_first = True, 
                                         hidden_size = h_sz, 
                                         num_layers = 2,
                                         dropout = dropout,
                                         cell = cell.LSTM_Vanilla),
                                'dropout': nn.Dropout(p = dropout),
                                'relu': nn.ReLU(),
                                'linear': nn.Linear(h_sz, output_size)
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
h_sz = 264 
for gate in ['input', 'output', 'forget']:
    networks.update({f'lstm_n{gate[0]}gaf':
                     nn.ModuleDict({'lstm': layers.LSTM(input_size = input_size, 
                                            hidden_size = h_sz,
                                            cell = cell.LSTM_NXGAF, 
                                            num_layers = 2,
                                            dropout = dropout,
                                            **dict(naf_gate = gate)),
                                    'dropout': nn.Dropout(p = dropout),
                                    'relu': nn.ReLU(),
                                    'linear': nn.Linear(h_sz, output_size)
                                    }),
                    
                    })
    
#%% LSTM layer with Peephole Connections cell architecture
h_sz = 217
networks.update({'lstm_pc':
                 nn.ModuleDict({'lstm': layers.LSTM(input_size = input_size,
                                         batch_first = True, 
                                         hidden_size = h_sz, 
                                         num_layers = 2,
                                         dropout = dropout,
                                         cell = cell.LSTM_PC),
                                'dropout': nn.Dropout(p = dropout),
                                'relu': nn.ReLU(),
                                'linear': nn.Linear(h_sz, output_size)
                                })
                })
#%% LSTM layer with Forget Bias 1 cell architecture
h_sz = 264
networks.update({'lstm_fb1':
                 nn.ModuleDict({'lstm': layers.LSTM(input_size = input_size,
                                         batch_first = True, 
                                         hidden_size = h_sz, 
                                         num_layers = 2,
                                         dropout = dropout,
                                         cell = cell.LSTM_FB1),
                                'dropout': nn.Dropout(p = dropout),
                                'relu': nn.ReLU(),
                                'linear': nn.Linear(h_sz, output_size)
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
    
#%% Self-Attention layer with LSTM Vanilla layers as encoders
h_sz = 230
networks.update({f'lstm_attn-vanilla':
                 nn.ModuleDict({'lstm_encoder': 
                                    layers.LSTM(
                                        input_size = input_size,
                                        hidden_size = h_sz,
                                        cell = cell.LSTM_Vanilla),
                                'attention': 
                                    layers.SelfAttentionLayer(
                                        input_size = h_sz),
                                'lstm_decoder': 
                                    layers.LSTM(
                                        input_size = h_sz, 
                                        hidden_size = h_sz,
                                        cell = cell.LSTM_Vanilla),
                                'dropout': nn.Dropout(p = dropout),
                                'relu': nn.ReLU(),
                                'linear': nn.Linear(h_sz, output_size)
                                })
                })

#%% GRU layer with Vanilla cell architecture
h_sz = 306
networks.update({'gru_vanilla':
                 nn.ModuleDict({'gru': layers.GU(input_size = input_size,
                                         batch_first = True, 
                                         hidden_size = h_sz, 
                                         num_layers = 2,
                                         dropout = dropout,
                                         cell = cell.GRU_Vanilla),
                                'dropout': nn.Dropout(p = dropout),
                                'relu': nn.ReLU(),
                                'linear': nn.Linear(h_sz, output_size)
                                })
                })

#%% GRU layer with SLIMx (x = 1, 2 & 3) cell architecture
h_szs = [354, 354, 898]
for v in [1, 2, 3]:
    networks.update({f'gru_slim{v}':
                     nn.ModuleDict({'gru': layers.GU(input_size = input_size, 
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
h_sz = 306
networks.update({f'gru_mut3':
                    nn.ModuleDict({'gru': layers.GU(input_size = input_size, 
                                        hidden_size = h_sz,
                                        cell = cell.GRU_MUTX, 
                                        num_layers = 2,
                                        dropout = dropout,
                                        **dict(version = 3)),
                                    'dropout': nn.Dropout(p = dropout),
                                    'relu': nn.ReLU(),
                                    'linear': nn.Linear(h_sz, output_size)
                                })
                })

#%% Self-Attention layer with GRU Vanilla layers as encoders
h_sz = 256
networks.update({f'gru_attn-vanilla':
                 nn.ModuleDict({'gru_encoder': 
                                    layers.GU(
                                        input_size = input_size,
                                        hidden_size = h_sz,
                                        cell = cell.GRU_Vanilla),
                                'attention': 
                                    layers.SelfAttentionLayer(
                                        input_size = h_sz),
                                'gru_decoder': 
                                    layers.GU(
                                        input_size = h_sz, 
                                        hidden_size = h_sz,
                                        cell = cell.GRU_Vanilla),
                                'dropout': nn.Dropout(p = dropout),
                                'relu': nn.ReLU(),
                                'linear': nn.Linear(h_sz, output_size)
                                })
                })

#%% MGU layer with Vanilla cell architecture
h_sz = 376
networks.update({'mgu_vanilla':
                 nn.ModuleDict({'mgu': layers.GU(input_size = input_size,
                                         batch_first = True, 
                                         hidden_size = h_sz, 
                                         num_layers = 2,
                                         dropout = dropout,
                                         cell = cell.MGU_Vanilla),
                                'dropout': nn.Dropout(p = dropout),
                                'relu': nn.ReLU(),
                                'linear': nn.Linear(h_sz, output_size)
                                })
                })

#%% GRU layer with SLIMx (x = 1, 2 & 3) cell architecture
h_szs = [417, 417, 897]
for v in [1, 2, 3]:
    networks.update({f'mgu_slim{v}':
                     nn.ModuleDict({'mgu': layers.GU(input_size = input_size, 
                                            hidden_size = h_szs[v-1],
                                            cell = cell.MGU_SLIMX, 
                                            num_layers = 2 if v < 3 else 1,
                                            dropout = dropout,
                                            **dict(version = v)),
                                    'dropout': nn.Dropout(p = dropout),
                                    'relu': nn.ReLU(),
                                    'linear': nn.Linear(h_szs[v-1], output_size)
                                    })
                    })
    
#%% Self-Attention layer with MGU Vanilla layers as encoders
h_sz = 294
networks.update({f'mgu_attn-vanilla':
                 nn.ModuleDict({'mgu_encoder': 
                                    layers.GU(
                                        input_size = input_size,
                                        hidden_size = h_sz,
                                        cell = cell.MGU_Vanilla),
                                'attention': 
                                    layers.SelfAttentionLayer(
                                        input_size = h_sz),
                                'mgu_decoder': 
                                    layers.GU(
                                        input_size = h_sz, 
                                        hidden_size = h_sz,
                                        cell = cell.MGU_Vanilla),
                                'dropout': nn.Dropout(p = dropout),
                                'relu': nn.ReLU(),
                                'linear': nn.Linear(h_sz, output_size)
                                })
                })
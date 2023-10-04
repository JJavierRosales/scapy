# Libraries used for hinting
from __future__ import annotations
from typing import Union

import pandas as pd
import numpy as np
from datetime import datetime, timedelta

import torch
import torch.nn as nn
from torch.utils.data import TensorDataset
from sklearn.model_selection import train_test_split

import sys
sys.path.append("..")

# Import local modules.
from . import utils
from .cdm import ConjunctionDataMessage as CDM
from .event import ConjunctionEvent as CE 
from .event import ConjunctionEventsDataset as CED

#%% FUNCTION: kelvins_challenge_events
def kelvins_challenge_events(filepath:str, num_events:int = None, 
        date_tca:datetime = None, remove_outliers:bool = True,
        drop_features:list = ['c_rcs_estimate', 't_rcs_estimate'], 
        print_log:bool=False) -> CED:
    """Import Kelvins Challenge dataset as a ConjunctionEventsDataset object.

    Args:
        filepath (str): Path where Kelvins dataset is located.
        num_events (int, optional): Number of events to import. Defaults to None.
        date_tca (datetime, optional): _description_. Defaults to None.
        remove_outliers (bool, optional): Flag to remove outliers. Defaults to 
        True.
        drop_features (list, optional): List of features from original dataset 
        to remove. Defaults to ['c_rcs_estimate', 't_rcs_estimate'].
        print_log (bool, optional): Print description of the process. Defaults 
        to False.

    Returns:
        ConjunctionEventsDataset: Object containing all Events objects.
    """

    # Import Kelvins dataset using pandas.
    print('Loading Kelvins Challenge dataset from external file ...', end='\r')
    kelvins = pd.read_csv(filepath)
    print('Kelvins Challenge dataset imported from external file ' + \
          '({} entries):\n{}'.format(len(kelvins), filepath))

    utils.tabular_list(drop_features,n_cols=1)
    # Drop features if passed to the function
    if len(drop_features)>0:
        kelvins = kelvins.drop(drop_features, axis=1)
        if print_log: print('Features removed:\n{}' \
            .format(utils.tabular_list(drop_features, n_cols=2, col_sep=' - ')))

    # Remove rows containing NaN values.
    if print_log: print('Dropping rows with NaNs...', end='\r')
    kelvins = kelvins.dropna()
    if print_log: print(f'Dropping rows with NaNs... {len(kelvins)} '
                        f'entries remaining.')

    if remove_outliers:

        if print_log: print('Removing outliers...', end='\r')
        kelvins = kelvins[kelvins['t_sigma_r'] <= 20]
        kelvins = kelvins[kelvins['c_sigma_r'] <= 1000]
        kelvins = kelvins[kelvins['t_sigma_t'] <= 2000]
        kelvins = kelvins[kelvins['c_sigma_t'] <= 100000]
        kelvins = kelvins[kelvins['t_sigma_n'] <= 10]
        kelvins = kelvins[kelvins['c_sigma_n'] <= 450]

        if print_log: print(f'Removing outliers... {len(kelvins)} '
                            f'entries remaining.')

    # Shuffle data.
    kelvins = kelvins.sample(frac=1, axis=1).reset_index(drop=True)

    # Get CDMs grouped by event_id
    kelvins_events = kelvins.groupby('event_id').groups
    if print_log: print('Grouped rows into {} events'.format(len(kelvins_events)))

    # Get TCA as current datetime (not provided in Kelvins dataset).
    if date_tca is None: date_tca = datetime.now()
    if print_log: print('Taking TCA as current time: {}'.format(date_tca))

    # Get number of events to import from Kelvins dataset.
    num_events = len(kelvins_events) if num_events is None \
        else min(num_events, len(kelvins_events))

    # Initialize array of RTN reference frame for position and velocity.
    rtn_components = ['R', 'T', 'N', 'RDOT', 'TDOT', 'NDOT']

    # Iterate over all features to get the time series subsets
    pb_events = utils.ProgressBar(iterations = range(num_events),
                    title = 'KELVINS DATASET IMPORT:', 
                    description='Importing Events from Kelvins dataset...')

    # Initialize counter for progressbar
    n = 0

    # Initialize events list to store all Event objects.
    events = []

    # Iterate over all events in Kelvins dataset
    for event_id, rows in kelvins_events.items():

        # Update counter for progress bar.
        n += 1
        if n > num_events: break

        # Initialize CDM list to store all CDMs contained in a single event.
        event_cdms = []

        # Iterate over all CDMs in the event.
        for _, k_cdm in kelvins.iloc[rows].iterrows():

            # Initialize CDM object.
            cdm = CDM()

            time_to_tca = k_cdm['time_to_tca']  # days
            date_creation = date_tca - timedelta(days=time_to_tca)

            cdm['CREATION_DATE'] = CDM.datetime_to_str(date_creation)
            cdm['TCA'] = CDM.datetime_to_str(date_tca)

            cdm['MISS_DISTANCE'] = k_cdm['miss_distance']
            cdm['RELATIVE_SPEED'] = k_cdm['relative_speed']


            # Get relative state vector components.
            for state in ['POSITION', 'VELOCITY']:
                for rtn in rtn_components[:3]:
                    feature = 'RELATIVE_{}_{}'.format(state, rtn)
                    cdm[feature] = k_cdm[feature.lower()] 

            # Get object specific features.
            for k, v in {'OBJECT1':'t', 'OBJECT2':'c'}.items():

                # Get covariance matrix elements for both objects (lower 
                # diagonal).
                for i, i_rtn in enumerate(rtn_components):
                    for j, j_rtn in enumerate(rtn_components):
                        if j>i: continue

                        # Get feature label in uppercase
                        feature = '_C{}_{}'.format(i_rtn, j_rtn)

                        if i_rtn == j_rtn:
                            cdm[k + feature] = \
                                k_cdm['{}_sigma_{}' \
                                      .format(v,i_rtn).lower()]**2.0
                        else:
                            # Re-scale non-diagonal elements of the covariance
                            # matrix using the variances of the variable (i.e
                            # CR_T = t_cr_t * t_sigma_r * t_sigma_t)
                            cdm[k + feature] = \
                                k_cdm[v + feature.lower()] * \
                                k_cdm['{}_sigma_{}'.format(v,j_rtn).lower()] * \
                                k_cdm['{}_sigma_{}'.format(v,i_rtn).lower()]

                cdm[k+'_RECOMMENDED_OD_SPAN'] = k_cdm[v+'_recommended_od_span']
                cdm[k+'_ACTUAL_OD_SPAN'] = k_cdm[v+'_actual_od_span']
                cdm[k+'_OBS_AVAILABLE'] = k_cdm[v+'_obs_available']
                cdm[k+'_OBS_USED'] = k_cdm[v+'_obs_used']
                cdm[k+'_RESIDUALS_ACCEPTED'] = k_cdm[v+'_residuals_accepted']
                cdm[k+'_WEIGHTED_RMS'] = k_cdm[v+'_weighted_rms']
                cdm[k+'_SEDR'] = k_cdm[v+'_sedr']

                # Get number of days until CDM creation.
                for t in ['start', 'end']:
                    time_lastob = k_cdm['{}_time_lastob_{}'.format(v,t)]
                    time_lastob = date_creation - timedelta(days=time_lastob)
                    cdm['{}_TIME_LASTOB_{}'.format(k, t).upper()] = \
                        CDM.datetime_to_str(time_lastob)

            cdm['OBJECT1_OBJECT_TYPE'] = 'PAYLOAD'
            cdm['OBJECT2_OBJECT_TYPE'] = k_cdm['c_object_type']

            cdm['COLLISION_PROBABILITY'] = k_cdm['risk']

            values_extra = {}
            for feature in ['max_risk_estimate', 'max_risk_scaling']:
                if not feature in list(k_cdm.keys()): continue
                values_extra.update({f'__{feature.upper()}': k_cdm[feature]})

            cdm._values_extra.update(values_extra)

            # Append CDM object to the event list.
            event_cdms.append(cdm)

            # Update progress bar
            pb_events.refresh(i = n, nested_progress = True)

        # Append ConjunctionEvent object to events list.
        events.append(CE(event_cdms))

    # Update progress bar.
    pb_events.refresh(i = n-1, 
        description = f'Dataset imported ({len(events)} events).')

    return CED(events=events)

#%% CLASS: Dataset
class Dataset(TensorDataset):
    """Dataset object instanciator from TensorDataset.

    Args:
        TensorDataset (class): Pytorch TensorDataset loader.
    """
    def __init__(self, X:torch.Tensor, y:torch.Tensor) -> None:
        """Initialise Dataset object.

        Args:
            X (torch.Tensor): Input data.
            y (torch.Tensor): Output data (targets).
        """
        self.inputs = X
        self.outputs = y

        self.input_size = X.size(1)
        self.output_size = y.size(1)

        self.len = self.inputs.shape[0]

    def split(self, split_size:float=0.8, shuffle:bool=True, 
              random_state:int=42) -> None:
        """Split data into two groups.

        Args:
            split_size (float, optional): Split proportion. Defaults to 0.8.
            shuffle (bool, optional): Suffle data. Defaults to True.
            random_state (int, optional): Random state to enable 
            reproducibility. Defaults to 42.
        """

        Xi, Xj, yi, yj = train_test_split(self.inputs, self.outputs, 
                                        train_size = split_size, 
                                        random_state = random_state, 
                                        shuffle = shuffle)
        
        return Dataset(Xi, yi), Dataset(Xj, yj)
    
    def __getitem__(self, index:Union[int, slice]):
        """Get row from the tensor dataset.

        Args:
            index (Union[int,slice]): Index(es) to retrieve

        Returns:
            tuple: Inputs and output values.
        """
        return self.inputs[index], self.outputs[index]
    
    def __len__(self):
        """Get number of entries in the tensor dataset.
        """
        return self.len
    
    def __repr__(self) -> str:
        """Print information about the object constructor.

        Returns:
            str: Information of the object.
        """
        description = f'TensorDataset(Inputs: {self.inputs.size(1)} ' + \
                    f'| Outputs: {self.outputs.size(1)} ' + \
                    f'| Entries: {len(self.inputs)})'
        return description

#%% CLASS: TensorDatasetFromDataFrame
class TensorDatasetFromDataFrame():
    """Tensor Dataset from pandas DataFrame instanciator.
    """
    def __init__(self, df_input:pd.DataFrame, output_features:list, 
                 input_features:list, normalise_inputs:bool=True):
        """Initialise Tensor Dataset from pandas DataFrame object.

        Args:
            df_input (pd.DataFrame): Input pandas DataFrame.
            output_features (list): List of output features (targets).
            input_features (list): List of input features.
            normalise_inputs (bool, optional): Normalise inputs. Defaults to 
            True.

        Raises:
            ValueError: Some of the features are not present in the input 
            DataFrame.
        """
        
        # Remove empty columns
        df = df_input.copy()
        df.dropna(inplace=True, axis=1, how='all')

        # Initialise list with the categorical features and numerical features
        self.cat_features = []
        self.num_features = []
        undefined_features = []

        for f in input_features:
            if not f in df.columns: continue

            if df[f].dtype=='category':
                self.cat_features.append(f)
            elif f in df._get_numeric_data().columns:
                self.num_features.append(f)
            elif not f in output_features:
                undefined_features.append(f)

        if len(undefined_features)>0:
            unsupported_dtypes = []
            for f in undefined_features:
                unsupported_dtypes.append(df[f].dtype)

            raise ValueError(f'Some features ({undefined_features}) have ' + \
                             f'unsupported dtypes ({set(unsupported_dtypes)}).')
        
        # Check that all output features are part of the input dataframe.
        for o in output_features:
           if not o in df.columns:
              undefined_features.append(o)
        
        self.input_features = self.cat_features + self.num_features
        self.output_features = output_features
        

        self.df_outputs = df[self.output_features]
        self.df_inputs = df[self.input_features]

        # Get Torch for output features
        self.outputs = torch.tensor(self.df_outputs.to_numpy(), 
                        dtype=torch.float, 
                        requires_grad=True) 
        
        

        # Get Torch for all continuous features
        X_num = torch.tensor(self.df_inputs[self.num_features].to_numpy(), 
                            dtype=torch.float, 
                            requires_grad=True)
        X_num = torch.nan_to_num(X_num)

        # Normalise continuous variables
        if normalise_inputs:
            bn_nums = nn.BatchNorm1d(len(self.num_features))
            X_num = bn_nums(X_num)


        # Get Torch for all categorical features
        if len(self.cat_features)>0:
            cats = [self.df_inputs[f].cat.codes.values for f in self.cat_features]
            cats = np.stack(cats, 1)
            cats = torch.tensor(cats, dtype=torch.int)

            # This will set embedding sizes for the categorical columns:
            # an embedding size is the length of the array into which every category
            # is converted
            cat_szs = [len(self.df_inputs[f].cat.categories) for f in self.cat_features]
            emb_szs = [(size, min(50, (size+1)//2)) for size in cat_szs]

            # Initialize list of embedding operations.
            #  - nuv = Number of unique vectors
            #  - nvc = Number of vector componets
            embeddings = nn.ModuleList([nn.Embedding(nuv, nvc) for nuv, nvc in emb_szs])

            # Initialize embeddings list
            embeddings_tensors = []

            # Apply embedding operation to every column of categorical tensor
            for i, embedding in enumerate(embeddings):
                #print(cats[:,i])
                embeddings_tensors.append(embedding(cats[:,i]))

            # Concatenate embedding sections into 1
            X_cat = torch.cat(embeddings_tensors, 1)
        
            # Concatenate embeddings with continuous variables into one torch
            self.inputs = torch.cat([X_cat, X_num], 1)
        else:
            self.inputs = X_num

        # Get final dataset with inputs and targets
        self.data = Dataset(self.inputs, self.outputs)

        self.len = self.inputs.shape[0]
        self.input_size = self.data.input_size
        self.output_size = self.data.output_size

    def __repr__(self) -> str:
        """Print information about the object constructor.

        Returns:
            str: Information of the object.
        """
        description = f'TensorDatasetFromDataFrame(' + \
                     f'Inputs: {self.input_size} ' + \
                     f'| Outputs: {self.output_size} ' + \
                     f'| Entries: {len(self.inputs)})'
        return description
    
    def __getitem__(self, index:Union[int,slice]) -> tuple:
        """Get row from the tensor dataset.

        Args:
            index (Union[int,slice]): Index(es) to retrieve

        Returns:
            tuple: Inputs and output values.
        """
        return self.inputs[index], self.outputs[index]
    
    def __len__(self):
        """Get number of entries in the tensor dataset.
        """
        return self.len
# if __name__ == "__main__":

#     # y_true = np.array([1, 2, 3, 4])
#     # y_pred = np.array([1, 4, 3, 4])

#     # print(coefficient_of_determination(y_true, y_pred))
#     # print(r2_sklearn(y_true, y_pred))

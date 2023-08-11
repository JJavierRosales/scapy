# Libraries used for hinting
from __future__ import annotations
from typing import Type, Union


import numpy as np
import warnings
from datetime import datetime as dt
import copy
import pandas as pd
from . import utils, cfg
from .cfg import *



#%%
# Based on CCSDS 508.0-B-1
# https://public.ccsds.org/Pubs/508x0b1e2c1.pdf
class ConjunctionDataMessage():
    def __init__(self, filepath:str = None, set_defaults:bool = True):
        # Header
        # Relative metadata
        # Object 1
        #  Metadata, OD, State, Covariance
        # Object 2
        #  Metadata, OD, State, Covariance
        # Comments are optional and not currently supported by this class

        for cluster, keys in cdm_clusters.items():
            # Set clusters of CDM fields (header, relative_metadata, ... etc.)
            # containing the list of features embeded in every cluster.
            setattr(self, '_keys_' + cluster, keys)

            if 'obligatory' in cluster: continue

            # Initialize dictionary with the list of features contained in the
            # cluster (i.e. tca, miss_distance, relative_speed, ... etc.)
            dict_values = dict.fromkeys(keys)

            # Initialize variable to store the values of the CDM per cluster
            if cluster in ['header', 'relative_metadata']:
                setattr(self, '_values_' + cluster, dict_values)
            else:
                setattr(self, '_values_object_' + cluster, 
                        [dict_values, dict_values])

<<<<<<< HEAD
=======
        for cluster, keys in cdm_clusters_obligatory.items():
            # Set clusters of CDM fields (header, relative_metadata, ... etc.)
            # containing the list of features embeded in every cluster.
            setattr(self, '_keys_' + cluster, keys)

>>>>>>> dev
        # This holds extra key, value pairs associated with each CDM object, 
        # used internally by the Kessler codebase and not a part of the CDM 
        # standard
        self._values_extra = {}

        self._keys_with_dates = \
            ['CREATION_DATE', 'TCA', 'SCREEN_ENTRY_TIME', 'START_SCREEN_PERIOD', 
            'STOP_SCREEN_PERIOD','SCREEN_EXIT_TIME','OBJECT1_TIME_LASTOB_START', 
            'OBJECT1_TIME_LASTOB_END', 'OBJECT2_TIME_LASTOB_START', 
            'OBJECT2_TIME_LASTOB_END']


        if set_defaults:
            self.set_header('CCSDS_CDM_VERS', '1.0')
            self.set_header('CREATION_DATE', dt.utcnow().isoformat())
            self.set_object(0, 'OBJECT', 'OBJECT1')
            self.set_object(1, 'OBJECT', 'OBJECT2')

        if filepath:
            self.copy_from(ConjunctionDataMessage.load(filepath))

    def copy(self) -> ConjunctionDataMessage:
        """Creates a deep copy of a CDM object.

        Returns:
            ConjunctionDataMessage: Deep copy of CDM object.
        """

        # Instanciate new empty CDM object
        ret = ConjunctionDataMessage()

        for cluster in cdm_clusters.keys():
            if 'obligatory' in cluster: continue

            # Set clusters of CDM fields (header, relative_metadata, ... etc.)
            # containing the list of features embeded in every cluster.
            if cluster in ['header', 'relative_metadata']:
                cluster = '_values_' + cluster
            else:
                cluster = '_values_object_' + cluster

            # Set values in the new CDM object from self
            setattr(ret, cluster, copy.deepcopy(getattr(self, cluster)))

        return ret

    def copy_from(self, other_cdm) -> None:
        """Copies CDM object into self attribute within class.

        Args:
            other_cdm (Type[ConjunctionDataMessage]): External CDM to create the
            copy from.
        """

        for cluster in cdm_clusters.keys():
            if 'obligatory' in cluster: continue

            # Set clusters of CDM fields (header, relative_metadata, ... etc.)
            # containing the list of features embeded in every cluster.
            if cluster in ['header', 'relative_metadata']:
                cluster = '_values_' + cluster
            else:
                cluster = '_values_object_' + cluster

            # Create a deep copy of CDM object containing values.
            setattr(self, cluster, copy.deepcopy(getattr(other_cdm, cluster)))


    def to_dict(self) -> dict:
        """Convert CDM object to dictionary containing all the values.

        Returns:
            dict: Dictionary containing all the values of the CDM.
        """

        # Initialize output dictionary
        data = {}

        # Iterate over all clusters and keys from cdm_clusters.
        for cluster, keys in cdm_clusters.items():
            if 'obligatory' in cluster: continue

            # Set clusters of CDM fields (header, relative_metadata, ... etc.)
            # containing the list of features embeded in every cluster.
            if cluster in ['header', 'relative_metadata']:

                # Add values to dictionary for header and relative_metadata 
                # clusters.
                data.update(getattr(self, '_values_' + cluster))

            else:
                
                # Iterate over the two objects for object-specific features in
                # a CDM
                for i in [0, 1]:
                    # Create the prefix for the object in the format OBJECT_X.
                    prefix = 'OBJECT{}_'.format(i+1)

                    # Redefine keys in cluster to include the object prefix.
                    keys = map(lambda x: prefix+x, keys)

                    # Initialize dictionary for the cluster.
                    data_object = dict.fromkeys(keys)

                    # Get values dictionaty from cluster specific to the object.
                    values = getattr(self, '_values_object_' + cluster)[i]

                    # Fill cluster dictionary with all the values.
                    for key, value in values.items():
                        data_object[prefix+key] = value

                    # Update final dictionary.
                    data.update(data_object)

        # Add any extra values.
        data.update(self._values_extra)

        return data

    def to_dataframe(self) -> pd.DataFrame:
        """Convert CDM to pandas DataFrame.

        Returns:
            pd.DataFrame: Pandas DataFrame containing all CDM information.
        """
        data = self.to_dict()
        return pd.DataFrame(data, index=[0])

    def load(filepath:str) -> ConjunctionDataMessage:
        """Create a CDM object from a text file in KVN format.

        Args:
            filepath (str): Path where the file is located.

        Returns:
            ConjunctionDataMessage: CDM object.
        """

        # Initialize list to store the content of the text file containing CDM
        # data in KVN format.
        content = []
        with open(filepath) as f:
            lines = f.readlines()
            for line in lines:
                line = line.replace(u'\ufeff', '')
                line = line.strip()
                if line.startswith('COMMENT') or len(line) == 0:
                    continue

                # Split line into a (key, value) pair
                key, value = line.split('=')
                key, value = key.strip(), value.strip()

                # Convert value as float if it is a number
                value = float(value) if utils.is_number(value) else value
    #             print(line)
                content.append((key, value))

        # Instanciate CDM object.
        cdm = ConjunctionDataMessage(set_defaults=False)

        currently_parsing = 'header_and_relative_metadata'
        for key, value in content:
            if currently_parsing == 'header_and_relative_metadata':
                if key in cdm._keys_header:
                    cdm.set_header(key, value)
                elif key in cdm._keys_relative_metadata:
                    cdm.set_relative_metadata(key, value)
                elif key == 'OBJECT' and value == 'OBJECT1':
                    cdm.set_object(0, key, value)
                    currently_parsing = 'object1'
                    continue
                elif key == 'OBJECT' and value == 'OBJECT2':
                    cdm.set_object(1, key, value)
                    currently_parsing = 'object2'
                    continue
            elif currently_parsing == 'object1':
                if key == 'OBJECT' and value == 'OBJECT2':
                    cdm.set_object(1, key, value)
                    currently_parsing = 'object2'
                    continue
                try:
                    cdm.set_object(0, key, value)
                except:
                    continue
            elif currently_parsing == 'object2':
                if key == 'OBJECT' and value == 'OBJECT1':
                    cdm.set_object(0, key, value)
                    currently_parsing = 'object1'
                    continue
                try:
                    cdm.set_object(1, key, value)
                except:
                    continue
        return cdm

    def save(self, filepath:str) -> None:
        """Save CDM in KVN format into a external text file.

        Args:
            filepath (str): System path where the text file shall be saved.
        """
        content = self.kvn()
        with open(filepath, 'w') as f:
            f.write(content)

    def __hash__(self) -> int:
        """Return a hashed integer value of the object we pass as a parameter 
        into it.

        Returns:
            int: Hash of the CDM in KVN format.
        """
        return hash(self.kvn(show_all=True))

    def __eq__(self, other: ConjunctionDataMessage) -> bool:
        """Check if CDM object is exactly the same as other passed as a 
        parameter.

        Args:
            other (ConjunctionDataMessage): CDM object to check vs self.

        Returns:
            bool: True if CDM is exactly the same, False otherwise.
        """
        if isinstance(other, ConjunctionDataMessage):
            return hash(self) == hash(other)
        return False


    def set_value(self, key:str, value:str, object_id:int = None) -> None:
        """Set/assigns value to a CDM field (key).

        Args:
            key (str): Field of the CDM to which the value shall be assigned.
            value (str): Value to be assigned.
            object_id (int, optional): Object ID of the object for which the 
            value belongs to (OBJECT1 or OBJECT2). This parameter is only 
            applicable to keys relative to objects CDM data. Defaults to None.

        Raises:
            ValueError: Invalid object ID.
            RuntimeError: Format of the value not matching a datetime string 
            type.
        """

        # Get cluster to which the value is set
        cluster = cluster_from_key(key)

        # Check if data cluster belongs to those concerning objects' data.
        if not cluster in ['header', 'metadata']:

            # Check if object_id is either 0 or 1. Raise error otherwise.
            if not object_id in [0, 1]:
                raise ValueError(f'Invalid object_id ({object_id}) for ' + \
                                 f'{key}. Expecting object_id to be 0 or 1.')

            # Set values for the key field in the CDM.
            getattr(self, '_values_object_' + cluster)[key][object_id] = value

        else:
            
            # If key is one of the CDM fields with a date.
            if key in self._keys_with_dates:
                # We have a field with a date string as the value. Check if the 
                # string is in the format needed by the CCSDS 508.0-B-1 standard
                time_format = utils.get_ccsds_time_format(value)
                idx = time_format.find('DDD')
                if idx!=-1:
                    value = utils.doy_2_date(value, value[idx:idx+3], 
                                            value[0:4], idx)
                try:
                    _ = dt.strptime(value, '%Y-%m-%dT%H:%M:%S.%f')
                except Exception as e:
                    raise RuntimeError('{} ({}) is not in the expected ' + \
                                       'format.\n{}'.format(key, value, str(e)))

            getattr(self, '_values_' + cluster)[key] = value

    def get_value(self, key:str, object_id:int = None) -> Union[str, float, int]:
        """Get value from a CDM field (key).

        Args:
            key (str): Field of the CDM to be retrieved.
            object_id (int, optional): Object ID of the object the key belongs 
            to (OBJECT1 or OBJECT2). This parameter is only applicable to keys 
            relative to objects CDM data. Defaults to None.

        Raises:
            ValueError: Invalid object ID.

        Returns:
            Union[str, float, int]: Value assigned to the CDM key.
        """

        # Get cluster to which the value is set
        cluster = cluster_from_key(key)

        # Check if data cluster belongs to those concerning objects' data.
        if not cluster in ['header', 'metadata']:

            # Check if object_id is either 0 or 1. Raise error otherwise.
            if not object_id in [0, 1]:
                raise ValueError(f'Invalid object_id ({object_id}) for ' + \
                                 f'{key}. Expecting object_id to be 0 or 1.')

            # Return value assigned to the CDM key.
            return getattr(self, '_values_object_' + cluster)[key][object_id]

        else:
            # Return value assigned to the CDM key.
            return getattr(self, '_values_' + cluster)[key]

    
    def set_header(self, key:str, value:str) -> None:
        if key in self._keys_header:
            if key in self._keys_with_dates:
                # We have a field with a date string as the value. Check if the 
                # string is in the format needed by the CCSDS 508.0-B-1 standard
                time_format = utils.get_ccsds_time_format(value)
                idx = time_format.find('DDD')
                if idx!=-1:
                    value = utils.doy_2_date(value, value[idx:idx+3], value[0:4], idx)
                try:
                    _ = dt.strptime(value, '%Y-%m-%dT%H:%M:%S.%f')
                except Exception as e:
                    raise RuntimeError('{} ({}) is not in the expected format.\n{}'.format(key, value, str(e)))
            self._values_header[key] = value
        else:
            raise ValueError('Invalid key ({}) for header'.format(key))

    def set_relative_metadata(self, key, value):
        if key in self._keys_relative_metadata:
            self._values_relative_metadata[key] = value
        else:
            raise ValueError(f'Invalid key ({key}) for relative metadata')

    def set_object(self, object_id, key, value):
        if object_id != 0 and object_id != 1:
            raise ValueError('Expecting object_id to be 0 or 1')
        if key in self._keys_metadata:
            self._values_object_metadata[object_id][key] = value
        elif key in self._keys_data_od:
            self._values_object_data_od[object_id][key] = value
        elif key in self._keys_data_state:
            self._values_object_data_state[object_id][key] = value
        elif key in self._keys_data_covariance:
            self._values_object_data_covariance[object_id][key] = value
        else:
            raise ValueError('Invalid key ({}) for object data'.format(key))

    def get_object(self, object_id, key):
        if object_id != 0 and object_id != 1:
            raise ValueError('Expecting object_id to be 0 or 1')
        if key in self._keys_metadata:
            return self._values_object_metadata[object_id][key]
        elif key in self._keys_data_od:
            return self._values_object_data_od[object_id][key]
        elif key in self._keys_data_state:
            return self._values_object_data_state[object_id][key]
        elif key in self._keys_data_covariance:
            return self._values_object_data_covariance[object_id][key]
        else:
            raise ValueError(f'Invalid key ({key}) for object data')

    def get_relative_metadata(self, key):
        if key in self._keys_relative_metadata:
            return self._values_relative_metadata[key]
        else:
            raise ValueError(f'Invalid key ({key}) for relative metadata')

    def set_state(self, object_id, state):
        self.set_object(object_id, 'X', state[0, 0])
        self.set_object(object_id, 'Y', state[0, 1])
        self.set_object(object_id, 'Z', state[0, 2])
        self.set_object(object_id, 'X_DOT', state[1, 0])
        self.set_object(object_id, 'Y_DOT', state[1, 1])
        self.set_object(object_id, 'Z_DOT', state[1, 2])
        self._update_state_relative()
        self._update_miss_distance()

    def _update_miss_distance(self):
        state_object1 = self.get_state(0)
        if np.isnan(state_object1.sum()):
            warnings.warn('state_object1 has NaN')
        state_object2 = self.get_state(1)
        if np.isnan(state_object2.sum()):
            warnings.warn('state_object2 has NaN')

        miss_distance = np.linalg.norm(state_object1[0] - state_object2[0])
        self.set_relative_metadata('MISS_DISTANCE', miss_distance)

    def _update_state_relative(self):
        def uvw_matrix(r:np.ndarray, v:np.ndarray) -> np.ndarray:
            """Get reference frame from position and velocity vectors.

            Args:
                r (np.ndarray): Radial vector (direction of orbit radius).
                v (np.ndarray): Velocity vector (tangent to orbit).

            Returns:
                np.ndarray: _description_
            """
            u = r / np.linalg.norm(r)
            w = np.cross(r, v)
            w = w / np.linalg.norm(w)
            v = np.cross(w, u)
            return np.vstack((u, v, w))


        def relative_state(state_obj_1:np.ndarray, 
                           state_obj_2:np.ndarray) -> np.ndarray:
            """Takes states in ITRF and returns relative state in RTN with 
            target as reference

            Args:
                state_obj_1 (np.ndarray): State vectors for OBJECT1.
                state_obj_2 (np.ndarray): State vectors for OBJECT2.

            Returns:
                np.ndarray: Relative state vector in targets RTN reference 
                frame.
            """

            # Get unit vectors from RTN frame of OBJECT1 (target).
            rot_matrix = uvw_matrix(state_obj_1[0], state_obj_1[1])

            # Get relative position and velocities in XYZ coordinates
            relative_state = state_obj_2 - state_obj_1

            # Express relative state vectors (position and velocity) in the 
            # OBJECT1 framework using the dot product.
            for state in [0, 1]:
                for component in [0, 1, 2]:
                    relative_state[state, component] = \
                        np.dot(rot_matrix[component], relative_state[sate])

            return relative_state

        # Get state vector from OBJECT1 and raise warning is Not a Number.
        state_object1 = self.get_state(0)
        if np.isnan(state_object1.sum()): warnings.warn('state_object1 has NaN')

        # Get state vector from OBJECT2 and raise warning is Not a Number.
        state_object2 = self.get_state(1)
        if np.isnan(state_object2.sum()): warnings.warn('state_object2 has NaN')

        # Get relative state vector.
        relative_state = relative_state(state_object1, state_object2)

        # Set the values of the relative state vector features in the CDM.
        for i, state in enumerate(['POSITION', 'VELOCITY']):
            for j, component in enumerate(['R', 'T', 'N']):
                self.set_relative_metadata(f'RELATIVE_{state}_{component}', 
                                           relative_state[i, j])

        # Set relative speed using the relative velocity components.
        self.set_relative_metadata('RELATIVE_SPEED', 
                                   np.linalg.norm(relative_state[1]))

    def get_state_relative(self) -> np.ndarray:
        """Get relative state vector between OBJECT1 and OBJECT2.

        Returns:
            np.ndarray: Relative state vector.
        """

        # Initialize relative state vector.
        relative_state = np.zeros([2, 3])

        # Iterate over position and velocity for the 3 components in the RTN
        # framework.
        for i, state in enumerate(['POSITION', 'VELOCITY']):
            for j, component in enumerate(['R', 'T', 'N']):
                label = f'RELATIVE_{state}_{component}'
                relative_state[i, j] = self.get_relative_metadata(label)

        return relative_state

    def get_state(self, object_id:int) -> np.ndarray:
        """Get state vector for a specific object.

        Args:
            object_id (int): Object ID to which the state vector belongs to.

        Returns:
            np.ndarray: State vector.
        """

        # Initialize state vector
        state_vector = np.zeros([2, 3])
        for i, state in enumerate(['', '_DOT']):
            for j, component in enumerate(['X', 'Y', 'Z']):
                state_vector[i, j] = self.get_object(object_id, 
                                                     component + state) 

        return state_vector

    def get_covariance(self, object_id:int) -> np.ndarray:
        """Get covariance matrix

        Args:
            object_id (int): Object ID to which the covarianve matrix belongs to

        Returns:
            np.ndarray: Covariance matrix.
        """

        # Initialize covariance matrix with 0s
        covariance = np.zeros([6, 6])

        # Initialize list of positiona and velocity components in the RTN 
        # framework
        components = ['R', 'T', 'N', 'RDOT', 'TDOT', 'NDOT']

        # Iterate over the components to fill the covariance matrix
        for i, i_name in enumerate(components):
            for j, j_name in enumerate(components):
                if j>i: break
                covariance[i, j] = self.get_object(object_id, 
                                                   f'C{i_name}_{j_name}')

        # Copies lower triangle to the upper part
        covariance = covariance + covariance.T - np.diag(np.diag(covariance))

        return covariance

    def set_covariance(self, object_id:int, covariance:np.ndarray) -> None:
        """Set the covariance matrix values in the CDM object.

        Args:
            object_id (int): Object ID the covariance matrix belongs to.
            covariance (np.ndarray): Covariance matrix to assign to the CDM
            object.
        """

        # Initialize list of positiona and velocity components in the RTN 
        # framework
        components = ['R', 'T', 'N', 'RDOT', 'TDOT', 'NDOT']

        # Iterate over the components to set the covariance matrix
        for i, i_rtn in enumerate(components):
            for j, j_rtn in enumerate(components):
                if j>i: break # Fill only the lower triangle.
                self.set_object(object_id, 
                                f'C{i_rtn}_{j_rtn}',
                                covariance[i, j])
    @staticmethod
    def datetime_to_str(input_datetime:datetime) -> str:
        return input_datetime.strftime('%Y-%m-%dT%H:%M:%S.%f')

    def validate(self) -> None:
        """Check all compulsary data in CDM is provided.
        """
        def check(keys:list, values:dict, cluster_name:str) -> None:
            """Print missing obligatory values in a CDM.

            Args:
                keys (list): Obligatory features in a CDM.
                values (dict): Values contained in the CDM.
                cluster_name (str): Name of the data cluster to print.
            """
            for key in keys:
                if values[key] is None:
                    print(f'Missing obligatory value in {cluster_name}: {key}')

        # Iterate over all data clusters in a CDM.
        for cluster in cdm_clusters.keys():

            # Get obligatory keys and values from a given cluster
            keys  = getattr(self,'_keys_' + cluster + '_obligatory') 
            values  = getattr(self,'_values_' + cluster)

            # Format cluster name for printing.
            cluster_name = cluster.replace('_',' ').capitalize()

            if len(values)>1:
                for object_id, value in enumerate(values):
                    check(keys, value, 
                          f'OBJECT{object_id} data ({cluster_name})')
            else:
                check(keys, values, cluster_name)

    def kvn(self, show_all:bool = False) -> str:
        """Convert CDM object to string in KVN format.

        Args:
            show_all (bool, optional): Flag to determine if KVN entries with 
            empty values are included. Defaults to False.

        Returns:
            str: CDM in KVN format.
        """

        # Define internal function to concatenate KVN entries.
        def append(kvn_input:str, cluster_values:dict, 
                   obligatory_keys:list) -> str:
            """Append new entry to string in KVN format.

            Args:
                kvn_input (str): KVN entries from a given CDM.
                cluster_values (dict): Values to append to the KVN.
                obligatory_keys (list): List of compulsory keys that shall be
                included in the KVN file regardless of their values.

            Returns:
                str: Concatenated KVN entries.
            """

            # Iterate over all data provided for a given cluster (i.e. header,
            # metadata, ... etc).
            for k, v in cluster_values.items():
                # Format key as a string with 37 characters (fill with spaces)
                k_str = k.ljust(37, ' ')

                # Format value depending on content.
                if v is None:
                    # If no value is provided, include it if show_all parameter
                    # is True or is a compulsory CDM key.
                    if show_all or k in obligatory_keys:
                        kvn_input += '{} =\n'.format(k_str)
                else:
                    # If the value is provided, convert to string taking care of 
                    # scientific notation.
                    if isinstance(v, float) or isinstance(v, int):
                        v_str = '{}'.format(v)
                        if 'e' in v_str:
                            v_str = '{:.3E}'.format(v)
                    else:
                        v_str = str(v)
                    kvn_input += '{} = {}\n'.format(k_str, v_str)
                    
            return kvn_input

        # Initialize empty return variable ret.
        content = ''

        # Iterate over all clusters of data to create the KVN content.
        for cluster in cdm_clusters.keys():
            if 'obligatory' in cluster: continue
            obligatory_keys = getattr(self, '_keys_' + cluster + '_obligatory')

            if cluster in ['header', 'relative_metadata']:
                values = getattr(self, '_values_' + cluster)
                content = append(content, values, obligatory_keys)
            else:
                object_values = getattr(self, '_values_object_' + cluster)
                for values in object_values:
                    content = append(content, values, obligatory_keys)

        return content

    def __repr__(self):
        return self.kvn()

    def __getitem__(self, key):
        return self.to_dict()[key]

    def __setitem__(self, key, value):
        if key in self._keys_header:
            self.set_header(key, value)
        elif key in self._keys_relative_metadata:
            self.set_relative_metadata(key, value)
        elif key.startswith('OBJECT1_'):
            key = key.split('OBJECT1_')[1]
            self.set_object(0, key, value)
        elif key.startswith('OBJECT2_'):
            key = key.split('OBJECT2_')[1]
            self.set_object(1, key, value)
        else:
            raise ValueError('Invalid key: {}'.format(key))


CDM = ConjunctionDataMessage

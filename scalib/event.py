# Libraries used for hinting
from __future__ import annotations
from typing import Type, Union

import pandas as pd
import numpy as np
import matplotlib as mpl
import matplotlib.pyplot as plt
from glob import glob
import copy
import os
import re

<<<<<<< HEAD
from . import utils
from .cdm import CDM
=======
from . import utils, cfg
from .cdm import ConjunctionDataMessage as CDM
>>>>>>> dev

mpl.rcParams['axes.unicode_minus'] = False
plt_default_backend = plt.get_backend()

#%%
<<<<<<< HEAD
class Event():
    def __init__(self, cdms:list = None, filepaths:list = None) -> None:
        """Initialize event object. Defaults to empty Event (0 CDMs).
=======
class EventsPlotting():

    def plot_features(self, features:Union[list, str], figsize:tuple=None, 
                      axs:Union[np.ndarray, plt.Axes] = None, 
                      return_axs:bool = False, filepath:str = None, 
                      sharex:bool = True, 
                      *args, **kwargs) -> Union[None, np.ndarray]:
        """Generate a matrix of charts plotting the evolution of all the 
        features for a single (Event) or multiple events (EventsDataset). 

        Args:
            features (Union[list, str]): List of features to plot.
            figsize (tuple, optional): Size per figure. Defaults to None.
            axs (Union[np.ndarray, plt.Axes], optional): Axis object. Defaults 
            to None.
            return_axs (bool, optional): Flag to return axis object (i.e. from 
            other chart). Defaults to False.
            filepath (str, optional): Path to save the plot. Defaults to None.
            sharex (bool, optional): Flag to share X-axis. Defaults to True.

        Returns:
            Union[None, np.ndarray]: Array of axis objects if return_axs = True, 
            None otherwise.
        """


        # Convert features to list if only one is provided.
        if not isinstance(features, list): features = [features]

        # Check if axs parameter is provided and not None
        if axs is None:
            # Get square matrix dimensions to arrange the subplots.
            rows, cols = utils.plt_matrix(len(features))

            # Set dimension of the final plot containing all subplots.
            if figsize is None: figsize = (cols*20/7, rows*12/6)

            fig, axs = plt.subplots(nrows = rows, ncols = cols, 
                                    figsize = figsize, sharex = sharex)

        # Convert axs to array in case only one plot is created.
        if not isinstance(axs, np.ndarray): axs = np.array(axs)

        # Iterate over all subplots from left to right, from top to bottom.
        for i, ax in enumerate(axs.flat):
            if i < len(features):

                # Plot legend only in the first subplot.
                if i != 0 and 'legend' in kwargs: kwargs['legend'] = False

                # Plot feature evolution: 
                #  - If invoked by Event class plots the evolution of all the 
                # features from a single event (1 event from 1 feature per 
                # chart).
                # - If invoked by ConjunctionEventsDataset class plots the 
                # evolution of all features from all events (All events from 1 
                # feature per chart).
                self.plot_feature(features[i], ax = ax, *args, **kwargs)
            else:
                # Cancel out any extra subplot available in the axs object.
                ax.axis('off')

        # Remove blank spaces around the plot to be more compact.
        plt.tight_layout()

        # Save plot only if filepath is provided.
        if filepath is not None:
            print('Plotting to file: {}'.format(filepath))
            plt.savefig(filepath)

        if return_axs:
            return axs

    def plot_uncertainty(self, figsize:tuple = (20, 12), diagonal:bool = False, 
                         *args, **kwargs) -> None:
        """Plot uncertainty using the covariance matrix terms from the
         state vector.

        Args:
            figsize (tuple, optional): Figure size. Defaults to (20, 12).
            diagonal (bool, optional): Flag to determine if only the main 
            diagonal of the covariance matrix shall be plotted or not. Defaults 
            to False.
        """

        # Initialize list of position and velocity components in the RTN 
        # framework
        components = ['R', 'T', 'N', 'RDOT', 'TDOT', 'NDOT']

        # Iterate over the components to get the relevant features from the 
        # covariance matrix (diagonal or upper/lower triangle).
        features = []
        for i, i_name in enumerate(components):
            for j, j_name in enumerate(components):
                if j>i or (diagonal and j<i): continue
                features.append(f'C{i_name}_{j_name}')

        # Get the list of features for both objects.
        features = list(map(lambda f: 'OBJECT1_'+f, features)) + \
                   list(map(lambda f: 'OBJECT2_'+f, features))
        return self.plot_features(features, figsize = figsize, *args, **kwargs)

#%%
class ConjunctionEvent(EventsPlotting):
    def __init__(self, cdms:list = None, filepaths:list = None) -> None:
        """Initialize event object. Defaults to empty ConjunctionEvent (0 CDMs).
>>>>>>> dev

        Args:
            cdms (list, optional): List of CDM objects belonging to the event. 
            Defaults to None.
            filepaths (list, optional): List of file paths of CDM text files
            that belongs to an event. Defaults to None.

        Raises:
            RuntimeError: cdms and filepaths parameter are not None. Only one of
            both parameters can be passed into the function.
        """
        if cdms is not None:
            if filepaths is not None:
                raise RuntimeError('Expecting only one of cdms, filepaths, not both')
            self._cdms = cdms
        elif filepaths is not None:
            self._cdms = [CDM(filepath) for filepath in filepaths]
        else:
            self._cdms = []
        self._update_cdm_extra_features()
        self._dataframe = None

    def _update_cdm_extra_features(self) -> None:
        """Add features to CDM object relative to the event:
         - __CREATION_DATE in days since the creation date from the first event.
         - __TCA estimated days from the creation date of the event to TCA.
         - __DAYS_TO_TCA estimated remaining days to TCA.
        """
        if len(self._cdms) > 0:

            # Store creation date of the first CDM object as the start date for
            # the entire Event. This assumes the first CDM object in the list is
            # the very first one.
            date0 = self._cdms[0]['CREATION_DATE']

            # Iterate over all CDM objects available in the event.
            for cdm in self._cdms:
                # Get creation date and TCA as days since creation of event.
                cdm._values_extra['__CREATION_DATE'] = \
                    utils.from_date_str_to_days(cdm['CREATION_DATE'],date0=date0)
                cdm._values_extra['__TCA'] = \
                    utils.from_date_str_to_days(cdm['TCA'], date0 = date0)

                # Compute estimated remaining days to TCA.
                cdm._values_extra['__DAYS_TO_TCA'] = \
                    cdm._values_extra['__TCA'] - \
                    cdm._values_extra['__CREATION_DATE']

    def add(self, cdm: Union[CDM, list], return_result:bool=False) \
<<<<<<< HEAD
            -> Union[Event, None]:
=======
            -> Union[ConjunctionEvent, None]:
        """Add one or multilple CDM objects to the Event object.

        Raises:
            ValueError: cdm parameter is not a CDM object or list of CDM objects.

        Returns:
            Union[ConjunctionEvent, None]: Event object if return_result = True, 
            None otherwise.
        """
>>>>>>> dev
        
        # Check if it cdm parameter is a single CDM or a list. If list, use
        # recursivity of the same function to add a single item.
        if isinstance(cdm, CDM):
            self._cdms.append(cdm)
        elif isinstance(cdm, list):
            for c in cdm:
                self.add(c)
        else:
            raise ValueError('Expecting a single CDM or a list of CDMs')

        # Update extra values for the new CDMs added to the event.
        self._update_cdm_extra_features()

        # Return Event object if required.
        if return_result: return self

<<<<<<< HEAD
    def copy(self) -> Event:
        return Event(cdms = copy.deepcopy(self._cdms))
=======
    def copy(self) -> ConjunctionEvent:
        """Create a deepcopy of the Event object.

        Returns:
            ConjunctionEvent: Event object.
        """
        return ConjunctionEvent(cdms = copy.deepcopy(self._cdms))
>>>>>>> dev

    def to_dataframe(self) -> pd.DataFrame:
        """Convert to pandas DataFrame.

        Returns:
            pd.DataFrame: Conjunction event containing one CDM per row.
        """
        if self._dataframe is None:
            if len(self) == 0:
                self._dataframe = pd.DataFrame()
            cdm_dataframes = []
            for cdm in self._cdms:
                cdm_dataframes.append(cdm.to_dataframe())
            self._dataframe = pd.concat(cdm_dataframes, ignore_index=True)
        return self._dataframe

    def plot_feature(self, feature:str, figsize:tuple = (5, 3), 
                     ax:plt.Axes = None, return_ax:bool = False, 
                     apply_func = (lambda x: x), filepath:str = None, 
<<<<<<< HEAD
                     legend:bool = False, xlim:tuple = (-0.01,7.01), 
                     ylims:Union[tuple, dict] = None, 
                     *args, **kwargs) -> Union[None, plt.Axes]:
        """_summary_
=======
                     legend:bool = False, xlim:tuple = (-0.5,7.5), 
                     ylims:Union[tuple, dict] = None, 
                     *args, **kwargs) -> Union[None, plt.Axes]:
        """Plot the evolution of a given feature during the entire duration of 
        a conjunction event.
>>>>>>> dev

        Args:
            feature (str): Name of the feature to plot.
            figsize (tuple, optional): Figure size. Defaults to (5, 3).
            ax (plt.Axes, optional): Axes object. Defaults to None.
            return_ax (bool, optional): Flag to return the axes of the plot 
            created. Defaults to False.
            apply_func (function, optional): Lambda function to process 
            feature's data. Defaults to (lambda x: x).
            filepath (str, optional): System path where the figure is saved. 
            Defaults to None.
            legend (bool, optional): Flag to show the legend. Defaults to False.
            xlim (tuple, optional): X-axis limits. Defaults to (-0.01,7.01).
<<<<<<< HEAD
            ylim (tuple, optional): Y-axis limits. Defaults to None.

        Raises:
            RuntimeError: _description_
            RuntimeError: _description_
=======
            ylims (tuple, optional): Y-axis limits. Defaults to None.

        Raises:
            RuntimeError: TCA not available in CDM object.
            RuntimeError: CREATION_DATE not available in CDM object.
>>>>>>> dev

        Returns:
            Union[None, plt.Axes]: _description_
        """
        # Initialize lists to store the X and Y data.
        data_x = []
        data_y = []
        for i, cdm in enumerate(self._cdms):
            # Check if TCA and CREATION_DATE features are provided in the CDM.
            # Raise an error otherwise.
            if cdm['TCA'] is None:
                raise RuntimeError(f'CDM {i} in event does not have TCA.')
            if cdm['CREATION_DATE'] is None:
                raise RuntimeError(f'CDM {i} in event does not have ' + \
                                   f'CREATION_DATE')

            time_to_tca = utils.from_date_str_to_days(cdm['TCA'], 
                                                    date0=cdm['CREATION_DATE'])
            data_x.append(time_to_tca)
            data_y.append(apply_func(cdm[feature]))

        # Create axes instance if not passed as a parameter.
        if ax is None: fig, ax = plt.subplots(figsize=figsize)

<<<<<<< HEAD
        ax.plot(data_x, data_y, marker='.', *args, **kwargs)
        # ax.scatter(data_x, data_y)
        
        ax.set_title(feature)
=======

        # Add name of the feature inside the chart to optimize space in the plot 
        # grid.
        # ax.set_title(r'\texttt{'+ feature + '}', fontsize=8)
        t = ax.text(x = 0.5, y = 0.975, s = r'\texttt{'+ feature + '}', 
                size=8, c='black', 
                ha = 'center', va='top', transform=ax.transAxes, 
                bbox = dict(facecolor = 'white', edgecolor = 'white',
                            alpha = 0.5, pad = 1))


        # Plot data 
        ax.plot(data_x, data_y, marker='.', *args, **kwargs)
        
        # Plot grid.
>>>>>>> dev
        ax.grid(True, linestyle='--')

        # Set Axes limits
        ax.set_xlim(xlim[0], xlim[1])
<<<<<<< HEAD
        ax.set_xlabel('Time to TCA (days)')

        # Set Y-axis limits if provided.
        if ylim is not None: ax.set_ylim(ylims[0], ylims[1])
=======
        ax.set_xlabel('Days to TCA', fontsize=8)

        # Set Y-axis limits if provided.
        if ylims is not None: 
            if isinstance(ylims, dict):
                ax.set_ylim(ylims[feature][0], ylims[feature][1])
            else:
                ax.set_ylim(ylims[0], ylims[1])

        # Adapt X and Y axes ticks for a better plot representation.
        ax.set_yticks(np.linspace(ax.get_ylim()[0],ax.get_ylim()[1],5))
        ax.set_xticks(np.arange(8))
>>>>>>> dev

        # Plot legend
        if legend: ax.legend(loc='best')

        # Save plot if filepath is provided.
        if filepath is not None:
            print(f'Plotting to file: {filepath}')
            plt.savefig(filepath)

        if return_ax:
            return ax

<<<<<<< HEAD
    def plot_features(self, features:Union[list, str], figsize:tuple=None, 
                      axs:Union[np.ndarray, plt.Axes] = None, 
                      return_axs:bool = False, filepath:str = None, 
                      sharex:bool = True, 
                      *args, **kwargs) -> Union[None, np.ndarray]:

        def plt_matrix(num_subplots:int) -> tuple:
            """Calculate number of rows and columns for a square matrix 
            containing subplots.

            Args:
                num_subplots (int): Number of subplots contained in the matrix.

            Returns:
                tuple: Number of rows and columns of the matrix.
            """
            if num_subplots < 5:
                return 1, num_subplots
            else:
                cols = math.ceil(math.sqrt(num_subplots))
                rows = 0
                while num_items > 0:
                    rows += 1
                    num_items -= cols
                return rows, cols

        # Convert features to list if only one is provided.
        if not isinstance(features, list): features = [features]

        # Check if axs parameter is provided and not None
        if axs is None:
            # Get square matrix dimensions to arrange the subplots.
            rows, cols = plt_matrix(len(features))

            # Set dimension of the final plot containing all subplots.
            if figsize is None: figsize = (cols*20/7, rows*12/6)

            fig, axs = plt.subplots(nrows = rows, ncols = cols, 
                                    figsize = figsize, sharex = sharex)

        # Convert axs to array in case only one plot is created.
        if not isinstance(axs, np.ndarray): axs = np.array(axs)

        # Iterate over all subplots from left to right, from top to bottom.
        for i, ax in enumerate(axs.flat):
            if i < len(features):

                # Plot legend only in the first subplot.
                if i != 0 and 'legend' in kwargs: kwargs['legend'] = False

                # Plot evolution of feature vs time.
                self.plot_feature(features[i], ax = ax, *args, **kwargs)
            else:
                # Cancel out any extra subplot available in the axs object.
                ax.axis('off')

        # Remove blank spaces around the plot to be more compact.
        plt.tight_layout()

        # Save plot only if filepath is provided.
        if filepath is not None:
            print('Plotting to file: {}'.format(filepath))
            plt.savefig(filepath)

        if return_axs:
            return axs

    def plot_uncertainty(self, figsize:tuple = (20, 12), diagonal:bool = False, 
                         *args, **kwargs):

        # Initialize list of position and velocity components in the RTN 
        # framework
        components = ['R', 'T', 'N', 'RDOT', 'TDOT', 'NDOT']

        # Iterate over the components to get the relevant features from the 
        # covariance matrix (diagonal or upper/lower triangle).
        features = []
        for i, i_name in enumerate(components):
            for j, j_name in enumerate(components):
                if j>i or (diagonal and j<i): continue
                features.append(f'C{i_name}_{j_name}')

        # if diagonal:
        #     features = ['CR_R', 'CT_T', 'CN_N', 'CRDOT_RDOT', 'CTDOT_TDOT', 
        #     'CNDOT_NDOT']
        # else:
        #     features = ['CR_R', 'CT_R', 'CT_T', 'CN_R', 'CN_T', 'CN_N', 
        #                 'CRDOT_R', 'CRDOT_T', 'CRDOT_N', 'CRDOT_RDOT', 
        #                 'CTDOT_R', 'CTDOT_T', 'CTDOT_N', 'CTDOT_RDOT', 
        #                 'CTDOT_TDOT', 'CNDOT_R', 'CNDOT_T', 'CNDOT_N', 
        #                 'CNDOT_RDOT', 'CNDOT_TDOT', 'CNDOT_NDOT']

        # Get the list of features for both objects.
        features = list(map(lambda f: 'OBJECT1_'+f, features)) + \
                   list(map(lambda f: 'OBJECT2_'+f, features))
        return self.plot_features(features, figsize = figsize, *args, **kwargs)

=======
>>>>>>> dev
    def __repr__(self) -> str:
        """Print readable information about the event.

        Returns:
            str: Class name with number of CDMs objects contained on it.
        """
<<<<<<< HEAD
        return 'Event(CDMs: {})'.format(len(self))
=======
        return 'ConjunctionEvent(CDMs: {})'.format(len(self))
>>>>>>> dev

    def __getitem__(self, index:Union[int, slice]) -> Union[CDM, list]:
        """Get CDM object from event given an index or list of indexes.

        Args:
            index (Union[int, slice]): Index or list of indexes of CDMs to 
            retrieve from the event.

        Returns:
            Union[CDM, list]: CDM or list of CDM objects.
        """
        if isinstance(index, slice):
<<<<<<< HEAD
            return Event(cdms = self._cdms[index])
=======
            return ConjunctionEvent(cdms = self._cdms[index])
>>>>>>> dev
        else:
            return self._cdms[index]

    def __len__(self) -> int:
        """Get number of events embeded in the conjunction event.

        Returns:
            int: Number of CDMs in the event.
        """
        return len(self._cdms)

#%%
<<<<<<< HEAD
class EventDataset():
    def __init__(self, cdms_dir:str = None, cdm_extension:str = '.cdm.kvn.txt',
                 events:list = None) -> None:
=======
class ConjunctionEventsDataset(EventsPlotting):
    def __init__(self, cdms_dir:str = None, cdm_extension:str = '.cdm.kvn.txt',
                 events:list = None) -> None:
        """Initializes ConjunctionEventsDataset object importing text files 
        containing CDM information in KVN format from a given folder or by 
        passing a list of ConjunctionEvent objects.

        Args:
            cdms_dir (str, optional): Folder directory where all text files 
            containing CDM information in KVN format are located. Defaults to 
            None.
            cdm_extension (str, optional): Text file extensions to use when 
            looking for CDMs. Defaults to '.cdm.kvn.txt'.
            events (list, optional): List of ConjunctionEvent objects. Defaults 
            to None.
        """
>>>>>>> dev
        if events is None:
            if cdms_dir is None:
                self._events = []
            else:
                print('Loading CDMS (with extension {}) from directory: {}' \
                      .format(cdm_extension, cdms_dir))
                
                # Get list of sorted file paths matching the file extension.
                filepaths = sorted(glob(os.path.join(cdms_dir, 
                                                     '*' + cdm_extension)))
                
                # Create RegularExpression to find files whose names' format is
                # XXXXXX_NN.cdm.kvn.txt where XXXXX can be any text containing
                # letters and numbers associated to a specific event, NN the 
                # number of CDM for that event.
                regex = r"(.*)_([0-9]+{})".format(cdm_extension)
                matches = re.finditer(regex, '\n'.join(filepaths))

                # Initialize empty list to store event prefixes in order to
                # identify CDMs belonging to specific Events.
                event_prefixes = []
                
                # Iterate over all files matching the regexp.
                for match in matches:
                    # Append the filename prefix to group events.
                    event_prefixes.append(match.groups()[0])
                
                # Remove duplicates and sort prefixes.
                event_prefixes = sorted(set(event_prefixes))

                # Using the prefixes (unique per event), iterate over all CDMs
                # contained in the event.
                event_cdm_filepaths = []
                for event_prefix in event_prefixes:
                    # Get list of CDM files whose names match the event_prefix.
                    cdm_filepaths.append(list(filter(lambda f: 
                                                     f.startswith(event_prefix), 
                                                     filepaths)))
                # Append event objects from the filepaths.
<<<<<<< HEAD
                self._events = [Event(cdm_filepaths=f) for f in cdm_filepaths]
=======
                self._events = [ConjunctionEvent(cdm_filepaths=f) \
                                for f in cdm_filepaths]
>>>>>>> dev
                print('Loaded {} CDMs grouped into {} events' \
                      .format(len(file_names), len(self._events)))
        else:
            self._events = events

    # Create utility function that does not access to any properties of the 
    # class with "staticmethod".
    # Proposed API for the pandas loader
    @staticmethod
    def from_pandas(df:pd.DataFrame, 
<<<<<<< HEAD
        cdm_compatible_fields:dict=cfg.df_to_ccsds_features, 
        group_events_by:str='event_id', 
        date_format:str='%Y-%m-%d %H:%M:%S.%f') -> EventDataset:

        # Remove columns with any NaN values.
        print('Dataframe with {} rows and {} columns'.format(len(df), len(df.columns)))
        print('Dropping columns with NaNs')
        df = df.dropna(axis=1)
        print('Dataframe with {} rows and {} columns'.format(len(df), len(df.columns)))
        pandas_column_names_after_dropping = list(df.columns)

        # Get dictionary of event_id's with row indexes per event.
        print('Grouping by {}'.format(group_events_by))
        df_events = df.groupby(group_events_by).groups
        print('Grouped into {} event(s)'.format(len(df_events)))
        
        
        events = []
        # util.progress_bar_init('Converting DataFrame to EventDataset', len(df_events), 'Events')
        i = 0
        for k, v in df_events.items():
            # util.progress_bar_update(i)
            i += 1
            
            # Get DataFrame from one single conjunction event using the indexes
            # from v
            df_event = df.iloc[v]
            
            # Initialize list to store all CDM objects.
            cdms = []
=======
        df_to_ccsds_name_mapping:dict=cfg.df_to_ccsds_features,
        dropna:bool=True, group_events_by:str='event_id', 
        date_format:str='%Y-%m-%d %H:%M:%S.%f') -> ConjunctionEventsDataset:
        """Import Conjunction Events Dataset from pandas DataFrame.

        Args:
            df (pd.DataFrame): Pandas DataFrame containing all events.
            df_to_ccsds_name_mapping (dict, optional): Dictionary containing the 
            mapping of pandas columns with CDM features names as instructed by 
            the CCSDS. Defaults to cfg.df_to_ccsds_features.
            dropna (bool, optional): Flag to drop columns containing NaN values.
            Defaults to True.
            group_events_by (str, optional): Column from DataFrame to use to 
            group events. Defaults to 'event_id'.
            date_format (_type_, optional): Date format to convert datetime 
            columns. Defaults to '%Y-%m-%d %H:%M:%S.%f'.

        Returns:
            ConjunctionEventsDataset: Conjunction Events dataset.
        """

        # Remove columns with any NaN values.
        print(f'Dataframe with {len(df)} rows and {len(df.columns)} columns')

        if dropna:
            print('Dropping columns with NaNs...', end='\r')
            df = df.dropna(axis=1)
            print('Dropping columns with NaNs... {} rows and {} columns' + \
                  ' remaining'.format(len(df), len(df.columns)))

        # Get dictionary of event_id's with row indexes per event.
        df_events = df.groupby(group_events_by).groups
        print('Grouped into {} event(s) by {} column.' \
                .format(len(df_events), group_events_by))
        
        # Initialize events list.
        events = []

        # Initialize counter and progress bar object
        n = 0
        pb_events = utils.ProgressBar(iterations = range(len(df_events)), 
            description='Importing Events from pandas DataFrame...', 
            desc_loc='right')

        # Iterate over all events
        for event_id, rows in df_events.items():
        
            n += 1
            
            # Get DataFrame from one single conjunction event using the indexes
            # from v
            df_event = df.iloc[rows]
            
            # Initialize list to store all CDM objects.
            event_cdms = []
>>>>>>> dev
            
            # Iterate over all CDMs contained in the event k.
            for _, df_cdm in df_event.iterrows():
                
                # Initialize single CDM object.
                cdm = CDM()
                
<<<<<<< HEAD
                for pandas_name, cdm_name in cdm_compatible_fields.items():
                    if pandas_name in pandas_column_names_after_dropping:
                        # Get the value from the single Event DataFrame.
                        value = df_cdm[pandas_name]
=======
                for df_name, ccsds_name in df_to_ccsds_name_mapping.items():
                    if df_name in list(df.columns):
                        # Get the value from the single Event DataFrame.
                        value = df_cdm[df_name]
>>>>>>> dev
                        # Check if the field is a date, if so transform to the 
                        # correct date string format expected in the CCSDS 
                        # 508.0-B-1 standard.
                        if utils.is_date(value, date_format):
<<<<<<< HEAD
                            value = utils.transform_date_str(value, date_format, '%Y-%m-%dT%H:%M:%S.%f')
                        cdm[cdm_name] = value
                        
                # Append CDM object to the list.
                cdms.append(cdm)
            # Append Event object to the list passing all CDM objects.
            events.append(Event(cdms))
        # util.progress_bar_end()
        
        # Create EventDataset object with the list of events extracted.
        event_dataset = EventDataset(events=events)
        print('\n{}'.format(event_dataset))
        return event_dataset

    def to_dataframe(self) -> pd.DataFrame:
=======
                            value = utils.transform_date_str(date_string = value, 
                                date_format_from = date_format, 
                                date_format_to = '%Y-%m-%dT%H:%M:%S.%f')

                        cdm[ccsds_name] = value
                        
                # Append CDM object to the list.
                event_cdms.append(cdm)
            # Append Cojunction Event object to the list passing all CDM objects.
            events.append(ConjunctionEvent(event_cdms))

            # Update progress bar.
            pb_events.refresh(i = n+1)

        # Print final message in progress bar.
        pb_events.refresh(i = n+1, 
            description='Conjunction Events dataset imported successfully.')
        
        # Create ConjunctionEventsDataset object with the list of events 
        # extracted.
        event_dataset = ConjunctionEventsDataset(events = events)
        print('\n{}'.format(event_dataset))

        return event_dataset

    def to_dataframe(self) -> pd.DataFrame:
        """Convert Conjunction Events dataset to pandas DataFrame

        Returns:
            pd.DataFrame: Pandas DataFrame with all CDMs from all events.
        """
>>>>>>> dev
        
        # Return empty dataframe if no events are available
        if len(self) == 0: return pd.DataFrame()

<<<<<<< HEAD
        # Iterate over all events objects contained in the class.
        event_dataframes = []
        # util.progress_bar_init('Converting EventDataset to DataFrame', len(self._events), 'Events')
        for i, event in enumerate(self._events):
            # util.progress_bar_update(i)
            
            # Append Event object as a dataframe to the list.
            event_dataframes.append(event.to_dataframe())
            
        # util.progress_bar_end()
=======
        # Initialize list to store events and progress bar.
        event_dataframes = []
        pb_events = utils.ProgressBar(iterations=self._events,
            description='Importing Conjunction Events dataset...', 
            desc_loc='right')

        # Iterate over all events objects contained in the class.
        for e, event in enumerate(pb_events.iterations):
            
            # Append Conjunction Event object as a dataframe to the list.
            event_dataframes.append(event.to_dataframe())

            # Update progress bar.
            pb_events.refresh(i = e+1)
        
        # Update progress bar with the last message.
        pb_events.refresh(i = e+1, 
            description='Conjunction Events dataset imported successfully.')
>>>>>>> dev
        
        return pd.concat(event_dataframes, ignore_index=True)

    def dates(self) -> None:
<<<<<<< HEAD
        print('CDM| CREATION_DATE (mean)       | Days (mean, std)  | Days to TCA (mean, std)')
        
        # Iterate over maximum number of CDMs.
        for i in range(self.event_lengths_max):
            
            # Initialize lists to store information on creation date and days
            # to TCA. 
            creation_date_days, days_to_tca = [], []
            for event in self:
                if i < len(event):
                    creation_date_days.append(event[i]['__CREATION_DATE'])
                    days_to_tca.append(event[i]['__DAYS_TO_TCA'])
                    
            # Convert creation date and days to TCA to numpy arrays.
            creation_date_days = np.array(creation_date_days)
            days_to_tca = np.array(days_to_tca)
            
            # Get creation date from the first CDM object.
            date0 = self[0][0]['CREATION_DATE']
            creation_date_days_mean_str = utils.add_days_to_date_str(date0, creation_date_days.mean())
            
            print('{:02d} | {} | {:.6f} {:.6f} | {:.6f} {:.6f}' \
                  .format(i+1, creation_date_days_mean_str, 
                          creation_date_days.mean(), creation_date_days.std(), 
                          days_to_tca.mean(), days_to_tca.std()))
=======
        """Print summary on datetime information about CDM generation and TCA 
        using all information contained in the Events dataset. 
        """

        # Print header.
        print('CDM | ' + \
              'Days from event creation (mean, std)  | ' + \
              'Days to TCA (mean, std)')
        
        # Iterate over CDMs indexes
        for i in range(self.event_lengths_max):
            
            # Initialize lists to store information on creation date and days
            # to TCA from all events.
            creation_date_days, days_to_tca = [], []

            # For the same CDM index, iterate over all events in the object.
            for event in self:
                # Continue loop if CDM index is greater than number of CDMs 
                # stored in the event.
                if i >= len(event): continue

                # Append CDM(i) creation date in days since Event creation. 
                creation_date_days.append(event[i]['__CREATION_DATE'])

                # Append number of days left to TCA (from CDM(i) creation).
                days_to_tca.append(event[i]['__DAYS_TO_TCA'])
                    
            # Cast both lists as numpy arrays. Both arrays contain the values 
            # for a specific CDM index.
            creation_date_days = np.array(creation_date_days)
            days_to_tca = np.array(days_to_tca)
            
            # Print results
            print('{3:>02d} | {18:^.6f} {18:^.6f} | {11:^.6f} {11:^.6f}' \
                .format(i+1, creation_date_days.mean(), creation_date_days.std(), 
                        days_to_tca.mean(), days_to_tca.std()))


    def common_features(self, only_numeric:bool=False) -> list:
        """Get list of features contained in the Conjunction Events dataset.

        Args:
            only_numeric (bool, optional): Flag to determine wether only numeric 
            columns shall be retrieved or not. Defaults to False.

        Returns:
            list: List of features contained in the dataset.
        """

        # Get Conjunction Event dataset as a pandas DataFrame.
        df = self.to_dataframe()

        # Remove any columns and rows containing NaN values.
        df = df.dropna(axis=1)

        # Return only numeric inputs if required.
        if only_numeric:
            df = df.select_dtypes(include=['int', 'float64', 'float32'])

        # Get all columns except from __DAYS_TO_TCA.
        features = [f for f in list(df.columns) if f!='__DAYS_TO_TCA']

        return features

    def get_CDMs(self) -> list:
        """Get all CDMs contained in all events in a list.

        Returns:
            list: List with all CDMs in the Conjunction Events dataset.
        """

        # Initialize list to store all CDMs.
        events_cdms = []

        # Iterate over all events.
        for event in self:
            # Iterate over all CDMs in the event.
            for cdm in event:
                # Append CDM to the output list.
                events_cdms.append(cdm)

        return events_cdms
>>>>>>> dev

    @property
    def event_lengths(self):
        return list(map(len, self._events))

    @property
    def event_lengths_min(self):
        return min(self.event_lengths)

    @property
    def event_lengths_max(self):
        return max(self.event_lengths)

    @property
    def event_lengths_mean(self):
        return np.array(self.event_lengths).mean()

    @property
    def event_lengths_stddev(self):
        return np.array(self.event_lengths).std()

<<<<<<< HEAD
    def common_features(self, only_numeric:bool=False):
        df = self.to_dataframe()
        df = df.dropna(axis=1)
        if only_numeric:
            df = df.select_dtypes(include=['int', 'float64', 'float32'])
        features = list(df.columns)
        if '__DAYS_TO_TCA' in features:
            features.remove('__DAYS_TO_TCA')
        return features

    def get_CDMs(self):
        cdms = []
        for event in self:
            for cdm in event:
                cdms.append(cdm)
        return cdms

    def plot_event_lengths(self, figsize=(6, 4), file_name=None, *args, **kwargs):
        fig, ax = plt.subplots(figsize=figsize)
        event_lengths = self.event_lengths()
        ax.hist(event_lengths, *args, **kwargs)
        ax.set_xlabel('Event length (number of CDMs)')
        if file_name is not None:
            print('Plotting to file: {}'.format(file_name))
            plt.savefig(file_name)

    def plot_feature(self, feature_name, figsize=None, ax=None, return_ax=False, file_name=None, *args, **kwargs):
        if ax is None:
            if figsize is None:
                figsize = 5, 3
            fig, ax = plt.subplots(figsize=figsize)
        for event in self:
            event.plot_feature(feature_name, ax=ax, *args, **kwargs)
            if 'label' in kwargs:
                kwargs.pop('label')  # We want to label only the first Event in this EventDataset, for not cluttering the legend

        if file_name is not None:
            print('Plotting to file: {}'.format(file_name))
            plt.savefig(file_name)

        if return_ax:
            return ax

    def plot_features(self, feature_names, figsize=None, axs=None, return_axs=False, file_name=None, sharex=True, *args, **kwargs):
        if not isinstance(feature_names, list):
            feature_names = [feature_names]
        if axs is None:
            rows, cols = utils.tile_rows_cols(len(feature_names))
            if figsize is None:
                figsize = (cols*20/7, rows*12/6)
            fig, axs = plt.subplots(rows, cols, figsize=figsize, sharex=sharex)

        if not isinstance(axs, np.ndarray):
            axs = np.array(axs)
        for i, ax in enumerate(axs.flat):
            if i < len(feature_names):
                if i != 0 and 'legend' in kwargs:
                    kwargs['legend'] = False

                self.plot_feature(feature_names[i], ax=ax, *args, **kwargs)
            else:
                ax.axis('off')
        plt.tight_layout()

        if file_name is not None:
            print('Plotting to file: {}'.format(file_name))
            plt.savefig(file_name)

        if return_axs:
            return axs

    def plot_uncertainty(self, figsize:tuple = (20, 12), diagonal:bool = False, 
                         *args, **kwargs):
        if diagonal:
            features = ['CR_R','CT_T', 'CN_N', 'CRDOT_RDOT', 'CTDOT_TDOT', 'CNDOT_NDOT']
        else:
            features = ['CR_R', 'CT_R', 'CT_T', 'CN_R', 'CN_T', 'CN_N', 'CRDOT_R', 'CRDOT_T', 'CRDOT_N', 'CRDOT_RDOT', 'CTDOT_R', 'CTDOT_T', 'CTDOT_N', 'CTDOT_RDOT', 'CTDOT_TDOT', 'CNDOT_R', 'CNDOT_T', 'CNDOT_N', 'CNDOT_RDOT', 'CNDOT_TDOT', 'CNDOT_NDOT']
        features = list(map(lambda f: 'OBJECT1_'+f, features)) + \
                   list(map(lambda f: 'OBJECT2_'+f, features))
        return self.plot_features(features, figsize=figsize, *args, **kwargs)

    def filter(self, filter_func) -> EventDataset:
        events = []
        for event in self:
            if filter_func(event):
                events.append(event)
        return EventDataset(events=events)

    def __getitem__(self, index: Union[int, slice]) -> Union[EventDataset, Event]:
        if isinstance(index, slice):
            return EventDataset(events=self._events[index])
=======
    def plot_event_lengths(self, figsize:tuple = (7, 3), filepath:str = None, 
        *args, **kwargs) -> None:
        """Plot histogram with the number of events producing a number n of CDMs.

        Args:
            figsize (tuple, optional): Figure size. Defaults to (6, 4).
            filepath (str, optional): Path to save the plot. Defaults to None.
        """

        # Initialize figure object.
        fig, ax = plt.subplots(figsize = figsize)

        # Get number of CDMs per event using the custom porperty event_lengths.
        event_lengths = self.event_lengths

        # Plot histogram with the number of CDMs:
        #  - X axis: Number of CDMs is represented (1, 2, ... max_events_length)
        #  - Y axis: Number of events containing that number of CDMs.

        labels, counts = np.unique(event_lengths, return_counts=True)
        ax.bar(labels, counts, align='center')
        # ax.hist(event_lengths, bins=np.arange(1,23), align='mid', *args, **kwargs)

        ax.set_xticks(labels)

        # Set title, and X and Y axis labels.
        ax.set_title(r'Conjunction Data Messages histogram.')
        ax.set_xlabel(r'Number of CDMs')
        ax.set_ylabel(r'Number of Conjunction Events')

        # Set grid.
        ax.grid(True, linestyle='--')

        # Save figure if path is provided.
        if filepath is not None:
            print('Plotting to file: {}'.format(filepath))
            plt.savefig(filepath)

    def plot_feature(self, feature:str, figsize:tuple=(6, 4), 
        ax:plt.Axes = None, return_ax:bool = False, filepath:str = None, 
        *args, **kwargs) -> Union[None, plt.Axes]:
        """Plot evolution of a given feature throughout all ConjunctionEvents 
        contained in the dataset.

        Args:
            feature (str): Feature to plot.
            figsize (tuple, optional): Figure size. Defaults to (6, 4).
            ax (plt.Axes, optional): Axes object. Defaults to None.
            return_ax (bool, optional): Flag to return Axes object. Defaults to 
            False.
            filepath (str, optional): Path where the plot is saved. Defaults to 
            None.

        Returns:
            Union[None, plt.Axes]: Axes object if return_ax=True, None 
            otherwise.
        """

        # Initialize figure object if not passed into the function.
        if ax is None: fig, ax = plt.subplots(figsize = figsize)

        # Iterate over all events in the class to plot plot all events for a 
        # given feature.
        for event in self:

            # Plot evolution of the feature throughout the conjunctione event.
            # Use ConjunctionEvent class function plot_feature.
            event.plot_feature(feature, ax = ax, *args, **kwargs)

            # Label only the first event in the dataset. Remove label for the 
            # remaining plots. This prevents from cluttering the legend.
            if 'label' in kwargs: kwargs.pop('label')  

        # Save plot as a figure if filepath is provided.
        if filepath is not None:
            print('Plotting to file: {}'.format(filepath))
            plt.savefig(filepath)

        if return_ax: return ax

    def filter(self, filter_func) -> ConjunctionEventsDataset:
        """Filter Conjunction Events using a lambda function applied to the 
        ConjunctionEvent object.

        Args:
            filter_func (_type_): Function to apply upon ConjunctionEvent. When
             applying, it must return a boolean value.

        Returns:
            ConjunctionEventsDataset: Conjunction Events dataset filtered.
        """
        # Initialize events list.
        events = []

        # Iterate over all events contained in the dataset
        for event in self:
            
            # Apply filter function and append to the output list if True
            if filter_func(event): events.append(event)

        return ConjunctionEventsDataset(events=events)

    def __getitem__(self, index: Union[int, slice]) \
        -> Union[ConjunctionEventsDataset, ConjunctionEvent]:
        """Redefine magic attribute to get events from a given index or slice.

        Args:
            index (Union[int, slice]): Index(es) of ConjunctionEvent objects to 
            retrieve.

        Returns:
            Union[ConjunctionEventsDataset, ConjunctionEvent]: ConjunctionEvent 
            object if index is integer, ConjunctionEventsDataset otherwise.
        """
        if isinstance(index, slice):
            return ConjunctionEventsDataset(events=self._events[index])
>>>>>>> dev
        else:
            return self._events[index]

    def __len__(self) -> int:
<<<<<<< HEAD
        return len(self._events)

    def __repr__(self) -> str:
        if len(self) == 0:
            return 'EventDataset()'
        else:
            event_lengths = list(map(len, self._events))
            event_lengths_min = min(event_lengths)
            event_lengths_max = max(event_lengths)
            event_lengths_mean = sum(event_lengths)/len(event_lengths)
            return 'EventDataset(Events:{}, number of CDMs per event: {} (min), {} (max), {:.2f} (mean))'.format(len(self._events), event_lengths_min, event_lengths_max, event_lengths_mean)
=======
        """Get number of Conjunction Events contained in the dataset.

        Returns:
            int: Number of Conjunction Events.
        """
        return len(self._events)

    def __repr__(self) -> str:
        """Print detailed description about object class.

        Returns:
            str: Information about the ConjunctionEventsDataset object class.
        """
        if len(self) == 0:
            return 'ConjunctionEventsDataset()'
        else:
            event_lengths = list(map(len, self._events))
            return f'ConjunctionEventsDataset(Events:{len(self._events)} | ' + \
                   f'Number of CDMs per event: ' + \
                   f'{min(event_lengths)} (min), ' + \
                   f'{max(event_lengths)} (max), ' + \
                   f'{sum(event_lengths)/len(event_lengths):.2f} (mean))'
>>>>>>> dev

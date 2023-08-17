# Libraries used for hinting
from __future__ import annotations
from typing import Type, Union

import os
import pandas as pd
import requests
import time
import scipy.stats as st
import numpy as np
import warnings
import matplotlib.pyplot as plt
from datetime import datetime, timedelta

from . import utils

from sklearn.metrics import r2_score


#%%
from .cdm import CDM
from .event import ConjunctionEvent, ConjunctionEventsDataset

def kelvins_challenge_events(filepath:str, num_events:int = None, 
        date_tca:datetime = None, remove_outliers:bool = True,
        drop_features:list = ['c_rcs_estimate', 't_rcs_estimate'], 
        print_log:bool=False) -> ConjunctionEventsDataset:
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

            # Get object specific compulsory features.
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

            cdm['OBJECT2_OBJECT_TYPE'] = k_cdm['c_object_type']

            # Append CDM object to the event list.
            event_cdms.append(cdm)

            # Update progress bar
            pb_events.refresh(i = n, nested_progress = True)

        # Append ConjunctionEvent object to events list.
        events.append(ConjunctionEvent(event_cdms))

    # Update progress bar.
    pb_events.refresh(i = n-1, 
        description = f'Dataset imported ({len(events)} events).')

    return ConjunctionEventsDataset(events=events)


#%%
def import_cdm_data(filepath: str) -> pd.DataFrame:
    """Import CDM dataset from the Collision Avoidance Challenge and cast 
    features into the correct data types.

    Args:
        filepath (str): Path of the CSV file containing CDM data.

    Returns:
        pd.DataFrame: Dataframe object containing the dataset.
    """

    # Import training dataset
    df = pd.read_csv(filepath, sep=',', header=0, skipinitialspace=False)

    # Cast categorical features as category type.
    for feature in ['event_id', 'mission_id', 'c_object_type', 
                    't_time_lastob_start', 'c_time_lastob_start',
                    't_time_lastob_end', 'c_time_lastob_end']:
        df[feature] = df[feature].fillna(0, inplace=False).astype('category')
        #df[feature] = df[feature].fillna("UNKNOWN")

    # Cast indexes and integer values to int type.
    for feature in ['t_obs_available', 't_obs_used',
                    'c_obs_available', 'c_obs_used',
                    'F10', 'AP', 'F3M', 'SSN']:
        df[feature] = df[feature].fillna(0, inplace=False).astype('int16')

    # Sort values of dataframe by event_id and time_to_tca and re-index
    df.sort_values(by=['event_id', 'time_to_tca'], axis='index', 
                ascending=[True,False], inplace=True, ignore_index=True)

    return df

#%%
# Define Collision Risk Probability Estimator
class RobustScalerClipper():

    def __init__(self, data:np.ndarray, quantiles:tuple=(0.01, 0.99), 
                 with_offset:bool=False):

        # Instanciate data, quantiles, scale, center (if applicable), and 
        # scaled data
        self.data = data[~np.isnan(data)] 
        self.quantiles = quantiles
        self.quantiles_values = (np.quantile(self.data, quantiles[0]), 
                                 np.quantile(self.data, quantiles[1]))
        
        self.scale = utils.round_by_om(self.quantiles_values[1] - \
                                       self.quantiles_values[0])

        if with_offset:
            if np.sum(self.data<0)==0:
                self.offset = np.min(self.data)
            elif np.sum(self.data>0)==0:
                self.offset = np.max(self.data)
            else:
                self.offset = utils.round_by_om(np.quantile(self.data, 0.5))
        else:
            self.offset = None

        if with_offset:
            self.scaled_data = (self.data-self.offset)/self.scale
        else: 
            self.scaled_data = self.data/self.scale

    def clip(self, clip_lims:tuple = (-1.0, 1.0)):

        self.clip_lims = clip_lims
        self.clipped_data = np.clip(self.scaled_data, 
                                    a_min=clip_lims[0], 
                                    a_max=clip_lims[1])

        self.outliers = np.sum(self.scaled_data>clip_lims[1]) + \
                        np.sum(self.scaled_data<clip_lims[0])
        
        return self
#%%
class FitScipyDistribution:
    def __init__(self, data, distribution):

        # Instanciate distribution and distribution name
        self.dist = distribution
        self.name = distribution.name

        # Convert data input to numpy array and keep only finites values to 
        # instanciate data
        data = np.asarray(data, dtype=np.float64)
        data = data[np.isfinite(data)]
        self.data = data

        # Ignore warnings from data that can't be fit
        warnings.filterwarnings("ignore")
        
        # Try to fit the distribution
        tic = time.process_time()
        
        try:
            
            # Fit distribution to data
            fitting_output = distribution.fit(self.data)

            # Separate parts of parameters
            params = {'loc':   fitting_output[-2], 'scale': fitting_output[-1],
                      'arg_values': [], 'arg_names': []}

            if distribution.shapes!=None: 
                params['arg_names']  = distribution.shapes.split(", ") 
                params['arg_values'] = list(fitting_output[:-2])

            params['names'] = params['arg_names'] + ['loc', 'scale']
            params['values'] = params['arg_values'] + \
                               [params['loc'], params['scale']]
            self.params = params
                
        except Exception:

            self.params = None
            pass
        
        # Re-enable warnings
        warnings.filterwarnings('default')
        
        toc = time.process_time()
        self.processing_time = toc-tic
        
    def rvs(self, **kwargs):
        """Generate data distribution from stats function."""
        
        # Get arguments and keyword based arguments for the random data 
        # generation
        args = self.params['arg_values']
        kwds = {'loc': self.params['loc'], 'scale': self.params['scale']}
        
        if 'limits' in kwargs.keys():
            # Get the norm to scale probabilities from the PDF; this transforms 
            # the range from [0, 1] to [0, norm]
            limits = kwargs['limits']
            norm  = self.dist.cdf(limits[1], *args, **kwds) - \
                    self.dist.cdf(limits[0], *args, **kwds)
            
            # Get sample using the inverse CDF: 
            # 1.- Scale probabilities from PDF using the Norm.
            # 2.- Add the lowest probability to ensure lower value in data 
            # corresponds matches the lower boundary

            scaled_pdf = np.random.rand(size)*norm + \
                         self.dist.cdf(limits[0], *args, **kwds)
            data       = self.dist.ppf(scaled_pdf, *args, **kwds)
        else:
            
            # If no limit or data range is required, used built-in RVS function 
            # from SciPy.
            kwds.update({'size': kwargs.get('size', int(1e3)), \
                         'random_state':kwargs.get('random_state', 1)})
            data = self.dist.rvs(*args, **kwds)
        
        return data

        
    def r2_score(self):
        """Compute coefficient of determination (R2 score)"""
        
        if self.params == None: return 0
        
        # Disable warnings
        warnings.filterwarnings("ignore")
        
        # Get histogram of original data
        try:
            
            y, bin_edges = np.histogram(self.data, 
                                        bins=utils.nbins(self.data,'fd')['n'], 
                                        density=True)
            bin_centers = (bin_edges + np.roll(bin_edges, -1))[:-1] / 2.0

            # Calculate fitted PDF and error with fit in distribution
            pdf = self.dist.pdf(bin_centers, loc=self.params['loc'], 
                                scale=self.params['scale'], 
                                *self.params['arg_values'])

            self.n_bins = len(bin_centers)
            self.bins_width = bin_edges[1]-bin_edges[0]

            r2 = r2_score(y, pdf)
            r2 = r2 if abs(r2)<=1 else r2/abs(r2)
        except Exception:
            r2 = 0
            pass
        
        # Re-enable warnings
        warnings.filterwarnings("default")
        
        return r2
    
    def pdf(self, size=10000, tol=1e-4):
        """Generate distributions's Probability Distribution Function"""
        
        if self.params == None: return None
        
        # Separate parts of parameters
        arg = self.params['arg_values']
        loc = self.params['loc']
        scale = self.params['scale']
        
        # Get standard data limits for better representation
        std_lims, std_data, outliers = \
            utils.outliers_boundaries(data = self.data.flatten(), 
                                      threshold = 1.5, 
                                      positive_only = False)
        
        # Build PDF and turn into pandas Series
        x = np.linspace(std_lims[0], std_lims[1], size)
        y = self.dist.pdf(x, loc=loc, scale=scale, *arg)
        
        pdf = pd.Series(y, x)
        
        return pdf

#%%
def get_scipy_distributions(cwd:str, 
    stdists_exc:list = ['studentized_range', 'levy_l_gen', 'levy_stable']) \
    -> list:
    """Get list of continuous distributions from SciPy.org website

    Args:
        cwd (str): Current working directory path.
        stdists_exc (list, optional): List of distributions names to exclude 
        from list. Defaults to ['studentized_range', 'levy_l_gen', 
        'levy_stable'].

    Returns:
        list: List of distributions available in SciPy.org.
    """
    
    filepath = os.path.join(cwd,'notebooks','nbtemp','scipy_distributions.csv')
    try:
        # Get Continuous distributions table from SciPy website.
        url = 'https://docs.scipy.org/doc/scipy/reference/stats.html'
        tbody = pd.read_html(requests.get(url).content)[1]
        
        # Create pandas dataframe and save CSV local file to be able to use the 
        # notebook offline.
        df_stdist = pd.DataFrame(data = tbody[0].to_list(), 
                                 columns=['scipy_distributions'])
        
        if not os.path.exists(os.path.join(cwd,'notebooks','nbtemp')):
            os.mkdir(os.path.join(cwd,'notebooks','nbtemp'))
        
        # Export dataframe as CSV file in the temporary folder
        df_stdist.to_csv(filepath, sep=',')
        
    except Exception:
        print("Could not read SciPy website. Importing data from local file...")
        
        # Import scipy distributions if working offline
        df_stdist = pd.read_csv(filepath, sep=',', header=0, index_col=None, 
                                skipinitialspace=False)

        pass
    
    # Evaluate list of objects in str format to convert it into a Python list
    stdists_list = []
    
    # Iterate through all the continous distributions on the website and 
    # evaluate it to discard those that are not compatible with the library 
    # version installed.
    for stdist_i in df_stdist['scipy_distributions'].to_list():
        try:
            if not stdist_i in stdists_exc: 
                stdists_list.append(eval('st.' + stdist_i))
        except Exception:
            pass

    return stdists_list

#%%
def find_best_distribution(data: pd.Series, scipy_distributions:list):
    """Find best fitted distribution (higher R2 score vs actual probability 
    density).

    Args:
        data (pd.Series): Array upon which find the best SciPy distribution.
        scipy_distributions (list): List of SciPy distributions to fit.

    Returns:
        Union[instance, pd.DataFrame]: Best distribution object and ranking of 
        all distributions with its R2 score.
    """
    
    # Remove non-finite values and initialize best holder using the norm 
    # distribution
    data = data[data.notnull()]
    best_stdist = FitScipyDistribution(data, st.norm)
    
    fitting_results = []

    # Estimate distribution parameters from data
    for i, stdist_i in enumerate(scipy_distributions):
        
        # Fit stdist to real data
        fitted_stdist = FitScipyDistribution(data, stdist_i)

        print(f'Progress: {(i+1)/len(scipy_distributions)*100:3.1f}% '
              f'({i:3d}/{len(scipy_distributions)-1:3d})'
              f' | Best dist. (R2={best_stdist.r2_score():.2f}) '
              f'-> {best_stdist.name:22s}'
              f' | New dist. (R2={fitted_stdist.r2_score():.2f}) '
              f'-> {fitted_stdist.name:22s} ', end='\r')
        
        # If it improves the current best distribution, reassign best 
        # distribution
        if best_stdist.r2_score() < fitted_stdist.r2_score(): 
            best_stdist = fitted_stdist
        
        params = {'params': fitted_stdist.params, 
                  'r2_score': fitted_stdist.r2_score()}
        fitting_results.append([fitted_stdist.name, params])
        
    # Sort values of dataframe by sse ascending and and re-index.
    ranking = pd.DataFrame(data=fitting_results, 
                           columns=['distribution', 'results'])
    
    
    return best_stdist, ranking

#%%
def plot_histogram(df_input:pd.DataFrame, features:list, 
                   bins_rule:str='fd', **kwargs) -> None:
    """Plot custom histogram for dataframe features. 

    Args:
        df_input (pd.DataFrame): Pandas dataframe to plot.
        features (list): Features from pandas dataframe to plot.
        bins_rule (str): Rule to compute number of bins. Defaults to 'fd'.

    Returns:
        None
    """

    # Check that all features are categorical or numerical
    if 'str' in df_input[features].dtypes.values: return None 
    if len(np.unique([str(t) for t in list(df_input[features].dtypes)])) > 1: 
        return None

    # Normalize data if passed as argument
    data = []
    for f, feature in enumerate(features):
        data.append(df_input[feature].dropna().to_numpy())

    all_data = df_input[features].dropna().to_numpy().flatten()

    # Create figure object
    plt.figure(figsize=kwargs.get('figsize', (8,3)))
    
    axes = plt.gca()

    # Get kwargs specific for the plot
    plt_kwargs = dict(edgecolor='white', align='mid', alpha=1.0, rwidth=1.0)
    # plt_kwargs.update(kwargs.get('plt_kwargs',dict()))

    if not 'category' in df_input[features].dtypes.values:
        # Compute number of outliers for better representation on histogram

        std_lims, std_data, outliers = utils.outliers_boundaries(all_data, 
                                            threshold = 1.5, 
                                            positive_only=np.sum(all_data<0)==0)

        # Compute new X-axis limits for a better plot representation.
        xlim = kwargs.get('xlim', 
                          (utils.round_by_om(max(std_lims[0], all_data.min()), 
                                             abs_method='floor'), 
                           utils.round_by_om(min(std_lims[1], all_data.max()), 
                                             abs_method='ceil')))
        plt.xlim(xlim)
        plt_kwargs.update(dict(range=xlim))

        # Calculate number of bins to plot histogram
        nbins =  utils.nbins(std_data, bins_rule)
        bins = kwargs.get('bins', nbins['range'])

        description_table = pd.DataFrame(data=df_input[features]) \
                            .describe(percentiles=[0.05, 0.25, 0.5, 0.75, 0.95])

    else:
        # Calculate number of bins to plot histogram
        bins = len(df_input[features[0]].cat.categories.values)

        description_table = pd.DataFrame(data=df_input[features]).describe()


    # Format values for the description table
    values=[[utils.number2latex(element) for element in array] \
            for array in description_table.to_numpy()]
    columns = kwargs.get('describe_colnames', 
                        [r'\texttt{' + feature + '}' for feature in features])
    text = utils.df2latex(pd.DataFrame(data=values, 
                                       index=list(description_table.index), 
                                       columns=columns))

    # Print statistical summary before the histogram
    if kwargs.get('describe', True): 
        t = axes.text(1.04, 0.5, text, size=10, ha='left', 
                  va='center', c='black', transform=axes.transAxes, 
                  bbox=dict(facecolor='white', edgecolor='black', 
                  alpha=0.75, pad=5))

    # Plot histogram
    if 'hist_kwargs' in list(kwargs.keys()):
        for f in range(len(features), 0, -1):
            plt_kwargs.update(kwargs['hist_kwargs'][f-1])
            plt.hist(data[f-1], bins=bins, **plt_kwargs)
    else:
        plt.hist(data, bins=bins, **plt_kwargs)
    
    # Compute new Y-axis limits for a better plot representation.
    ylim = (axes.get_ylim()[0], 
            utils.round_by_om(axes.get_ylim()[1], abs_method='ceil'))
    ylim = kwargs.get('ylim', ylim)
    plt.yticks(np.linspace(ylim[0],ylim[1],5))
    plt.ylim(ylim)

    # Set axis labels and title
    xlabel = kwargs.get('xlabel', r'\texttt{' + " ".join(features) + '}')
    ylabel = kwargs.get('ylabel', r'Number of objects')
    title  = kwargs.get('title',  r'Histogram')
    
    plt.ylabel(ylabel=ylabel, fontsize=12) 
    plt.xlabel(xlabel=xlabel, fontsize=12)
    plt.title(label=title,   fontsize=12)
    
    # Plot legend and print plot
    if kwargs.get('legend', True): plt.legend(loc='upper right', fontsize=10)
    plt.grid(True, linestyle='--')
    if 'filepath' in list(kwargs.keys()): 
        plt.savefig(kwargs['filepath'], bbox_inches='tight')
    plt.show()

    return None


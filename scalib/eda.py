# Libraries used for hinting
from __future__ import annotations
from typing import Union

import os
import pandas as pd
import time
import scipy.stats as st
import numpy as np
import warnings
import matplotlib.pyplot as plt
from datetime import datetime, timedelta

import scalib.utils as utils

from sklearn.metrics import r2_score as r2_sklearn

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

            cdm['OBJECT2_OBJECT_TYPE'] = k_cdm['c_object_type']

            values_extra = {}
            for feature in ['risk', 'max_risk_estimate', 'max_risk_scaling']:
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



#%% CLASS: RobustScalerClipper
# Define Collision Risk Probability Estimator
class RobustScalerClipper():

    def __init__(self, data:np.ndarray, quantiles:tuple=(0.01, 0.99), 
                 with_offset:bool=False) -> None:
        """Initialize object with scaled data and associated parameters.

        Args:
            data (np.ndarray): Data to scale.
            quantiles (tuple, optional): Quantiles used to normalize the data. 
            Defaults to (0.01, 0.99).
            with_offset (bool, optional): Offset scaled data. Defaults to False.
        """

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

    def clip(self, clip_lims:tuple = (-1.0, 1.0)) -> RobustScalerClipper:
        """Clip data using upper and/or lower limits.

        Args:
            clip_lims (tuple, optional): Limits to clip the data. Defaults to 
            (-1.0, 1.0).
        """

        self.clip_lims = clip_lims
        self.clipped_data = np.clip(self.scaled_data, 
                                    a_min=clip_lims[0], 
                                    a_max=clip_lims[1])

        self.outliers = np.sum(self.scaled_data>clip_lims[1]) + \
                        np.sum(self.scaled_data<clip_lims[0])
        
#%% CLASS: FitScipyDistribution
class FitScipyDistribution:

    def __init__(self, data:np.ndarray, distribution:st) -> None:
        """Initialize distribution parameters and information.

        Args:
            data (np.ndarray): NumPy array with the values to fit.
            distribution (st): SciPy distribution to fit to the data.
        """

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

            if distribution.shapes != None: 
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
        
    def rvs(self, size:int=1000, random_state:int=1, **kwargs) -> np.ndarray:
        """Generate data distribution from stats function. Random Variates of 
        given Size.

        Args:
            size (int, optional): Size of NumPy array to generate. Defaults to 
            1000.
            random_state (int, optional): Random state for reproducibility. 
            Defaults to 1.

        Returns:
            np.ndarray: Random data as a NumPy array following the distribution.
        """
        
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
            kwds.update({'size': size, 'random_state':random_state})
            data = self.dist.rvs(*args, **kwds)
        
        return data

        
    def r2_score(self) -> float:
        """Compute coefficient of determination (R2 score)
        """
        
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

            r2 = r2_sklearn(y, pdf)
            r2 = r2 if abs(r2)<=1 else r2/abs(r2)
        except Exception:
            r2 = 0
            pass
        
        # Re-enable warnings
        warnings.filterwarnings("default")
        
        return r2
    
    def pdf(self, size:int=10000) -> pd.Series:
        """Generate distributions's Probability Distribution Function (PDF).

        Args:
            size (int, optional): Size of the array resulting from the PDF. 
            Defaults to 10000.

        Returns:
            pd.Series: Series with the PDF values over an array of X values.
        """
        
        if self.params == None: return None
        
        # Separate parts of parameters
        arg = self.params['arg_values']
        loc = self.params['loc']
        scale = self.params['scale']
        
        # Get standard data limits for better representation
        std_lims, _, _ = \
            utils.outliers_boundaries(data = self.data.flatten(), 
                                      threshold = 1.5, 
                                      positive_only = False)
        
        # Build PDF and turn into pandas Series
        x = np.linspace(std_lims[0], std_lims[1], size)
        y = self.dist.pdf(x, loc=loc, scale=scale, *arg)
        
        pdf = pd.Series(y, x)
        
        return pdf



#%% FUNCTION: find_best_distribution
def find_best_distribution(data: np.ndarray, 
                           stdists_exc:list = ['studentized_range', 
                                               'levy_l_gen', 
                                               'levy_stable']) -> tuple:
    """Find best fitted distribution (higher R2 score vs actual probability 
    density).

    Args:
        data (np.ndarray): Array upon which find the best SciPy distribution.
        stdists_exc (list, optional): List of distributions names to exclude 
        from list. Defaults to ['studentized_range', 'levy_l_gen', 
        'levy_stable'].

    Returns:
        tuple: Best distribution object and ranking of all distributions with 
        its R2 score.
    """

    # Get Continuous distributions available in SciPy stats module
    scipy_distributions = {}
    for i in dir(st):
        # Get name of the class
        class_name = str(getattr(st,i).__class__)

        # If it is a continuous distribution class, add it to the dictionary
        if 'scipy.stats._continuous_distns' in class_name:
            scipy_distributions[i] = getattr(st,i)
    
    # Remove non-finite values and initialize best holder using the norm 
    # distribution
    best_stdist = FitScipyDistribution(data, st.norm)

    pb_dist = utils.ProgressBar(iterations = scipy_distributions.values(), 
                    title='FITTING SCIPY DISTRIBUTIONS:',
                    description = "Looking for best distribution")
    
    fitting_results = []

    # Estimate distribution parameters from data
    for i, stdist_i in enumerate(pb_dist.iterations):

        # If distribution is one of the excluded skip fitting process.
        if stdist_i.name in stdists_exc: continue
        
        # Fit stdist to real data.
        fitted_stdist = FitScipyDistribution(data, stdist_i)

        # Update progress bar.
        description = f"Max R2 = {best_stdist.r2_score():6.4e} > " + \
                      f"Fitting {fitted_stdist.name}"
        
        pb_dist.refresh(i = i+1, 
                        description = description,
                        nested_progress = True)
        
        # If it improves the current best distribution, reassign best 
        # distribution
        if best_stdist.r2_score() < fitted_stdist.r2_score(): 
            best_stdist = fitted_stdist
        
        params = {'params': fitted_stdist.params, 
                  'r2_score': fitted_stdist.r2_score()}
        fitting_results.append([fitted_stdist.name, params])

    # Show completed process
    pb_dist.refresh(i = i+1, description = description.split('>')[0])
        
    # Sort values of dataframe by sse ascending and and re-index.
    ranking = pd.DataFrame(data = fitting_results, 
                           columns=['distribution', 'results'])
    
    
    return best_stdist, ranking

#%% FUNCTION: plot_histogram
def plot_histogram(data:np.ndarray, features:list, figsize:tuple = (6, 3), 
                   return_ax:bool=False, bins:int=40, show_stats:bool = False, 
                   show:bool=True, **kwargs):
    """Plot custom histogram for dataframe features. 

    Args:
        df_input (pd.DataFrame): Pandas dataframe to plot.
        features (list): Features from pandas dataframe to plot.
        figsize (tuple, optional): Figure size. Defaults to (8, 3).
        return_ax (bool, optional): Return Axes object if True. Defaults to 
        False.
        bins (int, optional): Number of bins to use in the histogram. Defaults 
        to 40.
        show_stats (bool, optional): Add table with statistics to the right
        hand size of the plot. Defaults to False.
        show (bool, optional): Show plot inline. Defaults to True.

    Returns:
        Union[None, plt.Axes]: Axes object if return_ax = True, None otherwise.
    """

    if not isinstance(data, list):
        data = [data]
        features = [features]

    # Create figure object
    fig, ax = plt.subplots(figsize = figsize)

    # Get kwargs specific for the plot
    plt_kwargs = dict(edgecolor='white', align='mid', alpha=1.0, rwidth=1.0)

    if isinstance(data[0], str):

        # Calculate number of bins to plot histogram
        if bins!=len(set(data)):
            warnings.warn(f'\nNumber of bins for categorical data shall be '
                          f'equal to the number of unique categories.'
                          f'\n Setting bins={len(set(data))}.')
            bins = len(set(data))

        df_stats = pd.DataFrame(data = data).describe()

    else:

        # Compute new X-axis limits for a better plot representation.
        xlim = kwargs.get('xlim', None)

        if xlim is None:
            xlim_percentiles = kwargs.get('xlim_percentiles', (2, 98))
            xlim = (utils.round_by_om(np.percentile(data, xlim_percentiles[0])), 
                    utils.round_by_om(np.percentile(data, xlim_percentiles[1])))

        # Set the limits of representation for the X-axis
        ax.set_xlim(xlim)
        plt_kwargs.update(dict(range=xlim))

        # Calculate number of bins to plot histogram
        df_stats = pd.DataFrame(data = data[0], columns = features) \
                            .describe(percentiles=[0.5, 0.05, 0.95])
        df_stats.rename(index={'count': 'Count', 'mean':'Mean','std':'Std',
                               '50%':'Median', 'min':'Min','max':'Max'}, 
                        inplace=True)

    # Add summary before the histogram
    if show_stats: 
        # Format numbers to be in LaTeX format
        values = [[utils.number2latex(number) for number in row] \
                for row in df_stats.to_numpy()]
        
        # Get table in LaTeX format
        columns = [r'\texttt{' + feature + '}' for feature in features]
        text = utils.df2latex(pd.DataFrame(data = values, 
                                        index = list(df_stats.index), 
                                        columns=columns))

        # Add LaTeX table to the right of the plot
        ax.text(1.04, 0.5, text, size=10, ha='left', 
                va='center', c='black', transform=ax.transAxes, 
                bbox=dict(facecolor='white', edgecolor='black', 
                          alpha=0.75, pad=5))


    # Plot histogram
    ax.hist(data, bins=bins, color = kwargs.get('color', 'tab:blue'), 
            **plt_kwargs)
    
    # Compute new Y-axis limits for a better plot representation.
    ylim = kwargs.get('ylim', ax.get_ylim())
    ax.set_yticks(np.linspace(ylim[0], ylim[1], 5))
    ax.set_ylim(ylim)

    # Set axis labels and title
    xlabel = kwargs.get('xlabel', r'\texttt{' + " ".join(features) + '}')
    ylabel = kwargs.get('ylabel', r'Number of objects')
    title  = kwargs.get('title',  r'Histogram')
    
    ax.set_ylabel(ylabel = ylabel, fontsize = 12) 
    ax.set_xlabel(xlabel = xlabel, fontsize = 12)
    ax.set_title(label = title,   fontsize = 12)
    
    # Plot legend and print plot
    if kwargs.get('legend', False): ax.legend(loc='upper right', fontsize=10)
    ax.grid(True, linestyle='--')

    # Remove blank spaces around the plot to be more compact.
    plt.tight_layout()

    if 'filepath' in list(kwargs.keys()):
        fig.savefig(kwargs['filepath'], bbox_inches='tight')

    if return_ax: return ax
    if show: plt.show()

#%% FUNCTION: plot_scipy_pdf
def plot_scipy_pdf(data: np.ndarray, stdist: st, figsize:tuple=(7,3),
                   return_ax:bool = False, filepath:str = None, show:bool=True, 
                   **kwargs) -> Union[None, plt.Axes]:
    """Plot histogram and PDF based on a given distribution and its parameters.

    Args:
        data (np.ndarray): Actual data to plot in the histogram.
        stdist (st): SciPy distribution to plot.
        figsize (tuple, optional): Figure size. Defaults to (7, 3).
        return_ax (bool, optional): Return ax object.
        filepath (str, optional): Path of the folder where the figure is saved.
        show (bool, optional): Show plot inline. Defaults to True.

    Returns:
        None: None
    """

    # Display plot
    fig, ax = plt.subplots(figsize = figsize)

    # Describe PDF fitting parameters if user wants
    if kwargs.get('show_parameters', False):
        
        # Create description table to print statistical model used
        index  = ['Data points', 'Function', '$R^2$ score']
        values = [utils.number2latex(len(data)), 
                  r'\texttt{' + stdist['dist'].name + r'}', 
                  utils.number2latex(stdist['r2_score'])]

        # Include information on the parameters
        if stdist['params']!=None:
            index = index + ['', r'\textbf{Parameters:}'] + \
            [r'\texttt{' + n + '}' for n in stdist['params']['names']]

            values  = values + ['', ''] + \
                [utils.number2latex(v) for v in stdist['params']['values']]

        # Get the latex output to include table on the right hand side of the 
        # chart.
        text = utils.df2latex(df = pd.DataFrame(index=index, 
                                              data=values, 
                                              columns=['']), 
                                column_format='cc')

        ax.text(1.04, 0.5, text, size=10, transform=ax.transAxes,
                ha='left', va='center', c='black',  
                bbox=dict(facecolor='white', edgecolor='black', 
                          alpha=0.75, pad=5))
    
    # Calculate number of bins to use in the histogram
    bins = utils.nbins(data, kwargs.get('bins_method', 'fd'))

    # Get standard data limits for better representation
    std_lims, _, _ = utils.outliers_boundaries(data.flatten(), 
                                               threshold = 1.5, 
                                               positive_only=False)

    # Separate parts of parameters
    arg     = stdist['params']['arg_values'] 
    loc     = stdist['params']['loc'] 
    scale   = stdist['params']['scale']
    
    # Build PDF and turn into pandas Series
    x = np.linspace(std_lims[0], std_lims[1], 1000)
    y = stdist['dist'].pdf(x, loc=loc, scale=scale, *arg)
    
    # Plot the Probability density function
    ax.plot(x, y, lw = 1.5, color = "orange", label="Estimated")
    
    # Plot histogram
    n, bin_edges, _ = ax.hist(data, bins = bins['n'], density = True, 
                               histtype='bar', color="dimgrey", 
                               edgecolor = "white", label="Actual")
    
    # Get standard data boundaries for better readability of the plot
    positive_only = (np.sum(data<0)==0)
    std_lims, _, _ = utils.outliers_boundaries(data, 
                                               positive_only = positive_only)
    
    # Set X and Y-axis limits
    ax.set_xlim(max(bin_edges[0]  - bins['width'],std_lims[0]) , 
             min(bin_edges[-1] + bins['width'],std_lims[1]))
    ax.set_ylim(0, np.max(n)*1.5)
    
    # Set title and axis labels
    ax.set_title(r'Actual vs estimated probability density', fontsize=10)
    ax.set_xlabel(kwargs.get('xlabel', 'Feature'))
    ax.set_ylabel(r'Probability density')
    ax.grid(True, linestyle="dashed", alpha=0.5)
    ax.legend(loc="best", fontsize=10)

    # Remove blank spaces around the plot to be more compact.
    plt.tight_layout()
    
    # Save plot only if filepath is provided.
    if filepath is not None:
        print('Plotting to file: {}'.format(filepath))
        fig.savefig(filepath)

    if return_ax: return ax
    if show: plt.show()
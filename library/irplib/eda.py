
import os
import pandas as pd
import requests
import time
import scipy.stats as st
import numpy as np
import warnings
import matplotlib.pyplot as plt

from . import utils

from sklearn.metrics import r2_score

# Import function to clear output
from IPython.display import clear_output

from typing import Union

#%%
class FitScipyDistribution:
    def __init__(self, data, distribution):
        self.dist = distribution
        self.data = data[~np.isnan(data)]
        self.name = distribution.name
        
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
            params['values'] = params['arg_values'] + [params['loc'], params['scale']]
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
        
        # Get arguments and keyword based arguments for the random data generation
        args = self.params['arg_values']
        kwds = {'loc': self.params['loc'], 'scale': self.params['scale']}
        
        if 'limits' in kwargs.keys():
            # Get the norm to scale probabilities from the PDF; this transforms the range 
            # from [0, 1] to [0, norm]
            limits = kwargs['limits']
            norm  = self.dist.cdf(limits[1], *args, **kwds)-self.dist.cdf(limits[0], *args, **kwds)
            
            # Get sample using the inverse CDF: 
            # 1.- Scale probabilities from PDF using the Norm.
            # 2.- Add the lowest probability to ensure lower value in data corresponds 
            #     matches the lower boundary

            scaled_pdf = np.random.rand(size)*norm + self.dist.cdf(limits[0], *args, **kwds)
            data       = self.dist.ppf(scaled_pdf, *args, **kwds)
        else:
            
            # If no limit or data range is required, used built-in RVS function from SciPy.
            kwds.update({'size': kwargs.get('size', int(1e3)), 'random_state':kwargs.get('random_state', 1)})
            data = self.dist.rvs(*args, **kwds)
        
        return data

        
    def r2_score(self):
        """Compute coefficient of determination (R2 score)"""
        
        if self.params == None: return 0
        
        # Disable warnings
        warnings.filterwarnings("ignore")
        
        # Get histogram of original data
        try:
            
            y, bin_edges = np.histogram(self.data, bins=utils.nbins(self.data,'fd')['n'], density=True)
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
        arg, loc, scale = self.params['arg_values'], self.params['loc'], self.params['scale']
        
        # Get standard data limits for better representation
        std_lims = utils.outliers_boundaries(self.data, threshold = 1.5, positive_only=False)
        
        # Build PDF and turn into pandas Series
        x = np.linspace(std_lims[0], std_lims[1], size)
        y = self.dist.pdf(x, loc=loc, scale=scale, *arg)
        
        pdf = pd.Series(y, x)
        
        return pdf

#%%
def get_scipy_distributions(cwd:str, stdists_exc:list = ['studentized_range', 'levy_l_gen', 'levy_stable']) -> list:
    """Get list of continuous distributions from SciPy.org website

    Args:
        cwd (str): Current working directory path.
        stdists_exc (list, optional): List of distributions names to exclude from list. Defaults to ['studentized_range', 'levy_l_gen', 'levy_stable'].

    Returns:
        list: List of distributions available in SciPy.org.
    """
    
    filepath = os.path.join(cwd,'notebooks','nbtemp','scipy_distributions.csv')
    try:
        # Get Continuous distributions table from SciPy website.
        url = 'https://docs.scipy.org/doc/scipy/reference/stats.html'
        tbody = pd.read_html(requests.get(url).content)[1]
        
        # Create pandas dataframe and save CSV local file to be able to use the notebook offline.
        df_stdist = pd.DataFrame(data = tbody[0].to_list(), columns=['scipy_distributions'])
        
        if not os.path.exists(os.path.join(cwd,'notebooks','nbtemp')):
            os.mkdir(os.path.join(cwd,'notebooks','nbtemp'))
        
        # Export dataframe as CSV file in the temporary folder
        df_stdist.to_csv(filepath, sep=',')
        
    except Exception:
        print("Could not read SciPy.org website. Importing data from local file...")
        
        # Import scipy distributions if working offline
        df_stdist = pd.read_csv(filepath, sep=',', header=0, index_col=None, skipinitialspace=False)

        pass
    
    # Evaluate list of objects in str format to convert it into a Python list
    stdists_list = []
    
    # Iterate through all the continous distributions on the website and evaluate it
    # to discard those that are not compatible with the library version installed.
    for stdist_i in df_stdist['scipy_distributions'].to_list():
        try:
            if not stdist_i in stdists_exc: stdists_list.append(eval('st.' + stdist_i))
        except Exception:
            pass

    return stdists_list

#%%
def find_best_distribution(data: pd.Series, scipy_distributions:list):
    """Find best fitted distribution (higher R2 score vs actual probability density).

    Args:
        data (pd.Series): Array upon which find the best SciPy distribution.
        scipy_distributions (list): List of SciPy distributions to fit.

    Returns:
        Union[instance, pd.DataFrame]: Best distribution object and ranking of all distributions with its R2 score.
    """
    
    # Remove non-finite values and initialize best holder using the norm distribution
    data = data[np.isfinite(data)]
    best_stdist = FitScipyDistribution(data, st.norm)
    
    fitting_results = []

    # Estimate distribution parameters from data
    for i, stdist_i in enumerate(scipy_distributions):
        
        # Fit stdist to real data
        fitted_stdist = FitScipyDistribution(data, stdist_i)

        # clear_output(wait=True)

        print('Progress %5.1f%% (%3d/%3d)  Best: %10s (R2: %0.2f)  Distribution: %10s (R2=%0.2f) \t' %
              ((i+1)/len(scipy_distributions)*100, i, len(scipy_distributions)-1, 
               best_stdist.name, best_stdist.r2_score(), 
               fitted_stdist.name, fitted_stdist.r2_score()), end='\r')
        
        # If it improves the current best distribution, reassign best distribution
        if best_stdist.r2_score() < fitted_stdist.r2_score(): best_stdist = fitted_stdist
        
        fitting_results.append([fitted_stdist.name, fitted_stdist.r2_score()])
        
    # Sort values of dataframe by sse ascending and and re-index.
    ranking = pd.DataFrame(data=fitting_results, columns=['distribution', 'r2_score'])
    ranking.sort_values(by=['r2_score'], axis='index', ascending=[False], 
                              inplace=True, ignore_index=True)
    
    # Clear output to print final results
    # clear_output(wait=True)
    
    return best_stdist, ranking

#%%
def plot_histogram(df_input:pd.DataFrame, features:list, bins_rule:str='fd', **kwargs) -> None:
    """Plot custom histogram for dataframe features. 

    Args:
        df_input (pd.DataFrame): Pandas dataframe to plot.
        features (list): Features from pandas dataframe to plot.
        bins_rule (str): Rule to compute number of histogram bins. Defaults to 'fd'.

    Returns:
        None
    """

    # Check that all features are categorical or numerical
    if 'str' in df_input[features].dtypes.values: return None 
    if len(np.unique(df_input[features].dtypes.values)) > 1: return None

    # Normalize data if passed as argument
    data = []
    for f, feature in enumerate(features):
        data.append(df_input[feature].to_numpy())
        data[-1] = data[-1][~pd.isnull(data[-1])]

    all_data = df_input[features].to_numpy().flatten()
    all_data = all_data[~pd.isnull(all_data)]

    # Create figure object
    plt.figure(figsize=kwargs.get('figsize', (8,3)))
    
    axes = plt.gca()

    # Get kwargs specific for the plot
    plt_kwargs = dict(edgecolor='white', align='mid', alpha=1.0, rwidth=1.0)
    # plt_kwargs.update(kwargs.get('plt_kwargs',dict()))

    if not 'category' in df_input[features].dtypes.values:
        # Compute number of outliers for better representation on histogram

        std_lims = utils.outliers_boundaries(all_data, threshold = 1.5, positive_only=np.sum(all_data<0)==0)

        # Identify outliers
        outliers = (all_data<std_lims[0]) | (all_data>std_lims[1])

        # Compute new X-axis limits for a better plot representation.
        xlim = kwargs.get('xlim', (utils.round_by_mo(max(std_lims[0], all_data.min()), abs_method='floor'), 
                                utils.round_by_mo(min(std_lims[1], all_data.max()), abs_method='ceil')))
        plt.xlim(xlim)
        plt_kwargs.update(dict(range=xlim))

        # Calculate number of bins to plot histogram 
        bins = kwargs.get('bins', utils.nbins(all_data[~outliers], bins_rule)['n'])

        om = max(utils.order_of_magnitude(all_data.min()), utils.order_of_magnitude(all_data.max()))
        om = '{:.3e}' if om>=5 else '{:.3f}'
        
        text = df_to_latex(pd.DataFrame(data=df_input[features]).describe(percentiles=[0.05, 0.25, 0.5, 0.75, 0.95]).applymap(om.format))

    else:
        # Calculate number of bins to plot histogram
        bins = len(df_input[features[0]].cat.categories.values)

        text = df_to_latex(pd.DataFrame(data=df_input[features]).describe())

    # Print statistical summary before the histogram
    if kwargs.get('describe', True): 
        t = axes.text(1.04, 0.5, text, size=10, ha='left', va='center', c='black', transform=axes.transAxes, 
                  bbox=dict(facecolor='white', edgecolor='black', alpha=0.75, pad=5))

    # Plot histogram
    if 'hist_kwargs' in list(kwargs.keys()):
        for f in range(len(features), 0, -1):
            plt_kwargs.update(kwargs['hist_kwargs'][f-1])
            plt.hist(data[f-1], bins=bins, **plt_kwargs)
    else:
        plt.hist(data, bins=bins, **plt_kwargs)
    
    # Compute new Y-axis limits for a better plot representation.
    ylim = kwargs.get('ylim',(axes.get_ylim()[0], utils.round_by_mo(axes.get_ylim()[1], abs_method='ceil')))
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
    plt.show()
    
    return None

#%%
def df_to_latex(df: pd.DataFrame, column_format:str='c') -> str:
    """Convert pandas DataFrame to latex table.

    Args:
        df (pd.DataFrame): DataFrame to convert to LaTeX format.
        column_format (str, optional): Columns alignment (left 'l', center 'c', or right 'r'). Defaults to 'c'.

    Returns:
        str: DataFrame in string format.
    """

    column_format = 'c'*(len(df.columns)+1) if column_format=='c' else column_format

    new_column_names = dict(zip(df.columns, ["\textbf{" + c + "}" for c in df.columns]))
    
    df.rename(new_column_names, axis='columns', inplace=True)
    
    table = df.style.to_latex(column_format=column_format)
    table = table.replace('\n', '').encode('unicode-escape').decode()\
            .replace('%', '\\%').replace('\\\\', '\\') \
            .replace('\\\\count', '\\\\\\hline count')
        
    return table
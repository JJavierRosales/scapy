# Libraries used for type hinting
from __future__ import annotations
from typing import Union

import torch
import pandas as pd
import numpy as np
import random
import math
import warnings
import time
import scipy.stats as st
from typing import Union

from sklearn.linear_model import LinearRegression
from sklearn.metrics import roc_auc_score
from sklearn.neighbors import KernelDensity

import statsmodels.api as smapi
import statsmodels as sm

import datetime
import matplotlib.pyplot as plt

# Import json library and create function to format dictionaries.
import json

# Get current working directory path for the tool parent folder and print it.
from pathlib import Path
import os
parent_folder = 'scapy'
cwd = str(Path(os.getcwd()[:os.getcwd().index(parent_folder)+len(parent_folder)]))


#%% FUNCTION: format_json
def format_json(input:dict) -> str:
    """Convert dictionary to a prettify JSON string.

    Args:
        input (dict): Dictionary to convert.

    Returns:
        str: String in JSON format.
    """
    return json.dumps(input, indent=4)
#%% CLASS: FitScipyDistribution
class FitScipyDistribution:
    """SciPy fitted distribution instanciator.
    """
    def __init__(self, data:np.ndarray, distribution:st) -> None:
        """Initialise distribution parameters and information.

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
                                        bins=nbins(self.data,'fd')['n'], 
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
            outliers_boundaries(data = self.data.flatten(), 
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

    pb_dist = ProgressBar(iterations = scipy_distributions.values(), 
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

#%% FUNCTION: bws_statsmodels
def bws_statsmodels(data: np.ndarray, 
                    methods: Union[list, str]=['normal_reference', 'cv_ml'], 
                    **sm_kwargs) -> dict:
    """Statsmodels multivariate kernel density estimator.

    Args:
        data (np.ndarray): Data distribution to approach.
        methods (Union[list, str], optional): Estimator method. Defaults to 
        ['normal_reference', 'cv_ml'].

    Returns:
        dict: Dictionary with estimated bandwidths per method.
    """

    # Bandwidth selection methods:
    # normal_reference: normal reference rule of thumb (default)
    # cv_ml: cross validation maximum likelihood
    # cv_ls: cross validation least squares
    methods = [methods] if isinstance(methods, str) else methods

    # Set the default values for (efficient) bandwidth estimation.
    settings = sm.nonparametric.kernel_density.EstimatorSettings(
                efficient = sm_kwargs.get('efficient', True), 
                n_sub = sm_kwargs.get('n_sub', len(data)//10))

    # Ignore warnings from data that can't be fit
    warnings.filterwarnings("ignore")

    bandwidths = {}
    for method in methods:
        # Compute estimated probability density using method "method"
        epd = smapi.nonparametric.KDEMultivariate(data = data, 
                                                  var_type = 'c', 
                                                  bw = method, 
                                                  defaults = settings)

        # Get the bandwidth parameters.
        bandwidths[method] = epd.bw[0]

    # Re-enable warnings
    warnings.filterwarnings('default')
        
    return bandwidths
#%% FUNCTION: bws_msecv
def bws_msecv(data: np.ndarray, kernel:str='gaussian', bins_rule:str='fd', 
              conv_accuracy:float = 1e-5, n_batches_min:int = 2, 
              n_batches_max:int = 10, underfitting_factor:float=2, 
              print_log:bool = False) -> tuple:
    """Computes optimal bandwidth minimizing MSE actual vs estimated density 
    through cross-validation.

    Args:
        data (np.ndarray): Array containing all input data.
        kernel (str, optional): Kernel to use. Defaults to 'gaussian'.
        bins (str, optional): Rule to use to compute the bin size. It can be one 
        of the following options:
            - Sturge ('sturge')
            - Scott ('scott')
            - Rice ('rice')
            - Freedman-Diaconis ('fd') - Default
        conv_accuracy (float, optional): Convergence accuracy. Defaults to 1e-5.
        n_batches_min (int, optional): Minimum number of batches for the 
        cross-validation. Defaults to 2.
        n_batches_max (int, optional): Maximum number of batches for the 
        cross-validation. Defaults to 10.
        underfitting_factor (float, optional): Factor to prevent bandwidth 
        overfitting. Defaults to 2.
        print_log (bool, optional): Print computational log. Defaults to True.

    Returns:
        tuple: Bandwidth and estimated R2 score.
    """

    # Define local function to compute KDE
    def kde(x:np.ndarray, x_grid:np.ndarray, bandwidth:float, 
            kernel:str='gaussian', **kwargs) -> np.ndarray:
        """Get probability density function using Scikit-learn KernelDensity 
        class. 

        Args:
            x (np.ndarray): Array of data to describe.
            x_grid (np.ndarray): Grid of points upon which the estimation will 
            be done.
            bandwidth (float): Kernel bandwidth.
            kernel (str, optional): Kernel to use. Defaults to gaussian.

        Returns:
            pdf (np.ndarray): Estimated probability densities over x_grid.
        """
        
        # Instanciate KernelDensity class from Scikit-learn (kernel defaults to 
        # 'gaussian')
        kde = KernelDensity(bandwidth = bandwidth, kernel = kernel, **kwargs)
        
        # Fit Kernel
        kde.fit(np.array(x).reshape(-1,1))
        
        # Get log-likelihood of the samples
        log_pdf = kde.score_samples(x_grid[:, np.newaxis])
        epd     = np.exp(log_pdf)
        
        return epd
    
    # Set the seed for reproducibility
    np.random.seed(42)

    # Exclude NaN and non-finite numbers from data
    data = data[np.isfinite(data)]

    # Get bins
    bins = nbins(data, bins_rule)

    # Create an array with the number of batches to process per iteration 
    batches_list = np.arange(start = n_batches_min, 
                             stop = n_batches_max + 1, 
                             step = 1, 
                             dtype = np.int32)
    
    # Initialize arry to store best bandwidth per group of batches
    best_bandwidths = np.zeros(len(batches_list))

    for i, batches in enumerate(batches_list):
        
        # Create an array with random values between 0 and the total number of 
        # batches of the size of the entire array of data (following a uniform 
        # distribution).
        batches_categories = np.random.randint(low = 0, 
                                               high = batches, 
                                               size = len(data))

        # Initialize array to store actual probabilty densities (apds) from 
        # every batch
        apds = np.zeros((bins['n'], batches))

        # Iterate over all batches categories (0 to number of batches)
        for c in range(batches):
            
            # Get sample of data corresponding to a specific batch category "c"
            sample = data[batches_categories==c]
            
            # Get actual probability density from the sample using number of 
            # bins obtained from the entire array of data and store it in the 
            # apds array.
            apds[:,c], bin_edges = np.histogram(sample, bins = bins['n'], 
                                                density=True)

        # Get array of bin centers to which the actual probability density are 
        # associated
        bin_centers = bin_edges[:-1] + bins['width']/2

        # Compute average of all actual probability densities from all batches
        avg_apd = np.mean(apds, axis=1, dtype=np.float64)

        # Initialize bandwidth array to evaluate estimated probability density 
        # from kernel
        bandwidths, step = np.linspace(start = bins['width']/10, 
                                       stop = bins['width'], 
                                       num = 50, 
                                       retstep = True)

        # Initialize best_bw
        best_bw, bw = (0.0, np.inf)
        while True:

            # Initialize mean squared errors array associated to every bandwidth
            mse = np.zeros(len(bandwidths))

            # Iterate over all the bandwidths to compute the MSE from the actual 
            # vs estimated probability densities
            for b, bandwidth in enumerate(bandwidths):

                # Get estimated probability distribution using the bandwidth "b"
                epd = kde(data, bin_centers, bandwidth=bandwidth)
                
                # Compute MSE from actual vs estimated probability densities
                mse[b] = ((epd - avg_apd)**2).mean()

            # Get index of minimum MSE. If there are more than 1 local minimum, 
            # choose the one with the highest index (higher bandwidth) to 
            # prevent overfitting.
            argmin_mse = np.argmin(mse) if len(arglocmin(mse))==1 \
                                        else arglocmin(mse)[-1]

            # Get bandwidth that minimizes MSE and check accuracy vs best_bw
            bw = bandwidths[argmin_mse]

            # Check if convergence accuracy is achieved to stop iterations
            if abs(1 - best_bw/bw) <= conv_accuracy: break

            # Update best_bw and bandwidths array to increase final bandwidth 
            # accuracy
            best_bw = bw
            bandwidths, step = np.linspace(start = bw - step, 
                                           stop = bw + step, 
                                           num = len(bandwidths) + 10, 
                                           retstep = True)

        # Add best bandwidth from this group of batches to final array
        best_bandwidths[i] = best_bw 

        if print_log: 
            print(f'Batches = {batches:2d} '
                  f'({len(data)//batches:5d} d.p. per batch)  '
                  f'Best bw = {best_bw:.5f} '
                  f'Conv. accuracy = {abs(1 - best_bw/bw):.4e}  '
                  f'MSE(apd, epd) = {mse.min():.4e}')

    # Round-up best bandwidth from all groups of batches using one order of 
    # magnitude less
    # scale = 10**utils.om(best_bandwidths.mean())
    # best_bw = (math.ceil(best_bandwidths.mean()/scale)*scale)
    best_bw = round_by_om(best_bandwidths.mean()*underfitting_factor)
    
    
    # Compute final estimated probability density using the best bandwidth and 
    # compare it with actual probability density using MSE
    epd = kde(data, bin_centers, kernel='gaussian', bandwidth=best_bw)
    apd, bin_edges = np.histogram(data, bins = bins['n'], density=True)
    estimated_mse = ((epd - apd)**2).mean()
    
    if print_log: print(f'\nFinal Optimal bandwidth = {best_bw}\t '
                        f'MSE(apd, epd) = {estimated_mse}')
    
    return best_bw, r2_score(apd,epd)

#%% FUNCTION: plot_histogram
def plot_histogram(data:np.ndarray, features:list, figsize:tuple = (6, 3), 
                   return_ax:bool=False, bins:int=40, show_stats:bool = False, 
                   show:bool=True, **kwargs):
    """Plot custom histogram for dataframe features. 

    Args:
        data (np.ndarray): Array of data to plot.
        features (list): Features names.
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
            xlim = (round_by_om(np.percentile(data, xlim_percentiles[0])), 
                    round_by_om(np.percentile(data, xlim_percentiles[1])))

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
        values = [[number2latex(number) for number in row] \
                for row in df_stats.to_numpy()]
        
        # Get table in LaTeX format
        columns = [r'\texttt{' + feature + '}' for feature in features]
        text = df2latex(pd.DataFrame(data = values, 
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

#%% FUNCTION: sse
def sse(y_true:np.array, y_pred:np.array) -> float:
    """Compute Sum of Squared Errors (SSE).

    Args:
        y_true (np.array): Array with true values.
        y_pred (np.array): Array with predicted values.

    Returns:
        np.array: Sum of Squared Errors (SSE).
    """
    
    return sum((y_pred - y_true) * (y_pred - y_true))

#%% FUNCTION: r2_score
def r2_score(y_true:np.array, y_pred:np.array) -> float:
    """Compue coefficient of determination.

    Args:
        y_true (np.array): Array with true values.
        y_pred (np.array): Array with predicted values.

    Returns:
        float: Coefficient of determination.
    """
    y_mean = [np.mean(y_true) for y in y_true]
    sse_regr = sse(y_pred, y_true)
    sse_x_mean = sse(y_mean, y_true)

    return 1 - (sse_regr/sse_x_mean)

#%% FUNCTION: binary_auc_roc
def binary_auc_roc(outputs:torch.Tensor, targets:torch.Tensor) -> float:
    """Compute area under the curve ROC.

    Args:
        outputs (torch.Tensor): Predicted values.
        targets (torch.Tensor): Target values

    Returns:
        float: Area under the curve ROC.
    """

    y_pred = [int(torch.argmax(t)) for t in outputs]
    y_true = [int(torch.argmax(t)) for t in targets]

    try:
        aucroc = roc_auc_score(y_true, y_pred)
    except:
        aucroc = np.nan

    return aucroc
#%% FUNCTION: binary_confusion_matrix
def binary_confusion_matrix(outputs:torch.Tensor, targets:torch.Tensor, 
                          return_metrics:bool = True) -> dict:
    """Compute confusion matrix from two classification vectors.

    Args:
        outputs (torch.Tensor): Predicted categories.
        targets (torch.Tensor): True categories.
        return_metrics (bool, optional): Return derived metrics Accuracy, 
        Precision, Recall and F1-Score. Defaults to True.

    Returns:
        dict: Confusion matrix results (tp, fp, fn, tn) and classification 
        metrics when applicable.
    """
    
    y_true = targets.detach().numpy()
    y_pred = outputs.detach().numpy()

    y_true = np.asarray([int(np.argmax(i)) for i in y_true])
    y_pred = np.asarray([int(np.argmax(i)) for i in y_pred])

    cfm = {}
    cfm['tp'] = np.sum((y_true==1)*(y_pred==1))
    cfm['tn'] = np.sum((y_true==0)*(y_pred==0))
    cfm['fp'] = np.sum((y_true==0)*(y_pred==1))
    cfm['fn'] = np.sum((y_true==1)*(y_pred==0))


    if return_metrics:
        cfm['accuracy'] = (cfm['tp'] + cfm['tn'])/targets.size(0)*100
        cfm['precision'] = cfm['tp']/(cfm['tp'] + cfm['fp'])*100
        cfm['recall'] = cfm['tp']/(cfm['tp'] + cfm['fn'])*100
        cfm['f1'] = 2*cfm['precision'] * cfm['recall'] / \
                    (cfm['precision'] + cfm['recall'])

    return cfm
#%% FUNCTION: mape
def mape(output:torch.Tensor, target:torch.Tensor, 
                epsilon:float=1e-8) -> torch.Tensor:
    """Mean Absolute Percentage Error (MAPE)

    Args:
        output (torch.Tensor): Predicted values.
        target (torch.Tensor): True values.
        epsilon (float, optional): Infinitesimal delta to skip 
        singularities caused by target=0. Defaults to 1e-8.

    Returns:
        torch.Tensor: _description_
    """
    
    return float(torch.mean(torch.abs((target - output) / \
                                torch.add(target, epsilon))*100))
#%% FUNCTION: pocid
def pocid(output:torch.Tensor, target:torch.Tensor) -> float:
    """Prediction Of Change In Direction (POCID)

    Args:
        output (torch.Tensor): Predicted value.
        target (torch.Tensor): True value.

    Returns:
        float: Percentage of predictions whose direction matches those of the 
        true values.
    """

    curr_output, prev_output = output[:, 1:], output[:, :-1]
    curr_target, prev_target = target[:, 1:], target[:, :-1]

    d = 0
    for i, cout in enumerate(curr_output):
        pout = prev_output[i]
        ctar, ptar = curr_target[i], prev_target[i]

        # Get boolean tensor
        pocid = (cout-pout)*(ctar-ptar)>0

        # Count number of conditions matching True
        d += int(torch.sum(pocid))

    return float(d/torch.numel(curr_output)*100)
#%% FUNCTION: get_lr
def get_lr(optimizer:torch.nn) -> float:
    """Get learning rate of optimizer.

    Args:
        optimizer (torch.nn): Optimizer object.

    Returns:
        float: Learning rate.
    """
    for param_group in optimizer.param_groups:
        return param_group['lr']
#%% FUNCTION: seed
def seed(seed:int = None) -> None:
    """Set the seed in numpy and pytorch for reproducibility.

    Args:
        seed (int, optional): Seed value. Defaults to None.
    """
    if seed is None:
        seed = int((time.time()*1e6) % 1e8)
    global _random_seed
    _random_seed = seed
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available(): torch.cuda.manual_seed(seed)

#%% FUNCTION: docstring
def docstring(item, internal_attr:bool=False, builtin_attr:bool=False) -> None:
    """Print DocString from a specific Module, Class, or Function.

    Args:
        item (_type_): Module, Class, or Function to print documentation from. 
        internal_attr (bool, optional): Flag to include internal attributes. 
        Defaults to False.
        builtin_attr (bool, optional): Flag to include built-in attributes. 
        Defaults to False.
    """

    # Initialize methods list.
    methods = []

    # Iterate over all methods available in the item.
    for method in dir(item):
        if (method.startswith('_') and internal_attr) or \
           (method.startswith('__') and builtin_attr) or \
            not method.startswith('_'):
            methods.append(method)

    # Sort methods by alphabetic order
    methods = sorted(set(methods))

    for method in methods:  
        print('Method: {}\n\n{}\n{}' \
            .format(method,getattr(item, method).__doc__, "_"*80))
#%% FUNCTION: plt_matrix
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
        while num_subplots > 0:
            rows += 1
            num_subplots -= cols
            
        return rows, cols  
#%% FUNCTION: from_date_str_to_days
def from_date_str_to_days(date:datetime.datetime, 
                          date0:str='2020-05-22T21:41:31.975', 
                          date_format:str='%Y-%m-%dT%H:%M:%S.%f') -> float:
    """Convert date in string format to date.

    Args:
        date (datetime): Final date.
        date0 (str, optional): Initial date. Defaults to 
        '2020-05-22T21:41:31.975'.
        date_format (str, optional): Format of initial and final date. Defaults 
        to '%Y-%m-%dT%H:%M:%S.%f'.

    Returns:
        float: Number of days.
    """
    
    # Convert date in string format as datetime type
    date = datetime.datetime.strptime(date, date_format)
    date0 = datetime.datetime.strptime(date0, date_format)

    # Get time between final and initial dates.
    dd = date-date0
    days = dd.days
    days_fraction = (dd.seconds + dd.microseconds/1e6) / (60*60*24)

    return days + days_fraction
#%% FUNCTION: doy_2_date
def doy_2_date(value:str, doy:str, year:int, idx:int) -> str:
    """Converts Day of Year (DOY) date format to date format.

    Args:
        value (str): Original date time string with day of year format 
        "YYYY-DDDTHH:MM:SS.ff"
        doy (str): The day of year in the DOY format.
        year (int): The year.
        idx (int): Index of the start of the original "value" string at which 
        characters 'DDD' are found.

    Returns:
        str: Transformed date in traditional date format.
    """
    # Calculate datetime format
    date_num = datetime.datetime(int(year), 1, 1) + \
               datetime.timedelta(int(doy) - 1)

    # Split datetime object into a date list
    date_vec = [date_num.year, date_num.month, date_num.day, 
                date_num.hour, date_num.minute]
    
    # Extract final date string. Use zfill() to pad year, month and day fields 
    # with zeroes if not filling up sufficient spaces. 
    value = str(date_vec[0]).zfill(4) +'-' + \
            str(date_vec[1]).zfill(2) + '-' + \
            str(date_vec[2]).zfill(2) + 'T' + \
            value[idx+4:-1] 
    
    return value
#%% FUNCTION: get_ccsds_time_format
def get_ccsds_time_format(time_string:str) -> str:
    """Get the format of the datetime string in the CDM. The CCSDS time format 
    is required to be of the general form yyyy-[mm-dd|ddd]THH:MM:SS[.F*][Z]. The
    following considerations are taken into account:

    (1) The date and time fields are separated by a "T".
    (2) The date field has a four digit year followed by either a two digit 
        month and two digit day, or a three digit day-of-year.  
    (3) The year, month, day, and day-of-year fields are separated by a dash.
    (4) The hours, minutes and seconds fields are each two digits separated 
        by colons.
    (5) The fraction of seconds is optional and can have any number of
        digits.
    (6) If a fraction of seconds is provided, it is separated from the two
        digit seconds by a period.
    (7) The time string can end with an optional "Z" time zone indicator

    Args:
        time_string (str): Original time string stored in CDM. It must be of the 
        form yyyy-[mm-dd|ddd]THH:MM:SS[.F*][Z], otherwise it is invalid and a 
        RuntimeError is raised.

    Raises:
        RuntimeError: 0 or more than 1 T separator.
        RuntimeError: Datetime format not recognised.
        RuntimeError: Second decimals not identified.

    Returns:
        str: Outputs the format of the time string. 
    """

    # Count number of T separators.
    numT = time_string.count('T')

    # Raise error if found 0 or more than one separator.
    if numT == -1 or numT > 1:
        # Raise error if time_string does not contain a single 'T'.
        raise RuntimeError(f"Invalid CCSDS time string: {time_string}."
                           f" {numT} 'T' separator(s) found between date and "
                           f"time portions of the string.")
    
    # Get the position of the separator T (separates date from time).
    idx_T = time_string.find('T')

    if idx_T ==10:
        time_format = "yyyy-mm-ddTHH:MM:SS"
    elif idx_T ==8:
        time_format = "yyyy-DDDTHH:MM:SS"
    else: 
        raise RuntimeError(f"Invalid CCSDS time string: {time_string}. "
                           f"Date format not one of yyyy-mm-dd or yyyy-DDD.")
    
    # Check if 'Z' time zone indicator appended to the string
    z_opt = (time_string[-1]=='Z') 

    # Get location of the fraction of seconds decimal separator
    num_decimal = time_string.count('.')
    if num_decimal > 1:
        raise RuntimeError(f"Invalid CCSDS time string: {time_string}. "
                           f"More than one fraction of seconds decimal "
                           f"separator ('.') found.")
    
    # Get the position of the dot in the seconds decimals
    idx_decimal = time_string.find('.')

    # If datetime has seconds decimals, get the number of decimals.
    n_decimals = 0
    if num_decimal != 0:
        n_decimals = len(time_string) - 1 - idx_decimal - (1 if z_opt else 0)

    # Get second decimals format if CDM datetime has decimals
    frac_str = ('.' + ('F'*n_decimals)) if n_decimals > 0 else ""

    # Add Z time zone indicator if present.
    frac_str = frac_str + ('Z' if z_opt else '')

    # Join time_format, seconds fraction (if any) and time zone (if any).
    time_format = time_format + frac_str

    return time_format

#%% FUNCTION: has_nan_or_inf
def has_nan_or_inf(value: Union[float, torch.TensorFloat]) -> bool:
    """Check if value(s) contain any infinite or Not a Number (NaN).

    Args:
        value (Union[float, torch.TensorFloat]): Value(s) to check.

    Returns:
        bool: True if value(s) has infinite or NaN value(s), False otherwise.
    """

    if torch.is_tensor(value):
        value = torch.sum(value)
        isnan = int(torch.isnan(value)) > 0
        isinf = int(torch.isinf(value)) > 0
        return isnan or isinf
    else:
        value = float(value)
        return math.isnan(value) or math.isinf(value)
#%% FUNCTION: tile_rows_cols
def tile_rows_cols(num_items:int) -> tuple:
    """Get number of rows and columns a series of items can be organised to 
    follow a square shaped frame.

    Args:
        num_items (int): Number of items to tile.

    Returns:
        tuple: tuple with the number of rows and columns the items can be 
        organised in a squared shape frame.
    """

    if num_items < 5:
        return 1, num_items
    else:
        cols = math.ceil(math.sqrt(num_items))
        rows = 0
        while num_items > 0:
            rows += 1
            num_items -= cols
        return rows, cols
#%% FUNCTION: add_days_to_date_str
def add_days_to_date_str(date0:datetime, days:float) -> str:
    """Add/Substract natural date from initial date.

    Args:
        date0 (datetime): Initial date.
        days (float): Natural days to add/substract.

    Returns:
        str: Datetime as a string.
    """
    date0 = datetime.datetime.strptime(date0, '%Y-%m-%dT%H:%M:%S.%f')
    date = date0 + datetime.timedelta(days=days)

    return date.strftime('%Y-%m-%dT%H:%M:%S.%f')
#%% FUNCTION: transform_date_str
def transform_date_str(date_string:str, from_format:str, to_format:str) -> str:
    """Convert date in string format from certain format to another format.

    Args:
        date_string (str): Date as string type to change the format.
        from_format (str): Format in which the date is provided.
        to_format (str): String datetime format of conversion

    Returns:
        str: Date in the new string datetime format.
    """
    date = datetime.datetime.strptime(date_string, from_format)

    return date.strftime(to_format)
#%% FUNCTION: is_date
def is_date(date_string:str, date_format:str) -> bool:
    """Check a date in string format is an actual date.

    Args:
        date_string (str): Date in string format to check.
        date_format (str): Format string in which the date is provided.

    Returns:
        bool: True if it is an actual date, False otherwise.
    """
    try:
        datetime.datetime.strptime(date_string, date_format)
        return True
    except:
        return False
#%% FUNCTION: is_number
def is_number(s:Union[float, str]) -> bool:
    """Check if value is a number or a string.

    Args:
        s (Union[float, str]): Value to check.

    Returns:
        bool: True if is a number, False otherwise.
    """
    try:
        float(s)
        return True
    except ValueError:
        return False
#%% FUNCTION: arglocmax
def arglocmax(a:np.ndarray) -> np.ndarray:
    """Find the indeces of the local maximums of an array.

    Args:
        a (np.ndarray): Numpy array to check.

    Returns:
        np.ndarray: Indeces of the local maximum.
    """

    condition = np.r_[True, a[1:] > a[:-1]] & np.r_[a[:-1] > a[1:], True]
    index_array = np.asarray([i for i, c in enumerate(condition) if c], 
                             dtype=np.int32)

    return index_array

#%% FUNCTION: arglocmin
def arglocmin(a:np.ndarray) -> np.ndarray:
    """Find the indeces of the local minimums of an array.

    Args:
        a (np.ndarray): Numpy array to check.

    Returns:
        np.ndarray: Indeces of the local maximum.
    """

    condition = np.r_[True, a[1:] < a[:-1]] & np.r_[a[:-1] < a[1:], True]
    index_array = np.asarray([i for i, c in enumerate(condition) if c], 
                             dtype=np.int32)

    return index_array

#%% FUNCTION: nbins
def nbins(data:np.ndarray, rule:str = 'fd') -> dict:
    """Calculate number of bins and bin width for histograms.

    Args:
        data (array_like): Array containing all data.
        rule (str): Rule to use to compute the bin size. It can be one of the 
        following options:
            - Sturge ('sturge')
            - Scott ('scott')
            - Rice ('rice')
            - Freedman-Diaconis ('fd') - Default

    Returns:
        dict: Dictionary containing with the number of bins 'n' and bin width 
        'width'.
    """

    # Check if rule passed by user is valid
    if not rule in ['sturge', 'scott', 'rice', 'fd']: return None

    # Get the number of items within the dataset
    isinteger = isinstance(data[0], np.integer)
    data = data.astype(float)
    n = len(data)
    
    # Compute the histogram bins size (range)
    bins_width = {'sturge': 1 + 3.322*np.log(n),    
                 'scott': 3.49*np.std(data)*n**(-1.0/3.0),                       
                 'rice': 2*n**(1.0/3.0),                         
                 'fd': 2*st.iqr(data)*n**(-1.0/3.0)}
    
    # Compute number of bins
    n_bins =  math.ceil((data.max() - data.min())/bins_width[rule])

    # Compute range of bins
    if isinteger:
        steps = int(math.ceil((data.max() - data.min())/n_bins))
        bins_range = range(int(data.min()), int(data.max()), steps)
    else:
        bins_range = np.linspace(data.min(),data.max()+bins_width[rule], n_bins)
    
    return {'n': n_bins, 'width': bins_width[rule], 'range': bins_range}
#%% FUNCTION: om
def om(value: float) -> int:
    """Get order of magnitude of value.

    Args:
        value (float): Value to get the order of magnitude from.

    Returns:
        int: Order of magnitude.
    """
    if not (isinstance(value, float) or \
            isinstance(value, np.integer) or \
            isinstance(value, int)): 
        return np.nan
    if value==0 or np.isnan(value): return 0
    if abs(value)==np.inf: return np.inf
    
    return int(math.floor(math.log(abs(value), 10)))
#%% FUNCTION: round_by_om
def round_by_om(value:float, abs_method:str='ceil', **kwargs) -> float:
    """Round up/down float by specifying rounding of magnitude.

    Args:
        value (float): Value to round up/down.
        abs_method (str, optional): Method to round considering the absolute 
        value (up or down). Defaults to 'ceil'.

    Returns:
        float: Value rounded.
    """


    # Return None if method is not valid
    if not abs_method in ['ceil', 'floor']: return None
    
    # Return 0 if value is 0
    if value==0: return 0

    # Compute order of magnitude
    initial_om = 10**kwargs.get('rounding_om', om(value))
    
    # Initialize dictionary with round methods
    methods = {'ceil':math.ceil, 'floor':math.floor}

    # Invert method if value is negative
    if value<0: abs_method = 'floor' if abs_method=='ceil' else 'ceil'

    return methods[abs_method](value/initial_om)*initial_om

#%% FUNCTION: df2latex
def df2latex(df: pd.DataFrame, column_format:str='c', 
             include_header:bool=False) -> str:
    """Convert pandas DataFrame to latex table.

    Args:
        df (pd.DataFrame): DataFrame to convert to LaTeX format.
        column_format (str, optional): Columns alignment (left 'l', center 'c', 
        or right 'r'). Defaults to 'c'.
        include_header (bool, optional): Include header of DataFrame if True.
        Defaults to False

    Returns:
        str: DataFrame in string format.
    """

    column_format = 'c'*(len(df.columns)+1) if column_format=='c' \
        else column_format


    new_column_names = dict(zip(df.columns, 
                            ["\textbf{" + c + "}" for c in df.columns]))
    
    df.rename(new_column_names, axis='columns', inplace=True)
    
    table = df.style.to_latex(column_format=column_format)
    table = table.replace('\n', '').encode('unicode-escape').decode()\
            .replace('%', '\\%').replace('\\\\', '\\')

    if not include_header:
        header_index = (table.index('{' + column_format + '}') + len(column_format) + 2,
                        table.index('\\\\') + 4)
        table = table[:header_index[0]+1] + table[header_index[1]-2:]
    else:
        table = table.replace('\\\\', '\\\\\\hline ', 1)

        
    return table

#%% FUNCTION: number2latex
def number2latex(value:Union[int, float]) -> str:
    """Format a given value depending on its order of magnitude.

    Args:
        value (Union[int, float]): Value to format as a string.

    Returns:
        str: Return value with specific format.
    """

    output = f'{value}'

    # Check input is a number
    if not (isinstance(value, int) or \
            isinstance(value, float)): return output
    if not np.isfinite(value): return output

    if (value%1==0 or isinstance(value, np.integer)) and om(value)<=5:
        # If integer, show no decimals
        output = '{:d}'.format(int(value))
    elif (om(value)>-2 and om(value)<5):
        # If absolute value is in the range (0.01, 10000) show 3 decimals
        output = '{:.3f}'.format(value)
    elif om(value)>5 or om(value)<=-2:
        # If absolute value is in the range (0, 0.01] or [10000, inf) show 
        # scientific notation with 3 decimals
        # output = r'$' + '{:.3e}'.format(value).replace('e',r'\cdot10^{')
        output = r'$' + '{:.3e}'.format(value).replace('e',r'\mathrm{e}{')

        output = output.replace('{+0','{').replace('{-0','{-') + r'}$'

    return output
#%% FUNCTION: outliers_boundaries
def outliers_boundaries(data: np.ndarray, threshold: Union[tuple, float]=1.5, 
        positive_only:bool=False) -> Union[tuple, np.ndarray, np.ndarray]:
    """Compute limits of standard data within a given data distribution.

    Args:
        data (np.ndarray): Data to get the outliers boundaries from.
        threshold (float, optional): Proportion of IQR to take into account. 
        Defaults to 1.5.
        positive_only (bool, optional): Force negative lower boundary to be 0 if
         data can only be positive. Defaults to False.

    Returns:
        tuple: Range of values between which satandard data is comprised.
        np.ndarray: Array contaning standard data (non outliers).
        np.ndarray: Array containing outliers.
    """
    
    Q1 = np.quantile(data,0.25)
    Q3 = np.quantile(data,0.75)
    IQR = st.iqr(data)

    positive_only = sum(data<0)==0 if positive_only else positive_only

    threshold = (threshold, threshold) if isinstance(threshold, float) \
        else threshold
    
    if positive_only:
        std_lims = (max(0,(Q1-IQR*threshold[0])), (Q3+IQR*threshold[1]))
    else:
        std_lims = ((Q1-IQR*threshold[0]), (Q3+IQR*threshold[1]))

    # Get outliers and standard data
    filter_outliers = (data<std_lims[0]) | (data>std_lims[1])
    outliers = data[filter_outliers]
    std_data = data[~filter_outliers]

    return std_lims, std_data, outliers
#%% FUNCTION: compute_vif
def compute_vif(df_input: pd.DataFrame) -> pd.DataFrame:
    """Compute Variance Inflation Factor to evaluate multicolinearity.

    Args:
        df_input (pd.DataFrame): Input dataframe to compute VIF.

    Returns:
        pd.DataFrame: DataFrame with all the features as keys and VIF scores as 
        values.
    """
    
    # Create deep copy of input DataFrame
    data = df_input.copy(deep=True).dropna()
    
    # Get the features and model object
    features = [feature for feature in data.columns \
                if not data[feature].dtype in ['category', 'str']]

    model = LinearRegression()
    
    # Create empty list to store R2 scores
    r2_scores = []
    
    # Iterate through all features to evaluate its VIF
    warnings.filterwarnings("ignore") # Disable warnings if division by 0
    
    for y_feature in features:
        
        x_features = [f for f in features if not f==y_feature]
        
        x = data[x_features]
        y = data[y_feature]
        
        # Fit the model and calculate VIF
        model.fit(x, y)
        
        r2_scores.append(1/(1-model.score(x, y)))
    
    warnings.filterwarnings("default") # Restore warnings

    # Create dataframe with the results of the VIF computation
    result = pd.DataFrame(index=features, columns=['VIF'], data=r2_scores)
        
    return result
#%% FUNCTION: vif_selection
def vif_selection(df_input:pd.DataFrame, maxvif:float=5.0) -> dict:
    """Variable selection using Variance Inflation Factor (VIF) threshold.

    Args:
        df_input (pd.DataFrame): Input dataframe.
        maxvif (float, optional): Maximum VIF score. Defaults to 0.8.

    Returns:
        dict: Dictionary containing correlated and independent features with 
        their VIF scores.
    """
    
    # Create a deep copy of the input DataFrame
    df = df_input.copy(deep=True).dropna()

    # Compute all VIF values and all features with the maximum VIF
    vif = compute_vif(df)
    vif_values = np.sort(np.unique(np.asarray(vif.values).flatten()))[::-1]
    maxvif_features = list(vif[vif['VIF']==vif_values[0]].index.values)

    correlated = {}
    while vif_values[0] > maxvif:

        # Initialize tuple to evaluate which feature (among those with max VIF) 
        # that should be removed, based on the maximum VIF, number of maximum 
        # VIFs and second maximum VIF after feature removal.
        collinear_feature = {'feature':maxvif_features[0],
                             'maxvif':np.inf,
                             'n_features_maxvif':len(df.columns),
                             '2nd_maxvif':np.inf}

        # Iterate over all features with maximum VIF
        for feature in maxvif_features:

            vif = compute_vif(df[[f for f in df.columns if f!= feature]])
            vif_values = np.sort(np.unique(np.asarray(vif.values).flatten()))[::-1]
            n_features_maxvif = np.sum(vif.values==vif_values[0])

            # Check if VIF analysis is producing lower values when removing the 
            # feature
            if (collinear_feature['maxvif'] > vif_values[0]) or \
               ((collinear_feature['maxvif'] == vif_values[0]) and \
               (collinear_feature['n_features_maxvif'] > n_features_maxvif)) or \
               ((collinear_feature['maxvif'] == vif_values[0]) and \
               (collinear_feature['n_features_maxvif'] == n_features_maxvif) and \
               (collinear_feature['2nd_maxvif'] >= vif_values[1])):
                
                collinear_feature['feature'] = feature
                collinear_feature['maxvif'] = vif_values[0]
                collinear_feature['n_features_maxvif'] = n_features_maxvif
                collinear_feature['2nd_maxvif'] = vif_values[1]


        # Store correlated values
        correlated[collinear_feature['feature']] = collinear_feature['maxvif']

        # Update dataframe to exclude correlated feature
        df.drop(columns=[collinear_feature['feature']], inplace=True)

        # Compute VIF values excluding the correlated feature
        vif = compute_vif(df)
        vif_values = np.sort(np.unique(np.asarray(vif.values).flatten()))[::-1]
        maxvif_features = list(vif[vif['VIF']==vif_values[0]].index.values)

    # Get information on the independent and correlated variables
    output = {'independent': vif.to_dict('index'),
              'correlated': correlated}
              
    return output
#%% FUNCTION: tabular_list
def tabular_list(input:list, n_cols:int = 3, **kwargs) -> str:
    """Format list as a tabular table in string format.

    Args:
        input_list (list): List of items to format.
        n_cols (int, optional): Number of columns to print. Defaults to 3.

    Returns:
        str: String with the list shown as a table.
    """

    input_list = input.copy() 

    # Get column separator string. Defaults to empty.
    col_sep = kwargs.get('col_sep', f'') 

    # Get horizontal separation between columns. Defaults to 4.
    hsep = kwargs.get('hsep', len(col_sep)+2)

    # Get text alignment. Defaults to < (left).
    alignment = kwargs.get('alignment', '<') 

    # Get maximum number of chars allowed per column to not exceed 80 chars 
    # width.
    chars_per_col = (80-(n_cols-1)*hsep)//n_cols
    if chars_per_col > 6:
        # Get maximum length allowed for the items. Defaults to 30.
        max_len = kwargs.get('max_len', chars_per_col)
    else:
        max_len = kwargs.get('max_len', min([len(item) for item in input_list]))

    # Get display order (0 for columns, 1 for rows).
    axis = kwargs.get('axis', 1) 

    # Shorten item of list if it exceeds max_len parameter
    for i, item in enumerate(input_list):
        if max_len > len(item) or max_len<4: continue
        out = len(item) - max_len
        n_char = (len(item)//2+len(item)%2) - (out//2+out%2)-1
        input_list[i] = item[:n_char] + '..' + item[-(max_len - n_char - 2):]

    output_list = []

    n_rows = len(input_list)//n_cols + (1 if len(input_list)%n_cols>0 else 0)
    for r in range(n_rows):
        row = []

        col_range = np.arange(r*n_cols, r*n_cols + n_cols, 1) if axis==0 else \
                    np.arange(r, r+n_cols*n_rows, n_rows)
        
        for c in col_range:
            if len(input_list)>=c+1:
                row.append(input_list[c])
            else:
                row.append('')

        output_list.append(row) 

    output = f''
    for r, row in enumerate(output_list):
        for c, item in enumerate(row):
            if item=='': continue
            output = output + col_sep + f'{item:{alignment}{max_len + hsep}}\t'
        output = output + f'\n'

    return output
#%% CLASS: ProgressBar
class ProgressBar():
    """Progress bar instanciator.
    """
    def __init__(self, iterations:Union[int,list],title:str = None, 
                 description:str = None):
        """Initialise progress bar instanciator.

        Args:
            iterations (Union[int,list]): Number of iterations to perform or 
            list of elements upon which the iterations are taking place.
            title (str, optional): Title of the progress bar log. Defaults to 
            None.
            description (str, optional): Additional comments. Defaults to None.
        """
        
        # Define list of sectors and range of values they apply to
        self._sectors_list = list(['', '\u258F', '\u258D', '\u258C', '\u258B', 
                                  '\u258A', '\u2589', '\u2588'])
        self._sectors_range = np.linspace(0,1,8)

        self.iterations = iterations

        if isinstance(self.iterations, np.integer):
            self._n_iterations = self.iterations
        else: 
            self._n_iterations = len(self.iterations)
        self.description = description if description is not None else ""

        self._title = ("\n" if title != None else "") + title
        self._header = None
        self._log = ""
        self._i = 0

        # Initialize start time of iteration
        self._it_start_time = time.time()
        self._start_time = time.time()

        # Initialize average duration of iteration
        self.avg_it_duration = 0.0

        # Initialize number of iterations per second
        self._its_per_second = 0.0


    def get_progress(self) -> tuple:
        """Compute progress (0 to 100) and subprogress (0 to 1).

        Returns:
            tuple: Values of the progress and subprogress.
        """

        self._progress = (self._i/self._n_iterations)
        progress = int(((self._i/self._n_iterations)*100//10))
        subprogress = (self._i/self._n_iterations*100 - progress*10)/10

        return (progress, subprogress)

    @staticmethod
    def format_time(duration:float) -> str:
        """Convert miliseconds to time format 'hh:mm:ss'.

        Args:
            duration (float): Number of miliseconds

        Returns:
            str: Time as a string.
        """
        m, s = divmod(duration, 60)
        h, m = divmod(m, 60)

        return f'{int(h):02d}h:{int(m):02d}m:{int(s):02d}s'

    
    def refresh(self, i:int, description:str = None, 
            nested_progress:bool = False) -> None:
        """Update progress bar information.

        Args:
            i (int): Iteration number.
            description (str, optional): Description of the iteration. Defaults 
            to None.
            nested_progress (bool, optional): Nested progress: used for nested 
            loops. Defaults to False.
        """

        # Update description if new description is given
        if description == None:
            self.description = self.description  
        else: 
            self.description = description

        # Get order of magnitude of the number of iterations and number of 
        # iterations per second to improve readability of the log message.
        om_iterations = om(self._n_iterations+1) + 1
        
        if i > self._i:

            if self._header is None:
                self._header = self._title + \
                    f"\n| {'Progress':<{19 + 2*om_iterations+1}}" + \
                    f" | {'Time':^11} | " + \
                    f"{'Iters/sec':^{max(9, om_iterations+3)}} | " + \
                    f"{'Comments':<}"
                print(self._header)

            self._i = i

            # Get duration of iteration, average of duration per iteration and 
            # iterations per second.
            self._it_duration = time.time() - self._it_start_time \
                               if self._i > 1 else 0.0
            self.avg_it_duration = (time.time() - self._start_time)/self._i

            if self.avg_it_duration > 0:
                self._its_per_second = 1.0/self.avg_it_duration
            else:
                self._its_per_second = self._n_iterations
            
            # Compute estimated time remaining and overall duration.
            self.ert = self.avg_it_duration * \
                (self._n_iterations - self._i) if self._i > 1 else 0.0
            self.edt = self.avg_it_duration*self._n_iterations \
                if self._i > 1 else 0.0

            # Update iteration start time
            self._it_start_time = time.time()

        # Calculate how many entire sectors and type of subsector to display 
        progress, subprogress = self.get_progress()
        sectors   = self._sectors_list[-1]*progress

        # Get number of 8th sections as subsectors
        idx_subsector = np.sum(self._sectors_range <= subprogress) - 1
        subsector = self._sectors_list[idx_subsector]

        # Check if it is the last iteration.
        last_iter = (i==self._n_iterations and not nested_progress)

        # Create log concatenating the description and defining the end of the 
        # print log depending on the number of iteration
        pb_progress = f'{self._progress*100:>3.0f}%'
        pb_bar = f'|{sectors}{subsector}{" "*(10-len(sectors)-len(subsector))}|'
        pb_counter = f'{i}/{self._n_iterations}'
        pb_iter = f'{self._its_per_second:.2f}'
        pb_time = f'{self.format_time(self.edt)}' if last_iter \
             else f'{self.format_time(self.ert)}'
        
        log = f"| {pb_progress:>4} {pb_bar:^12} " + \
              f"({pb_counter:>{2*om_iterations+1}})" + \
              f"| {pb_time:^11} " + \
              f"| {pb_iter:^{max(9, om_iterations+3)}} " + \
              f"| {self.description}"

        # Ensure next log has the same number of characters to so that no 
        # residuals from previous log are left in the screen.
        if len(self._log) > len(log):
            self._log = f'{log:<{len(self._log)}}'
        else:
            self._log = log

        # Determine end character depending on the number of iterations.
        end = '\n' if (i==self._n_iterations and not nested_progress) else '\r'
        
        # Print progress log
        print('{:<}'.format(self._log), end = end)
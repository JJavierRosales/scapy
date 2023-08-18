# Libraries used for type hinting
from __future__ import annotations
from typing import Union

from sklearn.neighbors import KernelDensity
import matplotlib.pyplot as plt
from . import utils 
import numpy as np
import pandas as pd
import scipy.stats as st
import math
import warnings
import os

from scalib import eda

import statsmodels.api as smapi
import statsmodels as sm

#%% FUNCTION: kde
# Define function to compute KDE
def kde(x:np.ndarray, x_grid:np.ndarray, bandwidth:float, **kwargs) -> np.ndarray:
    """Get probability density function using Scikit-learn KernelDensity class. 

    Args:
        x (np.ndarray): Array of data to describe.
        x_grid (np.ndarray): Grid of points upon which the estimation will be done.
        bandwidth (float): Kernel bandwidth.

    Returns:
        pdf (np.ndarray): Estimated probability densities over x_grid.
    """
    
    # Instanciate KernelDensity class from Scikit-learn (kernel defaults to 
    # 'gaussian')
    kde = KernelDensity(bandwidth=bandwidth, **kwargs)
    
    # Fit Kernel
    kde.fit(np.array(x).reshape(-1,1))
    
    # Get log-likelihood of the samples
    log_pdf = kde.score_samples(x_grid[:, np.newaxis])
    epd     = np.exp(log_pdf)
    
    return epd
#%% FUNCTION: plot_scipy_pdf
def plot_scipy_pdf(data: np.ndarray, stdist: st, figsize:tuple=(7,3),
                   return_ax:bool = False, filepath:str = None, 
                   **kwargs) -> Union[None, plt.Axes]:
    """Plot histogram and PDF based on a given distribution and its parameters.

    Args:
        data (np.ndarray): Actual data to plot in the histogram.
        stdist (st): SciPy distribution to plot.
        figsize (tuple, optional): Figure size. Defaults to (7, 3).
        return_ax (bool, optional): Return ax object.
        filepath (str, optional): Path of the folder where the figure is saved.

    Returns:
        None: None
    """

    # Display plot
    fig, ax = plt.subplots(figsize = figsize)

    # Describe PDF fitting parameters if user wants
    if kwargs.get('describe', False)==True:
        
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
        text = utils.df2latex(df=pd.DataFrame(index=index, 
                                              data=values, 
                                              columns=['']
                                              ), column_format='cc')


        t = ax.text(1.04, 0.5, text, size=10, transform=ax.transAxes,
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
        plt.savefig(filepath)

    if return_ax:
        return ax
#%% FUNCTION: plot_kde
def plot_kde(data:np.ndarray, bandwidths:np.ndarray, figsize:tuple=(6, 3), 
             filepath:str = None, return_ax:bool = False, 
             **kwargs) -> Union[None, plt.Axes]:
    """Plot barchart with actual and estimated probability density.

    Args:
        data (np.ndarray): Actual data from which the probability density is 
        computed.
        bandwidths (np.ndarray): Array of bandwidths to evaluate on the kernel.

    Returns:
        Union[None, plt.Axes]: Axes object if return_ax = True.
    """
    
    # Plot the histogram and pdf

    bins = kwargs.get('bins', utils.nbins(data, 'fd'))

    fig, ax = plt.subplots(figsize = figsize )
    n, bin_edges, patches = plt.hist(data, bins = bins['n'], 
                                     density=True, color='gray', 
                                     ec='white', label='Actual data')
    
    
    labels = kwargs.get('bw_labels', [])
    kernel = kwargs.get('kernel', 'gaussian')
    pdf_grid_size = kwargs.get('pdf_grid_size', 200)

    # Iterate over all bandwidths
    for b, bw in enumerate(bandwidths):
        
        # Fit Kernel Density Estimator
        model = KernelDensity(bandwidth = bw, kernel = kernel)
        data = data.reshape((len(data), 1))
        model.fit(data)

        # Sample probabilities for a range of outcomes
        values = np.asarray([v for v in np.linspace(start = min(data), 
                                                    stop = max(data), 
                                                    num = pdf_grid_size)])
        values = values.reshape((len(values), 1))
        probabilities = np.exp(model.score_samples(values))

        if len(labels) < b+1:
            labels.append(f'bw = {bw:.4f}' if utils.om(bw)>-4 \
                                           else f'bw = {bw:.3e}')

        ax.plot(values, probabilities, label = labels[b])

    std_lims, _, _ = utils.outliers_boundaries(data, 
                                               threshold = 1.5, 
                                               positive_only=np.sum(data<0)==0)

    # Set X and Y-axis limits
    xlim = (max(bin_edges[0]  - bins['width'], std_lims[0]), 
            min(bin_edges[-1] + bins['width'], std_lims[1]))
    
    ax.set_xlim(kwargs.get('xlim', xlim))

    # Set axis labels and plot title.
    ax.set_xlabel(kwargs.get('xlabel', 'Feature'))
    ax.set_ylabel(kwargs.get('ylabel', 'Probability density'))
    ax.set_title('Probability Density analysis', fontsize=10)

    ax.grid(True, linestyle="dashed", alpha=0.5)
    ax.legend(loc='best', fontsize=10)

    # Remove blank spaces around the plot to be more compact.
    plt.tight_layout()
    
    # Save plot only if filepath is provided.
    if filepath is not None:
        print('Plotting to file: {}'.format(filepath))
        plt.savefig(filepath)

    if return_ax:
        return ax

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
def bws_msecv(data: np.ndarray, bins:dict, conv_accuracy:float = 1e-5, 
              n_batches_min:int = 2, n_batches_max:int = 10, 
              print_log:bool = False) -> dict:
    """Computes optimal bandwidth minimizing MSE actual vs estimated density 
    through cross-validation.

    Args:
        data (np.ndarray): Array containing all input data.
        bins (dict): _description_
        conv_accuracy (float, optional): Convergence accuracy. Defaults to 1e-5.
        n_batches_min (int, optional): Minimum number of batches for the 
        cross-validation. Defaults to 2.
        n_batches_max (int, optional): Maximum number of batches for the 
        cross-validation. Defaults to 10.
        print_log (bool, optional): Print computational log. Defaults to True.

    Returns:
        dict: Dictionary with the bandwidth value that minimizes the MSE and the 
        estimated MSE.
    """
    
    # Set the seed for reproducibility
    np.random.seed(42)

    # Exclude NaN and non-finite numbers from data
    data = data[np.isfinite(data)]

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
            argmin_mse = np.argmin(mse) if len(utils.arglocmin(mse))==1 \
                                        else utils.arglocmin(mse)[-1]

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
    scale = 10**utils.om(best_bandwidths.mean())
    best_bw = (math.ceil(best_bandwidths.mean()/scale)*scale)
    
    
    # Compute final estimated probability density using the best bandwidth and 
    # compare it with actual probability density using MSE
    epd = kde(data, bin_centers, kernel='gaussian', bandwidth=best_bw)
    apd, bin_edges = np.histogram(data, bins = bins['n'], density=True)
    estimated_mse = ((epd - apd)**2).mean()
    
    if print_log: print(f'\nFinal Optimal bandwidth = {best_bw}\t '
                        f'MSE(apd, epd) = {estimated_mse}')
    
    return {'bw': best_bw, 'estimated_mse':estimated_mse}

#%% FUNCTION: import_stdists_ranking
def import_stdists_ranking(df_input:pd.DataFrame, filepath:str, 
                           scipy_distributions:list = None) -> tuple:
    """Import satistical fitting results per feature in a DataFrame.

    Args:
        df_input (pd.DataFrame): Pandas DataFrame containing the data to fit to 
        the SciPy statistical distributions.
        filepath (str): Filepath from/to where the ranking DataFrame shall be 
        imported/exported as a CSV file.
        scipy_distributions (list, optional): List of SciPy distributions for 
        which the ranking is retrieved. Defaults to None.

    Raises:
        ValueError: Parent folder of filepath does not exist.

    Returns:
        tuple: Tuple with two outputs: 
            - DataFrame with the results of the distribution fitting process 
            (features as columns, distributions as indexes, R2 and parameters as 
            values in a dictionary).
            - Dictionary containing the SciPy distribution object, R2 score 
            results and parameters for the best distribution per feature.
    """

    # Get parent folder path and check if exists.
    folderpath = os.path.dirname(filepath)
    if not os.path.exists(folderpath):
        raise ValueError(f'Folder ({folderpath}) does not exist.')

    # Check if list of SciPy distributions is provided
    if scipy_distributions is None:
        scipy_distributions = eda.get_scipy_distributions(cwd=utils.cwd)
    
    # Import ranking dataframe if already available
    if os.path.exists(filepath):
        ranking = pd.read_csv(filepath_or_buffer=filepath, sep=',', 
                              header='infer', index_col=0, decimal='.', 
                              encoding='utf-8')
    else:

        # Initialize ranking dataframe to evaluate best distributions per 
        # feature
        ranking = pd.DataFrame(index=[dist.name for dist in scipy_distributions])

    # Get all remaining features to rank if any.
    features = [f for f in df_input.columns if not f in ranking.columns]

    # For all remaining features to rank, process them and add them to the file.
    for feature in features:

        # Skip categorical features.
        if df_input[feature].dtypes=='category':continue

        # Set column name to study and remove outliers. 
        data = df_input[feature].dropna().to_numpy()

        # Find best distribution that describes the feature
        _, ranking_i = eda.find_best_distribution(data, scipy_distributions)
        
        # Prepare ranking dataset to be exported as CSV
        ranking_i.set_index('distribution', inplace=True)
        ranking_i.rename(columns={'results':feature}, inplace=True)
        ranking = ranking.join(ranking_i)

        # Export results to a CSV file
        ranking.to_csv(path_or_buf=filepath, sep=',', header=True, 
                        index=True, decimal='.')


    # Initialize dictionary with the best fitted distribution per feature.
    best_dists = {}

    # Iterate over all the features in the ranking dataset.
    for feature in ranking.columns:

        # Get R2 scores as an array (using dictionary).
        r2_scores = np.asarray([eval(stdist)['r2_score'] \
                                for stdist in ranking[feature]])
        
        # Get array index for the distribution with the best R2 score
        idxmax = np.argmax(r2_scores)[0]

        # Get the distribution names as an array.
        dists = np.asarray(ranking.index.values)
        
        # Get parameters of the fitted distribution.
        params = np.asarray([eval(stdist)['params'] \
                             for stdist in ranking[feature]])
        
        # Add information to the dictionary.
        best_dists[feature] = {'dist': eval('st.' + dists[idxmax][0]), 
                               'r2_score': r2_scores[idxmax][0], 
                               'params': params[idxmax][0]}
    
    return ranking, best_dists
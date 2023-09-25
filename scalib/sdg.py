# Libraries used for type hinting
from __future__ import annotations
from typing import Union

from sklearn.neighbors import KernelDensity
from sklearn.metrics import r2_score as r2_sklearn
import matplotlib.pyplot as plt

import numpy as np
import pandas as pd
import scipy.stats as st
import math
import warnings
import os
import pickle

import scalib.eda as eda
import scalib.utils as utils 

import statsmodels.api as smapi
import statsmodels as sm





#%% FUNCTION: kde
# Define function to compute KDE
def kde(x:np.ndarray, x_grid:np.ndarray, bandwidth:float, kernel:str='gaussian',
         **kwargs) -> np.ndarray:
    """Get probability density function using Scikit-learn KernelDensity class. 

    Args:
        x (np.ndarray): Array of data to describe.
        x_grid (np.ndarray): Grid of points upon which the estimation will be done.
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


#%% FUNCTION: plot_kde
def plot_kde(data:np.ndarray, bandwidth:float, figsize:tuple=(6, 3), 
             filepath:str = None, return_ax:bool = False, 
             **kwargs) -> Union[None, plt.Axes]:
    """Plot barchart with actual and estimated probability density.

    Args:
        data (np.ndarray): Actual data from which the probability density is 
        computed.
        bandwidth (float): Bandwidth value.

    Returns:
        Union[None, plt.Axes]: Axes object if return_ax = True.
    """
    
    # Plot the histogram and pdf

    bins = kwargs.get('bins', utils.nbins(data, 'fd'))

    fig, ax = plt.subplots(figsize = figsize )
    n, bin_edges, patches = plt.hist(data, bins = bins['n'], 
                                     density=True, color='gray', 
                                     ec='white', label='Actual data')
    
    

    kernel = kwargs.get('kernel', 'gaussian')
    pdf_grid_size = kwargs.get('pdf_grid_size', 200)


        
    # Fit Kernel Density Estimator
    model = KernelDensity(bandwidth = bandwidth, kernel = kernel)
    data = data.reshape((len(data), 1))
    model.fit(data)

    # Sample probabilities for a range of outcomes
    values = np.asarray([v for v in np.linspace(start = min(data), 
                                                stop = max(data), 
                                                num = pdf_grid_size)])
    values = values.reshape((len(values), 1))
    epd = np.exp(model.score_samples(values))

    label = kwargs.get('label',f'BW = {bandwidth:.2e}')


    ax.plot(values, epd, label = label, color='tab:orange')

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

    if 'title' in kwargs.keys():
        ax.set_title(kwargs['title'], fontsize=10)

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

    
    
    # Set the seed for reproducibility
    np.random.seed(42)

    # Exclude NaN and non-finite numbers from data
    data = data[np.isfinite(data)]

    # Get bins
    bins = utils.nbins(data, bins_rule)

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
    # scale = 10**utils.om(best_bandwidths.mean())
    # best_bw = (math.ceil(best_bandwidths.mean()/scale)*scale)
    best_bw = utils.round_by_om(best_bandwidths.mean()*underfitting_factor)
    
    
    # Compute final estimated probability density using the best bandwidth and 
    # compare it with actual probability density using MSE
    epd = kde(data, bin_centers, kernel='gaussian', bandwidth=best_bw)
    apd, bin_edges = np.histogram(data, bins = bins['n'], density=True)
    estimated_mse = ((epd - apd)**2).mean()
    r2_score = r2_sklearn(apd,epd)
    
    if print_log: print(f'\nFinal Optimal bandwidth = {best_bw}\t '
                        f'MSE(apd, epd) = {estimated_mse}')
    
    return best_bw, r2_score

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

#%% CLASS: SyntheticDataGenerator
class SyntheticDataGenerator():
    def __init__(self, data:np.ndarray, r2_threhold:float = 0.95, 
                 kernel:str = 'gaussian', underfitting_factor:float=3.0,
                 filepath:str = None):
        """Initializes SyntheticDataGenerator class.

        Args:
            data (np.ndarray): Array of numerical values to approach.
            r2_threhold (float, optional): _description_. Defaults to 0.95.
            kernel (str, optional): _description_. Defaults to 'gaussian'.
            underfitting_factor (float, optional): _description_. Defaults to 
            3.0.
            filepath (str, optional): Path of the file with the fitted 
            parameters. Defaults to None.
        """


        # Initialize attributes
        self.data = data.flatten()
        self.r2_threshold = r2_threhold

        if filepath is not None:
            folderpath = os.path.dirname(filepath)
            if not os.path.exists(folderpath):
                raise ValueError(f'Parent folder {folderpath} does not exist.')
            else:
                self.filepath = filepath




        if filepath is None or \
            (os.path.exists(folderpath) and not os.path.exists(filepath)):
            load_model = False
        else:
            load_model = True

        if load_model:
            try:
                # Load parameters from existing files
                self.load(filepath = self.filepath)
            
                # print(f'Parameters loaded from {self.filepath}')
            except:
                load_model = False
                pass

        if not load_model:
            # Find best distribution that describes the feature
            best_stdist, _ = eda.find_best_distribution(data)

            # Generate synthetic data
            self.dist = best_stdist.dist
            self.dist_params = best_stdist.params
            self.dist_r2_score = best_stdist.r2_score()

            # Set kernel function and underfitting factor
            self.kernel = kernel
            self.underfitting_factor = underfitting_factor

            # Save parameters in the filepath if provided.
            if filepath is not None: self.save(filepath=self.filepath)


        

    # Define kernel and underfitting_factor parameters as properties in the 
    # class to attach behaviours. Setting a new value in either r2_threshold,
    # underfitting_factor or kernel parameters will check if the best fitted 
    # SciPy distribution is above the R2 score threshold and update the 
    # bandwidth and KDE function accordingly.
    @property
    def r2_threshold(self):
        return self._r2_threshold
    
    @r2_threshold.setter
    def r2_threshold(self, new_threshold):
        self._r2_threshold = new_threshold
        if hasattr(self, 'dist_r2_score') and \
           hasattr(self, 'data') and \
           hasattr(self, 'kernel') and \
           hasattr(self, 'underfitting_factor'):
            if self.dist_r2_score < new_threshold:
                self.set_kde()

    @property
    def underfitting_factor(self):
        return self._underfitting_factor
    
    @underfitting_factor.setter
    def underfitting_factor(self, new_factor:float):
        self._underfitting_factor = new_factor
        if hasattr(self, 'data') and \
           hasattr(self, 'kernel') and \
           hasattr(self, 'r2_threshold') and \
           hasattr(self, 'dist_r2_score'):
            if self.dist_r2_score < self.r2_threshold:
                self.set_kde()

    # @property
    # def kernel(self):
    #     return self._kernel
    
    # @kernel.setter
    # def kernel(self, new_kernel:str):
    #     self._kernel = new_kernel
    #     if self.dist_r2_score < self.r2_threshold and \
    #         (hasattr(self, 'data') and \
    #          hasattr(self, 'underfitting_factor')):
    #         self.set_kde()



    def save(self, filepath:str):

        if not filepath.endswith('.pkl'):
            filepath = filepath + '.pkl'

        src_dict = {'dist':{'name': self.dist.name,
                            'params': self.dist_params,
                            'r2_score': self.dist_r2_score}}
        if hasattr(self, 'kde'):
            src_dict['kde'] = {'kernel': self.kernel,
                            'params': self.kde_params,
                            'r2_score': self.kde_r2_score,
                            'underfitting_factor':self.underfitting_factor}

        with open(filepath, 'wb') as f:
            pickle.dump(src_dict, f)
        


    def load(self, filepath:str):

        with open(filepath, 'rb') as f:
            loaded_dict = pickle.load(f)

        self.dist = getattr(st, loaded_dict['dist']['name'])
        self.dist_params = loaded_dict['dist']['params']
        self.dist_r2_score = loaded_dict['dist']['r2_score']
 
        if 'kde' in list(loaded_dict.keys()):
            self.underfitting_factor = loaded_dict['kde']['underfitting_factor']
            self.kde_params = loaded_dict['kde']['params']
            self.kde_r2_score = loaded_dict['kde']['r2_score']
            self.kde = KernelDensity(**self.kde_params).fit(self.data.reshape(-1,1))
            self.kernel = loaded_dict['kde']['kernel']
        

    def set_kde(self):

        print(f'\nEstimating optimal bandwidth for {self.kernel.capitalize()} '
              f'Kernel Density...', end='\r')
        # Get bandwidth results from MSE Cross-Validation function.
        bandwidth, r2_score = bws_msecv(data = self.data, 
                                kernel = self.kernel, 
                                underfitting_factor=self.underfitting_factor)

        print(f'Estimating optimal bandwidth for {self.kernel.capitalize()} '
              f'Kernel Density... Optimal bandwidth = {bandwidth:4.2e}.')

        # Fit KernelDensity estimator
        self.kde = KernelDensity(kernel = self.kernel, bandwidth = bandwidth) \
                                .fit(self.data.reshape(-1,1))
        
        self.kde_params = self.kde.get_params(deep=True)
        self.kde_r2_score = r2_score

        # Update synthetic data if already computed.
        if hasattr(self, 'synthetic_data'):
            self.generate_data(n_samples=len(self.synthetic_data))

        # Update estimated probability density values if it was already computed.
        if hasattr(self, 'epd_values'):
            self.probability_density(n_samples_pd = len(self.epd_values))

        # Save parameters if filepath is provided.
        if hasattr(self, 'filepath'): self.save(filepath=self.filepath)

        
    def generate_data(self, n_samples:int = 1000, random_state:int = None):

        # Check which method is used to produce synthetic data: SciPy 
        # distribution or Kernel Density Estimator.
        if self.dist_r2_score >= self.r2_threshold:

            # Get loc, scale and arg parameters
            loc = self.dist_params['loc']
            scale = self.dist_params['scale']
            arg = self.dist_params['arg_values']

            synthetic_data = self.dist.rvs(*arg, 
                                           loc = loc, 
                                           scale = scale, 
                                           size = n_samples*2, 
                                           random_state = random_state)
        else:

            synthetic_data = self.kde.sample(n_samples = n_samples*2, 
                                             random_state = random_state)
            
        # Limit the synthetic data to be always within the boundaries of
        # actual data.
        boundary_condition = (synthetic_data >= min(self.data)) & \
                                (synthetic_data <= max(self.data))
        
        synthetic_data = synthetic_data[boundary_condition]
        
        # Select random values from the array.
        synthetic_data = np.random.choice(synthetic_data, 
                                            size=n_samples, 
                                            replace = False)
                
        self.synthetic_data = synthetic_data.flatten()


    def probability_density(self, n_samples_pd:int = 1000):
        
        if self.dist_r2_score >= self.r2_threshold:

            # Get loc, scale and arg parameters
            loc = self.dist_params['loc']
            scale = self.dist_params['scale']
            arg = self.dist_params['arg_values']

            # Initialize tolerance for the start and end limits
            tol = 1e-3

            while True:
                try:
                    # Get sane start and end points of distribution
                    start = self.dist.ppf(tol, *arg, loc=loc, scale=scale) \
                        if arg else self.dist.ppf(tol, loc=loc, scale=scale)
                    end = self.dist.ppf(1-tol, *arg, loc=loc, scale=scale) \
                        if arg else self.dist.ppf(1-tol, loc=loc, scale=scale)
                    break
                except:
                    pass
                tol *= 10

            # Build PDF and turn into pandas Series
            values = np.linspace(min(self.data), max(self.data), n_samples_pd)
            if min(self.data) > start:
                values = np.linspace(start, end, n_samples_pd)

            epd = self.dist.pdf(values, loc=loc, scale=scale, *arg)

        else:
            
            # Initialize KDE object if has not been done already.
            if self.kde is None: self.set_kde()

            # Estimated Probability density for visualization purposes only.
            values = np.linspace(min(self.data), max(self.data), n_samples_pd)

            # Get log-likelihood of the samples
            log_pdf = self.kde.score_samples(values[:, np.newaxis])
            epd = np.exp(log_pdf)

        self.epd_values = pd.Series(epd, values)


    def plot_histogram(self, show_data:str='all', bins:int=25, xlabel:str=None, 
                       random_state:int = None, figsize:tuple=(6,3), 
                       n_synthetic_samples:int = 2000, n_epd_samples:int = 1000,
                       show_probabilities:bool=True, show:bool=False,
                       filepath:str=None, return_ax:bool = False, 
                       show_stats:bool=False, ax:plt.Axes = None, 
                       show_legend:bool=True):

        
        # Get synthetic data if required.
        if show_data in ['all', 'synthetic']:
            self.generate_data(n_synthetic_samples, random_state)

        # Initialize plot object.
        if ax is None:
            fig, ax = plt.subplots(figsize = figsize)

        if show_data=='all':
            hist_data = [self.data, self.synthetic_data]
            hist_labels = ["Actual", "Synthetic"]
            hist_colors = ["dimgrey","lightskyblue"]
        elif show_data=='synthetic':
            hist_data = [self.synthetic_data]
            hist_labels = ["Synthetic"]
            hist_colors = ["lightskyblue"]
        elif show_data=='actual':
            hist_data = [self.data]
            hist_labels = ["Actual"]
            hist_colors = ["dimgrey"]
        else:
            raise ValueError(f'Parameter show_data ({show_data}) shall be'
                             f' either "all", "actual", or "synthetic".')
            
        # Create histogram.
        n_hist_data, _, _ = ax.hist(hist_data, bins, label=hist_labels, 
                                color=hist_colors, density=show_probabilities)
        
        # Add statistical summary next to the plot
        if show_stats: 

            # Initialize dataframe for statistics
            df_stats = pd.DataFrame()

            # Iterate over all histogram data to plot
            for h in range(len(hist_data)):
                df_h = pd.DataFrame(data = hist_data[h], 
                                    columns = [hist_labels[h]]) \
                                .describe(percentiles=[0.5])
                df_stats = pd.concat([df_stats, df_h], axis=1)

            # Rename stats names in the DataFrame for better readability.    
            df_stats.rename(index={'count': 'Count', 'mean':'Mean','std':'Std',
                                '50%':'Median', 'min':'Min','max':'Max'}, 
                            inplace=True)
            
            # Format numbers to be in LaTeX format
            values = [[utils.number2latex(number) for number in row] \
                    for row in df_stats.to_numpy()]
            
            # Get table in LaTeX format
            text = utils.df2latex(pd.DataFrame(data = values, 
                                            index = list(df_stats.index), 
                                            columns=hist_labels),
                                  include_header=True)

            # Add LaTeX table to the right of the plot
            ax.text(1.04, 0.5, text, size=10, ha='left', 
                    va='center', c='black', transform=ax.transAxes, 
                    bbox=dict(facecolor='white', edgecolor='black', 
                            alpha=0.75, pad=5))

        # Plot density probabilities if required.
        if show_probabilities:
            # Get estimated probability density
            self.probability_density(n_epd_samples)

            ax.plot(self.epd_values, label="PDF", color = "tab:orange")
            ax.set_ylim(0, max([max(hd) for hd in n_hist_data])*1.25)

        # Set Y-axis label
        ylabel = 'Probability density' if show_probabilities else 'Count'
        ax.set_ylabel(ylabel)

        if xlabel is not None: ax.set_xlabel(xlabel)
            
        ax.grid(True, linestyle="dashed", alpha=0.5)
        if show_legend: ax.legend(loc="best", fontsize=8)

        plt.tight_layout()

        if filepath is not None: 
            # print(f'Saving plot to: {filepath}')
            plt.savefig(fname = filepath, bbox_inches='tight')

        if return_ax: return ax

        # if show:
        #     plt.show()
        # else:
        #     plt.close()

    def __repr__(self) -> str:
        """Print readable information about the synthetic data.

        Returns:
            str: Class name with number of CDMs objects contained on it.
        """

        if self.dist_r2_score >= self.r2_threshold:
            # Get length of longest name for formatting
            s = len(max(self.dist_params['names'], key = len))
            
            params = zip(self.dist_params['names'], self.dist_params['values']) 
            params = '\n   - '.join([f'{n:{s}} = {v:}' for n, v in params])

            description = f'\n  Method:       Parametric' + \
                          f'\n  Distribution: {self.dist.name}' + \
                          f'\n  Parameters:\n   - {params}'
        else:
            description = f'\n  Method:    Non-Parametric' + \
                          f'\n  Kernel:    {self.kde_params["kernel"]}' + \
                          f'\n  Bandwidth: {self.kde_params["bandwidth"]}' + \
                          f'\n  Algorithm: {self.kde_params["algorithm"]}'
            
        return f'\nSyntheticDataGenerator:{description}'
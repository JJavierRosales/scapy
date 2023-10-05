# Libraries used for type hinting
from __future__ import annotations
from typing import Union


import matplotlib.pyplot as plt

import numpy as np
import pandas as pd
import scipy.stats as st
import os
import pickle


from . import utils 

from sklearn.neighbors import KernelDensity

#%% CLASS: SyntheticDataGenerator
class SyntheticDataGenerator():
    def __init__(self, data:np.ndarray, r2_threhold:float = 0.95, 
                 kernel:str = 'gaussian', underfitting_factor:float=3.0,
                 filepath:str = None):
        """Initialises generator constructor.

        Args:
            data (np.ndarray): Array of numerical values to approach.
            r2_threhold (float, optional): Coefficient of determination 
            threshold (above which it is considered a valid approximation). 
            Defaults to 0.95.
            kernel (str, optional): Kernel to use: 'gaussian', 'tophat', 
            'epanechnikov', 'exponential', 'linear', 'cosine'. Defaults to 
            'gaussian'.
            underfitting_factor (float, optional): Factor to reduce overfitting 
            (multiplies the optimal bandwidth computed through MSE cross 
            validation). Defaults to 3.0.
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
            best_stdist, _ = utils.find_best_distribution(data)

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
        """Save generator parameters into an external pickle file.

        Args:
            filepath (str): File path where the parameters are saved.
        """

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
        


    def load(self, filepath:str) -> None:
        """Load generator parameters.

        Args:
            filepath (str): Origin file path.
        """

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
        

    def set_kde(self) -> None:
        """Set kernel density estimator and compute PDF.
        """

        print(f'\nEstimating optimal bandwidth for {self.kernel.capitalize()} '
              f'Kernel Density...', end='\r')
        # Get bandwidth results from MSE Cross-Validation function.
        bandwidth, r2_score = utils.bws_msecv(data = self.data, 
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

        
    def generate_data(self, n_samples:int = 1000, random_state:int = None) \
        -> None:
        """Generate synthetic data.

        Args:
            n_samples (int, optional): Number of data points to create. Defaults 
            to 1000.
            random_state (int, optional): Random state to ensure 
            reproducibility. Defaults to None.
        """

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


    def probability_density(self, n_samples_pd:int = 1000) -> None:
        """Compute probability density points.

        Args:
            n_samples_pd (int, optional): Number of points to compute. Defaults 
            to 1000.
        """
        
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
                       show_probabilities:bool=True, filepath:str=None, 
                       return_ax:bool = False, show_stats:bool=False, 
                       ax:plt.Axes = None, show_legend:bool=True) -> None:
        """Plot histogram.

        Args:
            show_data (str, optional): Type of data to display: 'all', 
            'synthetic' or 'actual. Defaults to 'all'.
            bins (int, optional): Number of bins. Defaults to 25.
            xlabel (str, optional): X-axis label. Defaults to None.
            random_state (int, optional): Random state to ensure 
            reproducibility. Defaults to None.
            figsize (tuple, optional): Figure size. Defaults to (6,3).
            n_synthetic_samples (int, optional): Number of synthetic data points 
            to plot. Defaults to 2000.
            n_epd_samples (int, optional): Number of estimated probability 
            density points. Defaults to 1000.
            show_probabilities (bool, optional): Show probabilities (density 
            plot). Defaults to True.
            filepath (str, optional): Path to save the figure. Defaults to None.
            return_ax (bool, optional): Return axis object. Defaults to False.
            show_stats (bool, optional): Show main statistics from distributon: 
            mean, median, standard deviation and quartiles. Defaults to False.
            ax (plt.Axes, optional): Axis object. Defaults to None.
            show_legend (bool, optional): Show legend in plot. Defaults to True.

        Raises:
            ValueError: show_data parameter is invalid.
        """

        
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

#%% Initialise generator object as a shortcut
generator = SyntheticDataGenerator
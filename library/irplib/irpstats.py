import time
import warnings
import numpy as np
import pandas as pd
import scipy.stats as st
from .main import outliers_boundaries, nbins

from sklearn.metrics import r2_score

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
            
            y, bin_edges = np.histogram(self.data, bins=nbins(self.data,'fd')['n'], density=True)
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
        std_lims = outliers_boundaries(self.data, threshold = 1.5, positive_only=False)
        
        # Build PDF and turn into pandas Series
        x = np.linspace(std_lims[0], std_lims[1], size)
        y = self.dist.pdf(x, loc=loc, scale=scale, *arg)
        
        pdf = pd.Series(y, x)
        
        return pdf
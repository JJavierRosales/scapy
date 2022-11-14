#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Nov  5 16:34:03 2022

@author: jjrr
"""


import numpy as np
import pandas as pd
import warnings
import scipy.stats as st
import os
import requests
from pathlib import Path
from matplotlib import rc, pyplot as plt

rc("font",**{"family":"serif","serif":["Computer Modern"],"size" : 14})
rc("text", usetex=True)

# Import custom library
import general as gn

#%%
# Initialize global variables for paths

cwd         = str(Path(os.getcwd()).parents[0])
cwd_images  = os.path.join(cwd, 'images')
cwd_data    = os.path.join(cwd, 'data')

#%%
# Create models from data
def best_fit_distribution(data, bins=200, exclude=[]):
    """Model data by finding best fit distribution to data"""
    # Get histogram of original data
    y, x = np.histogram(data, bins=bins, density=True)
    x = (x + np.roll(x, -1))[:-1] / 2.0
    
    # Get Continuous distributions table from SciPy website.
    url = 'https://docs.scipy.org/doc/scipy/reference/stats.html'
    dist_table = pd.read_html(requests.get(url).content)[1]
    
    # Evaluate list of objects in str format to convert it into a Python list
    dist_list = eval('[' + (', ').join('st.' + dist_table[0]) + ']')
    
    # Best holders
    best = {'dist': st.norm,
            'params': (0.0, 1.0),
            'sse': np.inf}
    
    dists_checked = []

    # Estimate distribution parameters from data
    for i, dist in enumerate(dist_list):
        
        
        print('Progress %5.1f%% (%3d/%3d) \tDistribution: %16s \tBest: %s' %
              ((i+1)/len(dist_list)*100, i, len(dist_list)-1, dist.name, 
               best['dist'].name))
        
        if dist.name in exclude: continue
        
        # Try to fit the distribution
        try:
            # Ignore warnings from data that can"t be fit
            with warnings.catch_warnings():
                warnings.filterwarnings("ignore")

                # fit dist to data
                params = dist.fit(data)

                # Separate parts of parameters
                arg = params[:-2]
                loc = params[-2]
                scale = params[-1]

                # Calculate fitted PDF and error with fit in distribution
                pdf = dist.pdf(x, loc=loc, scale=scale, *arg)
                sse = np.sum(np.power(y - pdf, 2.0))

                # identify if this distribution is better
                if best['sse'] > sse > 0:
                    best['dist']= dist
                    best['params'] = params
                    best['sse'] = sse

        except Exception:
            pass
        
        dists_checked.append(dist.name)
        
    print("\nMinimum error: %1.3e" % best['sse'])
    return (best['dist'].name, best['params'])

#%%
def make_pdf(dist, params, size=10000):
    """Generate distributions's Probability Distribution Function """

    # Separate parts of parameters
    arg = params[:-2]
    loc = params[-2]
    scale = params[-1]

    # Get same start and end points of distribution
    start = dist.ppf(1e-4, *arg, loc=loc, scale=scale) \
        if arg else dist.ppf(1e-4, loc=loc, scale=scale)
    end = dist.ppf(1-1e-4, *arg, loc=loc, scale=scale) \
        if arg else dist.ppf(1-1e-4, loc=loc, scale=scale)

    # Build PDF and turn into pandas Series
    x = np.linspace(start, end, size)
    y = dist.pdf(x, loc=loc, scale=scale, *arg)
    pdf = pd.Series(y, x)

    return pdf


#%%
def get_probabilities(dist, arg, loc, scale, size=10000):
    """Generate distributions's Probability Distribution Function """

    # Get sane start and end points of distribution
    start = dist.ppf(1e-4, *arg, loc=loc, scale=scale) \
        if arg else dist.ppf(1e-4, loc=loc, scale=scale)
    end = dist.ppf(1-1e-4, *arg, loc=loc, scale=scale) \
        if arg else dist.ppf(1-1e-4, loc=loc, scale=scale)

    # Build PDF and turn into pandas Series
    values = np.linspace(start, end, size)

    return values, dist.pdf(values, loc=loc, scale=scale, *arg)


#%%
if __name__ == "__main__":
     
    
    
    n_bins = 50
    column = "miss_distance"
    
    # Import Kelvins data
    df = pd.read_csv(os.path.join(cwd_data, 
                                  'esa-challenge',
                                  'train_data.csv'), 
                     sep=',', header=0, index_col=None, skipinitialspace=False)
    
    # Sort values of dataframe by event_id and time_to_tca and re-index
    df.sort_values(by=['event_id', 'time_to_tca'], axis='index', 
                   ascending=[True,False], inplace=True, ignore_index=True)

    # Get only last CDM data from every event_id
    df = df.drop_duplicates('event_id', keep='last')
    
    data = gn.remove_outliers(df[column], 1.5)
    

    # Find best fit distribution
    dist_name, dist_params = best_fit_distribution(data, 200, ['studentized_range'])
    dist = getattr(st, dist_name)
    
    # Get information of the parameters
    param_names = (dist.shapes + ", loc, scale").split(", ") \
        if dist.shapes else ["loc", "scale"]
    param_str = "\quad ".join(["{}={:0.5f}".format(k,v) \
                               for k,v in zip(param_names, dist_params)])

    print("Best distribution:", dist.name)
    for k,v in zip(param_names, dist_params): print(k,"=", v)
        
    # Make PDF with best params 
    pdf = make_pdf(dist, dist_params)
    
    figurename = column + "_" + dist_name + ".pdf"
    
    # Display  plot
    plt.figure(figsize=(7,5))

    ax = pdf.plot(lw=1.5, color = "orange", 
                  label=dist_name.capitalize() +" PDF")
    data.plot(kind="hist", bins=n_bins, ax=ax, 
              density=True, label="Kelvins data", color="dimgrey")
    
    plt.title("{}".format(param_str), fontsize=12)
    plt.xlabel(column)
    plt.ylabel(r"Probability density")
    plt.grid(True, linestyle="dashed", alpha=0.5)
    plt.legend(loc="best")
    plt.savefig(os.path.join(cwd_images, 
                             'probability-density', 
                             figurename),
                bbox_inches="tight")
    plt.show()
    
    

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
def fit_distribution(stdist, data, bins):
    
    # Get histogram of original data
    y, x = np.histogram(data, bins=bins, density=True)
    x = (x + np.roll(x, -1))[:-1] / 2.0
    
    # Best holders
    dist = {'dist': st.norm,
            'params': (0.0, 1.0),
            'sse': np.inf}
    
    # Try to fit the distribution
    try:
        # Ignore warnings from data that can"t be fit
        with warnings.catch_warnings():
            warnings.filterwarnings("ignore")

            # fit dist to data
            params = stdist.fit(data)

            # Separate parts of parameters
            arg = params[:-2]
            loc = params[-2]
            scale = params[-1]

            # Calculate fitted PDF and error with fit in distribution
            pdf = stdist.pdf(x, loc=loc, scale=scale, *arg)
            sse = np.sum(np.power(y - pdf, 2.0))

            # identify if this distribution is better
            dist['dist']    = stdist
            dist['params']  = params
            dist['sse']     = sse
            
    except Exception:
        pass
    
    return dist

#%%
# Create models from data
def find_best_distribution(data, bins=200, st_excluded=[]):
    """Model data by finding best fit distribution to data"""
    
    # Get Continuous distributions table from SciPy website.
    url = 'https://docs.scipy.org/doc/scipy/reference/stats.html'
    dist_table = pd.read_html(requests.get(url).content)[1]
    
    # Evaluate list of objects in str format to convert it into a Python list
    stdist_list = eval('[' + (', ').join('st.' + dist_table[0]) + ']')
    
    # Best holders
    best_dist = {'dist': st.norm,
            'params': (0.0, 1.0),
            'sse': np.inf}
    
    dists_checked = []

    # Estimate distribution parameters from data
    for i, stdist in enumerate(stdist_list):
        
        
        print('Progress %5.1f%% (%3d/%3d) \tDistribution: %16s \tBest: %s' %
              ((i+1)/len(stdist_list)*100, i, len(stdist_list)-1, stdist.name, 
               best_dist['dist'].name))
        
        if stdist.name in st_excluded: continue
        
        dist = fit_distribution(stdist, data, bins)
        
        if best_dist['sse'] > dist['sse'] > 0: best_dist = dist
        
        dists_checked.append(stdist.name)
        
    # Get information of the parameters
    param_names = (dist['dist'].shapes + ", loc, scale").split(", ") \
                    if dist['dist'].shapes else ["loc", "scale"]
        
    
    params = ""
    for p, v in zip(param_names, best_dist['params']): 
        params = params + "\n - "+ p + "=" + str(v)
        
    print("\nBest dist.: %s\nSSE: %1.3e\nParameters:%s" % \
          (best_dist['dist'].name, best_dist['sse'], params))
    
    return best_dist

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
def plot_dist(data, dist, xlabel, bins, figurename):
    
    # Make Probability Density Function with distribution parameters 
    pdf = make_pdf(dist['dist'], dist['params'])
    
    # Get information of the parameters
    param_names = (dist['dist'].shapes + ", loc, scale").split(", ") \
        if dist['dist'].shapes else ["loc", "scale"]
            
    # Create string with all parameters to print into the plot's title       
    title = (dist['dist'].name).capitalize() + '\n' + \
                "\quad ".join(["{}={:0.5f}".format(k,v) \
                for k,v in zip(param_names, dist['params'])])

    # Display plot
    plt.figure(figsize=(7,5))

    ax = pdf.plot(lw=1.5, color = "orange", label="PDF")
    data.plot(kind="hist", bins=bins, ax=ax, density=True, 
              label="Data", color="dimgrey")
    
    plt.title("{}".format(title), fontsize=12)
    plt.xlabel(xlabel)
    plt.ylabel(r"Probability density")
    plt.grid(True, linestyle="dashed", alpha=0.5)
    plt.legend(loc="best", fontsize=10)
    plt.savefig(os.path.join(cwd_images, 
                             'probability-density', 
                             figurename + ".pdf"),
                bbox_inches="tight")
    plt.show()
    
    return


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
     
    # Import Kelvins data
    df = pd.read_csv(os.path.join(cwd_data, 'esa-challenge', 'train_data.csv'), 
                     sep=',', header=0, index_col=None, skipinitialspace=False)
    
    # Sort values of dataframe by event_id and time_to_tca and re-index
    df.sort_values(by=['event_id', 'time_to_tca'], axis='index', 
                   ascending=[True,False], inplace=True, ignore_index=True)

    # Get only last CDM data from every event_id
    df = df.drop_duplicates('event_id', keep='last')
    
    # Set column name to study and remove outliers.
    column = "miss_distance"
    data = gn.remove_outliers(df[column], 1.5)
    
    # Find distribution that best fits the data
    best_dist = find_best_distribution(data, 200, ['studentized_range'])
    
    # Fit distribution to the data
    dist = fit_distribution(st.expon, data, 200)
    
    # Print plot including histogram and distribution fitted
    plot_dist(data, dist, column, 
              50, column + '-' + dist['dist'].name)
    

    
    

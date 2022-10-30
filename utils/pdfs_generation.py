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
from pathlib import Path
from matplotlib import rc, pyplot as plt

rc("font",**{"family":"serif","serif":["Computer Modern"],"size" : 14})
rc("text", usetex=True)

# Import custom library
import general as gn

#%%
# Initialize global variables for paths

root_path = str(Path(os.getcwd()).parents[0])
root_path_images = os.path.join(root_path, 'images')
root_path_data = os.path.join(root_path, 'data')

#%%
# Create models from data
def best_fit_distribution(data, bins=200):
    """Model data by finding best fit distribution to data"""
    # Get histogram of original data
    y, x = np.histogram(data, bins=bins, density=True)
    x = (x + np.roll(x, -1))[:-1] / 2.0

    # Distributions to check
    distributions_list = [        
        st.alpha,st.anglit,st.arcsine,st.beta,st.betaprime,st.bradford,st.burr,
        st.cauchy,st.chi,st.chi2,st.cosine,st.dgamma,st.dweibull,st.erlang,
        st.expon,st.exponnorm,st.exponweib,st.exponpow,st.f,st.fatiguelife,
        st.fisk,st.foldcauchy,st.foldnorm,
        st.genlogistic,st.genpareto,st.gennorm,st.genexpon,st.genextreme,
        st.gausshyper,st.gamma,st.gengamma,st.genhalflogistic,st.gilbrat,
        st.gompertz,st.gumbel_r,st.gumbel_l,st.halfcauchy,st.halflogistic,
        st.halfnorm,st.halfgennorm,st.hypsecant,st.invgamma,st.invgauss,
        st.invweibull,st.johnsonsb,st.johnsonsu,st.ksone,st.kstwobign,
        st.laplace,st.levy,st.levy_l,st.logistic,st.loggamma,st.loglaplace,
        st.lognorm,st.lomax,st.maxwell,st.mielke,st.nakagami,st.norm,st.pareto,
        st.pearson3,st.powerlaw,st.powerlognorm,st.powernorm,st.rdist,
        st.reciprocal,st.rayleigh,st.rice,st.recipinvgauss,st.semicircular,
        st.t,st.triang,st.truncexpon,st.truncnorm,st.tukeylambda,st.uniform,
        st.vonmises,st.vonmises_line,st.wald,st.weibull_min,st.weibull_max,
        st.wrapcauchy#,st.ncx2,st.ncf,st.nct,st.levy_stable,st.frechet_r,st.frechet_l
    ]

    # Best holders
    best_distribution = st.norm
    best_params = (0.0, 1.0)
    best_sse = np.inf

    # Estimate distribution parameters from data
    for i in range(len(distributions_list)):
        
        distribution = distributions_list[i]
        
        print("Progress", round((i+1)/len(distributions_list)*100,2), 
              "%   \tDistribution:", distribution.name)
        
        # Try to fit the distribution
        try:
            # Ignore warnings from data that can"t be fit
            with warnings.catch_warnings():
                warnings.filterwarnings("ignore")

                # fit dist to data
                params = distribution.fit(data)

                # Separate parts of parameters
                arg = params[:-2]
                loc = params[-2]
                scale = params[-1]

                # Calculate fitted PDF and error with fit in distribution
                pdf = distribution.pdf(x, loc=loc, scale=scale, *arg)
                sse = np.sum(np.power(y - pdf, 2.0))

                # identify if this distribution is better
                if best_sse > sse > 0:
                    best_distribution = distribution
                    best_params = params
                    best_sse = sse

        except Exception:
            pass
        
    print("Minimum error:", best_sse)
    return (best_distribution.name, best_params)

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
    df = pd.read_csv(os.path.join(root_path_data, 
                                  'esa-challenge',
                                  'train_data.csv'), 
                     sep=',', header=0, index_col=None, skipinitialspace=False)
    
    data = gn.remove_outliers(df[column], 1.5)
    

    # Find best fit distribution
    best_fit_name, best_fit_params = best_fit_distribution(data, 200)
    best_dist = getattr(st, best_fit_name)
    
    # Get information of the parameters
    param_names = (best_dist.shapes + ", loc, scale").split(", ") \
        if best_dist.shapes else ["loc", "scale"]
    param_str = "\quad ".join(["{}={:0.5f}".format(k,v) \
                               for k,v in zip(param_names, best_fit_params)])

    print("Best distribution:", best_dist.name)
    for k,v in zip(param_names, best_fit_params): print(k,"=", v)
        
    # Make PDF with best params 
    pdf = make_pdf(best_dist, best_fit_params)
    
    figurename = column + "_" + best_fit_name + ".pdf"
    
    # Display  plot
    plt.figure(figsize=(7,5))

    ax = pdf.plot(lw=1.5, color = "orange", 
                  label=best_fit_name.capitalize() +" PDF")
    data.plot(kind="hist", bins=n_bins, ax=ax, 
              density=True, label="Kelvins data", color="dimgrey")
    
    plt.title("{}".format(param_str), fontsize=12)
    plt.xlabel(column)
    plt.ylabel(r"Probability density")
    plt.grid(True, linestyle="dashed", alpha=0.5)
    plt.legend(loc="best")
    plt.savefig(os.path.join(root_path_images, 
                             'probability-density', 
                             figurename),
                bbox_inches="tight")
    plt.show()
    
    

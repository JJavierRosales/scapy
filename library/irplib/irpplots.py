# Import Scikit-learn required libraries
from sklearn.neighbors import KernelDensity
import matplotlib.pyplot as plt
from .main import nbins, order_of_magnitude
import numpy as np

#%%
def plot_kde(data:np.ndarray, bandwidths:np.ndarray, **kwargs) -> None:
    """Plot barchart with actual and estimated probability density.

    Args:
        data (np.ndarray): Actual data from which the probability density is computed
        bandwidths (np.ndarray): Array of bandwidths to evaluate on the kernel.

    Returns:
        None: No output.
    """
    
    # Plot the histogram and pdf
    plt.figure(figsize = kwargs.get('figsize',(6, 3)) )
    plt.hist(data, bins = kwargs.get('bins', nbins(data)['n']), 
            density=True, ec='white', label='Actual')
    
    # Iterate over all bandwidths and plot
    for bw in bandwidths:
        
        # Fit Kernel Density Estimator
        model = KernelDensity(bandwidth=bw, kernel=kwargs.get('kernel', 'gaussian'))
        data = data.reshape((len(data), 1))
        model.fit(data)

        # Sample probabilities for a range of outcomes
        values = np.asarray([value for value in np.linspace(min(data), max(data), kwargs.get('pdf_grid_size', 200))])
        values = values.reshape((len(values), 1))
        probabilities = np.exp(model.score_samples(values))

        label = f'bw = {bw:.4f}' if  order_of_magnitude(bw)>-4 else f'bw = {bw:.3e}'

        plt.plot(values, probabilities, label=label)

    plt.xlabel(kwargs.get('xlabel', 'Feature'))
    plt.ylabel(kwargs.get('ylabel', 'Probability density'))
    plt.title('Probability Density analysis', fontsize=10)
    plt.grid(True, linestyle="dashed", alpha=0.5)
    plt.legend(loc='best', fontsize=10)
    plt.show()
    
    return None
# Import Scikit-learn required libraries
from sklearn.neighbors import KernelDensity
import matplotlib.pyplot as plt
from . import utils 
import numpy as np

#%%
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
    
    # Instanciate KernelDensity class from Scikit-learn (kernel defaults to 'gaussian')
    kde = KernelDensity(bandwidth=bandwidth, **kwargs)
    
    # Fit Kernel
    kde.fit(np.array(x).reshape(-1,1))
    
    # Get log-likelihood of the samples
    log_pdf = kde.score_samples(x_grid[:, np.newaxis])
    epd     = np.exp(log_pdf)
    
    return epd
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
    plt.hist(data, bins = kwargs.get('bins', utils.nbins(data)['n']), 
            density=True, ec='white', label='Actual')
    
    # Iterate over all bandwidths and plot
    labels = kwargs.get('bw_labels', [])

    for b, bw in enumerate(bandwidths):
        
        # Fit Kernel Density Estimator
        model = KernelDensity(bandwidth=bw, kernel=kwargs.get('kernel', 'gaussian'))
        data = data.reshape((len(data), 1))
        model.fit(data)

        # Sample probabilities for a range of outcomes
        values = np.asarray([value for value in np.linspace(min(data), max(data), kwargs.get('pdf_grid_size', 200))])
        values = values.reshape((len(values), 1))
        probabilities = np.exp(model.score_samples(values))

        if len(labels)<b+1:
            labels.append(f'bw = {bw:.4f}' if  utils.order_of_magnitude(bw)>-4 else f'bw = {bw:.3e}')

        plt.plot(values, probabilities, label=labels[b])

    plt.xlabel(kwargs.get('xlabel', 'Feature'))
    plt.ylabel(kwargs.get('ylabel', 'Probability density'))
    plt.title('Probability Density analysis', fontsize=10)
    plt.grid(True, linestyle="dashed", alpha=0.5)
    plt.legend(loc='best', fontsize=10)
    plt.show()
    
    return None

#%%
def msecv(data: np.ndarray, bins:dict, conv_accuracy:float = 1e-5, 
                    n_batches_max:int = 10, print_log:bool = True) -> dict:
    """Computes optimal bandwidth minimizing MSE actual vs estimated density through cross-validation.

    Args:
        data (np.ndarray): Array containing all input data.
        bins (dict): _description_
        conv_accuracy (float, optional): Convergence accuracy. Defaults to 1e-5.
        n_batches_max (int, optional): Maximum number of batches for the cross-validation. Defaults to 10.
        print_log (bool, optional): Print computational log. Defaults to True.

    Returns:
        dict: Dictionary with the bandwidth value that minimizes the MSE and the estimated MSE.
    """
    
    # Exclude NaN and non-finite numbers from data
    data = data[np.isfinite(data)]

    # Create an array with the number of batches to process per iteration 
    batches_list = np.arange(start=2, stop=n_batches_max + 1, step=1, dtype=np.int32)
    
    # Initialize arry to store best bandwidth per group of batches
    best_bandwidths = np.zeros(len(batches_list))

    for i, batches in enumerate(batches_list):
        
        # Create an array with random values between 0 and the total number of batches
        # of the size of the entire array of data (following a uniform distribution).
        batches_categories = np.random.randint(low=0, high=batches, size=len(data))

        # Initialize array to store actual probabilty densities (apds) from every batch
        apds = np.zeros((bins['n'], batches))

        # Iterate over all batches categories (0 to number of batches)
        for c in range(batches):
            
            # Get sample of data corresponding to a specific batch category "c"
            sample = data[batches_categories==c]
            
            # Get actual probability density from the sample using number of bins obtained
            # from the entire array of data and store it in the apds array.
            apds[:,c], bin_edges = np.histogram(sample, bins = bins['n'], density=True)

        # Get array of bin centers to which the actual probability density are associated
        bin_centers = bin_edges[:-1] + bins['width']/2

        # Compute average of all actual probability densities from all batches
        avg_apd = np.mean(apds, axis=1, dtype=np.float64)

        # Initialize bandwidth array to evaluate estimated probability density from kernel
        bandwidths, step = np.linspace(bins['width']/100, bins['width'], 100, retstep=True)

        # Initialize best_bw
        best_bw = 0.0
        while True:

            # Initialize mean squared errors array associated to every bandwidth
            mse = np.zeros(len(bandwidths))

            # Iterate over all the bandwidths to compute the MSE from the actual vs estimated 
            # probability densities
            for b, bandwidth in enumerate(bandwidths):

                # Get estimated probability distribution using the bandwidth "b"
                epd = bws.kde(data, bin_centers, bandwidth=bandwidth)
                
                # Compute MSE from actual vs estimated probability densities
                mse[b] = ((epd - avg_apd)**2).mean()
                
            # Get bandwidth that minimizes MSE and check accuracy vs best_bw
            bw = bandwidths[np.argmin(mse)]

            # Check if convergence accuracy is achieved to stop iterations
            if abs(1 - best_bw/bw) <= conv_accuracy: break
            
            # Update best_bw and bandwidths array to increase final bandwidth accuracy
            best_bw = bw
            bandwidths, step = np.linspace(bw-step, bw+step, 100, retstep = True)
            
        # Add best bandwidth from this group of batches to final array
        best_bandwidths[i] = best_bw 

        if print_log: print(f'Batches = {batches:2} ({len(data)//batches} d.p. per batch)  '
                            f'Best bandwidth = {best_bw:.5f}  MSE(apd, epd) = {mse.min():.4e}')

    # Round-up best bandwidth from all groups of batches using one order of magnitude less
    scale = 10**main.order_of_magnitude(best_bandwidths.mean())
    best_bw = (math.ceil(best_bandwidths.mean()/scale)*scale)
    
    
    # Compute final estimated probability density using the best bandwidth and compare it
    # with actual probability density using MSE
    epd = bws.kde(data, bin_centers, kernel='gaussian', bandwidth=best_bw)
    apd, bin_edges = np.histogram(data, bins = bins['n'], density=True)
    estimated_mse = ((epd - apd)**2).mean()
    
    if print_log: print(f'\nFinal Optimal bandwidth = {best_bw}\t MSE(apd, epd) = {estimated_mse}')
    
    return {'bw': best_bw, 'estimated_mse':estimated_mse}
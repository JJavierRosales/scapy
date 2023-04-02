import pandas as pd
import numpy as np
import math
import warnings
import scipy.stats as st
from sklearn.linear_model import LinearRegression

#%%
def nbins(data: np.ndarray, rule:str = 'fd') -> dict:
    """Calculate number of bins and bin width for histograms.

    Args:
        data (np.float32): Array containing all data.
        rule (str): Rule to use to compute the bin size. It can be one of the following options:
            - Sturge ('sturge')
            - Scott ('scott')
            - Rice ('rice')
            - Freedman-Diaconis ('fd') - Default

    Returns:
        dict: Dictionary containing with the number of bins 'n' and bin width 'width'.
    """

    # Check if rule passed by user is valid
    if not rule in ['sturge', 'scott', 'rice', 'fd']: return None

    # Get the number of items within the dataset
    data = data[~np.isnan(data)]
    n = len(data)
    
    # Compute the histogram bins size (range)
    bins_width = {'sturge': 1 + 3.322*np.log(n),    
                 'scott': 3.49*np.std(data)*n**(-1/3),                       
                 'rice': 2*n**(1/3),                         
                 'fd': 2*st.iqr(data)*n**(-1/3)}
    
    # Compute number of bins
    n_bins =  math.ceil((data.max() - data.min())/bins_width[rule])
    
    return {'n': n_bins, 'width': bins_width[rule]}
#%%
def order_of_magnitude(value: float) -> int:
    """Get order of magnitude of value.

    Args:
        value (float): Value to get the order of magnitude from.

    Returns:
        int: Order of magnitude.
    """
    if value==0: return 0
    if abs(value)==np.inf: return np.inf
    
    return int(math.floor(math.log(abs(value), 10)))
#%%
def outliers_boundaries(data: np.ndarray, threshold:float = 1.5, positive_only:bool=False) -> tuple:
    """Compute limits of standard data within a given data distribution.

    Args:
        data (np.ndarray): Data to get the outliers boundaries from.
        threshold (float, optional): Proportion of IQR to take into account. Defaults to 1.5.
        positive_only (bool, optional): Force negative lower boundary to be 0 if data can only be positive. Defaults to False.

    Returns:
        tuple: Range of values between which satandard data is comprised.
    """
    
    # Remove NaN from data
    data = data[~np.isnan(data)]
    
    Q1 = np.quantile(data,0.25)
    Q3 = np.quantile(data,0.75)
    IQR = st.iqr(data)
    
    if positive_only:
        return (max(0,(Q1-IQR*threshold)), (Q3+IQR*threshold))
    else:
        return ((Q1-IQR*threshold), (Q3+IQR*threshold))
#%%
def compute_vif(df_input: pd.DataFrame) -> dict:
    """Compute Variance Inflation Factor to evaluate multicolinearity.

    Args:
        df_input (pd.DataFrame): Input dataframe to compute VIF.

    Returns:
        dict: Dictionary with all the features as keys and VIF scores as values.
    """
    
    # Create deep copy of input DataFrame
    data = df_input.copy(deep=True).dropna()
    
    # Get the features and model object
    features = list(data.columns)
    model = LinearRegression()
    
    # Create empty dataset with 0s
    result = pd.DataFrame(index=['VIF'], columns=features)
    result = result.fillna(0)
    
    # Iterate through all features to evaluate its VIF
    for y_feature in features:
        
        x_features = [f for f in features if not f==y_feature]
        
        x = data[x_features]
        y = data[y_feature]
        
        # Fit the model and calculate VIF
        model.fit(x, y)
        
        # Disable warnings in case of division by 0
        warnings.filterwarnings("ignore")
        result[y_feature] =  (1/(1-model.score(x, y)))
        warnings.filterwarnings("default")
        
    return result
#%%
def vif_selection(df_input:pd.DataFrame, max_vif:float=0.8) -> dict:
    """Variable selection using Variance Inflation Factor (VIF) threshold.

    Args:
        df_input (pd.DataFrame): Input dataframe upon which VIF selection is performed.
        max_vif (float, optional): VIF score used to select variables. Defaults to 0.8.

    Returns:
        dict: Dictionary containing correlated and independent features with their VIF scores.
    """
    
    # Create a deep copy of the input DataFrame
    result = df_input.copy(deep=True).dropna()
    
    # Compute initial VIF from the entire dataset
    vif = compute_vif(result)
    
    correlated = {}
    while vif.values.max() > max_vif:
        
        collinear_feature = vif.idxmax(axis="columns").values[0]
        correlated[collinear_feature] = vif.loc['VIF', collinear_feature]
        
        features = [c for c in list(result.columns) if not c==collinear_feature]
        
        # Compute VIF with one less feature
        result = result[features]
        vif = compute_vif(result)
    
    # Get information on the independent and correlated variables
    output = {'independent': vif.to_dict('records')[0],
              'correlated': correlated}
              
    return output
#%%
import pandas as pd
import numpy as np
import math
import warnings
import time
import scipy.stats as st
from typing import Union
from sklearn.linear_model import LinearRegression

#%%
def arglocmax(a:np.ndarray):

    condition = np.r_[True, a[1:] > a[:-1]] & np.r_[a[:-1] > a[1:], True]
    index_array = np.asarray([i for i, c in enumerate(condition) if c], dtype=np.int32)

    return index_array

#%%
def arglocmin(a:np.ndarray):

    condition = np.r_[True, a[1:] < a[:-1]] & np.r_[a[:-1] < a[1:], True]
    index_array = np.asarray([i for i, c in enumerate(condition) if c], dtype=np.int32)

    return index_array

#%%
def nbins(data:np.ndarray, rule:str = 'fd') -> dict:
    """Calculate number of bins and bin width for histograms.

    Args:
        data (array_like): Array containing all data.
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
    data = data.astype(np.float64)
    n = len(data)
    
    # Compute the histogram bins size (range)
    bins_width = {'sturge': 1 + 3.322*np.log(n),    
                 'scott': 3.49*np.std(data)*n**(-1.0/3.0),                       
                 'rice': 2*n**(1.0/3.0),                         
                 'fd': 2*st.iqr(data)*n**(-1.0/3.0)}
    
    # Compute number of bins
    n_bins =  math.ceil((data.max() - data.min())/bins_width[rule])
    
    return {'n': n_bins, 'width': bins_width[rule]}
#%%
def om(value: float) -> int:
    """Get order of magnitude of value.

    Args:
        value (float): Value to get the order of magnitude from.

    Returns:
        int: Order of magnitude.
    """
    if not isinstance(value, float) and not isinstance(value, int): return np.nan
    if value==0 or np.isnan(value): return 0
    if abs(value)==np.inf: return np.inf
    
    return int(math.floor(math.log(abs(value), 10)))
#%%
def round_by_om(value:float, abs_method:str='ceil', **kwargs) -> float:
    """Round up/down float by specifying rounding of magnitude.

    Args:
        value (float): Value to round up/down.
        abs_method (str, optional): Method to round considering the absolute value (up or down). Defaults to 'ceil'.

    Returns:
        float: Value rounded.
    """


    # Return None if method is not valid
    if not abs_method in ['ceil', 'floor']: return None
    
    # Return 0 if value is 0
    if value==0: return 0

    # Compute order of magnitude
    initial_om = 10**kwargs.get('rounding_om', om(value))
    
    # Initialize dictionary with round methods
    methods = {'ceil':math.ceil, 'floor':math.floor}

    # Invert method if value is negative
    if value<0: abs_method = 'floor' if abs_method=='ceil' else 'ceil'

    return methods[abs_method](value/initial_om)*initial_om

#%%
#%%
def df2latex(df: pd.DataFrame, column_format:str='c') -> str:
    """Convert pandas DataFrame to latex table.

    Args:
        df (pd.DataFrame): DataFrame to convert to LaTeX format.
        column_format (str, optional): Columns alignment (left 'l', center 'c', or right 'r'). Defaults to 'c'.

    Returns:
        str: DataFrame in string format.
    """

    column_format = 'c'*(len(df.columns)+1) if column_format=='c' else column_format

    new_column_names = dict(zip(df.columns, ["\textbf{" + c + "}" for c in df.columns]))
    
    df.rename(new_column_names, axis='columns', inplace=True)
    
    table = df.style.to_latex(column_format=column_format)
    table = table.replace('\n', '').encode('unicode-escape').decode()\
            .replace('%', '\\%').replace('\\\\', '\\') \
            .replace('\\\\count', '\\\\\\hline count')
        
    return table

#%%
def number2latex(value) -> str:
    """Format a given value depending on its order of magnitude.

    Args:
        value: Value to format as a string.

    Returns:
        str: Return value with specific format.
    """
    # Check input is a number
    if not (isinstance(value, int) or isinstance(value, float)): return value
    if not np.isfinite(value): return value

    if (value%1==0 or isinstance(value, int)) and om(value)<5:
        # If integer, show no decimals
        output = '{:d}'.format(int(value))
    elif (om(value)>-2 and om(value)<5):
        # If absolute value is in the range (0.01, 10000) show 3 decimals
        output = '{:.3f}'.format(value)
    elif om(value)>=5 or om(value)<=-2:
        # If absolute value is in the range (0, 0.01] or [10000, inf) show scientific notation with 3 decimals
        output = r'$' + '{:.3e}'.format(value).replace('e',r'\cdot10^{').replace('{+0','{').replace('{-0','{-') + r'}$'

    return output
#%%
def outliers_boundaries(data: np.ndarray, threshold: Union[tuple, float]=1.5, positive_only:bool=False) -> Union[tuple, np.ndarray, np.ndarray]:
    """Compute limits of standard data within a given data distribution.

    Args:
        data (np.ndarray): Data to get the outliers boundaries from.
        threshold (float, optional): Proportion of IQR to take into account. Defaults to 1.5.
        positive_only (bool, optional): Force negative lower boundary to be 0 if data can only be positive. Defaults to False.

    Returns:
        tuple: Range of values between which satandard data is comprised.
        np.ndarray: Array contaning standard data (non outliers).
        np.ndarray: Array containing outliers.
    """
    
    Q1 = np.quantile(data,0.25)
    Q3 = np.quantile(data,0.75)
    IQR = st.iqr(data)

    threshold = (threshold, threshold) if isinstance(threshold, float) else threshold
    
    if positive_only:
        std_lims = (max(0,(Q1-IQR*threshold[0])), (Q3+IQR*threshold[1]))
    else:
        std_lims = ((Q1-IQR*threshold[0]), (Q3+IQR*threshold[1]))

    # Get outliers and standard data
    filter_outliers = (data<std_lims[0]) | (data>std_lims[1])
    outliers = data[filter_outliers]
    std_data = data[~filter_outliers]

    return std_lims, std_data, outliers
#%%
def compute_vif(df_input: pd.DataFrame) -> pd.DataFrame:
    """Compute Variance Inflation Factor to evaluate multicolinearity.

    Args:
        df_input (pd.DataFrame): Input dataframe to compute VIF.

    Returns:
        pd.DataFrame: DataFrame with all the features as keys and VIF scores as values.
    """
    
    # Create deep copy of input DataFrame
    data = df_input.copy(deep=True).dropna()
    
    # Get the features and model object
    features = [feature for feature in data.columns if not data[feature].dtype in ['category', 'str']]
    model = LinearRegression()
    
    # Create empty list to store R2 scores
    r2_scores = []
    
    # Iterate through all features to evaluate its VIF
    warnings.filterwarnings("ignore") # Disable warnings in case of division by 0
    
    for y_feature in features:
        
        x_features = [f for f in features if not f==y_feature]
        
        x = data[x_features]
        y = data[y_feature]
        
        # Fit the model and calculate VIF
        model.fit(x, y)
        
        r2_scores.append(1/(1-model.score(x, y)))
    
    warnings.filterwarnings("default") # Restore warnings

    # Create dataframe with the results of the VIF computation
    result = pd.DataFrame(index=features, columns=['VIF'], data=r2_scores)
        
    return result
#%%
def vif_selection(df_input:pd.DataFrame, maxvif:float=5.0) -> dict:
    """Variable selection using Variance Inflation Factor (VIF) threshold.

    Args:
        df_input (pd.DataFrame): Input dataframe upon which VIF selection is performed.
        maxvif (float, optional): VIF score used to select variables. Defaults to 0.8.

    Returns:
        dict: Dictionary containing correlated and independent features with their VIF scores.
    """
    
    # Create a deep copy of the input DataFrame
    df = df_input.copy(deep=True).dropna()

    # Compute all VIF values and all features with the maximum VIF
    vif = compute_vif(df)
    vif_values = np.sort(np.unique(np.asarray(vif.values).flatten()))[::-1]
    maxvif_features = list(vif[vif['VIF']==vif_values[0]].index.values)

    correlated = {}
    while vif_values[0] > maxvif:

        # Initialize tuple to evaluate which feature (among those with max VIF) that should be removed,
        # based on the maximum VIF, number of maximum VIFs and second maximum VIF after feature removal
        collinear_feature = {'feature':maxvif_features[0],
                             'maxvif':np.inf,
                             'n_features_maxvif':len(df.columns),
                             '2nd_maxvif':np.inf}

        # Iterate over all features with maximum VIF
        for feature in maxvif_features:

            vif = compute_vif(df[[f for f in df.columns if f!= feature]])
            vif_values = np.sort(np.unique(np.asarray(vif.values).flatten()))[::-1]
            n_features_maxvif = np.sum(vif.values==vif_values[0])

            # Check if VIF analysis is producing lower values when removing the feature
            if (collinear_feature['maxvif'] > vif_values[0]) or \
               ((collinear_feature['maxvif'] == vif_values[0]) and \
               (collinear_feature['n_features_maxvif'] > n_features_maxvif)) or \
               ((collinear_feature['maxvif'] == vif_values[0]) and \
               (collinear_feature['n_features_maxvif'] == n_features_maxvif) and \
               (collinear_feature['2nd_maxvif'] >= vif_values[1])):
                
                collinear_feature['feature'] = feature
                collinear_feature['maxvif'] = vif_values[0]
                collinear_feature['n_features_maxvif'] = n_features_maxvif
                collinear_feature['2nd_maxvif'] = vif_values[1]


        # Store correlated values
        correlated[collinear_feature['feature']] = collinear_feature['maxvif']

        # Update dataframe to exclude correlated feature
        df.drop(columns=[collinear_feature['feature']], inplace=True)

        # Compute VIF values excluding the correlated feature
        vif = compute_vif(df)
        vif_values = np.sort(np.unique(np.asarray(vif.values).flatten()))[::-1]
        maxvif_features = list(vif[vif['VIF']==vif_values[0]].index.values)

    # Get information on the independent and correlated variables
    output = {'independent': vif.to_dict('index'),
              'correlated': correlated}
              
    return output
#%%
def tabular_list(input:list, n_cols:int = 3, **kwargs) -> str:
    """Format list as a tabular table in string format.

    Args:
        input_list (list): List of items to format.
        n_cols (int, optional): Number of columns to segment the list. Defaults to 3.

    Returns:
        str: String with the list shown as a table.
    """

    input_list = input.copy()

    hsep = kwargs.get('hsep', 4) # Get horizontal separation between columns. Defaults to 4.
    col_sep = kwargs.get('col_sep', f'') # Get column separator string. Defaults to empty
    alignment = kwargs.get('alignment', '<') # Get text alignment. Defaults to < (left).
    max_len = kwargs.get('max_len', max([len(item) for item in input_list])) # Get maximum length allowed for the items. Defaults to 30.
    axis = kwargs.get('axis', 1) # Get display order (0 for columns, 1 for rows)

    # Shorten item of list if it exceeds max_len parameter
    for i, item in enumerate(input_list):
        if max_len > len(item) or max_len<4: continue
        out = len(item) - max_len
        n_char = (len(item)//2+len(item)%2) - (out//2+out%2)-1
        input_list[i] = item[:n_char] + '..' + item[-(max_len - n_char - 2):]

    output_list = []

    n_rows = len(input_list)//n_cols + (1 if len(input_list)%n_cols>0 else 0)
    for r in range(n_rows):
        row = []

        col_range = np.arange(r*n_cols, r*n_cols + n_cols, 1) if axis==0 else \
                    np.arange(r, r+n_cols*n_rows, n_rows)
        
        for c in col_range:
            if len(input_list)>=c+1:
                row.append(input_list[c])
            else:
                row.append('')

        output_list.append(row) 

    output = f''
    for r, row in enumerate(output_list):
        for c, item in enumerate(row):
            if item=='': continue
            output = output + col_sep + f'{item:{alignment}{max_len + hsep}}\t'
        output = output + f'\n'

    return output
#%%

# Define progressbar class
class progressbar():
    def __init__(self, iterations, description="", desc_loc='left'):

        # Define list of sectors and range of values they apply to
        self.sectors_list = list(['', '\u258F', '\u258D', '\u258C', '\u258B', '\u258A', '\u2589', '\u2588'])
        self.sectors_range = np.linspace(0,1,8)

        self.iterations = iterations
        self.n_iterations = self.iterations if isinstance(self.iterations, int) else len(self.iterations)
        self.description = description
        self.desc_loc = desc_loc
        self.log = ""
        self.i = 0

        # Initialize start time of iteration
        self.it_start_time = None

        # Initialize average duration of iteration
        self.avg_it_duration = 0.0


    def get_progress(self):

        # Compute progress (0 to 100) and subprogress (0 to 1)
        self.progress = (self.i/self.n_iterations)
        progress = int(((self.i/self.n_iterations)*100//10))
        subprogress = (self.i/self.n_iterations*100 - progress*10)/10

        return (progress, subprogress)

    def format_time(self, duration):

        m, s = divmod(duration, 60)
        h, m = divmod(m, 60)

        if int(h)==0:
            return f'{int(m):02d}:{int(s):02d}'
        else:
            return f'{int(h):02d}:{int(m):02d}:{int(s):02d}'

    
    def refresh(self, i, description:str = None):

        # Update description if new description is given
        self.description = self.description if description==None else description
        

        if i>self.i:

            self.i = i

            # Get duration of iteration, average of duration per iteration and iterations per second.
            self.it_duration = time.time()-self.it_start_time if self.i > 1 else 0.0
            self.its_per_second = 1.0/self.it_duration if self.i > 1 else 0.0

            self.avg_it_duration = (self.avg_it_duration*(self.i-1)+self.it_duration)/self.i if self.i > 2 else self.it_duration
            
            # Compute estimated remaining time and overall duration.
            self.estimated_remaining_time = self.avg_it_duration*(self.n_iterations-self.i) if self.i > 1 else 0.0
            self.estimated_duration = self.avg_it_duration*self.n_iterations if self.i > 1 else 0.0

            # Update iteration start time
            self.it_start_time = time.time()

        # Calculate how many entire sectors and type of subsector to display 
        progress, subprogress = self.get_progress()
        sectors     = self.sectors_list[-1]*progress
        subsector   = self.sectors_list[np.sum(self.sectors_range<=subprogress)-1]

        # Create log concatenating the description and defining the end of the print log depending on the number of iteration
        log = f' {self.progress*100:>3.0f}%' + \
              f' |{sectors}{subsector}{" "*(10-len(sectors)-len(subsector))}|' + \
              f' {i:>{om(self.n_iterations)}}/{self.n_iterations:<{om(self.n_iterations)}}' + \
              f' | Remaining time: {self.format_time(self.estimated_remaining_time)}' + \
              f' ({self.its_per_second:>{om(self.its_per_second) if om(self.its_per_second)>=1 else 1}.2f} it/s) '

        log = self.description + log if self.desc_loc=='left' else f'>' + log + self.description

        # Ensure next log has the same characters so that no residuals from previous log are left in the screen.
        self.log = log + f'{" "*(len(self.log)-len(log))}' if len(self.log)>len(log) else log

        end = '\n' if i==self.n_iterations else '\r'
        
        # Print progress log
        print(self.log, end=end)
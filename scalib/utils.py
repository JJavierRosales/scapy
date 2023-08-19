# Libraries used for type hinting
from __future__ import annotations
from typing import Union

import torch
import pandas as pd
import numpy as np
import random
import math
import warnings
import time
import scipy.stats as st
from typing import Union
from sklearn.linear_model import LinearRegression
import datetime

# Get current working directory path for the tool parent folder and print it.
from pathlib import Path
import os
parent_folder = 'scalib'
cwd = str(Path(os.getcwd()[:os.getcwd().index(parent_folder)+len(parent_folder)]))

#%% FUNCTION: get_lr
def get_lr(optimizer):
    for param_group in optimizer.param_groups:
        return param_group['lr']
#%% FUNCTION: seed
def seed(seed:int=None):
    if seed is None:
        seed = int((time.time()*1e6) % 1e8)
    global _random_seed
    _random_seed = seed
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available(): torch.cuda.manual_seed(seed)

#%% FUNCTION: docstring
def docstring(item, internal_attr:bool=False, builtin_attr:bool=False) -> None:
    """Print DocString from a specific Module, Class, or Function.

    Args:
        item (_type_): Module, Class, or Function to print documentation from. 
        internal_attr (bool, optional): Flag to include internal attributes. 
        Defaults to False.
        builtin_attr (bool, optional): Flag to include built-in attributes. 
        Defaults to False.
    """

    # Initialize methods list.
    methods = []

    # Iterate over all methods available in the item.
    for method in dir(item):
        if (method.startswith('_') and internal_attr) or \
           (method.startswith('__') and builtin_attr) or \
            not method.startswith('_'):
            methods.append(method)

    # Sort methods by alphabetic order
    methods = sorted(set(methods))

    for method in methods:  
        print('Method: {}\n\n{}\n{}' \
            .format(method,getattr(item, method).__doc__, "_"*80))
#%% FUNCTION: plt_matrix
def plt_matrix(num_subplots:int) -> tuple:
    """Calculate number of rows and columns for a square matrix 
    containing subplots.

    Args:
        num_subplots (int): Number of subplots contained in the matrix.

    Returns:
        tuple: Number of rows and columns of the matrix.
    """
    if num_subplots < 5:
        return 1, num_subplots
    else:
        cols = math.ceil(math.sqrt(num_subplots))
        rows = 0
        while num_subplots > 0:
            rows += 1
            num_subplots -= cols
            
        return rows, cols
        
#%% FUNCTION: from_date_str_to_days
def from_date_str_to_days(date:datetime.datetime, 
                          date0:str='2020-05-22T21:41:31.975', 
                          date_format:str='%Y-%m-%dT%H:%M:%S.%f') -> float:
    """Convert date in string format to date.

    Args:
        date (datetime): Final date.
        date0 (str, optional): Initial date. Defaults to 
        '2020-05-22T21:41:31.975'.
        date_format (str, optional): Format of initial and final date. Defaults 
        to '%Y-%m-%dT%H:%M:%S.%f'.

    Returns:
        float: Number of days.
    """
    
    # Convert date in string format as datetime type
    date = datetime.datetime.strptime(date, date_format)
    date0 = datetime.datetime.strptime(date0, date_format)

    # Get time between final and initial dates.
    dd = date-date0
    days = dd.days
    days_fraction = (dd.seconds + dd.microseconds/1e6) / (60*60*24)

    return days + days_fraction
#%% FUNCTION: doy_2_date
def doy_2_date(value:str, doy:str, year:int, idx:int) -> str:
    """Converts Day of Year (DOY) date format to date format.

    Args:
        value (str): Original date time string with day of year format 
        "YYYY-DDDTHH:MM:SS.ff"
        doy (str): The day of year in the DOY format.
        year (int): The year.
        idx (int): Index of the start of the original "value" string at which 
        characters 'DDD' are found.

    Returns:
        str: Transformed date in traditional date format.
    """
    # Calculate datetime format
    date_num = datetime.datetime(int(year), 1, 1) + \
               datetime.timedelta(int(doy) - 1)

    # Split datetime object into a date list
    date_vec = [date_num.year, date_num.month, date_num.day, 
                date_num.hour, date_num.minute]
    
    # Extract final date string. Use zfill() to pad year, month and day fields 
    # with zeroes if not filling up sufficient spaces. 
    value = str(date_vec[0]).zfill(4) +'-' + \
            str(date_vec[1]).zfill(2) + '-' + \
            str(date_vec[2]).zfill(2) + 'T' + \
            value[idx+4:-1] 
    
    return value
#%% FUNCTION: get_ccsds_time_format
def get_ccsds_time_format(time_string:str) -> str:
    """Get the format of the datetime string in the CDM. The CCSDS time format 
    is required to be of the general form yyyy-[mm-dd|ddd]THH:MM:SS[.F*][Z]. The
    following considerations are taken into account:

    (1) The date and time fields are separated by a "T".
    (2) The date field has a four digit year followed by either a two digit 
        month and two digit day, or a three digit day-of-year.  
    (3) The year, month, day, and day-of-year fields are separated by a dash.
    (4) The hours, minutes and seconds fields are each two digits separated 
        by colons.
    (5) The fraction of seconds is optional and can have any number of
        digits.
    (6) If a fraction of seconds is provided, it is separated from the two
        digit seconds by a period.
    (7) The time string can end with an optional "Z" time zone indicator

    Args:
        time_string (str): Original time string stored in CDM. It must be of the 
        form yyyy-[mm-dd|ddd]THH:MM:SS[.F*][Z], otherwise it is invalid and a 
        RuntimeError is raised.

    Raises:
        RuntimeError: 0 or more than 1 T separator.
        RuntimeError: Datetime format not recognised.
        RuntimeError: Second decimals not identified.

    Returns:
        str: Outputs the format of the time string. 
    """

    # Count number of T separators.
    numT = time_string.count('T')

    # Raise error if found 0 or more than one separator.
    if numT == -1 or numT > 1:
        # Raise error if time_string does not contain a single 'T'.
        raise RuntimeError(f"Invalid CCSDS time string: {time_string}."
                           f" {numT} 'T' separator(s) found between date and "
                           f"time portions of the string.")
    
    # Get the position of the separator T (separates date from time).
    idx_T = time_string.find('T')

    if idx_T ==10:
        time_format = "yyyy-mm-ddTHH:MM:SS"
    elif idx_T ==8:
        time_format = "yyyy-DDDTHH:MM:SS"
    else: 
        raise RuntimeError(f"Invalid CCSDS time string: {time_string}. "
                           f"Date format not one of yyyy-mm-dd or yyyy-DDD.")
    
    # Check if 'Z' time zone indicator appended to the string
    z_opt = (time_string[-1]=='Z') 

    # Get location of the fraction of seconds decimal separator
    num_decimal = time_string.count('.')
    if num_decimal > 1:
        raise RuntimeError(f"Invalid CCSDS time string: {time_string}. "
                           f"More than one fraction of seconds decimal "
                           f"separator ('.') found.")
    
    # Get the position of the dot in the seconds decimals
    idx_decimal = time_string.find('.')

    # If datetime has seconds decimals, get the number of decimals.
    if num_decimal != 0:
        n_decimals = len(time_string) - 1 - idx_decimal - (1 if z_opt else 0)

    # Get second decimals format if CDM datetime has decimals
    frac_str = ('.' + ('F'*n_decimals)) if n_decimals > 0 else ""

    # Add Z time zone indicator if present.
    frac_str = frac_str + ('Z' if z_opt else '')

    # Join time_format, seconds fraction (if any) and time zone (if any).
    time_format = time_format + frac_str

    return time_format


#%% FUNCTION: has_nan_or_inf
def has_nan_or_inf(value: Union[float, torch.TensorFloat]) -> bool:
    """Check if value(s) contain any infinite or Not a Number (NaN).

    Args:
        value (Union[float, torch.TensorFloat]): Value(s) to check.

    Returns:
        bool: True if value(s) has infinite or NaN value(s), False otherwise.
    """

    if torch.is_tensor(value):
        value = torch.sum(value)
        isnan = int(torch.isnan(value)) > 0
        isinf = int(torch.isinf(value)) > 0
        return isnan or isinf
    else:
        value = float(value)
        return math.isnan(value) or math.isinf(value)
#%% FUNCTION: tile_rows_cols
def tile_rows_cols(num_items:int) -> tuple:
    """Get number of rows and columns a series of items can be organised to 
    follow a square shaped frame.

    Args:
        num_items (int): Number of items to tile.

    Returns:
        tuple: tuple with the number of rows and columns the items can be 
        organised in a squared shape frame.
    """

    if num_items < 5:
        return 1, num_items
    else:
        cols = math.ceil(math.sqrt(num_items))
        rows = 0
        while num_items > 0:
            rows += 1
            num_items -= cols
        return rows, cols
#%% FUNCTION: add_days_to_date_str
def add_days_to_date_str(date0:datetime, days:float) -> str:
    """Add/Substract natural date from initial date.

    Args:
        date0 (datetime): Initial date.
        days (float): Natural days to add/substract.

    Returns:
        str: Datetime as a string.
    """
    date0 = datetime.datetime.strptime(date0, '%Y-%m-%dT%H:%M:%S.%f')
    date = date0 + datetime.timedelta(days=days)

    return date.strftime('%Y-%m-%dT%H:%M:%S.%f')
#%% FUNCTION: transform_date_str
def transform_date_str(date_string:str, from_format:str, to_format:str) -> str:
    """Convert date in string format from certain format to another format.

    Args:
        date_string (str): Date as string type to change the format.
        from_format (str): Format in which the date is provided.
        to_format (str): String datetime format of conversion

    Returns:
        str: Date in the new string datetime format.
    """
    date = datetime.datetime.strptime(date_string, from_format)

    return date.strftime(to_format)
#%% FUNCTION: is_date
def is_date(date_string:str, date_format:str) -> bool:
    """Check a date in string format is an actual date.

    Args:
        date_string (str): Date in string format to check.
        date_format (str): Format string in which the date is provided.

    Returns:
        bool: True if it is an actual date, False otherwise.
    """
    try:
        datetime.datetime.strptime(date_string, date_format)
        return True
    except:
        return False
#%% FUNCTION: is_number
def is_number(s:Union[float, str]) -> bool:
    """Check if value is a number or a string.

    Args:
        s (Union[float, str]): Value to check.

    Returns:
        bool: True if is a number, False otherwise.
    """
    try:
        float(s)
        return True
    except ValueError:
        return False
#%% FUNCTION: arglocmax
def arglocmax(a:np.ndarray) -> np.ndarray:
    """Find the indeces of the local maximums of an array.

    Args:
        a (np.ndarray): Numpy array to check.

    Returns:
        np.ndarray: Indeces of the local maximum.
    """

    condition = np.r_[True, a[1:] > a[:-1]] & np.r_[a[:-1] > a[1:], True]
    index_array = np.asarray([i for i, c in enumerate(condition) if c], 
                             dtype=np.int32)

    return index_array

#%% FUNCTION: arglocmin
def arglocmin(a:np.ndarray) -> np.ndarray:
    """Find the indeces of the local minimums of an array.

    Args:
        a (np.ndarray): Numpy array to check.

    Returns:
        np.ndarray: Indeces of the local maximum.
    """

    condition = np.r_[True, a[1:] < a[:-1]] & np.r_[a[:-1] < a[1:], True]
    index_array = np.asarray([i for i, c in enumerate(condition) if c], 
                             dtype=np.int32)

    return index_array

#%% FUNCTION: nbins
def nbins(data:np.ndarray, rule:str = 'fd') -> dict:
    """Calculate number of bins and bin width for histograms.

    Args:
        data (array_like): Array containing all data.
        rule (str): Rule to use to compute the bin size. It can be one of the 
        following options:
            - Sturge ('sturge')
            - Scott ('scott')
            - Rice ('rice')
            - Freedman-Diaconis ('fd') - Default

    Returns:
        dict: Dictionary containing with the number of bins 'n' and bin width 
        'width'.
    """

    # Check if rule passed by user is valid
    if not rule in ['sturge', 'scott', 'rice', 'fd']: return None

    # Get the number of items within the dataset
    isinteger = isinstance(data[0], np.integer)
    data = data.astype(np.float64)
    n = len(data)
    
    # Compute the histogram bins size (range)
    bins_width = {'sturge': 1 + 3.322*np.log(n),    
                 'scott': 3.49*np.std(data)*n**(-1.0/3.0),                       
                 'rice': 2*n**(1.0/3.0),                         
                 'fd': 2*st.iqr(data)*n**(-1.0/3.0)}
    
    # Compute number of bins
    n_bins =  math.ceil((data.max() - data.min())/bins_width[rule])

    # Compute range of bins
    if isinteger:
        steps = int(math.ceil((data.max() - data.min())/n_bins))
        bins_range = range(int(data.min()), int(data.max()), steps)
    else:
        bins_range = np.linspace(data.min(),data.max()+bins_width[rule], n_bins)
    
    return {'n': n_bins, 'width': bins_width[rule], 'range': bins_range}
#%% FUNCTION: om
def om(value: float) -> int:
    """Get order of magnitude of value.

    Args:
        value (float): Value to get the order of magnitude from.

    Returns:
        int: Order of magnitude.
    """
    if not (isinstance(value, np.float) or \
            isinstance(value, np.integer) or \
            isinstance(value, int)): 
        return np.nan
    if value==0 or np.isnan(value): return 0
    if abs(value)==np.inf: return np.inf
    
    return int(math.floor(math.log(abs(value), 10)))
#%% FUNCTION: round_by_om
def round_by_om(value:float, abs_method:str='ceil', **kwargs) -> float:
    """Round up/down float by specifying rounding of magnitude.

    Args:
        value (float): Value to round up/down.
        abs_method (str, optional): Method to round considering the absolute 
        value (up or down). Defaults to 'ceil'.

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

#%% FUNCTION: df2latex
def df2latex(df: pd.DataFrame, column_format:str='c') -> str:
    """Convert pandas DataFrame to latex table.

    Args:
        df (pd.DataFrame): DataFrame to convert to LaTeX format.
        column_format (str, optional): Columns alignment (left 'l', center 'c', 
        or right 'r'). Defaults to 'c'.

    Returns:
        str: DataFrame in string format.
    """

    column_format = 'c'*(len(df.columns)+1) if column_format=='c' \
        else column_format

    new_column_names = dict(zip(df.columns, 
                            ["\textbf{" + c + "}" for c in df.columns]))
    
    df.rename(new_column_names, axis='columns', inplace=True)
    
    table = df.style.to_latex(column_format=column_format)
    table = table.replace('\n', '').encode('unicode-escape').decode()\
            .replace('%', '\\%').replace('\\\\', '\\') \
            .replace('\\\\count', '\\\\\\hline count')
        
    return table

#%% FUNCTION: number2latex
def number2latex(value) -> str:
    """Format a given value depending on its order of magnitude.

    Args:
        value: Value to format as a string.

    Returns:
        str: Return value with specific format.
    """

    output = f'{value}'

    # Check input is a number
    if not (isinstance(value, np.integer) or \
            isinstance(value, np.float)): return output
    if not np.isfinite(value): return output

    if (value%1==0 or isinstance(value, np.integer)) and om(value)<=5:
        # If integer, show no decimals
        output = '{:d}'.format(int(value))
    elif (om(value)>-2 and om(value)<5):
        # If absolute value is in the range (0.01, 10000) show 3 decimals
        output = '{:.3f}'.format(value)
    elif om(value)>5 or om(value)<=-2:
        # If absolute value is in the range (0, 0.01] or [10000, inf) show 
        # scientific notation with 3 decimals
        output = r'$' + '{:.3e}'.format(value).replace('e',r'\cdot10^{')
        output = output.replace('{+0','{').replace('{-0','{-') + r'}$'

    return output
#%% FUNCTION: outliers_boundaries
def outliers_boundaries(data: np.ndarray, threshold: Union[tuple, float]=1.5, 
        positive_only:bool=False) -> Union[tuple, np.ndarray, np.ndarray]:
    """Compute limits of standard data within a given data distribution.

    Args:
        data (np.ndarray): Data to get the outliers boundaries from.
        threshold (float, optional): Proportion of IQR to take into account. 
        Defaults to 1.5.
        positive_only (bool, optional): Force negative lower boundary to be 0 if
         data can only be positive. Defaults to False.

    Returns:
        tuple: Range of values between which satandard data is comprised.
        np.ndarray: Array contaning standard data (non outliers).
        np.ndarray: Array containing outliers.
    """
    
    Q1 = np.quantile(data,0.25)
    Q3 = np.quantile(data,0.75)
    IQR = st.iqr(data)

    threshold = (threshold, threshold) if isinstance(threshold, float) \
        else threshold
    
    if positive_only:
        std_lims = (max(0,(Q1-IQR*threshold[0])), (Q3+IQR*threshold[1]))
    else:
        std_lims = ((Q1-IQR*threshold[0]), (Q3+IQR*threshold[1]))

    # Get outliers and standard data
    filter_outliers = (data<std_lims[0]) | (data>std_lims[1])
    outliers = data[filter_outliers]
    std_data = data[~filter_outliers]

    return std_lims, std_data, outliers
#%% FUNCTION: compute_vif
def compute_vif(df_input: pd.DataFrame) -> pd.DataFrame:
    """Compute Variance Inflation Factor to evaluate multicolinearity.

    Args:
        df_input (pd.DataFrame): Input dataframe to compute VIF.

    Returns:
        pd.DataFrame: DataFrame with all the features as keys and VIF scores as 
        values.
    """
    
    # Create deep copy of input DataFrame
    data = df_input.copy(deep=True).dropna()
    
    # Get the features and model object
    features = [feature for feature in data.columns \
                if not data[feature].dtype in ['category', 'str']]

    model = LinearRegression()
    
    # Create empty list to store R2 scores
    r2_scores = []
    
    # Iterate through all features to evaluate its VIF
    warnings.filterwarnings("ignore") # Disable warnings if division by 0
    
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
#%% FUNCTION: vif_selection
def vif_selection(df_input:pd.DataFrame, maxvif:float=5.0) -> dict:
    """Variable selection using Variance Inflation Factor (VIF) threshold.

    Args:
        df_input (pd.DataFrame): Input dataframe.
        maxvif (float, optional): Maximum VIF score. Defaults to 0.8.

    Returns:
        dict: Dictionary containing correlated and independent features with 
        their VIF scores.
    """
    
    # Create a deep copy of the input DataFrame
    df = df_input.copy(deep=True).dropna()

    # Compute all VIF values and all features with the maximum VIF
    vif = compute_vif(df)
    vif_values = np.sort(np.unique(np.asarray(vif.values).flatten()))[::-1]
    maxvif_features = list(vif[vif['VIF']==vif_values[0]].index.values)

    correlated = {}
    while vif_values[0] > maxvif:

        # Initialize tuple to evaluate which feature (among those with max VIF) 
        # that should be removed, based on the maximum VIF, number of maximum 
        # VIFs and second maximum VIF after feature removal.
        collinear_feature = {'feature':maxvif_features[0],
                             'maxvif':np.inf,
                             'n_features_maxvif':len(df.columns),
                             '2nd_maxvif':np.inf}

        # Iterate over all features with maximum VIF
        for feature in maxvif_features:

            vif = compute_vif(df[[f for f in df.columns if f!= feature]])
            vif_values = np.sort(np.unique(np.asarray(vif.values).flatten()))[::-1]
            n_features_maxvif = np.sum(vif.values==vif_values[0])

            # Check if VIF analysis is producing lower values when removing the 
            # feature
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
#%% FUNCTION: tabular_list
def tabular_list(input:list, n_cols:int = 3, **kwargs) -> str:
    """Format list as a tabular table in string format.

    Args:
        input_list (list): List of items to format.
        n_cols (int, optional): Number of columns to print. Defaults to 3.

    Returns:
        str: String with the list shown as a table.
    """

    input_list = input.copy() 

    # Get column separator string. Defaults to empty.
    col_sep = kwargs.get('col_sep', f'') 

    # Get horizontal separation between columns. Defaults to 4.
    hsep = kwargs.get('hsep', len(col_sep)+2)

    # Get text alignment. Defaults to < (left).
    alignment = kwargs.get('alignment', '<') 

    # Get maximum number of chars allowed per column to not exceed 80 chars 
    # width.
    chars_per_col = (80-(n_cols-1)*hsep)//n_cols
    if chars_per_col > 6:
        # Get maximum length allowed for the items. Defaults to 30.
        max_len = kwargs.get('max_len', chars_per_col)
    else:
        max_len = kwargs.get('max_len', min([len(item) for item in input_list]))

    # Get display order (0 for columns, 1 for rows).
    axis = kwargs.get('axis', 1) 

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
#%% CLASS: ProgressBar
class ProgressBar():
    def __init__(self, iterations:Union[int,list], description:str=""):

        # Define list of sectors and range of values they apply to
        self._sectors_list = list(['', '\u258F', '\u258D', '\u258C', '\u258B', 
                                  '\u258A', '\u2589', '\u2588'])
        self._sectors_range = np.linspace(0,1,8)

        self.iterations = iterations

        if isinstance(self.iterations, np.integer):
            self._n_iterations = self.iterations
        else: 
            self._n_iterations = len(self.iterations)
        self.description = description

        self._header = None
        self._log = ""
        self._i = 0

        # Initialize start time of iteration
        self._it_start_time = time.time()
        self._start_time = time.time()

        # Initialize average duration of iteration
        self.avg_it_duration = 0.0


    def get_progress(self):

        # Compute progress (0 to 100) and subprogress (0 to 1)
        self._progress = (self._i/self._n_iterations)
        progress = int(((self._i/self._n_iterations)*100//10))
        subprogress = (self._i/self._n_iterations*100 - progress*10)/10

        return (progress, subprogress)

    @staticmethod
    def format_time(duration:float) -> str:

        m, s = divmod(duration, 60)
        h, m = divmod(m, 60)

        return f'{int(h):02d}h:{int(m):02d}m:{int(s):02d}s'

    
    def refresh(self, i:int, description:str = None, 
            nested_progress:bool = False) -> None:

        # Update description if new description is given
        if description == None:
            self.description = self.description  
        else: 
            self.description = description

        # Get order of magnitude of the number of iterations and number of 
        # iterations per second to improve readability of the log message.
        om_iterations = om(self._n_iterations+1) + 1
        
        if i > self._i:

            if i==1 and self._header is None:
                self._header = f"\n| {'Progress':<{19 + 2*om_iterations+1}}" + \
                    f" | {'Time':^11} | " + \
                    f"{'Iters/sec':^{max(9, om_iterations+3)}} | " + \
                    f"{'Comments':<}"
                print(self._header)

            self._i = i

            # Get duration of iteration, average of duration per iteration and 
            # iterations per second.
            self._it_duration = time.time() - self._it_start_time \
                               if self._i > 1 else 0.0
            self.avg_it_duration = (time.time() - self._start_time)/self._i
            self._its_per_second = 1.0/self.avg_it_duration
            
            # Compute estimated time remaining and overall duration.
            self.ert = self.avg_it_duration * \
                (self._n_iterations - self._i) if self._i > 1 else 0.0
            self.edt = self.avg_it_duration*self._n_iterations \
                if self._i > 1 else 0.0

            # Update iteration start time
            self._it_start_time = time.time()

        # Calculate how many entire sectors and type of subsector to display 
        progress, subprogress = self.get_progress()
        sectors   = self._sectors_list[-1]*progress

        # Get number of 8th sections as subsectors
        idx_subsector = np.sum(self._sectors_range <= subprogress) - 1
        subsector = self._sectors_list[idx_subsector]

        # Check if it is the last iteration.
        last_iter = (i==self._n_iterations and not nested_progress)

        # Create log concatenating the description and defining the end of the 
        # print log depending on the number of iteration
        pb_progress = f'{self._progress*100:>3.0f}%'
        pb_bar = f'|{sectors}{subsector}{" "*(10-len(sectors)-len(subsector))}|'
        pb_counter = f'{i}/{self._n_iterations}'
        pb_iter = f'{self._its_per_second:.2f}'
        pb_time = f'{self.format_time(self.edt)}' if last_iter \
             else f'{self.format_time(self.ert)}'
        
        log = f"| {pb_progress:>4} {pb_bar:^12} " + \
              f"({pb_counter:>{2*om_iterations+1}})" + \
              f"| {pb_time:^11} " + \
              f"| {pb_iter:^{max(9, om_iterations+3)}} " + \
              f"| {self.description}"

        # Ensure next log has the same number of characters to so that no 
        # residuals from previous log are left in the screen.
        if len(self._log) > len(log):
            self._log = f'{log:<{len(self._log)}}'
        else:
            self._log = log

        # Determine end character depending on the number of iterations.
        end = '\n' if (i==self._n_iterations and not nested_progress) else '\r'
        
        # Print progress log
        print('{:<}'.format(self._log), end = end)
import os
import calendar
import pandas as pd
import warnings
import numpy as np
import operator
from collections import defaultdict

warnings.filterwarnings('ignore')

# Global variables
global center, event, months
center = 'LARC'
event = 'CDHE'
months = [6, 7, 8]

# Initialization
month_dict = {m: calendar.month_name[m].upper()[0] for m in range(1, 13)}
hist = (1961, 1990) # Historical range
temporal_res = ['daily', 'monthly_avg', 'annual_avg'][0] # Temporal resolution 
variables = ['pr_', 'tasmax_']

comp_ops = { # Comparative operators
    '<': operator.lt, '<=': operator.le,
    '>': operator.gt, '>=': operator.ge,
}


def get_files(center: str, variables: list = None):
    '''
    Returns list of filenames in a directory that contain a specific string.

    Args:
        center: the NASA center name in the filename
        variables: optional list of variables to filter files (default: None)
    '''
    return [os.path.join(center, f) for f in os.listdir(center) 
            if center in f and f.endswith('.csv') and temporal_res in f
            and any(v in f for v in variables)]


def preprocess_file(filename: str, is_hist: bool, percentiles:dict):
    '''
    Returns preprocessed pandas DataFrame based on CSV file type (historical or SSP)
    
    Args:
        filename: Path to the CSV file
        is_hist: Boolean indicating whether the file is historical (True) or SSP (False)
    
    Returns:
        - If historical: Processed DataFrame
        - If SSP: (variable name, processed DataFrame)
    '''
    name = filename[:-4].split('_')  # Extract variable name
    
    df = pd.read_csv(filename).rename(columns={'Unnamed: 0': 'date'})
    df['date'] = pd.to_datetime(df['date'])
    df = df.set_index('date')
    
    if name[1] in ['pr']:
        df = df * 86400 # Convert to ml/day
        # Calculate monthly mean per year while keeping original df dimensions
        if event == 'CDHE':
            df = df.groupby(df.index.strftime('%Y-%m')).transform('mean')
    
    if name[1] in ['tasmax', 'tas']:
        df = df - 273.15 # Convert Kelvin to Celsius

    if is_hist:
        # Calculate percentiles for each model
        op, p = percentiles[name[1]]
        df = df[(hist[0] <= df.index.year) & (df.index.year <= hist[1])]
        # df = df.groupby(df.index.strftime('%Y-%m')).mean()
        return pd.DataFrame({f'{name[1]}_{op}_{p}': df.quantile(p)})
    else: 
        df = df[df.index.month.isin(months)]        
        return ('_'.join(name[1:3]), df.add_suffix(f'_{name[1]}'))


def add_compound_flag(hist_df:pd.DataFrame, ssp_df:pd.DataFrame, percentiles:dict):
    '''
    Returns SSP dataframe with added compound flag based on historical percentiles
    
    Args:
        hist_df: Historical DataFrame
        ssp_df: SSP DataFrame
    '''
    for model in hist_df.index:
        flag = []
        for var, (op, p) in percentiles.items():
            # Check if ssp values exceed historical percentiles
            perc = hist_df[f'{var}_{op}_{p}'].loc[model]
            ssp_values = ssp_df[f'{model}_{var}'].to_list()
            
            # Apply the comparison function
            ssp_df[f'{model}_{var}_{op}_{p}'] = comp_ops[op](ssp_values, perc)
            flag.append(ssp_df[f'{model}_{var}_{op}_{p}'].to_list())
        
            # Calculate diff
            # ssp_df[f'{model}_{var}_diff'] = ssp_df[f'{model}_{var}'] - perc
        
        ssp_df[f'{model}_compound'] = np.all(flag, axis=0)
    return ssp_df


def process_data(files, percentiles):
    """
    Process historical and SSP climate data files.
    
    Args:
    files (dict): Dictionary containing 'historical' and 'ssp' file lists
    percentiles (dict): Dictionary of percentile thresholds for each variable
    
    Returns:
    tuple: (hist_df, ssp_dfs) where hist_df is the historical DataFrame and 
           ssp_dfs is a dictionary of SSP DataFrames
    """
    # Process historical files and merge into a single DataFrame
    hist_df = pd.concat([preprocess_file(f, True, percentiles) 
                         for f in files['historical']], axis=1).dropna()

    # Process SSP files and store in a dictionary
    ssp_dfs = {name: df for name, df in (preprocess_file(f, False, percentiles) 
                                         for f in files['ssp'])}

    # Merge all SSP data from the same scenario
    grouped_data = defaultdict(list)
    for key, df in ssp_dfs.items():
        grouped_data[key.split('_')[-1]].append(df)

    # Add flag to indicate compound events    
    ssp_dfs = {ssp: add_compound_flag(hist_df, pd.concat(dfs, axis=1), percentiles) 
               for ssp, dfs in grouped_data.items()}
    
    return hist_df, ssp_dfs


def max_consecutive(s):
    # Calculate consecutive Trues    
    return (s * (s.groupby((s != s.shift()).cumsum()).cumcount() + 1)).max()


def get_data(dfs:dict, suffix:str, agg:str, duration:bool=False):
    dict_df = {}
    for m, df in dfs.items():
        df_ = df.filter(regex=suffix)
        df_.index = pd.to_datetime(df_.index, format='%Y')
        if duration:
            df_ = df_.groupby(df_.index.year).apply(lambda x: x.apply(max_consecutive))
        else:
            df_ = df_.groupby(df_.index.year).agg(agg)
        dict_df[m] = df_
    return dict_df


def main(center_, event_, months_):
    global center, event, months
    center = center_
    event = event_
    months = months_

    # Percentile thresholds
    if event == 'CDHE':
        pr_op, tm_op = '<', '>'
        thresholds = [
            {'pr': [pr_op, 0.5], 'tasmax': [tm_op, 0.90]}, # base
            {'pr': [pr_op, 0.5], 'tasmax': [tm_op, 0.75]}, # least severe
            {'pr': [pr_op, 0.4], 'tasmax': [tm_op, 0.80]},
            {'pr': [pr_op, 0.3], 'tasmax': [tm_op, 0.85]},
            {'pr': [pr_op, 0.2], 'tasmax': [tm_op, 0.90]},
            {'pr': [pr_op, 0.1], 'tasmax': [tm_op, 0.95]}, # most severe
        ]
    elif event == 'CWHE':
        pr_op, tm_op = '>', '>'
        thresholds = [
            {'pr': [pr_op, 0.5], 'tasmax': [tm_op, 0.90]}, # base
            {'pr': [pr_op, 0.5], 'tasmax': [tm_op, 0.75]}, # least severe
            {'pr': [pr_op, 0.6], 'tasmax': [tm_op, 0.80]},
            {'pr': [pr_op, 0.7], 'tasmax': [tm_op, 0.85]},
            {'pr': [pr_op, 0.8], 'tasmax': [tm_op, 0.90]},
            {'pr': [pr_op, 0.9], 'tasmax': [tm_op, 0.95]}, # most severe
        ]
    else: 
        raise ValueError("Invalid event type")

    files = sorted([f for f in get_files(center, variables)])
    files = {key: [f for f in files if key in f] for key in ['historical', 'ssp']}
    
    dfs, hist, frequency, duration, pr, tm = {}, {}, {}, {}, {}, {}
    for threshold in thresholds:
        name = '_'.join([f'{v}{c}{p}' for v, (c, p) in threshold.items()])
        hist[name], dfs[name] = process_data(files, threshold)
        
        frequency[name] = get_data(dfs[name], f'_compound$', 'sum', False)
        duration[name] = get_data(dfs[name], f'_compound$', 'mean', True)
        pr[name] = get_data(dfs[name], f'_pr$', 'mean', False)
        tm[name] = get_data(dfs[name], f'_tasmax$', 'mean', False)
        
    return dfs, hist, frequency, duration, pr, tm

if __name__ == "__main__":
    
    results = main(center, event, months)
    dfs, hist, frequency, duration, pr, tm = results

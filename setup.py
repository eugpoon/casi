import os
import calendar
from typing import List, Dict, Tuple
from collections import defaultdict

import pandas as pd
import numpy as np
import operator
import warnings

warnings.filterwarnings('ignore')


global CENTER, EVENT, MONTHS
CENTER, EVENT, MONTHS = '', '', []

# Initialization
DATA_PATH = 'compound'
month_dict = {m: calendar.month_name[m].upper()[0] for m in range(1, 13)}
HIST = (1961, 1990) # Historical range
TEMPORAL_RES = ['daily', 'monthly_avg', 'annual_avg'][0] # Temporal resolution 
VARIABLES = ['pr_', 'tasmax_']

COMP_OPS = { # Comparative operators
    '<': operator.lt, '<=': operator.le,
    '>': operator.gt, '>=': operator.ge,
}


def get_files() -> List[str]:
    '''Returns list of filenames in a directory that contain a specific string'''
    return [os.path.join(DATA_PATH, f) for f in os.listdir(DATA_PATH) 
            if CENTER in f and f.endswith('.csv') and TEMPORAL_RES in f
            and any(v in f for v in VARIABLES)]

def validate_file_path(file_path: str) -> None:
    '''Validate if the file path exists.'''
    if not os.path.exists(file_path):
        raise FileNotFoundError(f'The file {file_path} does not exist.')

def preprocess_file(filename: str, is_hist: bool, percentiles: Dict[str, Tuple[str, float]]) -> pd.DataFrame:
    '''Returns preprocessed DataFrame based on CSV file type (historical or SSP)'''
    validate_file_path(filename)
    
    name = filename[:-4].split('_')  # Extract variable name
    
    df = pd.read_csv(filename).rename(columns={'Unnamed: 0': 'date'})
    df['date'] = pd.to_datetime(df['date'])
    df = df.set_index('date')
    
    if name[1] in ['pr']:
        df *= 86400 # Convert to ml/day
        # Calculate monthly mean per year while keeping original df dimensions
        if EVENT == 'CDHE':
            df = df.groupby(df.index.strftime('%Y-%m')).transform('mean')
    
    if name[1] in ['tasmax', 'tas']:
        df -= 273.15 # Convert Kelvin to Celsius

    if is_hist:
        # Calculate percentiles for each model
        op, p = percentiles[name[1]]
        df = df[(HIST[0] <= df.index.year) & (df.index.year <= HIST[1])]
        return pd.DataFrame({f'{name[1]}_{op}_{p}': df.quantile(p)})
    else: 
        df = df[df.index.month.isin(MONTHS)]     
        return ('_'.join(name[1:3]), df.add_suffix(f'_{name[1]}'))


def add_compound_flag(hist_df: pd.DataFrame, ssp_df: pd.DataFrame, percentiles: Dict[str, Tuple[str, float]]) -> pd.DataFrame:
    '''Returns SSP dataframe with added compound flag based on historical percentiles'''
    for model in hist_df.index:
        flag = []
        for var, (op, p) in percentiles.items():
            # Check if ssp values exceed historical percentiles
            perc = hist_df[f'{var}_{op}_{p}'].loc[model]
            ssp_values = ssp_df[f'{model}_{var}'].to_list()
            
            # Apply the comparison function
            ssp_df[f'{model}_{var}_{op}_{p}'] = COMP_OPS[op](ssp_values, perc)
            flag.append(ssp_df[f'{model}_{var}_{op}_{p}'].to_list())
        
            # Calculate diff
            # ssp_df[f'{model}_{var}_diff'] = ssp_df[f'{model}_{var}'] - perc
        
        ssp_df[f'{model}_compound'] = np.all(flag, axis=0)
    return ssp_df

def setup_thresholds(event: str) -> List[Dict[str, Tuple[str, float]]]:
    '''Set up thresholds for different severity levels of compound events.'''    
    if event == 'CDHE':
        pr_op, tm_op = '<', '>'
        return [
            {'pr': (pr_op, 0.5), 'tasmax': (tm_op, 0.90)},  # base
            {'pr': (pr_op, 0.5), 'tasmax': (tm_op, 0.75)},  # least severe
            {'pr': (pr_op, 0.4), 'tasmax': (tm_op, 0.80)},
            {'pr': (pr_op, 0.3), 'tasmax': (tm_op, 0.85)},
            {'pr': (pr_op, 0.2), 'tasmax': (tm_op, 0.90)},
            {'pr': (pr_op, 0.1), 'tasmax': (tm_op, 0.95)},  # most severe
        ]
        
    elif event == 'CWHE':
        pr_op, tm_op = '>', '>'
        return [
            {'pr': (pr_op, 0.5), 'tasmax': (tm_op, 0.90)},  # base
            {'pr': (pr_op, 0.5), 'tasmax': (tm_op, 0.75)},  # least severe
            {'pr': (pr_op, 0.6), 'tasmax': (tm_op, 0.80)},
            {'pr': (pr_op, 0.7), 'tasmax': (tm_op, 0.85)},
            {'pr': (pr_op, 0.8), 'tasmax': (tm_op, 0.90)},
            {'pr': (pr_op, 0.9), 'tasmax': (tm_op, 0.95)},  # most severe
        ]
        
    else:
        raise ValueError('Invalid event type')
        
def process_data(files: Dict[str, List[str]], percentiles: Dict[str, Tuple[str, float]]) -> Tuple[pd.DataFrame, Dict[str, pd.DataFrame]]:
    '''Process historical and SSP climate data files'''
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


def max_consecutive(s: pd.Series) -> int:
    '''Calculate maximum consecutive True values in a series.'''
    return (s * (s.groupby((s != s.shift()).cumsum()).cumcount() + 1)).max()


def get_data(dfs: Dict[str, pd.DataFrame], suffix: str, agg: str, duration: bool = False) -> Dict[str, pd.DataFrame]:
    '''Aggregate data from DataFrames based on specified criteria'''
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


def main(center: str, event: str, months: List[int]) -> Tuple[Dict, Dict, Dict, Dict, Dict, Dict]:
    '''Main function to process climate data'''
    global CENTER, EVENT, MONTHS
    CENTER, EVENT, MONTHS = center, event, months

    thresholds = setup_thresholds(EVENT)

    files = {key: [f for f in sorted(get_files()) if key in f] for key in ['historical', 'ssp']}
    
    dfs, hist, frequency, duration, pr, tm = {}, {}, {}, {}, {}, {}
    for threshold in thresholds:
        name = '_'.join([f'{v}{c}{p}' for v, (c, p) in threshold.items()])
        hist[name], dfs[name] = process_data(files, threshold)
        
        frequency[name] = get_data(dfs[name], f'_compound$', 'sum', False)
        duration[name] = get_data(dfs[name], f'_compound$', 'mean', True)
        pr[name] = get_data(dfs[name], f'_pr$', 'mean', False)
        tm[name] = get_data(dfs[name], f'_tasmax$', 'mean', False)
        
    return dfs, hist, frequency, duration, pr, tm

if __name__ == '__main__':
    dfs, hist, frequency, duration, pr, tm = main(center = 'LARC', event='CDHE', months='[6, 7, 8]')

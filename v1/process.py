import os, calendar, warnings
from typing import List, Dict, Tuple
from collections import defaultdict
from functools import reduce
import pandas as pd
import numpy as np
import operator
warnings.filterwarnings('ignore')


global CENTER, EVENT, MONTHS
CENTER, EVENT, MONTHS = '', '', []

DATA_PATH = '../data/compound'
HIST = (1961, 1990) # Historical range
TEMPORAL_RES = ['daily', 'monthly_avg', 'annual_avg'][0] # Temporal resolution 
VARIABLES = ['pr_', 'tasmax_']

COMP_OPS = { # Comparative operators
    '<': operator.lt, '<=': operator.le,
    '>': operator.gt, '>=': operator.ge,
}


def get_files():
    '''Returns list of filenames in a directory that contain a specific string'''
    return [os.path.join(DATA_PATH, f) for f in os.listdir(DATA_PATH) 
            if CENTER in f and f.endswith('.csv') and TEMPORAL_RES in f
            and any(v in f for v in VARIABLES)]


def validate_file_path(file_path: str) -> None:
    '''Validate if the file path exists.'''
    if not os.path.exists(file_path):
        raise FileNotFoundError(f'{file_path} does not exist.')


def preprocess_file(filename, is_hist, percentiles=None):
    '''Returns preprocessed DataFrame based on CSV file type (historical or SSP)'''
    validate_file_path(filename)
    name = filename[:-4].split('/')[-1].split('_')
    
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
        # df = df.groupby(df.index.strftime('%Y-%m')).mean()
        return pd.DataFrame({f'{name[1]}_{op}_{p}': df.quantile(p)})
    else: 
        df = df[df.index.month.isin(MONTHS)]
        df.insert(0, 'ssp', name[2])
        return '_'.join(name[1:3]), df


def setup_thresholds(event):
    '''Set up thresholds for different severity levels of compound events.'''    
    if event == 'CDHE':
        pr_op, tm_op = '<', '>'
        return [{'pr': (pr_op, p), 'tasmax': (tm_op, q)} for p, q in 
                # base; least --> most severe
                zip([0.50, 0.50, 0.40, 0.30, 0.20, 0.10],  # pr
                    [0.90, 0.75, 0.80, 0.85, 0.90, 0.95])] #tm  
    elif event == 'CWHE':
        pr_op, tm_op = '>', '>'
        return [{'pr': (pr_op, p), 'tasmax': (tm_op, q)} for p, q in 
                # base; least --> most severe
                zip([0.50, 0.50, 0.60, 0.70, 0.80, 0.90],  # pr
                    [0.90, 0.75, 0.80, 0.85, 0.90, 0.95])] # tm
    else:
        raise ValueError('Invalid event type')


def add_compound_flag(hist_df, ssp_df, percentiles):
    '''Returns SSP dataframe with added compound flag based on historical percentiles'''
    df = ssp_df[['date', 'ssp']]
    for model in hist_df.index:
        for var, (op, p) in percentiles.items():
            perc = hist_df[f'{var}_{op}_{p}'].loc[model]
            df[f'{model}_{var}_{op}_{p}'] = COMP_OPS[op](ssp_df[f'{model}_{var}'], perc)
        df[f'{model}_compound'] = np.all(df[[f'{model}_{var}_{op}_{p}' 
                                             for var, (op, p) in percentiles.items()]], axis=1)
    return df
    

def max_consecutive(s):
    '''Calculate maximum consecutive True values in a series.'''
    return (s * (s.groupby((s != s.shift()).cumsum()).cumcount() + 1)).max()


def group_data(df, suffix):
    '''Group data based on specified criteria'''
    df_ = df.copy()
    df_.date = df_.date.dt.year
    return (df_.set_index(['ssp', 'date']).filter(regex=suffix)
           .groupby(['ssp', 'date']))


def get_common_columns(dfs):
    return reduce(lambda x, y: x.intersection(y), (df.columns for df in dfs))


def main(center, event, months):

    global CENTER, EVENT, MONTHS
    CENTER, EVENT, MONTHS = center, event, months

    thresholds = setup_thresholds(EVENT)

    files = {key: [f for f in sorted(get_files()) if key in f] for key in ['historical', 'ssp']}
    
    # Process SSP files
    ssp_dfs = {name: df for name, df in (preprocess_file(f, False, None) for f in files['ssp'])}
    results = defaultdict(lambda: pd.DataFrame())
    for k, df in ssp_dfs.items():
        ssp = k.split('_')[0]
        results[ssp] = pd.concat([results[ssp], df], axis=0)
        
    cols = get_common_columns(list(results.values()))

    ssp_dfs = None
    for key, df in results.items():
        df = df[cols].reset_index()
        df.rename(columns={c: f'{c}_{key}' for c in df.columns 
                           if c not in ['date', 'ssp']}, inplace=True)
        ssp_dfs = df if ssp_dfs is None else ssp_dfs.merge(df, on=['date', 'ssp'])

    comp, hist, results = {}, {}, {}
    for threshold in thresholds:
        name = '_'.join([f'{v}{c}{p}' for v, (c, p) in threshold.items()])

        hist[name] = pd.concat([preprocess_file(f, True, threshold) 
                         for f in files['historical']], axis=1).dropna() 
        comp[name] = add_compound_flag(hist[name], ssp_dfs, threshold)

        comp_ = group_data(comp[name], f'_compound$')
        results[name] = comp_.sum().add_suffix('_total')
        results[name] = pd.concat(
            [comp_.apply(lambda x: x.apply(max_consecutive)).add_suffix('_duration'), 
             results[name]], axis=1)
        
    pr = group_data(ssp_dfs, f'_pr$').mean()
    tm = group_data(ssp_dfs, f'_tasmax$').mean()

    return ssp_dfs, hist, comp, results, pr, tm

if __name__ == '__main__':
    ssp_dfs, hist, comp, results, pr, tm = main(center = 'LARC', event='CDHE', months='[6, 7, 8]')





















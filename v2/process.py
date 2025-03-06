import os, calendar, warnings
from typing import List, Dict, Tuple
from collections import defaultdict
from functools import reduce
import pandas as pd
import numpy as np
import operator
from standard_precip.spi import SPI
warnings.filterwarnings('ignore')

##################################################
##            Variable initialization           ##
##################################################

DATA_PATH = '../compound'
HISTORICAL_YEARS, SSP_YEARS = (1981, 2020), (2021, 2100)
SPI_YEARS = (HISTORICAL_YEARS[0], SSP_YEARS[1])
VARIABLES = ['pr_', 'tasmax_'] # original variable in file names

COMP_OPS = { # Comparative operators
    '<': operator.lt, '<=': operator.le,
    '>': operator.gt, '>=': operator.ge,
}

def initialize(center_, event_, months_, freq_, scale_):
    global center, event, months, freq, scale, temporal_res, files, thresholds

    if freq_ == 'D':
        temporal_res = 'daily'
    elif freq_ == 'M':
        temporal_res = 'monthly_avg'
    else: 
        raise ValueError(f"{freq} should be one of ['D', 'M']")
    
    center, event, months, freq, scale = center_, event_, months_, freq_, scale_
    files = get_files()
    thresholds = setup_thresholds(event)
    
##################################################
##              Get and read files              ##
##################################################

def get_files():
    '''Returns list of filenames in a directory that contain a specific string'''
    return [os.path.join(DATA_PATH, f) for f in os.listdir(DATA_PATH) 
            if center in f and f.endswith('.csv') and temporal_res in f
            and any(v in f for v in VARIABLES)]

def validate_file_path(file_path: str) -> None:
    '''Validate if the file path exists.'''
    if not os.path.exists(file_path):
        raise FileNotFoundError(f'{file_path} does not exist.')

def read_data(filename):
    df = pd.read_csv(filename).rename(columns={'Unnamed: 0': 'date'})
    df['date'] = pd.to_datetime(df['date'])
    return df.sort_values('date').interpolate(method='linear', limit_direction='both')

def filter_dates(df, year_range:tuple, spi=False):
    if spi: # dates in column
        dates = pd.to_datetime(df.date).dt
        return df[(year_range[0] <= dates.year) & (dates.year <= year_range[1])]
    else: # dates in index
        dates = df.index.get_level_values('date')
        return df[(year_range[0] <= dates.year) & (dates.year <= year_range[1]) 
                & (dates.month.isin(months))]
    
##################################################
##         Process variables separately         ##
##################################################

def process_spi():
    # Get filenames
    pr_files = [file for file in files if 'pr_' in file]
    historical_file = next(file for file in pr_files if 'historical' in file)
    ssp_files = sorted([file for file in pr_files if 'ssp' in file])

    # Combine historical and ssp dataframes; Convert pr to mm/day
    historical_df = read_data(historical_file)
    spi_dfs = {os.path.basename(ssp_file).split('_')[2].split('.')[0]: 
               filter_dates(pd.concat([historical_df, read_data(ssp_file)], ignore_index=True), 
                            SPI_YEARS, spi=True).dropna(how='all')
               for ssp_file in ssp_files}
    
    # Calculate SPI
    for ssp_name, df in spi_dfs.items():
        print(f'Processing {ssp_name} spi ({min(df.date)} - {max(df.date)})...')        
        
        df[df.select_dtypes(include=['number']).columns] *= 86400 # Convert to mm/day
        columns = df.columns[1:]
        
        spi_df = pd.DataFrame({'date': df.date}).reset_index(drop=True)
        for col in columns:
            try:
                spi_df[f'{col}_spi'] = (
                    SPI().calculate(df, 'date', col, freq=freq, scale=scale, fit_type='lmom',
                                    dist_type='gam').filter(regex='_calculated_index$')
                    .reset_index(drop=True))
                
            except Exception as e:
                print(f'- Error calculating SPI for {col}: {e}')
                
        # Rename column names
        spi_dfs[ssp_name].columns = (spi_dfs[ssp_name].columns[:1].tolist() + 
                                     [f'{col}_pr' for col in spi_dfs[ssp_name].columns[1:]])
        spi_dfs[ssp_name] = spi_dfs[ssp_name].merge(spi_df, on='date')
    # Contains pr, scaled, and spi
    return (pd.concat([df.assign(ssp=ssp_name) for ssp_name, df in spi_dfs.items()],
                     ignore_index=True).set_index(['ssp', 'date']))

def process_tasmax():
    # Get filenames
    tm_files = [file for file in files if 'tasmax_' in file]
    historical_file = next(file for file in tm_files if 'historical' in file)
    ssp_files = sorted([file for file in tm_files if 'ssp' in file])
    
    # Combine ssp dataframes
    tm = pd.concat([read_data(ssp_file)
                      .assign(ssp=os.path.basename(ssp_file).split('_')[2].split('.')[0]) 
                      for ssp_file in ssp_files])
    
    # Convert Kelvin to Celsius
    numeric_columns = tm.select_dtypes(include=['number']).columns
    tm.loc[:, numeric_columns] -= 273.15
    return tm.set_index(['ssp', 'date']).add_suffix('_tasmax')

##################################################
##           Determine Compound Events          ##
##################################################

def setup_thresholds(event):
    '''Set up thresholds for different severity levels of compound events.'''
    def f_to_c(f):
        return (f - 32) * 5/9
    if event == 'CDHE':
        spi_op, tm_op = '<', '>'
        return [{'spi': (spi_op, p), 'tasmax': (tm_op, f_to_c(q))} for p, q in 
                # least --> most severe
                zip([-1, -2],  # spi (standardized)
                    [90, 90])] #tm (f)
    elif event == 'CWHE':
        spi_op, tm_op = '>', '>'
        return [{'spi': (spi_op, p), 'tasmax': (tm_op, f_to_c(q))} for p, q in 
                # least --> most severe
                zip([1, 2],  # spi (standardized)
                    [90, 90])] # tm (f)
    else:
        raise ValueError('Invalid event type')

def check_suffix(df, suffix):
    suffixes = [col.split('_')[-1] for col in df.columns if '_' in col]
    return all(s == suffix for s in suffixes) and len(set(suffixes)) == 1

def get_common_columns(dfs):
    return sorted(list(reduce(lambda x, y: x.intersection(y), 
                  (set(col.split('_')[0] for col in df.columns) for df in dfs))))

def process_compound(spi, tm, threshold):
    spi_, tm_ = None, None
    for var, (op, p) in threshold.items():
        if var == 'spi' and check_suffix(spi, 'spi'):
            spi_ = COMP_OPS[op](spi, p)
        elif var == 'tasmax' and check_suffix(tm, 'tasmax'):
            tm_ = COMP_OPS[op](tm, p)
        else:
            raise ValueError('Wrong order or update variable processing if not in [spi, tasmax]')
    compound = pd.concat([spi_, tm_], axis=1)
        
    # Determine if compound
    common_cols = get_common_columns([spi, tm])
    for col in common_cols:
        df = compound.filter(regex=col)
        compound[f'{col}_compound'] = df.all(axis=1)

    return compound
    
##################################################
##               Calculate Metrics              ##
##################################################

def group_data(df, suffix):
    '''Group data based on specified criteria'''
    df_ = df.copy().reset_index()
    df_.date = df_.date.dt.year
    return (df_.set_index(['ssp', 'date']).filter(regex=suffix)
           .groupby(['ssp', 'date']))

def max_consecutive(s):
    '''Calculate max consecutive True values in a series if max > 1'''
    result = (s * (s.groupby((s != s.shift()).cumsum()).cumcount() + 1)).max()
    return result if result > 1 else 0
    
def total_consecutive(s):
    '''Calculate total True values in a series if more than 1 consecutive True'''
    s = s.groupby((s != s.shift()).cumsum()).sum()
    return s[s > 1].sum()

##################################################
##                    Main                      ##
##################################################

def main():
    # Process variables
    pr_spi = filter_dates(process_spi(), SSP_YEARS)
    tm = filter_dates(process_tasmax(), SSP_YEARS)
    spi = pr_spi.filter(regex='_spi$')
    
    compound, results = {}, {}
    for threshold in thresholds:
        name = '_'.join([f'{v}{c}{round(p, 1)}' for v, (c, p) in threshold.items()])
        compound[name] = process_compound(spi, tm, threshold)
        grouped = group_data(compound[name], f'_compound$')
        
        # Total Compound Days
        results[name] = grouped.sum().add_suffix('_day_total')

        # Total Compound Events
        results[name] = pd.concat(
            [grouped.apply(lambda x: x.apply(total_consecutive)).add_suffix('_event_total'), 
             results[name]], axis=1)
        
        # Max Consecutive Compound Events
        results[name] = pd.concat(
            [grouped.apply(lambda x: x.apply(max_consecutive)).add_suffix('_event_max'), 
             results[name]], axis=1)
    
    pr_ = group_data(pr_spi, '^(?!.*scale).*_pr$').mean()
    tm_ = group_data(tm, '_tasmax$').mean()
    spi_ = group_data(spi, '_spi$').mean()

    return results, compound, pr_spi, tm, pr_, tm_, spi_






















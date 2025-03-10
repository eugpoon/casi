import os
# import calendar
import warnings
import pandas as pd
import numpy as np
import datetime
from itertools import groupby
import operator

# from typing import List, Dict, Tuple
# from collections import defaultdict
from functools import reduce
from spi import SPI

warnings.filterwarnings('ignore')

##################################################
##            Variable initialization           ##
##################################################

DATA_PATH = '../compound'

VARIABLES = ['pr_', 'tasmax_'] # original variable in file names

COMP_OPS = { # Comparative operators
    '<': operator.lt, '<=': operator.le,
    '>': operator.gt, '>=': operator.ge,
}

def initialize(center_, event_, months_, freq_, scale_):
    global center, event, months, freq, scale, temporal_res, files, thresholds
    global HISTORICAL_YEARS, SSP_YEARS, SPI_YEARS
    
    HISTORICAL_YEARS, SSP_YEARS = (1981, 2020), (2021, 2100)
    
    if freq_ == 'D':
        temporal_res = 'daily'
        delta = datetime.timedelta(days=scale_)
    elif freq_ == 'M':
        temporal_res = 'monthly_avg'
        delta = datetime.timedelta(days=30 * scale_)  # Approximation for months
    else: 
        raise ValueError(f"{freq} should be one of ['D', 'M']")

    # update start years
    SPI_YEARS = ((datetime.date(HISTORICAL_YEARS[0], 1, 1) - delta).year, SSP_YEARS[1])
    
    center, event, months, freq, scale = center_, event_, months_, freq_, scale_
    files = get_files()
    thresholds = setup_thresholds(event)

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
    df = df.set_index('date')
    return df.sort_values('date').interpolate(method='linear', limit_direction='both')

def filter_dates(df, year_range:tuple, year_month):
    dates = df.index.get_level_values('date')
    filtered = df[(year_range[0] <= dates.year) & (dates.year <= year_range[1])]
    return filtered[filtered.index.get_level_values('date').month.isin(months)] if year_month else filtered
    
    
##################################################
##               Process variables              ##
##################################################

def process_spi():
    # Get filenames
    pr_files = [file for file in files if 'pr_' in file]
    historical_file = next(file for file in pr_files if 'historical' in file)
    ssp_files = sorted([file for file in pr_files if 'ssp' in file])

    # Combine historical and ssp dataframes
    historical_df = read_data(historical_file)
    spi_dfs = {os.path.basename(ssp_file).split('_')[2].split('.')[0]: 
               filter_dates(pd.concat([historical_df, read_data(ssp_file)]), SPI_YEARS, year_month=False
                           ).dropna(how='all') * 86400 # Convert pr to mm/day
               for ssp_file in ssp_files}

    #Calculate SPI
    for ssp_name, df in spi_dfs.items():
        print(f'Processing {ssp_name} spi ({min(df.index.date)} to {max(df.index.date)})...') 
        spi_dfs[ssp_name] = pd.concat([SPI().calculate(df, HISTORICAL_YEARS, SPI_YEARS, freq=freq, scale=scale),
                                       df.add_suffix('_pr')], axis=1)
    
    # Combine all SSPs
    return (pd.concat([df.assign(ssp=ssp_name).reset_index() for ssp_name, df in spi_dfs.items()],
                     ignore_index=True).set_index(['ssp', 'date']))

def process_tasmax():
    # Get filenames
    tm_files = [file for file in files if 'tasmax_' in file]
    historical_file = next(file for file in tm_files if 'historical' in file)
    ssp_files = sorted([file for file in tm_files if 'ssp' in file])
    
    # Combine ssp dataframes
    tm = pd.concat([(read_data(ssp_file)-273.15) # Convert Kelvin to Celsius
                      .assign(ssp=os.path.basename(ssp_file).split('_')[2].split('.')[0]) 
                      for ssp_file in ssp_files])

    return tm.reset_index().set_index(['ssp', 'date']).add_suffix('_tasmax')

##################################################
##           Determine Compound Events          ##
##################################################



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

def total_sequence(s):
    return sum(1 for key, group in groupby(s) if key and sum(group) > 1)

def mean_duration(s):
    sequences = [len(list(g)) for k, g in groupby(s) if k]
    sequences = [s for s in sequences if s > 1]
    return round(np.mean(sequences), 2) if sequences else 0

##################################################
##                    Main                      ##
##################################################

def main():
    # Process variables
    pr_spi = filter_dates(process_spi(), SSP_YEARS, year_month=True)
    tm = filter_dates(process_tasmax(), SSP_YEARS, year_month=True)
    spi = pr_spi.filter(regex='_spi$')
    
    compound, results = {}, {}
    for threshold in thresholds:
        name = '_'.join([f'{v}{c}{round(p, 1)}' for v, (c, p) in threshold.items()])
        compound[name] = process_compound(spi, tm, threshold)
        grouped = group_data(compound[name], f'_compound$')
        
        # Total Compound Days
        results[name] = grouped.sum().add_suffix('_day_total')
        
        # Total Compound Events
        a = [grouped.apply(lambda x: x.apply(total_consecutive)).add_suffix('_event_total'), results[name]]
        results[name] = pd.concat(a, axis=1)
        
        # Max Consecutive Compound Events
        a = [grouped.apply(lambda x: x.apply(max_consecutive)).add_suffix('_event_max'), results[name]]
        results[name] = pd.concat(a, axis=1)

        # Total Compound Event Sequences
        a = [grouped.apply(lambda x: x.apply(total_sequence)).add_suffix('_sequence_total'), results[name]]
        results[name] = pd.concat(a, axis=1)
        
        # Average Compound Event Duration
        a = [grouped.apply(lambda x: x.apply(mean_duration)).add_suffix('_duration_mean'), results[name]]
        results[name] = pd.concat(a, axis=1)
    
    pr_  = group_data(pr_spi, '_pr$').mean()
    tm_  = group_data(tm, '_tasmax$').mean()
    spi_ = group_data(spi, '_spi$').mean()

    return results, compound, pr_spi, tm, pr_, tm_, spi_





















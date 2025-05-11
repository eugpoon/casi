"""
Combine CMIP6 daily variable files into a single dataset, apply unit conversions, 
and calculate SPI. Outputs raw and MME (multi-model ensemble) data.

Usage:
    python combine_raw.py -i ../data/compound -o ../data -v pr tasmax rzsm
"""

import pandas as pd
import os
import glob
import re
import argparse
import json
from spi import SPI 

with open('../data/defaults.json') as f:
    globals().update(json.load(f))

def read_and_concat_csv(input_dir, allowed_vars):
    """
    Read and combine daily CSVs into a single dataframe, converting units and computing SPI.

    Parameters:
        input_dir (str): Directory containing *_daily.csv files
        allowed_vars (list): List of variables to process (e.g., ['pr', 'tasmax', 'rzsm'])

    Returns:
        pd.DataFrame: Combined and processed dataframe
    """
    dfs = []
    for file in glob.glob(os.path.join(input_dir, '*_*_*_daily.csv')):
        name = os.path.basename(file)
        match = re.match(r'([^_]+)_([^_]+)_([^_]+)_daily\.csv', name)
        if not match: continue
        center, var, ssp = match.groups()
        if center not in centers or ssp not in ssps or var not in allowed_vars: continue

        try:
            df = pd.read_csv(file).rename(columns={'Unnamed: 0': 'date'})
            df.columns = df.columns.str.lower()
            if 'date' not in df: continue
            df['date'] = pd.to_datetime(df['date'], errors='coerce')
            df['center'], df['variable'], df['ssp'] = center, var, ssp
            df = df.sort_values('date')
            df.set_index(['center', 'variable', 'ssp', 'date'], inplace=True)
            
            # Unit conversions
            if var == 'pr': 
                df *= 86400  # kg/m²/s to mm/day
            elif var == 'tasmax': 
                df -= 273.15 # K to C
            
            # Interpolate missing values
            dfs.append(df.interpolate(method='linear', limit_direction='both'))
        except Exception as e:
            print(f'Error reading {name}: {e}')

    if not dfs:
        print('No valid dataframes to combine.')
        return None

    df = pd.concat(dfs).reset_index()
    cols = ['center', 'variable', 'ssp', 'date']
    df = df[cols + sorted(df.columns.difference(cols))].sort_values(cols).drop_duplicates()
    df = pd.concat([df, calculate_spi(df)]).sort_values(cols)

    out_path = os.path.join(args.output, 'compound_raw.parquet')
    df.to_parquet(out_path, index=False)
    
    numeric_df = df.set_index(['center', 'variable', 'ssp', 'date'])
    mme_df = pd.DataFrame({'mean': numeric_df.mean(axis=1, skipna=True),
                           'median': numeric_df.median(axis=1, skipna=True),
                           'q10': numeric_df.quantile(q=0.1, axis=1, interpolation='linear'),
                           'q90': numeric_df.quantile(q=0.9, axis=1, interpolation='linear')
                         })
    mme_df.reset_index().to_parquet('../data/compound_mme_raw.parquet')
    print(f'Saved to {out_path}')
    return df

def calculate_spi(df):
    """
    Compute SPI (Standardized Precipitation Index) for each center and SSP.

    Parameters:
        df (pd.DataFrame): Input dataframe including 'pr' variable

    Returns:
        pd.DataFrame: SPI values per date, center, and SSP
    """
    spi = SPI()
    dfs = []
    for ssp in ssps[:-1]:
        subset = df[(df['ssp'].str.contains(f'{ssp}|historical')) & (df['variable'] == 'pr')]
        for center in subset['center'].unique():
            sub = subset[subset['center'] == center].dropna(axis=1, how='all')
            if sub.empty: continue
            sub = sub.set_index('date').drop(columns=['center', 'variable', 'ssp'])
            result = spi.calculate(sub, base_years=spi_base, spi_years=spi_ssp, months=spi_months,
                                   freq=spi_freq, n=spi_n, gamma_n=spi_gamma_n)
            result['center'], result['variable'], result['ssp'] = center, 'spi', ssp
            dfs.append(result.reset_index())
    return pd.concat(dfs)

if __name__ == '__main__':
    p = argparse.ArgumentParser()
    p.add_argument('-i', '--input', required=True)
    p.add_argument('-o', '--output', required=True)
    p.add_argument('-v', '--variables', nargs='+', default=['pr', 'tasmax', 'rzsm'])
    args = p.parse_args()
    read_and_concat_csv(args.input, args.variables)
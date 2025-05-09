'''
Process RZSM data from zipped or folder input into daily CSVs per center.
Splits output into historical (pre-2015) and SSP245 (2015+). To 2099 because 2100 only has one day
Version: CASI2_RZSM_RAW_Mar25
'''

import os
import pandas as pd
import numpy as np
import argparse
import zipfile
from io import TextIOWrapper

MODELS = {i: name for i, name in enumerate([
            'ACCESS-CM2', 'ACCESS-ESM1-5', 'CESM2', 'CESM2-WACCM', 'CMCC-CM2-SR5', 'CMCC-ESM2', 'CNRM-CM6-1', 'CNRM-ESM2-1', 
            'EC-Earth3', 'FGOALS-g3','GFDL-CM4', 'GFDL-CM4_gr2', 'GFDL-ESM4', 'GISS-E2-1-G', 'IITM-ESM', 'INM-CM4-8', 'INM-CM5-0', 
            'KACE-1-0-G', 'MIROC-ES2L', 'MPI-ESM1-2-HR', 'MPI-ESM1-2-LR', 'MRI-ESM2-0', 'NorESM2-LM', 'NorESM2-MM', 'TaiESM1'
], start=1)}

def read_file(f, name, is_zip=False):
    '''Read a .dat file into a DataFrame with datetime index.'''
    df = pd.read_csv(f if is_zip else name, sep='\\s+', header=None, names=['year', 'month', 'day', name])
    df.index = pd.to_datetime(df[['year', 'month', 'day']])
    return df.drop(columns=['year', 'month', 'day'])

def process_file(file_name, open_fn, center_dfs):
    '''Extract center/model info and merge file into center-level DataFrame. Example filename: MEAN_AFRC_0001.dat'''
    parts = os.path.basename(file_name).split('_')
    center, model_idx = parts[1], int(parts[2].split('.')[0])
    model = MODELS.get(model_idx, f'Model_{model_idx}')
    df = read_file(open_fn(file_name), model, is_zip=callable(open_fn))
    center_dfs[center] = pd.merge(center_dfs.get(center, df.reindex([])), df, left_index=True, right_index=True, how='outer')

def main():
    parser = argparse.ArgumentParser(description='Process RZSM .dat files or ZIPs into daily CSVs.')
    parser.add_argument('-i', '--input_folder', required=True, help='Input folder or ZIP path.')
    parser.add_argument('-o', '--output_folder', required=True, help='Output CSV directory.')
    args = parser.parse_args()

    os.makedirs(args.output_folder, exist_ok=True)
    center_dfs = {}

    if zipfile.is_zipfile(args.input_folder):
        with zipfile.ZipFile(args.input_folder) as z:
            for name in z.namelist():
                if name.endswith('.dat') and not name.startswith('__MACOSX/'):
                    process_file(name, lambda n: TextIOWrapper(z.open(n), encoding='utf-8'), center_dfs)
    else:
        for name in os.listdir(args.input_folder):
            if name.startswith('MEAN_') and name.endswith('.dat'):
                process_file(os.path.join(args.input_folder, name), lambda n: n, center_dfs)

    full_index = pd.date_range('1950-01-01', '2099-12-31', freq='D')
    for center, df in center_dfs.items():
        df = df.replace(-9999, np.nan).reindex(full_index).ffill().bfill().sort_index(axis=1)
        df[df.index < '2015-01-01'].to_csv(f'{args.output_folder}/{center}_rzsm_historical_daily.csv')
        df[df.index >= '2015-01-01'].to_csv(f'{args.output_folder}/{center}_rzsm_ssp245_daily.csv')

if __name__ == '__main__':
    main()
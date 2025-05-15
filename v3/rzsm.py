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

MODELS = {
        1: 'ACCESS-CM2', 2: 'ACCESS-ESM1-5', 3: 'CESM2', 4: 'CESM2-WACCM', 
        5: 'CMCC-CM2-SR5', 6: 'CMCC-ESM2', 7: 'CNRM-CM6-1', 8: 'CNRM-ESM2-1', 
        9: 'EC-Earth3', 10: 'FGOALS-g3', 11: 'GFDL-CM4', 12: 'GFDL-CM4_gr2', 
        13: 'GFDL-ESM4', 14: 'GISS-E2-1-G', 15: 'IITM-ESM', 16: 'INM-CM4-8', 
        17: 'INM-CM5-0', 18: 'KACE-1-0-G', 19: 'MIROC-ES2L', 20: 'MPI-ESM1-2-HR', 
        21: 'MPI-ESM1-2-LR', 22: 'MRI-ESM2-0', 23: 'NorESM2-LM', 24: 'NorESM2-MM', 
        25: 'TaiESM1'
    }

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
    if len(df.dropna()) > 1:
        center_dfs[center] = pd.concat([center_dfs.get(center, pd.DataFrame(index=df.index)), df], axis=1)

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
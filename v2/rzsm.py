import os
import pandas as pd
import numpy as np
import argparse
# import warnings
# warnings.filterwarnings('ignore')

def main():
    parser = argparse.ArgumentParser(description='Process RZSM.')
    parser.add_argument('-i', '--input_folder', required=True, help='Input folder path.')
    parser.add_argument('-o', '--output_folder', required=True, help='Output folder path.')
    args = parser.parse_args()

    input_path = args.input_folder
    output_path = args.output_folder
    os.makedirs(output_path, exist_ok=True)

    models_dict = {
        1: 'ACCESS-CM2', 2: 'ACCESS-ESM1-5', 3: 'CESM2', 4: 'CESM2-WACCM', 
        5: 'CMCC-CM2-SR5', 6: 'CMCC-ESM2', 7: 'CNRM-CM6-1', 8: 'CNRM-ESM2-1', 
        9: 'EC-Earth3', 10: 'FGOALS-g3', 11: 'GFDL-CM4', 12: 'GFDL-CM4_gr2', 
        13: 'GFDL-ESM4', 14: 'GISS-E2-1-G', 15: 'IITM-ESM', 16: 'INM-CM4-8', 
        17: 'INM-CM5-0', 18: 'KACE-1-0-G', 19: 'MIROC-ES2L', 20: 'MPI-ESM1-2-HR', 
        21: 'MPI-ESM1-2-LR', 22: 'MRI-ESM2-0', 23: 'NorESM2-LM', 24: 'NorESM2-MM', 
        25: 'TaiESM1'
    }

    center_dfs = {}

    for file_name in os.listdir(input_path):
        if not file_name.startswith('MEAN_') or not file_name.endswith('.dat'):
            continue

        parts = file_name.split('_')
        center = parts[1]
        model_index = int(parts[2].split('.')[0])
        model_name = models_dict.get(model_index, f'Model_{model_index}')

        df = pd.read_csv(os.path.join(input_path, file_name), sep='\\s+', header=None, 
                         names=['year', 'month', 'day', model_name])

        df.insert(0, 'date', pd.to_datetime(df[['year', 'month', 'day']]))
        center_dfs[center] = pd.merge(center_dfs.setdefault(center, pd.DataFrame({'date': []})), 
                                      df.drop(columns=['year', 'month', 'day']), on='date', how='outer')

    for center, df in center_dfs.items():
        df = df.replace(-9999, np.nan).ffill()
        df.to_csv(os.path.join(output_path, f'{center}_rzsm_daily.csv'))

if __name__ == "__main__":
    main()

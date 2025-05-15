"""
Compute percentile-based thresholds for compound events using CMIP6 data.

Usage:
    python set_thresholds.py -i ../data/compound_raw.parquet -o ../data/compound_thresholds.csv
"""

import pandas as pd
import json
import argparse

# Load threshold rules for each event (e.g., CDHE, CWHE) from JSON
with open('../data/compound_events.json') as f:
    THRESHOLDS = json.load(f)

# Load default global configuration values (e.g., centers, SPI params)
with open('../data/defaults.json') as f:
    globals().update(json.load(f))

def set_thresholds(df):
    """
    Compute percentile thresholds for each variable and event, based on configuration.

    Parameters:
        df (pd.DataFrame): Combined CMIP6 data including columns ['center', 'variable', 'ssp', 'date', ...]

    Returns:
        pd.DataFrame: DataFrame containing thresholds per center, variable, SSP, and percentile
    """

    computed, results = {}, []

    for event, config in THRESHOLDS.items():
        print(event)
        rule, base_years = config['threshold'][0], config['base_years']
        dd = df.copy()
        
        months = config.get('months', None)
        month_str = str(months) if months else None
        if months:
            dd = df[df['date'].dt.month.isin(months)]

        for var, t in rule.items():
            if t['type'] != 'perc':
                continue
            dd_ = dd[(dd['variable'] == var)].copy()
            for ssp in sorted(filter(lambda x: 'ssp' in x, dd_['ssp'].unique())):
                # Filter for historical + target SSP and base year window
                subset = (dd_[(dd_.ssp.str.contains(f'{ssp}|historical')) & (dd_['date'].dt.year.between(*base_years))]
                          .drop(columns=['variable', 'ssp', 'date']))
                if subset.empty:
                    continue
                
                # Skip percentiles already computed
                key = (var, str(base_years), ssp, month_str)
                v_remain = list(set(t['values']) - computed.get(key, set()))
                if not v_remain:
                    continue
                
                # Compute quantiles per center
                subset = subset.groupby('center').quantile(v_remain).reset_index().rename(columns={'level_1': 'percentile'})
                subset['base'], subset['ssp'], subset['variable'], subset['months'] = str(base_years), ssp, var, month_str

                results.append(subset)
                computed.setdefault(key, set()).update(v_remain)

    return pd.concat(results, ignore_index=True)


if __name__ == '__main__':
    p = argparse.ArgumentParser()
    p.add_argument('-i', '--input', required=True, help='Path to input CSV file')
    p.add_argument('-o', '--output', required=True, help='Output file path for thresholds CSV')
    args = p.parse_args()

    df = pd.read_parquet(args.input)

    cols = ['center', 'variable', 'base' ,'ssp', 'percentile', 'months']
    thresholds_df = set_thresholds(df)
    thresholds_df = thresholds_df[cols + sorted(thresholds_df.columns.difference(cols))].sort_values(cols).drop_duplicates()

    thresholds_df.to_csv(args.output, index=False)
    print(f'Saved thresholds to {args.output}')

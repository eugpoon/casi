"""
Generate compound event metrics using thresholded CMIP6 data.

Usage:
    python get_compound.py -i ../data/ -o ../data/compound_results.parquet -e CDHE CWHE CFE
"""

import pandas as pd
import numpy as np
import argparse
import json
import re
from itertools import product, groupby
from functools import reduce
from pathlib import Path

# Comparison operators
OPS = {'<': pd.DataFrame.lt, '<=': pd.DataFrame.le, '>': pd.DataFrame.gt, '>=': pd.DataFrame.ge}

def load_data(input_dir):
    """Load required CMIP6 data, threshold values, and event configuration."""
    raw = pd.read_parquet(Path(input_dir) / 'compound_raw.parquet')
    thresholds_df = pd.read_csv(Path(input_dir) / 'compound_thresholds.csv')
    with open(Path(input_dir) / 'compound_events.json') as f:
        config = json.load(f)
    return raw, thresholds_df, config

def generate_thresholds(events, raw, thresholds_df, config):
    """Create thresholded binary masks for each event based on rules in config."""
    cols = [c for c in raw.columns if c not in ['center', 'variable', 'ssp', 'date']]
    results, combo = {}, {}

    for e in events:
        threshold, base = config[e]['threshold'], str(config[e]['base_years'])
        specs = threshold[0]
        raw_copy = raw.copy()

        if e == 'CFE': # calculate 7 day precip accumulation for CFE
            mask = (raw_copy.variable == 'pr')
            raw_copy.loc[mask, cols] = (raw_copy[mask].groupby(['center', 'ssp'])
                .apply(lambda g: g.set_index('date')[cols].rolling(7).sum().reset_index(drop=True), include_groups=False).values)

        # Generate all threshold combinations for compound events
        value_lists = [spec['values'] for spec in specs.values()]
        if threshold[1]:  # Product of values
            combo[e] = ['_'.join(f"{var}{specs[var]['op']}{round(v, 2)}" + (base if specs[var]['type'] == 'perc' else '')
                                 for var, v in zip(specs.keys(), values))
                        for values in product(*value_lists)]
        else: # position-wise merge
            if len(set(len(spec['values']) for spec in specs.values())) > 1:
                raise ValueError('All lists must have the same length')
            combo[e] = ['_'.join(f"{var}{specs[var]['op']}{round(specs[var]['values'][i], 2)}" +
                                 (base if specs[var]['type'] == 'perc' else '')
                                 for var in specs)
                        for i in range(len(next(iter(specs.values()))['values']))]
        
        # Create binary mask for each individual threshold
        for var, spec in specs.items():
            for v in spec['values']:
                name = f"{var}{spec['op']}{round(v, 2)}" + (base if spec['type'] == 'perc' else '')
                if name in results:
                    continue

                sub = raw_copy[raw_copy.variable == var].copy()
                original_nan_mask = sub[cols].isna()

                if spec['type'] == 'fixed':
                    sub_comp = OPS[spec['op']](sub[cols], v).astype('float')
                    sub[cols] = sub_comp.where(~original_nan_mask)

                elif spec['type'] == 'perc':
                    p = thresholds_df[(thresholds_df.variable == var) & (thresholds_df.percentile == v) & (thresholds_df.base == base)]
                    for c, s in p[['center', 'ssp']].drop_duplicates().values:
                        pc = p[(p.center == c) & (p.ssp == s)][cols]
                        rc_idx = sub[(sub.center == c) & (sub.ssp.str.contains(f'{s}|historical'))].index
                        rc = sub.loc[rc_idx, cols]
                        rc_mask = rc.isna()
                        rc_comp = OPS[spec['op']](rc, pc.values).astype('float')
                        sub.loc[rc_idx, cols] = rc_comp.where(~rc_mask)

                results[name] = sub.drop(columns='variable').set_index(['ssp', 'center', 'date']).dropna(axis=1, how='all')

    return results, combo

#########################
#    Compound Metrics   #
#########################

def total_consecutive(s):
    """Return total number of event days in sequences > 1 day."""
    if s.isna().all(): return np.nan
    s = s.groupby((s != s.shift()).cumsum()).sum()
    return s[s > 1].sum()

def total_sequence(s):
    """Return number of sequences with >1 event day."""
    if s.isna().all(): return np.nan
    return sum(1 for k, g in groupby(s) if k and sum(g) > 1)

def max_duration(s):
    """Return max duration of consecutive 1s (event days)."""
    if s.isna().all(): return np.nan
    return (s * (s.groupby((s != s.shift()).cumsum()).cumcount() + 1)).max() if (s == 1).sum() > 1 else 0
    
def mean_duration(s):
    """Return mean duration of event sequences > 1 day."""
    if s.isna().all(): return np.nan
    durations = [len(list(g)) for k, g in groupby(s) if k]
    durations = [d for d in durations if d > 1]
    return round(np.mean(durations), 2) if durations else 0

def compute_compound(results, combo):
    """Combine threshold masks and compute yearly compound event metrics."""
    compound = []
    for event, thresholds in combo.items():
        event_data = []
        for threshold in thresholds:
            # Match common index and columns
            dfs = [results[t] for t in threshold.split('_')]
            common_idx = reduce(lambda x, y: x.intersection(y), [df.index for df in dfs])
            common_cols = set.intersection(*[set(df.columns) for df in dfs])
            dfs = [df.loc[common_idx, sorted(common_cols)] for df in dfs]

            # Identify (index, column) positions where at least two conditions are simultaneously satisfied (two or more 1's)
            arr = np.stack([df.values for df in dfs])
            valid_count = np.sum(~np.isnan(arr), axis=0)
            ones_count = np.sum(arr == 1, axis=0)
            compound_vals = np.where(valid_count < 2, np.nan, (ones_count >= 2).astype(float))

            df_out = pd.DataFrame(compound_vals, index=dfs[0].index, columns=dfs[0].columns)
            df_out_reset = df_out.reset_index()
            df_out_reset['date'] = df_out_reset['date'].dt.year
            grouped = df_out_reset.groupby(['center', 'ssp', 'date'])

            # Calculate metrics
            result = pd.concat([
                grouped.sum(min_count=1).add_suffix('_day_total'),
                grouped.apply(lambda x: x.apply(total_consecutive)).add_suffix('_event_total'),
                grouped.apply(lambda x: x.apply(total_sequence)).add_suffix('_sequence_total'),
                grouped.apply(lambda x: x.apply(max_duration), include_groups=False).add_suffix('_duration_max'),
                grouped.apply(lambda x: x.apply(mean_duration), include_groups=False).add_suffix('_duration_mean')
            ], axis=1)

            result['threshold'] = re.sub(r'\[\d{4},\s*\d{4}\]', '', threshold)
            event_data.append(result)

        event_df = pd.concat(event_data).reset_index()
        event_df['event'] = event
        compound.append(event_df)

    compound = pd.concat(compound)
    cols = ['event', 'threshold', 'center', 'ssp', 'date']
    return compound[cols + sorted(compound.columns.difference(cols))].sort_values(cols)

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('-i', '--input', required=True, help='Input directory')
    parser.add_argument('-o', '--output', required=True, help='Output .parquet file path')
    parser.add_argument('-e', '--events', nargs='+', required=True, help='List of events to process')
    args = parser.parse_args()

    raw, thresholds_df, config = load_data(args.input)
    valid_events = [e for e in args.events if e in config]
    if not valid_events:
        raise ValueError('No valid events provided.')

    results, combo = generate_thresholds(valid_events, raw, thresholds_df, config)
    compound_df = compute_compound(results, combo)
    compound_df.to_parquet(args.output)
    
    df = compound_df.set_index(['event', 'threshold', 'center', 'ssp', 'date'])

    mme_df = []
    for suffix in ['_day_total', '_event_total', '_sequence_total', '_duration_max', '_duration_mean']:
        numeric_df = df.filter(regex=suffix)
        mme_df.append(pd.DataFrame({'mean': numeric_df.mean(axis=1, skipna=True),
                                    'median': numeric_df.median(axis=1, skipna=True),
                                    'q10': numeric_df.quantile(q=0.1, axis=1, interpolation='linear'),
                                    'q90': numeric_df.quantile(q=0.9, axis=1, interpolation='linear')
                                   }).add_suffix(suffix))
    pd.concat(mme_df, axis=1).reset_index().to_parquet('../data/compound_mme_results.parquet')

    print(f'Saved to {args.output}')

if __name__ == '__main__':
    main()

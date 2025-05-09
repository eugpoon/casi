import pandas as pd
import json
import argparse

with open('../data/compound_events.json') as f:
    THRESHOLDS = json.load(f)

with open('../data/defaults.json') as f:
    globals().update(json.load(f))

def set_thresholds(df):
    computed, results = {}, []

    for event, config in THRESHOLDS.items():
        print(event)
        rule, base_years = config['threshold'][0], config['base_years']
        dd = df[df['date'].dt.year.between(*base_years)]
        if config.get('months'):
            dd = dd[dd['date'].dt.month.isin(config.get('months'))]

        for var, t in rule.items():
            ttype = t['type']
            values = t['values']
            
            if ttype == 'perc':
                for ssp in ssps[:-1]:
                    subset = dd[(dd['ssp']==ssp) & (dd['variable'] == var)]
                    if subset.empty: 
                        continue
                    subset = pd.concat([subset, (dd[(dd['ssp'].str.contains('historical')) & (dd['variable'] == var)]
                                                )]).drop(columns=['variable', 'ssp', 'date'])
                    name = f'{var}{base_years}'
                    v_remain =  list(set(values) - computed.get(name, set()))
                    subset = subset.groupby(['center']).quantile(v_remain).reset_index().rename(columns={'level_1': 'percentile'})
                    subset['base'], subset['ssp'], subset['variable'] = f'{base_years}', ssp, var
                    results.append(subset)
                    computed.setdefault(name, set()).update(v_remain)
            # elif ttype == 'fixed':
            #     for v in values:
            #         subset = pd.DataFrame(v, columns=dd.columns, index=centers).drop(columns=['center', 'date'])
            #         subset['ssp'], subset['variable'], = None, var
            #         results.append(subset.reset_index().rename(columns={'index': 'center'}))
            else:
                continue

    return pd.concat(results, ignore_index=True)


if __name__ == '__main__':
    p = argparse.ArgumentParser()
    p.add_argument('-i', '--input', required=True, help='Path to input CSV file')
    p.add_argument('-o', '--output', required=True, help='Output file path for thresholds CSV')
    args = p.parse_args()

    df = pd.read_parquet(args.input)

    cols = ['center', 'variable', 'base' ,'ssp', 'percentile']
    thresholds_df = set_thresholds(df)
    thresholds_df = thresholds_df[cols + sorted(thresholds_df.columns.difference(cols))].sort_values(cols).drop_duplicates()

    thresholds_df.to_csv(args.output, index=False)
    print(f'Saved thresholds to {args.output}')

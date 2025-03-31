import os
import warnings
import json
from process import Compound
from itertools import product

warnings.filterwarnings('ignore')

def f_to_c(f):
    return (f - 32) * 5/9

outputs = ['_day_total', '_event_total', '_sequence_total', '_duration_max', '_duration_mean']
VAR = {
    'CDHE': { # Compound Dry Hot Event
        'months': [6,7,8], 'freq': 'D', 'gamma_n': [10, 20, 30][2], 
        'scale': 30, # pr accumulation period (days)
        'inputs': ['pr_', 'tasmax_'],
        'outputs': ['_pr', '_spi', '_tasmax']+outputs,
        'historical_years': (1981, 2020),
        'ssp_years': (2021, 2100),
        'threshold': ({'spi': ['<', [-1, -2]], # standardized
                       'tasmax': ['>', [f_to_c(90)]], # tm (f->c)
                      }, True) # whether to create all threshold combos
    },
    'CWHE': { # Compound Wet Hot Event
        'months': [6,7,8], 'freq': 'D', 'gamma_n': [10, 20, 30][2], 
        'scale': 30, # pr accumulation period (days)
        'inputs': ['pr_', 'tasmax_'],
        'outputs': ['_pr', '_spi', '_tasmax']+outputs,
        'historical_years': (1981, 2020),
        'ssp_years': (2021, 2100),
        'threshold':  ({'spi': ['>', [-1, -2]], # standardized
                        'tasmax': ['>', [f_to_c(90)]], # tm (f->c)
                       }, True)
    },
    'CFE': { # Compound Flooding Event
        'months': None, 'freq': 'D', 'gamma_n': None,
        'scale': 7, # pr accumulation period (days)
        'inputs': ['pr_', 'rzsm_'],
        'outputs': ['_pr', '_rzsm']+outputs,
        'historical_years': (1950, 2014),
        'ssp_years': (2015, 2099),
        # 'threshold':  ({'rzsm': ['>', [.7]], # percentiles
        #                 'pr': ['>', [50, 80]], # mm/day
        #                }, True),
        'threshold':  ({'rzsm': ['>', [.7, .8, .9, .95, .98]], # percentiles
                        'pr': ['>', [50, 80]], # mm/day
                       }, True)
    },
}

outdir = '../data/compound_results'
if not os.path.exists(outdir):
    os.makedirs(outdir, exist_ok=True)

with open('../data/events_vars.json', 'w') as file:
    json.dump(VAR, file)

def run_analysis(centers, events):
    # center_data = {}
    for center, event in product(centers, events):
        try:
            var = VAR[event]
            result, var_aggs, dfs = Compound(center, event, var).main()
            print(center, event)
            # center_data[center] = {'result': result, 'var_aggs': var_aggs, 'dfs': dfs}
            gam = f'_gamma{var['gamma_n']}' if var['gamma_n'] else ''
            result.to_csv(f'{outdir}/{center}_{event}_metrics_roll{var['scale']}{gam}.csv', index=False)
            var_aggs.to_csv(f'{outdir}/{center}_{event}_variables.csv', index=False)
            dfs['daily'].to_parquet(f'{outdir}/{center}_{event}_daily.parquet')
            if 'tres_rzsm' in dfs:
                dfs['tres_rzsm'].to_csv(f'{outdir}/{center}_{event}_tres_rzsm.csv')
            del var, result, var_aggs, dfs
        except Exception as e:
            print(f'{center} {event} skipped: {e}')














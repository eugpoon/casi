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
        'base_years': (1981, 2020),
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
        'base_years': (1981, 2020),
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
        'base_years': (1950, 2014),
        'ssp_years': (2015, 2099),
        # 'threshold':  ({'rzsm': ['>', [.7]], # percentiles
        #                 'pr': ['>', [50, 80]], # mm/day
        #                }, True),
        'threshold':  ({'rzsm': ['>', [.7, .8, .9, .95, .98]], # percentiles
                        'pr': ['>', [80]], # mm/day
                       }, True)
    },
    'flood': { # flood
        'months': None, 'freq': 'D', 'gamma_n': None, 'scale': 0,
        'inputs': ['rzsm_'],
        'outputs': ['_day_total'],
        'base_years': (1950, 2014),
        'ssp_years': (2015, 2099),
        'threshold':  ({'rzsm': ['>', [.7, .8, .9, .95, .98]], # percentiles
                       }, False)
    },
    'drought': { # drought
        'months': None, 'freq': 'D', 'gamma_n': None, 'scale': 0,
        'inputs': ['rzsm_'],
        'outputs': ['_day_total'],
        'base_years': (1950, 2014),
        'ssp_years': (2015, 2099),
        'threshold':  ({'rzsm': ['<', [.3, .2, .1, .05, .02]], # percentiles
                       }, False)
    },
    'CWHE2': { # Compound Wet Hot Event
        'months': None, 'freq': 'D', 'gamma_n': None, 'scale': 0,
        'inputs': ['rzsm_', 'tasmax_'],
        'outputs': ['_rzsm', '_tasmax']+outputs,
        'base_years': (1981, 2020),
        'ssp_years': (2021, 2100),
        'threshold':  ({'rzsm': ['>', [.7, .8, .9, .95, .98]], # percentiles
                        'tasmax': ['>', [f_to_c(90)]], # tm (f->c)
                       }, True)
    },
    'CDHE2': { # Compound Dry Hot Event
        'months': None, 'freq': 'D', 'gamma_n': None, 'scale': 0,
        'inputs': ['rzsm_', 'tasmax_'],
        'outputs': ['_rzsm', '_tasmax']+outputs,
        'base_years': (1981, 2020),
        'ssp_years': (2021, 2100),
        'threshold':  ({'rzsm': ['<', [.3, .2, .1, .05, .02]], # percentiles
                        'tasmax': ['>', [f_to_c(90)]], # tm (f->c)
                       }, True)
    },


}

outdir = '../data/compound_results1'
if not os.path.exists(outdir):
    os.makedirs(outdir, exist_ok=True)

with open('../data/events_vars.json', 'w') as file:
    json.dump(VAR, file)

def run_analysis(centers, events):
    for center, event in product(centers, events):
        try:
            var = VAR[event]
            result, var_aggs, dfs = Compound(center, event, var).main()
            print(center, event)
            gam = f'_gamma{var['gamma_n']}' if var['gamma_n'] else ''
            roll = f'_roll{var['scale']}' if var['scale'] != 0 else ''
            result.to_csv(f'{outdir}/{center}_{event}_metrics{roll}{gam}.csv', index=False)
            # if var_aggs:
            #     var_aggs.to_csv(f'{outdir}/{center}_{event}_variables.csv', index=False)
            # dfs['daily'].to_parquet(f'{outdir}/{center}_{event}_daily.parquet')
            dfs['compound'].to_parquet(f'{outdir}/{center}_{event}_compound.parquet')
            if 'thres_rzsm' in dfs:
                dfs['thres_rzsm'].to_csv(f'{outdir}/{center}_{event}_thres_rzsm.csv')
            del var, result, var_aggs, dfs
        except Exception as e:
            print(f'{center} {event} skipped: {e}')














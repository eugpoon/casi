import os
import warnings
import pandas as pd
import numpy as np
import datetime
import operator
from collections import Counter
from functools import reduce
from itertools import groupby, product
from spi import SPI

warnings.filterwarnings('ignore')

class Compound:
    def __init__(self, center, event, var):
        self.center = center
        self.event = event
        self.__dict__.update(var)
        self.center, self.event = center, event
        self.spi_years = None, None, None
        
        self.DATA_PATH = '../data/compound'
        
        self.COMP_OPS = {
            '<': lambda df1, df2: df1.lt(df2),
            '<=': lambda df1, df2: df1.le(df2),
            '>': lambda df1, df2: df1.gt(df2),
            '>=': lambda df1, df2: df1.ge(df2),
        }

        if self.threshold[1]: # create all threshold combos
            self.thresholds = [{key: (op, val[i]) for i, (key, (op, _)) in enumerate(self.threshold[0].items())}
                               for val in product(*[v[1] for v in self.threshold[0].values()])]
        else:
            if len(set([len(vals[1]) for vals in self.threshold[0].values()])) > 1:
                raise ValueError('All lists must have the same length')
            self.thresholds = [{key: (op, vals[i]) for key, (op, vals) in self.threshold[0].items()} 
                   for i in range(len(next(iter(self.threshold[0].values()))[1]))]
            
        if self.freq == 'D':
            self.temporal_res = 'daily'
            self.delta = datetime.timedelta(days=self.scale)
        elif self.freq == 'M':
            self.temporal_res = 'monthly_avg'
            self.delta = datetime.timedelta(days=30 * self.scale)  # approximation for months
        else:
            raise ValueError(f'{self.freq} should be one of [D, M]')
        
        self.files = self.get_files()
    
    ##################################################
    ##              Get and read files              ##
    ##################################################
    
    def get_files(self):
        '''Returns list of filenames in a directory that contain a specific string'''
        files = [os.path.join(self.DATA_PATH, f) for f in os.listdir(self.DATA_PATH) 
                 if self.center in f and f.endswith('.csv') and self.temporal_res in f
                 and any(v in f for v in self.inputs)]
    
        # Check if each variable is present in at least one file
        for var in self.inputs:
            if not any(var in f for f in files):
                raise ValueError(f'No files found for variable {var}')
        return files

    def validate_file_path(self, file_path: str) -> None:
        '''Validate if the file path exists.'''
        if not os.path.exists(file_path):
            raise FileNotFoundError(f'{file_path} does not exist.')

    def read_data(self, filename):
        df = pd.read_csv(filename).rename(columns={'Unnamed: 0': 'date'})
        df['date'] = pd.to_datetime(df['date'])
        df = df.set_index('date')
        return df.sort_values('date').interpolate(method='linear', limit_direction='both')

    def filter_dates(self, df, year_range: tuple, year_month):
        dates = df.index.get_level_values('date')
        filtered = df[(year_range[0] <= dates.year) & (dates.year <= year_range[1])]
        return (filtered[filtered.index.get_level_values('date').month.isin(self.months)] 
                if year_month and self.months else filtered)
    
    ##################################################
    ##               Process variables              ##
    ##################################################

    def get_filenames(self, prefix):
        '''Get historical and SSP files based on a prefix.'''
        files = [file for file in self.files if prefix in file]
        hist_file, ssp_files = None, None
        try:
            hist_file = next(file for file in files if 'historical' in file)
        except:
            pass
        ssp_files = sorted([file for file in files if 'ssp' in file])
        return files, hist_file, ssp_files
    
    def process_spi(self):
        files, hist_file, ssp_files = self.get_filenames('_pr')

        # Combine historical and ssp dataframes
        historical_df = self.read_data(hist_file)
        spi_dfs = {os.path.basename(ssp_file).split('_')[2].split('.')[0]: 
                   self.filter_dates(pd.concat([historical_df, self.read_data(ssp_file)]), self.spi_years,
                                     year_month=False).dropna(how='all') * 86400 # pr to mm/day
                   for ssp_file in ssp_files}

        # Calculate SPI
        for ssp_name, df in spi_dfs.items():
            # print(f'Processing {ssp_name} spi ({min(df.index.date)} to {max(df.index.date)})...') 
            spi_dfs[ssp_name] = pd.concat([SPI().calculate(df, self.historical_years, self.spi_years,
                    self.months, freq=self.freq, scale=self.scale, gamma_n=self.gamma_n),
                    df.add_suffix('_pr')], axis=1)
        
        return (pd.concat([df.assign(ssp=ssp_name).reset_index() for ssp_name, df in spi_dfs.items()],
                         ignore_index=True).set_index(['ssp', 'date']))

    def process_tasmax(self):
        files, hist_file, ssp_files = self.get_filenames('tasmax_')
        df = pd.concat([(self.read_data(ssp_file)-273.15) # Convert Kelvin to Celsius
                        .assign(ssp=os.path.basename(ssp_file).split('_')[2].split('.')[0]) 
                        for ssp_file in ssp_files])
        return df.reset_index().set_index(['ssp', 'date']).add_suffix('_tasmax')

    def process_rzsm(self):
        files, hist_file, ssp_files = self.get_filenames('rzsm_')
        df = pd.concat([self.read_data(ssp_file)
                        .assign(ssp=os.path.basename(ssp_file).split('_')[2].split('.')[0]) 
                        for ssp_file in ssp_files]
                      ).reset_index()
        # Use ssp245 data for 126 and 370
        # df = pd.concat([df, df.assign(ssp='ssp126').copy(), df.assign(ssp='ssp370').copy()]).dropna()
        '''
        ##################################################
        REMOVE LINE ABOVE
        ##################################################
        '''
        
        df = df.sort_values(['ssp', 'date']).set_index(['ssp', 'date']).add_suffix('_rzsm')
        hist_df = self.filter_dates(df, self.historical_years, False)
        ssp_df = self.filter_dates(df, self.ssp_years, False)
        thres_df = hist_df.groupby('ssp').quantile(self.threshold[0]['rzsm'][1])
        return ssp_df, thres_df

    def process_pr(self):
        files, hist_file, ssp_files = self.get_filenames('pr_')
        # Combine historical and ssp dataframes
        historical_df = self.read_data(hist_file)
        pr_dfs = {os.path.basename(ssp_file).split('_')[2].split('.')[0]: 
                  pd.concat([historical_df, self.read_data(ssp_file)]).dropna(how='all') * 86400 # to mm/day
                   for ssp_file in ssp_files}
        return (pd.concat([df.assign(ssp=ssp_name).reset_index() for ssp_name, df in pr_dfs.items()],
                         ignore_index=True).set_index(['ssp', 'date']).add_suffix('_pr'))
    
    ##################################################
    ##           Determine Compound Events          ##
    ##################################################

    def get_common_columns(self, df):
        return {col.split('_')[0] for col in df.columns 
                if Counter(c.split('_')[0] for c in df.columns)[col.split('_')[0]] > 1}

    def process_compound(self, dfs, threshold, thres_df:pd.DataFrame()=None):
        compound = []
        for var, (op, p) in threshold.items():
            df = dfs.filter(regex=var)
            if df.empty:
                raise ValueError('Empty dataframe')
            if thres_df is not None and var in ['rzsm']:
                cc = []
                for ssp in df.index.get_level_values('ssp').unique():
                    df_ssp = df.xs(ssp, level='ssp')
                    p_ssp = thres_df.loc[ssp, p]
                    if p_ssp.empty:
                        raise ValueError(f'No data found for {ssp} in thres_df')
                    c = self.COMP_OPS[op](df_ssp, p_ssp)
                    c.index = pd.MultiIndex.from_tuples([(ssp, i) for i in c.index], names=['ssp', 'date'])
                    cc.append(c)
                compound.append(pd.concat(cc, axis=0))
            else:
                compound.append(self.COMP_OPS[op](df, p))
        compound = pd.concat(compound, axis=1)
            
        # Determine if compound
        common_cols = self.get_common_columns(compound)
        for col in common_cols:
            df = compound.filter(regex=col)
            compound[f'{col}_compound'] = df.all(axis=1)
        return compound
    
    ##################################################
    ##               Calculate Metrics              ##
    ##################################################
    
    def group_data(self, df, suffix):
        '''Group data based on specified criteria'''
        df_ = df.copy().reset_index()
        df_.date = df_.date.dt.year
        return (df_.set_index(['ssp', 'date']).filter(regex=suffix)
               .groupby(['ssp', 'date']))

    def max_consecutive(self, s):
        '''Calculate max consecutive True values in a series if max > 1'''
        result = (s * (s.groupby((s != s.shift()).cumsum()).cumcount() + 1)).max()
        return result if result > 1 else 0

    def total_consecutive(self, s):
        '''Calculate total True values in a series if more than 1 consecutive True'''
        s = s.groupby((s != s.shift()).cumsum()).sum()
        return s[s > 1].sum()

    def total_sequence(self, s):
        return sum(1 for key, group in groupby(s) if key and sum(group) > 1)

    def mean_duration(self, s):
        sequences = [len(list(g)) for k, g in groupby(s) if k]
        sequences = [s for s in sequences if s > 1]
        return round(np.mean(sequences), 2) if sequences else 0

    ##################################################
    ##                    Main                      ##
    ##################################################
    
    def main(self):
        # Process variables
        tres_rzsm = None
        if self.event in ['CWHE','CDHE']:
            self.historical_years, self.ssp_years = (1981, 2020), (2021, 2100)
            self.spi_years = ((datetime.date(self.historical_years[0], 1, 1) - self.delta).year,
                              self.ssp_years[1])
            pr = self.filter_dates(self.process_spi(), self.ssp_years, year_month=True)
            tm = self.filter_dates(self.process_tasmax(), self.ssp_years, year_month=True)
            spi = pr.filter(regex='_spi$')
            dfs = pd.concat([spi, tm], axis=1).dropna()

            groups = {'daily':  pd.concat([spi, spi.mean(axis=1).rename(f'mean_spi'), 
                                           tm,  tm.mean(axis=1).rename(f'mean_tasmax')], axis=1),
                      'pr': self.group_data(pr.filter(regex='_pr$'), '_pr$').mean().reset_index(),
                      'spi': self.group_data(spi, '_spi$').mean().reset_index(),
                      'tasmax': self.group_data(tm, '_tasmax$').mean().reset_index()
                      }
            
        elif self.event in ['CFE']:
            self.historical_years, self.ssp_years = (1950, 2014), (2015, 2099)
            rzsm, tres_rzsm = self.process_rzsm() # ssp dailies, hist threshold
            pr = self.filter_dates(self.process_pr().rolling(self.scale).sum(), self.ssp_years, year_month=False)
            dfs = pd.concat([rzsm, pr], axis=1).dropna()
            groups = {'daily': pd.concat([rzsm, rzsm.mean(axis=1).rename(f'mean_rzsm'),
                                          pr,   pr.mean(axis=1).rename(f'mean_pr')], axis=1),
                      'pr': self.group_data(pr.filter(regex='_pr$'), '_pr$').mean().reset_index(),
                      'rzsm': self.group_data(rzsm, '_rzsm$').mean().reset_index(),
                      'tres_rzsm': tres_rzsm
                      }
            
        dfs = dfs.loc[:, ~dfs.columns.duplicated()]

        results, compounds = [], []
        for threshold in self.thresholds:
            # print(threshold)
            thres = '_'.join([f'{v}{c}{round(p, 2)}' for v, (c, p) in threshold.items()])
            compounds.append(self.process_compound(dfs, threshold, tres_rzsm))
            compounds[-1].insert(0, 'threshold', thres)  
            grouped = self.group_data(compounds[-1], f'_compound$')
            result = pd.concat([
                # Total Compound Days
                grouped.sum().add_suffix('_day_total'),
                # Total Compound Events
                grouped.apply(lambda x: x.apply(self.total_consecutive)).add_suffix('_event_total'),
                # Total Compound Event Sequences
                grouped.apply(lambda x: x.apply(self.total_sequence)).add_suffix('_sequence_total'),
                # Max Compound Event Sequence Duration
                grouped.apply(lambda x: x.apply(self.max_consecutive)).add_suffix('_duration_max'),
                # Average Compound Event Sequence Duration
                grouped.apply(lambda x: x.apply(self.mean_duration)).add_suffix('_duration_mean')
            ], axis=1)
            result.insert(0, 'threshold', thres)
            results.append(result)
        
        results = pd.concat(results).reset_index().set_index([ 'threshold', 'ssp', 'date'])
        groups['compound'] = pd.concat(compounds).reset_index()

        aggs = [[], []]
        for col in self.outputs:
            i = 0
            df = results.filter(regex=col)
            if df.shape[1]==0:
                df = groups[col[1:]].set_index(['ssp', 'date'])
                i = 1
            aggs[i].extend([
                df.mean(axis=1).rename(f'mean{col}'),
                df.median(axis=1).rename(f'med{col}'),
                df.quantile(0.1, axis=1).rename(f'p10{col}'),
                df.quantile(0.9, axis=1).rename(f'p90{col}'),
            ])
        results = pd.concat([results]+aggs[0], axis=1).reset_index()
        var_aggs = pd.concat(aggs[1], axis=1).reset_index()         
        return results, var_aggs, groups

        



















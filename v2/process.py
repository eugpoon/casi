import os
import warnings
import pandas as pd
import numpy as np
import datetime
import operator
from collections import Counter
from functools import reduce
from itertools import groupby
from spi import SPI

warnings.filterwarnings('ignore')

class Compound:
    def __init__(self, center, event, months, freq, scale, thresholds, gamma_n):
        self.center = center
        self.event = event
        self.months = months
        self.freq = freq
        self.scale = scale
        self.thresholds = thresholds
        self.gamma_n = gamma_n
        self.DATA_PATH = '../compound'
        self.VARIABLES = ['pr_', 'tasmax_']
        self.COMP_OPS = {
            '<': operator.lt, '<=': operator.le,
            '>': operator.gt, '>=': operator.ge,}
        
        if freq == 'D':
            self.temporal_res = 'daily'
            self.delta = datetime.timedelta(days=scale)
        elif freq == 'M':
            self.temporal_res = 'monthly_avg'
            self.delta = datetime.timedelta(days=30 * scale)  # approximation for months
        else:
            raise ValueError(f"{freq} should be one of ['D', 'M']")

        self.HISTORICAL_YEARS, self.SSP_YEARS, self.SPI_YEARS = None, None, None
        
        self.files = self.get_files()
       
    ##################################################
    ##              Get and read files              ##
    ##################################################
    
    def get_files(self):
        '''Returns list of filenames in a directory that contain a specific string'''
        return [os.path.join(self.DATA_PATH, f) for f in os.listdir(self.DATA_PATH) 
                if self.center in f and f.endswith('.csv') and self.temporal_res in f
                and any(v in f for v in self.VARIABLES)]

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

    def process_spi(self):
        # Get filenames
        files = [file for file in self.files if 'pr_' in file]
        historical_file = next(file for file in files if 'historical' in file)
        ssp_files = sorted([file for file in files if 'ssp' in file])

        # Combine historical and ssp dataframes
        historical_df = self.read_data(historical_file)
        spi_dfs = {os.path.basename(ssp_file).split('_')[2].split('.')[0]: 
                   self.filter_dates(pd.concat([historical_df, self.read_data(ssp_file)]), self.SPI_YEARS,
                                     year_month=False).dropna(how='all') * 86400 # Convert pr to mm/day
                   for ssp_file in ssp_files}

        # Calculate SPI
        for ssp_name, df in spi_dfs.items():
            print(f'Processing {ssp_name} spi ({min(df.index.date)} to {max(df.index.date)})...') 
            spi_dfs[ssp_name] = pd.concat([SPI().calculate(df, self.HISTORICAL_YEARS, self.SPI_YEARS,
                        self.months, freq=self.freq, scale=self.scale, gamma_n=self.gamma_n),
                                           df.add_suffix('_pr')], axis=1)
        
        # Combine all SSPs
        return (pd.concat([df.assign(ssp=ssp_name).reset_index() for ssp_name, df in spi_dfs.items()],
                         ignore_index=True).set_index(['ssp', 'date']))

    def process_tasmax(self):
        # Get filenames
        files = [file for file in self.files if 'tasmax_' in file]
        historical_file = next(file for file in files if 'historical' in file)
        ssp_files = sorted([file for file in files if 'ssp' in file])
        
        # Combine ssp dataframes
        tm = pd.concat([(self.read_data(ssp_file)-273.15) # Convert Kelvin to Celsius
                          .assign(ssp=os.path.basename(ssp_file).split('_')[2].split('.')[0]) 
                          for ssp_file in ssp_files])

        return tm.reset_index().set_index(['ssp', 'date']).add_suffix('_tasmax')

    def process_rzsm(self):
        # Get filenames
        files = [file for file in self.files if 'rzsm_' in file]
        historical_file = next(file for file in files if 'historical' in file)
        ssp_files = sorted([file for file in files if 'ssp' in file])
        
        # Combine ssp dataframes
        tm = pd.concat([(self.read_data(ssp_file)-273.15) # Convert Kelvin to Celsius
                          .assign(ssp=os.path.basename(ssp_file).split('_')[2].split('.')[0]) 
                          for ssp_file in ssp_files])

        return tm.reset_index().set_index(['ssp', 'date']).add_suffix('_tasmax')
    
    ##################################################
    ##           Determine Compound Events          ##
    ##################################################

    def get_common_columns(self, df):
        return {col.split('_')[0] for col in df.columns 
                    if Counter(c.split('_')[0] for c in df.columns)[col.split('_')[0]] > 1}

    def process_compound(self, dfs, threshold):
        compound = []
        for var, (op, p) in threshold.items():
            df = dfs.filter(regex=var)
            if df.empty:
                raise ValueError('Empty dataframe')
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
        if self.event in ['CWHE','CDHE']:
            self.HISTORICAL_YEARS, self.SSP_YEARS = (1981, 2020), (2021, 2100)
            self.SPI_YEARS = ((datetime.date(self.HISTORICAL_YEARS[0], 1, 1) - self.delta).year,
                              self.SSP_YEARS[1])
            
            pr = self.filter_dates(self.process_spi(), self.SSP_YEARS, year_month=True)
            tm = self.filter_dates(self.process_tasmax(), self.SSP_YEARS, year_month=True)
            spi = pr.filter(regex='_spi$')
            dfs = pd.concat([spi, tm], axis=1)
            
            groups = {'pr': self.group_data(pr.filter(regex='_pr$'), '_pr$').mean().reset_index(),
                      'spi': self.group_data(spi, '_spi$').mean().reset_index(),
                      'tasmax': self.group_data(tm, '_tasmax$').mean().reset_index()
                      }
            
        elif self.event in ['CFE']:
            self.HISTORICAL_YEARS, self.SSP_YEARS = (1950, 2014), (2015, 2100)
            # pr = self.filter_dates(self.process_spi(), self.SSP_YEARS, year_month=True)
            tm = self.filter_dates(self.process_tasmax(), self.SSP_YEARS, year_month=True)
            dfs = pd.concat([spi, tm], axis=1)
            
        dfs = dfs.loc[:, ~dfs.columns.duplicated()]

        results, compounds = [], []
        for threshold in self.thresholds:
            thres = '_'.join([f'{v}{c}{round(p, 1)}' for v, (c, p) in threshold.items()])
            compounds.append(self.process_compound(dfs, threshold))
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
        
        results = pd.concat(results).reset_index()
        compounds = pd.concat(compounds).reset_index()
        
        

        return results, compounds, groups

        



















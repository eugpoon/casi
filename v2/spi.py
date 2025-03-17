'''
Modified code from https://github.com/e-baumer/standard_precip/blob/master/standard_precip/base_sp.py
- original --> new
- Default df input: date as a column --> date as index
- Processes specified pr data one at a time --> processes whole dataframe (assumes all pr data)
- Fit distribution using all available data --> based on baseline year range
- If freq='D', use dayofyear --> use 'MM-DD'
- Includes n days prior rolling sums in gamma distribution
- Removed week option

Note: only daily version is updated, idk if it works the same on monthly
'''

from functools import reduce
import numpy as np
import pandas as pd
import scipy.stats as scs
import matplotlib.pyplot as plt

from standard_precip.lmoments import distr


class SPI():
    '''
    Calculate the SPI or SPEI index. A user specified distribution is fit to the precip data.
    The CDF of this distribution is then calculated after which the the standard normal
    distribution is calculated which gives the index. A distribution can be fit over the
    precipitation data either using MLE or L-moments. NCAR's SPI calculators and the SPI and
    SPEI R packages both use L-moments to fit the distribution. There are advantages and
    disadvantages to each technique.

    This calculation can be done on any time scale. Built in temporal scales include daily,
    weekly, and monthly; however, the user can define their own timescale.

    One should put some thought into the type of distribution fit to the
    data. Precipitation can have zero value and some distributions are only
    defined over interval (0, inf). Python's gamma distribution is defined
    over [0, inf). In addition SPEI which is constructed from precipitation
    - PET or (P-PET) can take on negative values.
    '''

    def __init__(self):
        self.distrb = None
        self.non_zero_distr = ['gam', 'pe3']

    @staticmethod
    def rolling_window_sum(df: pd.DataFrame, span: int=1, window_type: str=None,
                           center: bool=False, **kwargs):
        return df.rolling(window=span, win_type=window_type, center=center, **kwargs
                         ).sum().add_suffix(f'_roll{span}')

    def fit_distribution(self, data: pd.DataFrame, dist_type: str, fit_type: str='lmom', **kwargs):
        '''
        Parameters:
        - data: pd.DataFrame with multiple columns
        - dist_type: str, type of distribution
        - fit_type: str, method for fitting ('lmom' or 'mle')
        - **kwargs: additional keyword arguments for fitting
    
        Returns:
        - A DataFrame with parameters and p_zero for each column
        '''
    
        def fit_column(col_data):
            '''
            Fit given distribution to historical precipitation data.
            The fit is accomplished using either L-moments or MLE (Maximum Likelihood Estimation).

            For distributions that use the Gamma Function (Gamma and Pearson 3) remove observations
            that have 0 precipitation values and fit using non-zero observations. Also find probability
            of zero observation (estimated by number of zero obs / total obs). This is for latter use
            in calculating the CDF using (Thom, 1966. Some Methods of Climatological Analysis)
            '''
            # Get distribution type
            self.distrb = getattr(distr, dist_type)
        
            # Determine zeros if distribution can not handle x = 0
            p_zero = None
            if dist_type in self.non_zero_distr:
                p_zero = col_data[col_data == 0].shape[0] / col_data.shape[0]
                col_data = col_data[col_data != 0]
        
            if (col_data.shape[0]<4) or (p_zero==1):
                params = None
            else: # Fit distribution
                if fit_type == 'lmom':
                    params = self.distrb.lmom_fit(col_data, **kwargs)
                elif fit_type == 'mle':
                    params = self.distrb.fit(col_data, **kwargs)
                else:
                    raise AttributeError(f'{fit_type} not one of [mle, lmom]')
        
            return params, p_zero
    
        # Apply fit_column function to each column
        results = data.apply(lambda x: fit_column(x)).T
        results.columns = ['params', 'p_zero']
        return results # index are models

    def cdf_to_ppf(self, data: pd.DataFrame, params_df: pd.DataFrame):
        '''    
        Parameters:
        - data: pd.DataFrame with multiple columns
        - params_df: pd.DataFrame with params, p_zero for each column in data df
    
        Returns:
        - A DataFrame with SPI/SPEI values for each column in data df
        '''
        
        def cdf_to_ppf_column(col_data, params, p_zero):
            '''
            Take the specific distributions fitted parameters and calculate the
            cdf. Apply the inverse normal distribution to the cdf to get the SPI
            SPEI. This process is best described in Lloyd-Hughes and Saunders, 2002
            which is included in the documentation.
            '''
            # Calculate the CDF of observed precipitation
            if not (p_zero is None):
                if params:
                    cdf = p_zero + (1 - p_zero) * self.distrb.cdf(col_data, **params)
                else:
                    cdf = np.empty(col_data.shape)
                    cdf.fill(np.nan)
            else:
                cdf = self.distrb.cdf(col_data, **params)
        
            # Apply inverse normal distribution
            norm_ppf = scs.norm.ppf(cdf)
            norm_ppf[np.isinf(norm_ppf)] = np.nan
        
            return norm_ppf
    
        # Apply cdf_to_ppf_column function to each column    
        spi = data.apply(lambda x: cdf_to_ppf_column(x, params_df.loc[x.name, 'params'],
                                                         params_df.loc[x.name, 'p_zero']))
        return spi


    def extract_days(self, df, month_day, n):
        matching_dates = df[df['D'] == month_day].index
        start_dates = matching_dates - pd.Timedelta(days=n-1)    
        slices = []
        for start, end in zip(start_dates, matching_dates):
            slices.append(df.loc[start:end])
        return pd.concat(slices, ignore_index=False)

    def calculate(self, df: pd.DataFrame, base_years:tuple, spi_years:tuple, months:list, freq: str='D', 
                  scale: int=1, gamma_n:int=10, fit_type: str='lmom', dist_type: str='gam', 
                  **dist_kwargs) -> pd.DataFrame:
        column = df.columns
        
        if scale > 1:
            df = self.rolling_window_sum(df, scale)

        filtered = df[(df.index.year >= base_years[0]) & (df.index.year <= spi_years[1])]
        
        print(f'--> filtered: {min(filtered.index.date)} to {max(filtered.index.date)}')
        print(f'--> base: {base_years[0]} to {base_years[1]}')

        freq_map = {'D': '%m-%d', 'M': '%m'}
        if freq not in freq_map:
            raise AttributeError(f'{freq} not one of [M, D]')
        filtered[freq] = filtered.index.strftime(freq_map[freq])

        # Only include specified months in freq_range
        month_strings = [f'{m:02d}' for m in months]
        freq_range = [s for s in filtered[freq].unique() if s.split('-')[0] in month_strings]
        # freq_range = filtered[freq].unique()
        
        dfs = []
        for j in freq_range:
            precip = filtered.loc[(filtered.index.year > base_years[1]) & 
                                  (filtered[freq] == j)].drop(columns=freq).dropna()

            # Fit distribution using only baseline period data
            baseline_data = self.extract_days(filtered, j, gamma_n).dropna()
            baseline_data = baseline_data[(baseline_data.index.year >= base_years[0]) & 
                                          (baseline_data.index.year <= base_years[1])]
            params_df = self.fit_distribution(baseline_data.drop(columns=freq), dist_type, 
                                              fit_type, **dist_kwargs)
            # Calculate SPI
            spi = self.cdf_to_ppf(precip, params_df)
            spi.columns = column
            dfs.append(spi.add_suffix('_spi'))
        
        return filtered.merge(pd.concat(dfs, axis=0), left_index=True, right_index=True).drop(columns=freq)
            


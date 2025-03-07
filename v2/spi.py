'''
Modified code from https://github.com/e-baumer/standard_precip/blob/master/standard_precip/base_sp.py
- original --> new
- Default df input: date as a column --> date as index
- Specified pr data --> whole dataframe (all pr)
- Fit distribution using all available data --> based on baseline year range
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
        self._df_copy = None

    @staticmethod
    def rolling_window_sum(df: pd.DataFrame, span: int=1, window_type: str=None,
                           center: bool=False, **kwargs):
        return df.rolling(window=span, win_type=window_type, center=center, **kwargs
                         ).sum().add_suffix(f'_roll{span}')

    def fit_distribution(self, data: np.array, dist_type: str, fit_type: str='lmom', **kwargs):
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
            p_zero = data[data == 0].shape[0] / data.shape[0]
            data = data[data != 0]

        if (data.shape[0]<4) or (p_zero==1):
            params = None
        else: # Fit distribution
            if fit_type == 'lmom':
                params = self.distrb.lmom_fit(data, **kwargs)
            elif fit_type == 'mle':
                params = self.distrb.fit(data, **kwargs)
            else:
                raise AttributeError(f'{fit_type} not one of [mle, lmom]')

        return params, p_zero

    def cdf_to_ppf(self, data, params, p_zero):
        '''
        Take the specific distributions fitted parameters and calculate the
        cdf. Apply the inverse normal distribution to the cdf to get the SPI
        SPEI. This process is best described in Lloyd-Hughes and Saunders, 2002
        which is included in the documentation.

        '''

        # Calculate the CDF of observed precipitation on a given time scale
        if not (p_zero is None):
            if params:
                cdf = p_zero + (1 - p_zero) * self.distrb.cdf(data, **params)
            else:
                cdf = np.empty(data.shape)
                cdf.fill(np.nan)
        else:
            cdf = self.distrb.cdf(data, **params)

        # Apply inverse normal distribution
        norm_ppf = scs.norm.ppf(cdf)
        norm_ppf[np.isinf(norm_ppf)] = np.nan

        return norm_ppf

    def calculate(self, df: pd.DataFrame, base_years:tuple, spi_years:tuple, freq: str='M', scale: int=1, 
                  fit_type: str='lmom', dist_type: str='gam', **dist_kwargs) -> pd.DataFrame:
        
        if scale > 1:
            df = self.rolling_window_sum(df, scale)
        
        self._df_copy = df[(df.index.year >= base_years[0]) & (df.index.year <= spi_years[1])].copy()
        date_index = self._df_copy.index
        print(f'--> filtered: {min(date_index.date)} to {max(date_index.date)}')
        print(f'--> base: {base_years[0]} to {base_years[1]}')

        freq_map = {'D': 'dayofyear', 'W': 'isocalendar().week', 'M': 'month'}
        if freq not in freq_map:
            raise AttributeError(f'{freq} not one of [M, W, D]')
        
        self._df_copy[freq] = getattr(date_index, freq_map[freq])

        freq_range = self._df_copy[freq].unique()

        dfs = []
        for col in df.columns[:-1]:
            dfs_p = pd.DataFrame(index=self._df_copy.index)
            for j in freq_range:
                precip_single = self._df_copy.loc[self._df_copy[freq]==j, col].dropna()
        
                # Fit distribution using only baseline period data
                baseline_mask = ((precip_single.index.year >= base_years[0]) & 
                                 (precip_single.index.year <= base_years[1]))
                baseline_data = precip_single[baseline_mask]
                params, p_zero = self.fit_distribution(
                    np.sort(baseline_data)[::-1], dist_type, fit_type, **dist_kwargs)
        
                spi = self.cdf_to_ppf(precip_single, params, p_zero)
                dfs_p.loc[precip_single.index, f'{col.split('_')[0]}_spi'] = spi
        
            dfs.append(dfs_p)
        # display(pd.concat([self._df_copy] + dfs, axis=1).drop(columns=freq).head(40))

        return pd.concat([self._df_copy] + dfs, axis=1).drop(columns=freq)

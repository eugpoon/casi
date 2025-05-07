import argparse
import numpy as np
import pandas as pd
import requests
from scipy.stats import circmean
from datetime import datetime
from meteostat import Stations, Hourly
from noaa_coops import Station
import warnings
warnings.filterwarnings('ignore')

res = 'h'
agg = 'max'

def get_stations(stations, output_dir='data'):
    records = []
    for id in stations:    
        station = Station(id)
        meta = station.metadata
        name = meta.get('name')
        lat = meta.get('lat')
        lon = meta.get('lng')
        for var, dates in station.data_inventory.items():
            records.append({'id': id, 'name': name,
                            'latitude': lat, 'longitude': lon, 'variable': var,
                            'start_date': dates.get('start_date', 'N/A'),
                            'end_date': dates.get('end_date', 'N/A')})

    noaa = pd.DataFrame(records)
    exclude_keywords = r'Temperature|Monthly|Pressure|Conductivity|Rain Fall'
    noaa = (noaa[~noaa.variable.str.contains(exclude_keywords, case=False, na=False)]
          .sort_values(['variable', 'start_date']))
    noaa.to_csv(f'{output_dir}/noaa_tc.csv', index=False)
    

def clean_date(date_str):
    '''Convert ISO datetime string to YYYYMMDD format.'''
    return datetime.fromisoformat(date_str).strftime('%Y%m%d')

    
def fetch_and_process_station_data(id, start_date, end_date, product, 
                                  datum=None, rename_cols=None, filter_cols=None, 
                                  output_path=None):
    '''Fetch and process data for any station type.'''
    try:
        start_date = clean_date(start_date)
        end_date = clean_date(end_date)
        print(f'Station {id}: {start_date} to {end_date}')
        params = {'begin_date': start_date, 'end_date': end_date, 
                 'product': product, 'units': 'metric', 'time_zone': 'lst_ldt'}
        if datum:
            params['datum'] = datum
            
        station = Station(id)
        data = station.get_data(**params)
        
        if filter_cols:
            data = data.dropna(subset=filter_cols, how='all')
        if rename_cols:
            data = data.rename(columns=rename_cols)
            
        data.index.name = None
        
        if 'flag' in data.columns and len(data.flag.unique()) > 1:
            print(data[data.flag.str.contains('1')])
            print(data.flag.value_counts())
            
        if output_path:
            data.to_parquet(output_path)
            
        return data
    except Exception as e:
        print(f'Failed for station {id}: {e}')
        return None


def circular_mean(deg_series):
    '''Calculate circular mean for directional data.'''
    return circmean(deg_series.dropna(), high=360, low=0)


def process_wind_data(wind_df, output_dir='data'):
    ''' Process all wind stations.
        Field	Description
        t	    Time - Date and time of the observation
        s	    Speed - Measured wind speed
        d	    Direction - wind direction in degrees
        dr	    Direction - wind direction in text
        g	    Gust - Measured wind gust speed
        f	    Data Flags - in order of listing:
                -- (X) A flag that when set to 1 indicates that the maximum wind speed was exceeded
                -- (R) A flag that when set to 1 indicates that the rate of change tolerance limit was exceeded

    '''
    for _, row in wind_df.iterrows():
        fetch_and_process_station_data(
            row.id, row.start_date, row.end_date, 
            product='wind',
            rename_cols={'s':'speed', 'd':'dirdeg', 'dr':'dir', 'g':'gust', 'f':'flag'},
            filter_cols=['s', 'd', 'g'],
            output_path=f'{output_dir}/noaa_tc.wind.{row.id}.parquet'
        )

def aggregate_wind_data(wind_stations, output_dir='data'):
    '''Aggregate wind data across stations.'''
    wind_dfs = []
    for id in wind_stations:
        try:
            df = pd.read_parquet(f'{output_dir}/noaa_tc.wind.{id}.parquet').replace('', np.nan)
            df = df[(df.dirdeg >= 0) & (df.dirdeg <= 360) & (df.gust >= df.speed) & (df.gust >= 0) &  (df.speed >= 0)]
            
            # Resample hourly
            df = pd.concat([
                df[['speed', 'gust']].resample(res).agg(agg),
                df['dirdeg'].resample(res).apply(circular_mean),
                df.select_dtypes(exclude='number').resample(res).apply(lambda x: x.mode()[0] if not x.empty else np.nan)
            ], axis=1)
            
            wind_dfs.append(df.add_suffix(f'_{id}'))
        except Exception as e:
            print(f'Error aggregating wind data for {id}: {e}')
    
    if wind_dfs:
        result = pd.concat(wind_dfs, axis=1)
        result.to_parquet(f'{output_dir}/noaa_tc.wind.parquet')
        return result
    return None

def process_water_data(water_df, output_dir='data'):
    ''' Process water level data.
        Field	Description
        t	    Time - Date and time of the observation
        v	    Value - Measured water level height
        s	    Sigma - Standard deviation of 1 second samples used to compute the water level height
        f	    Data Flags - in order of listing:
                -- (I) A flag that when set to 1 indicates that the water level value has been inferred
                -- (F) A flag that when set to 1 indicates that the flat tolerance limit was exceeded
                -- (R) A flag that when set to 1 indicates that the rate of change tolerance limit was exceeded
                -- (T) A flag that when set to 1 indicates that either the maximum or minimum expected 
                       water level height limit was exceeded
        q	    Quality Assurance/Quality Control level
                -- p = preliminary
                -- v = verified
    '''
    for _, row in water_df.iterrows():
        fetch_and_process_station_data(
            row.id, row.start_date, row.end_date,
            product='water_level', datum='MHHW',
            rename_cols={'v': 'mhhw', 's': 'sigma', 'f': 'flag', 'q': 'quality'},
            filter_cols=['v'],
            output_path=f'{output_dir}/noaa_tc.waterlevel.mhhw.{row.id}.parquet'
        )

def aggregate_water_data(water_stations, output_dir='data'):
    '''Aggregate water data across stations.'''
    water_dfs = []
    for id in water_stations:
        try:
            mhhw = pd.read_parquet(f'{output_dir}/noaa_tc.waterlevel.mhhw.{id}.parquet')
            mhhw = mhhw[(mhhw.sigma <= 3) & (mhhw.quality=='v')][['mhhw']].resample(res).agg('max')
            water_dfs.append(mhhw.dropna().add_suffix(f'_{id}'))
        except Exception as e:
            print(f'Error aggregating water data for {id}: {e}')
    
    if water_dfs:
        result = pd.concat(water_dfs, axis=1)
        result.to_parquet(f'{output_dir}/noaa_tc.water.parquet')
        return result
    return None


def main(output_dir='data'):
    ####################
    #      NOAA        #
    ####################
    stations = [8637689, 8638610, 8638901, 8638614, 8638511, 8637624, 8638595]
    # stations = [8638901]
    get_stations(stations, output_dir)

    noaa = pd.read_csv(f'{output_dir}/noaa_tc.csv')
    noaa.id = noaa.id.astype(str)
    wind = noaa[noaa.variable=='Wind']
    water = noaa[noaa.variable=='Verified 6-Minute Water Level']
    wind_stations, water_stations = wind.id.values, water.id.values

    # Wind data
    print('Wind')
    process_wind_data(wind, output_dir)
    wind_data = aggregate_wind_data(wind_stations, output_dir)

    # Water data
    print('Water')
    process_water_data(water, output_dir)
    water_data = aggregate_water_data(water_stations, output_dir)

    ####################
    #    Meteostat     #
    ####################
    print('Meteostat')
    met_stations = Stations().nearby(37.0862, -76.3809).fetch(10)
    met_stations = met_stations[met_stations.distance <= 30000]  # km
    met_stations.to_csv(f'{output_dir}/meteo.csv')

    for id, row in met_stations.iterrows():
        print(id)
        dd = Hourly(id, row.hourly_start, row.hourly_end).fetch()
        dd.dropna(axis=1, how='all').to_parquet(f'{output_dir}/meteo.{id}.parquet')

    meteo = pd.read_csv(f'{output_dir}/meteo.csv')
    meteo_df = []
    for id in meteo.id:
        print(id)
        dd = (pd.read_parquet(f'{output_dir}/meteo.{id}.parquet')[['prcp', 'wdir', 'wspd']]
              .dropna(how='all').rename(columns={'prcp': 'pr', 'wdir': 'dir', 'wspd': 'speed'}))
        dd.speed = dd.speed / 3.6  # km/h to m/s
        if res != 'h':
            dd = pd.concat([
                    dd.pr.resample(res).sum(),
                    dd.dir.resample(res).apply(circular_mean),
                    dd.drop(columns=['pr', 'dir']).resample(res).agg(agg),
                ], axis=1)
        meteo_df.append(dd.add_suffix(f'_{id}'))
    pd.concat(meteo_df, axis=1).to_parquet(f'{output_dir}/meteo.pr_wind.parquet')

    ####################
    #      CMIP6       #
    ####################
    print('CMIP6')
    files = ['LARC_pr_historical_daily.csv', 'LARC_pr_ssp126_daily.csv', 'LARC_pr_ssp245_daily.csv',
             'LARC_pr_ssp370_daily.csv', 'LARC_sfcWind_historical_daily.csv', 'LARC_sfcWind_ssp245_daily.csv']
    cmip_df = {}
    for file in files:
        name, scenario = file.split('_')[1:3]
        dd = (pd.read_csv(f'data/compound/{file}')
              .rename(columns={'Unnamed: 0': 'date'})
              .assign(date=lambda d: pd.to_datetime(d['date']), scenario=scenario)
              .set_index(['scenario', 'date']).add_prefix(f'{name.lower()}_'))
        if name == 'pr':
            dd *= 86400
        dd.columns = dd.columns.str.lower()
        cmip_df.setdefault(name.lower(), []).append(dd)

    cmip_df = pd.concat([pd.concat(v) for v in cmip_df.values()], axis=1)
    cmip_df.to_parquet(f'{output_dir}/cmip.pr_wind.parquet')

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--output_dir', default='data', help='Output directory for saved data')
    args = parser.parse_args()
    main(output_dir=args.output_dir)
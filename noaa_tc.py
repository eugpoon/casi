import numpy as np
import pandas as pd
from scipy.stats import circmean
from datetime import datetime
from noaa_coops import Station

def get_stations(stations, output_dir='data'):
    records = []
    for station_id in stations:    
        station = Station(station_id)
        meta = station.metadata
        name = meta.get('name')
        lat = meta.get('lat')
        lon = meta.get('lng')
        for var, dates in station.data_inventory.items():
            records.append({'station_id': station_id, 'station_name': name,
                            'lat': lat, 'lon': lon, 'variable': var,
                            'start_date': dates.get('start_date', 'N/A'),
                            'end_date': dates.get('end_date', 'N/A')})

    noaa = pd.DataFrame(records)
    exclude_keywords = r'Temperature|Monthly|Pressure|Conductivity|Rain Fall'
    noaa = (noaa[~noaa.variable.str.contains(exclude_keywords, case=False, na=False)]
          .sort_values(['variable', 'start_date']))
    noaa.to_csv(f'{output_dir}/noaa_tc.csv', index=False)

def clean_date(date_str):
    """Convert ISO datetime string to YYYYMMDD format."""
    return datetime.fromisoformat(date_str).strftime('%Y%m%d')

def fetch_and_process_station_data(station_id, start_date, end_date, product, 
                                  datum=None, rename_cols=None, filter_cols=None, 
                                  output_path=None):
    """Generic function to fetch and process data for any station type."""
    try:
        start_date = clean_date(start_date)
        end_date = clean_date(end_date)
        print(f'Station {station_id}: {start_date} to {end_date}')
        params = {'begin_date': start_date, 'end_date': end_date, 
                 'product': product, 'units': 'metric', 'time_zone': 'lst'}
        if datum:
            params['datum'] = datum
            
        station = Station(station_id)
        data = station.get_data(**params)
        
        if filter_cols:
            data = data.dropna(subset=filter_cols, how='all')
        if rename_cols:
            data = data.rename(columns=rename_cols)
            
        data.index.name = None
        
        if 'flag' in data.columns and len(data.flag.unique()) > 1:
            display(data[data.flag.str.contains('1')])
            print(data.flag.value_counts())
            
        if output_path:
            data.to_parquet(output_path)
            
        return data
    except Exception as e:
        print(f"Failed for station {station_id}: {e}")
        return None

def circular_mean(deg_series):
    """Calculate circular mean for directional data."""
    return circmean(deg_series.dropna(), high=360, low=0)

def process_wind_data(wind_df, output_dir='data'):
    """ Process all wind stations.
        Field	Description
        t	    Time - Date and time of the observation
        s	    Speed - Measured wind speed
        d	    Direction - wind direction in degrees
        dr	    Direction - wind direction in text
        g	    Gust - Measured wind gust speed
        f	    Data Flags - in order of listing:
                -- (X) A flag that when set to 1 indicates that the maximum wind speed was exceeded
                -- (R) A flag that when set to 1 indicates that the rate of change tolerance limit was exceeded

    """
    for _, row in wind_df.iterrows():
        fetch_and_process_station_data(
            row.station_id, row.start_date, row.end_date, 
            product='wind',
            rename_cols={'s':'speed', 'd':'dirdeg', 'dr':'dir', 'g':'gust', 'f':'flag'},
            filter_cols=['s', 'd', 'g'],
            output_path=f"{output_dir}/noaa_tc.wind.{row.station_id}.parquet"
        )

def aggregate_wind_data(wind_stations, output_dir='data'):
    """Aggregate wind data across stations."""
    wind_dfs = []
    for station_id in wind_stations:
        try:
            df = pd.read_parquet(f'{output_dir}/noaa_tc.wind.{station_id}.parquet').replace('', np.nan)
            df = df[(df.dirdeg >= 0) & (df.dirdeg <= 360) & (df.gust >= df.speed)]
            
            # Resample hourly
            df = pd.concat([
                df[['speed', 'gust']].resample('h').max(),
                df['dirdeg'].resample('h').apply(circular_mean),
                df.select_dtypes(exclude='number').resample('h').apply(lambda x: x.mode()[0] if not x.empty else np.nan)
            ], axis=1)
            
            wind_dfs.append(df.add_suffix(f'_{station_id}'))
        except Exception as e:
            print(f"Error aggregating wind data for {station_id}: {e}")
    
    if wind_dfs:
        result = pd.concat(wind_dfs, axis=1)
        result.to_parquet(f'{output_dir}/noaa_tc.wind.parquet')
        return result
    return None

def process_water_data(water_df, output_dir='data'):
    """ Process tides and water level data.
        High Low Tides
        Field	Description
        t	    Time - Date and time of the observation
        v	    Value - Measured water level height
        ty	    Type - Designation of Water level height. 
                HH = Higher High water, H = High water, L = Low water, LL = Lower Low water
        f	    Data Flags - in order of listing:
                -- (I) A flag that when set to 1 indicates that the water level value has been inferred
                -- (L) A flag that when set to 1 indicates that either the maximum or minimum expected 
                       water level height limit was exceeded


        Water Level
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
    """
    # Process tides
    for _, row in water_df.iterrows():
        fetch_and_process_station_data(
            row.station_id, row.start_date, row.end_date,
            product='high_low', datum='MSL',
            rename_cols={'v':'msl', 'ty':'type', 'f':'flag'},
            filter_cols=['ty'],
            output_path=f"{output_dir}/noaa_tc.tides.{row.station_id}.parquet"
        )
        
        # Process water levels
        for var in ['msl', 'mhhw']:
            fetch_and_process_station_data(
                row.station_id, row.start_date, row.end_date,
                product='water_level', datum=var.upper(),
                rename_cols={'v':var, 's':'sigma', 'f':'flag', 'q':'quality'},
                filter_cols=['v'],
                output_path=f"{output_dir}/noaa_tc.waterlevel.{var}.{row.station_id}.parquet"
            )

def aggregate_water_data(water_stations, output_dir='data'):
    """Aggregate water data across stations."""
    water_dfs = []
    for station_id in water_stations:
        try:
            # Load and process tide data
            tide = pd.read_parquet(f'{output_dir}/noaa_tc.tides.{station_id}.parquet')
            tide.type = tide.type.str.strip().replace({'LL':0, 'L':1, 'H':3, 'HH':4})
            
            # Load and filter water level data
            mhhw = pd.read_parquet(f'{output_dir}/noaa_tc.waterlevel.mhhw.{station_id}.parquet')
            mhhw = mhhw[(mhhw.sigma <= 3) & (mhhw.quality=='v')]
            
            msl = pd.read_parquet(f'{output_dir}/noaa_tc.waterlevel.msl.{station_id}.parquet')
            msl = msl[(msl.sigma <= 3) & (msl.quality=='v')]
            
            # Combine and resample
            combined = pd.concat([msl.msl, mhhw.mhhw, tide.type], axis=1).resample('h').max()
            water_dfs.append(combined.dropna(subset=['msl', 'mhhw'], how='all')
                            .add_suffix(f'_{station_id}'))
        except Exception as e:
            print(f"Error aggregating water data for {station_id}: {e}")
    
    if water_dfs:
        result = pd.concat(water_dfs, axis=1)
        result.to_parquet(f'{output_dir}/noaa_tc.water.parquet')
        return result
    return None

# Example usage
def main():
    process_wind_data(wind)
    aggregate_wind_data(wind_stations)
    
    process_water_data(water)
    aggregate_water_data(water_stations)
    
if __name__ == "__main__":
    main()
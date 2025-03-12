# V2 (in progess)
## TODO:
- Look at other time scale: (7, 14) days and (3, 6, 9) months using daily data
- Move threshold dict to main notebook
- Apply to other centers

## Completed
- Use >90F (32.22C) for temperature threshold to capture extreme heat waves
- Change historical years: 1981-2020
- Show uncertainty/variability across models (10th and 90th) for time series
- Add metrics
	- total compound day: a day that satisfies all conditions
	- total compound event: a compound day that is part of a sequence of more than one consecutive days
    - total compound event sequences: a sequence of consecutive compound events with size>1
    - average duration of compound events
- Replace precipitation with standardized precipitation index (SPI)
- Current implementation uses daily spi and tasmax values
- Add variable comparison plot per ssp

## Modified SPI Calculations:
- Source: https://github.com/e-baumer/standard_precip/blob/master/standard_precip/base_sp.py
- Setup: dataframe of precipitation values; rows are days/months; columns are models
    - freq = ['D', 'M']; monthly looks questionable with only JJA --> ignore for now
    - scale = #; shorter time scale = short term drought
    - If freq == 'M': get monthly pr files
    - If freq == 'D': get daily pr files
- Calculate rolling sum of \<scale\> window size for accumulated precipitation
- Fit a gamma distribution for each day of the year using historical values (1981-2020)
    - Values: rolling sums for current day plus \<scale\> days prior
- Calculate SPI for 2021-2100 based on gamma distribution
    - +spi (wet), -spi (dry), magnitude (severity)

## Determine if compound:
- Extract results JJA for 2021-2100 for spi and tasmax
- Establish threshold conditions for each variable
- Filter by common model/column names
- Concat spi and tasmax axis=0
- Concat ssp's axis=1
- Determine if threshold conditions are met
- Output 1 df per threshold combo

## Problem:
### Missing values: issue with SPI calculation
- Handle missing data to calculate spi (last paragraph) [link](https://www.droughtmanagement.info/literature/WMO_standardized_precipitation_index_user_guide_en_2012.pdf#page=9)
- 1 row of missing pr results in \<scale\> rows of missing spi
- Columns with a lot of nan's may result in no spi (KACE-1-0-G for LARC)
- Why are there missing values?
    - KACE-1-0-G: no value for all 31st day of month
    - Models missing 37 values: no value for feb 29 (leap years)
        - [INM-CM4-8, INM-CM5-0, NorESM2-MM, NorESM2-LM, GFDL-ESM4, GISS-E2-1-G, FGOALS-g3, BCC-CSM2-MR, CMCC-ESM2, CESM2]
- How to handle missing values? 
    - Check original code
    - Options: interpolation, multivariate imputation (slow), back/front fill, ...
- Solution: interpolation by linear (df sorted by time)

## 02/26/25 meeting notes
- Rolling/moving window instead of monthly precipitation
- Use more recent base/historical period
- Dry/wet precipitation percentile: not 50th
- Show variability across models
- Drought index, humidity, soil moisture
- Different threshold based on seasonal/center
- ...


# V1
- methodology used in [Feng et al. 2020](https://www.sciencedirect.com/science/article/pii/S2212094720303121?via%3Dihub#bib17) with some changes made from 02/25/25 meeting
	- Feedback loops
	- Humidity
	- JPL
	- Most influential factors per center
- Modification:
	- CWHE uses daily precipitation instead of monthly
	- Show uncertainty/variability across models (10th and 90th) for time series


# Others
- ca_fires: main causes of California fires
- center_casi_projections and updated_extremes: for climate worksheets/report cards
 

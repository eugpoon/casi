# V2 (in progess)
## Completed
- Use >90F (32.22C) for temperature threshold to capture extreme heat waves
- Change historical years: 1981-2020
- Show uncertainty/variability across models (10th and 90th) for time series
- Add total compound event days
	- compound day: a day that satisfies all conditions
	- compound event: a compound day that is part of a sequence of more than one consecutive days
- Add standardized precipitation index
	- +spi (wet), -spi (dry), magnitude (severity)
    - shorter time scale = short term drought
- Current implementation uses daily spi and tasmax values
- Change SPI caluculations
    - fit gamma dist on 1981-2020
    - calculate spi for 2021-2100

## TODO:
- Monthly looks questionable with only JJA

## 02/26/25 meeting notes
- rolling/moving window instead of monthly precipitation
- use more recent base/historical period
- dry/wet precipitation percentile: not 50th
- show variability across models
- drought index, humidity, soil moisture
- different threshold based on seasonal/center
- ...


# V1
- methodology used in [Feng et al. 2020](https://www.sciencedirect.com/science/article/pii/S2212094720303121?via%3Dihub#bib17) with some changes made from 02/25/25 meeting
	- feedback loops
	- humidity
	- jpl
	- most influential factors per center
- modification:
	- cwhe uses daily precipitation instead of monthly
	- show uncertainty/variability across models (10th and 90th) for time series


# Others
- ca_fires: main causes of California fires
- center_casi_projections and updated_extremes: for climate worksheets/report cards
 

# V2 (in progess)
Incorporate feedback from meeting

## Completed
- change historical years: 1981-2020 (ssp 245 for last 5)
- show uncertainty/variability across models (10th and 90th) for time series
- add total compound event days
	- compound day: a day that satisfies all conditions
	- compound event: a compound day that is part of a sequence of more than one consecutive days

## TODO:
- over 90f temperature for threshold instead of percentile for extreme heat waves
- rolling mean: 29 days precipitation before current day (historical and ssp)
- plot total compound days vs events (separate ssp)
- add standardized precipitation index
	- +spi (wet), -spi (dry), magnitude (severity)
    - shorter time scale = short term drought
    - ? daily spi meausures accumulated drought severity based on the previous n days

## 02/26/25 meeting notes
- rolling/moving window instead of monthly precipitation
	- window size?
	- rolling sum for pr
- use more recent base/historical period
- dry/wet precipitation percentile: not 50th
- show variability across models
- drought index, humidity, soil moisture
- different threshold based on seasonal/center
...


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
 

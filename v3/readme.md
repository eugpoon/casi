# V3

Third/Last Iteration

---

### Notes

The following changes were made due to computational time

* **SPI configuration**:

  * Default: 3-day precipitation accumulation (`spi_n=3`) with gamma fitting window `gamma_n=10`
  * To modify, edit `spi_n` and `spi_gamma_n` in `defaults.json`

* **SPI months**:

  * By default, only JJA is processed
  * To include all months, set `"spi_months": []` or `None` in `defaults.json`

---

## Process All Data from All Centers

This repository contains scripts to preprocess and analyze compound climate events across all NASA centers. The workflow below is demonstrated in `compound.ipynb`.

> **Note:** All input and output files are located in the `data/` folder in the main directory.

---

### 1. Convert RZSM `.dat` Files to CSV

```bash
python rzsm.py -i '../data/CASI2_RZSM_RAW.zip' -o '../data/compound'
```

* **Required file**: CASI2_RZSM_RAW_Mar25.zip
* **Output**: CSV files for RZSM


### 2. Combine Daily CMIP6 Files into a Single Parquet File

```bash
python combine_raw.py -i '../data/compound' -o '../data' -v pr tasmax rzsm
```

* **Required files**:
  * compound/: contains daily CMIP6 data (historical, ssp126/245/370)
  * defaults.json

* **Output**:
  * compound_raw.parquet: combining selected variables across centers
  * compound_mme_raw.parquet: mean, median, q10, q90 across models from compound_raw


### 3. Set Thresholds Using Percentiles

```bash
python set_thresholds.py -i '../data/compound_raw.parquet' -o '../data/compound_thresholds.csv'
```

* **Required files**:
  * compound_raw.parquet (from Step 2)
  * defaults.json
  * compound_events.json

* **Output**:
  * compound_thresholds.csv: percentile-based threshold values for each event, variable, and model

### 4. Compute Compound Metrics

```bash
python get_compound.py -i '../data/' -o '../data/compound_results.parquet' -e CDHE CDHE2 CWHE CWHE2 CFE
```

* **Required files**:
  * compound_raw.parquet (from Step 2)
  * compound_thresholds.csv (from Step 3)
  * compound_events.json

* **Output**:
  * compound_results.parquet: summarizing compound event frequency and intensity
  * compound_mme_results.parquet: mean, median, q10, q90 across models from compound_results

---

### Preprocessed Data for Dashboard

To skip data processing and directly explore results, use the following files:

* compound_mme_raw.parquet
* compound_mme_results.parquet
* compound_events.json
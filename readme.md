# Compound Climate Events Analysis

Detection, analysis, and visualization of compound climate events using CMIP6 ensemble datasets

---

## Setup

To get started, create the environment using Conda:

```bash
conda env create -f environment.yml -n casi
conda activate casi
```
---

## Repository Structure

```bash
.
├── v1/               # Earliest version of compound event detection scripts
├── v2/               # Intermediate version with improved methodology
├── v3/               # Latest and active version for compound analysis
├── floods/           # Preliminary (compound) coastal flood analysis
├── worksheets/       # Risk assessment tables and metrics
├── data/             # Raw and processed input data (not tracked in Git)
└── environment.yml   # Conda environment specification
```

---

## Dashboards

Interactive dashboards for CMIP6-based event visualization:

> Run dashboards in a Jupyter Notebook from `v3/` or similar script folders.

* **Time Series**: Compare variable trends across models, years, or months.
* **Severity View**: Visualize yearly totals or extremes for a selected event.
* **Metric Comparison**: Compare all metrics for one event at a given threshold.

---

## Risk Worksheets

Located in `worksheets/`:

* Generates summary tables of projection extremes
* Supports export for climate risk assessments.
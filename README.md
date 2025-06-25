# No-Code Editor: CLI Command Reference

This README provides ready-to-use examples for running operations on your health dataset using the no-code CLI.

---

## 1. Download a Dataset

Download FHIR data and process it for analysis:

<pre>python cli.py download-data --base_url https://fhir.rs.adarv.in/fhir --dataset_name LeptoDemo</pre>


---

## 2. Basic Statistics

**Calculate the mean:**

<pre>python cli.py mean --input processed_data.csv --column House</pre>


**Calculate the median:**

<pre>python cli.py median --input processed_data.csv --column House</pre>

**Calculate the mode (all columns if not specified):**

<pre>python cli.py mode --input processed_data.csv</pre>


**Calculate the standard deviation:**

<pre>python cli.py std --input processed_data.csv --column Chills</pre>


**Calculate the range:**

<pre>python cli.py range --input processed_data.csv --column House</pre>


---

## 3. Clustering

**Cluster all suitable columns and show top 3 distinct clusters:**

<pre>python cli.py cluster --input processed_data.csv --clusters 3 --topx 3</pre>

**Cluster specific features and show top 2 clusters:**

<pre>python cli.py cluster --input processed_data.csv --features gender,Chills --clusters 3 --topx 2</pre>


---

## 4. Patient Segmentation

**Segment patients by a categorical variable (e.g., gender):**

<pre>python cli.py patient-segmentation --input processed_data.csv --groupby gender --obs-names-path obs_names.pkl --cond-names-path cond_names.pkl --top-n 5</pre>

- `--groupby` can be any categorical column (e.g., House, cluster_label).
- `--top-n` specifies how many of the most distinctive conditions/observations to report per group.

---

## 5. Frequency Analysis

**Frequency and proportion for a column:**

<pre>python cli.py frequency --input processed_data.csv --column Gender --proportion</pre>


---

## Notes

- All commands assume your working directory contains the relevant CSV and pickle files (e.g., `processed_data.csv`, `obs_names.pkl`, `cond_names.pkl`).
- For patient segmentation, the output will show for each group:
  - The prevalence (mean) of each condition/observation.
  - The top and bottom conditions that distinguish each group from the rest.
- For more details on each command, use the `--help` flag, e.g.:

<pre>python cli.py mean --help</pre>

- For more information on the source data, see the [FHIR API documentation](https://fhir.rs.adarv.in/fhir).

---

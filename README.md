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

**Cluster with patient segmentation:**

<pre>python cli.py cluster --file processed_data.csv --clusters 3 --topx 3 --segment-clusters --obs-names-path obs_names.pkl --cond-names-path cond_names.pkl --seg-top-n 5</pre>

- This will segment patients by cluster label and report, for each cluster, the most distinctive conditions and observations.
- `--segment-clusters` enables segmentation, and `--seg-top-n` controls how many top/bottom conditions are shown per cluster.

---

## 4. Frequency Analysis

**Frequency and proportion for a column:**

<pre>python cli.py frequency --input processed_data.csv --column Gender --proportion</pre>

---

## 5. Correlation Analysis

**Observation-condition correlation matrix:**

<pre>python cli.py correlation --input processed_data.csv --obs-names-path obs_names.pkl --cond-names-path cond_names.pkl</pre>

**Correlation with symptom co-occurrence matrix:**

<pre>python cli.py correlation --input processed_data.csv --obs-names-path obs_names.pkl --cond-names-path cond_names.pkl --symptom-cooccurrence</pre>

- This will output both the observation-condition correlation matrix and the symptom (condition) co-occurrence matrix.
- The output is a matrix saved to `symptom_cooccurrence.csv`, showing for each symptom pair the number of patients who had both.

---

## 6. Disease Prevalence

**Calculate the point prevalence of a disease or condition:**

<pre>python cli.py prevalence --conditions-pkl conditions.pkl --disease-col "Type of symptoms [Jaundice]" --case-value 1</pre>

`--input`: Path to the csv file containing your data.

`--disease-col`: Column name for the disease or symptom (must match exactly).

`--case-value`: Value indicating a positive case (default: 1).

- Output: Number of cases, total population, and prevalence (as proportion and percentage).

---

## 7. Correlation Coefficients (Pearson & Spearman)

**Calculate Pearson and Spearman correlation coefficients for two columns:**

<pre>python cli.py corr-coefficient --input-file processed_data.csv --col1 "Hepatocellular jaundice" --col2 "Chills"</pre>

**Calculate for all pairs of numeric/binary columns:**

<pre>python cli.py corr-coefficient --input-file processed_data.csv</pre>

`--input-file`: Path to your CSV or Excel file.

`--col1`, `--col2`: (Optional) Specify two columns for pairwise correlation. If omitted, all pairs are computed.

- Output: Table with col1, col2, pearson_coefficient, pearson_pvalue, spearman_coefficient, spearman_pvalue.
- Results are saved to `correlation_results.csv`.

---

## 8. Covariance
**Calculate covariance between two columns:**

<pre>python cli.py covariance --input-file processed_data.csv --col1 "Hepatocellular jaundice" --col2 "Livestock farmer"</pre>

**Calculate covariance for all pairs of numeric columns:**

<pre>python cli.py covariance --input-file processed_data.csv</pre>

`--input-file`: Path to your CSV or Excel file.

`--col1`, `--col2`: (Optional) Specify two columns for pairwise covariance. If omitted, all pairs are computed.

- Output: Table with col1, col2, and covariance.
- Results are saved to `covariance_results.csv`.

---

## Notes

- All commands assume your working directory contains the relevant CSV and pickle files (e.g., `processed_data.csv`, `obs_names.pkl`, `cond_names.pkl`).
- For clustering with segmentation, the output will show for each cluster:
  - The prevalence (mean) of each condition/observation.
  - The top and bottom conditions that distinguish each cluster from the rest.
- For correlation with symptom co-occurrence, the output will show:
  - The correlation matrix between observations and conditions.
  - The co-occurrence matrix for all pairs of symptoms/conditions.
- For more details on each command, use the `--help` flag:

<pre>python cli.py mean --help</pre>

- For more information on the source data, see the [FHIR API documentation](https://fhir.rs.adarv.in/fhir).

---

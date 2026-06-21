# Respiratory AI Landscape Analysis: Code Repository

This repository contains the Python analysis scripts used in the manuscript: **"Performance Inflation and Reporting Biases in Respiratory Artificial Intelligence: A Data-Driven Landscape Analysis."**

## Contents
* `01_descriptive_statistics.py`: Calculates data filtering metrics, missing data percentages, publication time bias, and the Spearman correlation with 95% confidence intervals.
* `02_publication_trends.py`: Generates the publication trends bar chart, filtering out invalid or incomplete temporal data.
* `03_advanced_analytics.py`: Generates the regression plot, disease vs. algorithm heatmap, and the evidence maturity matrix.

## Dataset Access
To comply with FAIR data standards, the underlying dataset (`Respiratory_AI_Database_v1.csv`) and the Semantic Ontology Mapping Table are hosted securely on Zenodo. 

**Dataset DOI:** https://doi.org/10.5281/zenodo.20784892

## Usage and Requirements
The scripts are written in Python 3. To reproduce the analyses, ensure the dataset is in the same directory as the scripts. 
**Required Libraries:** `pandas`, `numpy`, `matplotlib`, `seaborn`, `scipy`.

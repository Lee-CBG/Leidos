# Model Training and Evaluation Pipeline

This repository contains the code used to train, evaluate, and post-process machine learning models across multiple configurations and random seeds.

The pipeline is designed to support large-scale experimentation and comparative analysis across disease cohorts, timepoints, and model settings.

---

## Overview

- Trains models across multiple configurations
- Uses **100 random seeds** per configuration for robustness
- Produces final trained models and aggregated evaluation results
- Supports downstream analysis grouped by disease cohort, train/test timepoint, and model configuration

---

## Running the Pipeline

To train models for all configurations over 100 random seeds, run:

```bash
python main.py
```
This will generate all intermediate outputs and trained models required for analysis.
Requirements
All required Python packages are listed in requirements.txt.
Install dependencies with:

```bash
pip install -r requirements.txt
```
Trained Models
The final trained models are available for download at the following link:
https://drive.google.com/drive/u/1/folders/1jbK-60344zbNYwgxpiV_oz9djLEKwKF6

These models correspond to the full set of configurations and random seeds used in the study.

Post-processing and Analysis
The notebook final_leidos.ipynb is used to post-process and analyze model outputs.
Results are aggregated and grouped by:

Disease cohort
Train/test timepoint
Model configuration
This notebook generates the final summaries and figures used for evaluation.

Reproducibility
All experiments are run using fixed random seeds
Each configuration is evaluated over 100 independent seeds
Trained models and outputs are saved for reproducibility and downstream analysis
Notes
Ensure sufficient compute and storage before running the full pipeline
Downloading pre-trained models is recommended if retraining is not required

# Implementation Plan - Compare All Models

## Goal Description

Create a script to run and compare all 4 implemented time series models:

1.  **Chronos** (Zero-shot Foundation Model)
2.  **LTSF-Linear** (DLinear)
3.  **LSTM** (RNN)
4.  **TCN** (Dilated CNN)

## Proposed Changes

### Custom Model Directory

#### [NEW] [compare_all_models.py](file:///d:/INCOIS-internship/mystuff/custom%20model/compare_all_models.py)

- Imports all 4 forecaster classes.
- Loads the Airline dataset.
- Trainings and evaluates each model on the last 12 months.
- Computes MAPE, MAE, RMSE for each.
- Generates a combined plot `all_models_comparison.png`.
- Prints a text summary table.

## Verification Plan

### Automated Tests

- Run `python compare_all_models.py`
- Check for successful execution and generation of `all_models_comparison.png`.

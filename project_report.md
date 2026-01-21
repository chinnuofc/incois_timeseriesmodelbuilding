# Time Series Forecasting Model Expansion Project

**Comprehensive Report & User Guide**

## 1. Project Overview

We have successfully implemented and benchmarked a suite of diverse time series forecasting models, ranging from traditional deep learning baselines to state-of-the-art foundation models. The goal was to establish a robust forecasting playground compatible with `sktime` and optimized for CPU execution.

## 2. Implemented Models

### A. Chronos (Tiny)

- **Type**: Foundation Model (Transformer-based, pre-trained).
- **Why we used it**: To leverage "zero-shot" learning. Like GPT-4 for text, Chronos has seen millions of time series patterns and can forecast new data without training.
- **Implementation**: Wrapped `amazon/chronos-t5-tiny` using Hugging Face `transformers` inside a `sktime` forecaster.

### B. LTSF-Linear (DLinear)

- **Type**: Linear Decomposition Model.
- **Why we used it**: A "strong baseline" that often beats complex Transformers (like FEDformer/Autoformer) in long-term forecasting while being extremely fast.
- **Implementation**: Decomposes the series into **Trend** (Moving Average) and **Seasonality** (Residual) and applies simple Linear layers to each.
- **Key Feature**: Training time is milliseconds/seconds compared to minutes for others.

### C. LSTM (Long Short-Term Memory)

- **Type**: Recurrent Neural Network (RNN).
- **Why we used it**: The classic standard for time series. It processes data sequentially, maintaining an internal "memory" of past events.
- **Implementation**: A PyTorch-based Encoder-Decoder architecture.

### D. TCN (Temporal Convolutional Network)

- **Type**: Convolutional Neural Network (CNN).
- **Why we used it**: A modern alternative to RNNs. It uses "dilated convolutions" to look far back in history without the sequential processing bottleneck of RNNs.
- **Implementation**: Stacked residual blocks with causal convolutions (no future leakage).

## 3. Performance Benchmark

We compared all models on the **Airline Passengers dataset** (last 12 months held out for testing).

| Model              | MAPE (Error) | Time (s)  | Verdict                                            |
| :----------------- | :----------- | :-------- | :------------------------------------------------- |
| **Chronos (Tiny)** | **3.23%**    | 2.44s     | **Most Accurate** (Zero Training/Tuning)           |
| **DLinear**        | **3.48%**    | **0.54s** | **Most Efficient** (Near SOTA accuracy, 5x faster) |
| **TCN**            | 6.35%        | 3.33s     | Good baseline, stable                              |
| **LSTM**           | 8.84%        | 3.64s     | Standard baseline, requires more tuning            |

> [!TIP]
> **Conclusion**: For this dataset, **Chronos** provides the instant best accuracy, but **DLinear** is an incredible lightweight alternative for production systems where latency matters.

![Comparison Plot](file:///d:/INCOIS-internship/mystuff/custom%20model/all_models_comparison.png)

## 4. Dataset Used

- **Name**: Airline Passengers.
- **Source**: `sktime.datasets.load_airline`.
- **Characteristics**: Monthly data with strong trend and seasonality. This is a "textbook" dataset where DLinear's explicit trend decomposition shines.

## 5. Code Structure

The project is organized into self-contained modules (`.py`) and their corresponding tests.

```text
custom model/
├── chronos_sktime_forecaster.py  # Wrapper for Amazon Chronos
├── ltsf_linear_forecaster.py     # DLinear/NLinear implementation
├── lstm_forecaster.py            # LSTM implementation
├── tcn_forecaster.py             # TCN implementation
├── test_*.py                    # Individual test suites for each model
└── compare_all_models.py         # Master script to run the benchmark
```

## 6. Execution Guide (Commands)

Use these commands in your terminal to run the code. We utilize the virtual environment in `chronos_env`.

### Run the Benchmark (All Models)

This runs the full comparison and updates the plot.

```powershell
.\chronos_env\Scripts\python.exe compare_all_models.py
```

### Run Individual Tests

If you want to debug or test a specific model:

**1. LTSF-Linear (DLinear)**

```powershell
.\chronos_env\Scripts\python.exe test_ltsf_linear.py
```

**2. LSTM**

```powershell
.\chronos_env\Scripts\python.exe test_lstm.py
```

**3. TCN**

```powershell
.\chronos_env\Scripts\python.exe test_tcn.py
```

**4. Chronos**

```powershell
.\chronos_env\Scripts\python.exe test_chronos_forecaster.py
```

## 7. Refinements & Next Steps

- **Hyperparameter Tuning**: LSTM and TCN performance (8.8% / 6.3%) can likely be improved by tuning `hidden_size`, `layers`, or `learning_rate` in their respective files.
- **Multivariate Support**: Currently, `ltsf_linear_forecaster.py` is set up for univariate (single variable). It can be extended for multivariate forecasting if needed.
- **GPU Acceleration**: All scripts default to `device="cpu"`. If you have an NVIDIA GPU, change this to `device="cuda"` in the file `__init__` arguments for a speedup (though DLinear is already instant on CPU).

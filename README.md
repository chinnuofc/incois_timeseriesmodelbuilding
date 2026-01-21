# ChronosSktimeForecaster

A production-ready **sktime-compatible wrapper** for Amazon's **Chronos-T5** time series foundation model, optimized for CPU inference on resource-constrained systems (8GB RAM).

## Features

- ✅ Full `sktime` compatibility (inherits from `BaseForecaster`)
- ✅ Zero-shot forecasting (no training required)
- ✅ CPU-optimized (no GPU needed)
- ✅ Memory-efficient with garbage collection
- ✅ Prediction intervals and quantile forecasts
- ✅ Compatible with sktime's evaluation and model selection tools

## Hardware Requirements

| Resource | Minimum | Recommended |
|----------|---------|-------------|
| CPU | Any x86_64 | 4+ cores |
| RAM | 4 GB | 8 GB |
| Storage | 2 GB | 5 GB |
| GPU | Not required | - |

## Installation

### Option 1: Quick Install (Recommended)

```bash
# Create a virtual environment (recommended)
python -m venv chronos_env
chronos_env\Scripts\activate  # Windows
# source chronos_env/bin/activate  # Linux/Mac

# Install dependencies
pip install torch --index-url https://download.pytorch.org/whl/cpu
pip install sktime transformers accelerate chronos-forecasting matplotlib
```

### Option 2: Using requirements.txt

```bash
pip install -r requirements.txt
```

### Option 3: Minimal Install (for testing)

```bash
pip install torch --index-url https://download.pytorch.org/whl/cpu
pip install sktime chronos-forecasting
```

## Quick Start

```python
from chronos_sktime_forecaster import ChronosSktimeForecaster
from sktime.datasets import load_airline
from sktime.forecasting.model_selection import temporal_train_test_split

# Load data
y = load_airline()
y_train, y_test = temporal_train_test_split(y, test_size=12)

# Create and fit forecaster
forecaster = ChronosSktimeForecaster(
    model_name="amazon/chronos-t5-small",
    device="cpu",
)
forecaster.fit(y_train)

# Make predictions
y_pred = forecaster.predict(fh=list(range(1, 13)))

# Get prediction intervals
pred_int = forecaster.predict_interval(fh=list(range(1, 13)), coverage=[0.9])
```

## Available Models

| Model | Parameters | RAM Usage* | Speed |
|-------|-----------|-----------|-------|
| `amazon/chronos-t5-tiny` | ~8M | ~1 GB | Fastest |
| `amazon/chronos-t5-mini` | ~20M | ~1.5 GB | Fast |
| `amazon/chronos-t5-small` | ~46M | ~2 GB | Balanced |
| `amazon/chronos-t5-base` | ~200M | ~4 GB | Slower |
| `amazon/chronos-t5-large` | ~710M | ~8 GB | Slowest |

*Approximate RAM usage during inference with float32

## API Reference

### Constructor Parameters

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `model_name` | str | "amazon/chronos-t5-small" | HuggingFace model ID |
| `device` | str | "cpu" | Device for inference |
| `torch_dtype` | torch.dtype | torch.float32 | Model precision |
| `num_samples` | int | 20 | Samples for probabilistic forecasts |
| `temperature` | float | 1.0 | Sampling temperature |
| `context_length` | int | None | Max context length (memory saver) |

### Methods

- `fit(y, X=None, fh=None)` - Load model and store training data
- `predict(fh, X=None)` - Generate point forecasts
- `predict_interval(fh, X=None, coverage=[0.9])` - Prediction intervals
- `predict_quantiles(fh, X=None, alpha=[0.1, 0.5, 0.9])` - Quantile forecasts

## Usage with sktime Tools

### With Evaluation Metrics

```python
from sktime.performance_metrics.forecasting import mean_absolute_percentage_error

mape = mean_absolute_percentage_error(y_test, y_pred)
print(f"MAPE: {mape:.2%}")
```

### With Cross-Validation

```python
from sktime.forecasting.model_evaluation import evaluate
from sktime.split import ExpandingWindowSplitter

cv = ExpandingWindowSplitter(initial_window=100, step_length=12, fh=[1,2,3])
results = evaluate(forecaster, cv, y, scoring=mean_absolute_percentage_error)
```

## Memory Optimization Tips

1. **Use the smallest adequate model**: Start with `chronos-t5-tiny` and upgrade if needed
2. **Limit context length**: Set `context_length=512` to limit memory usage
3. **Reduce num_samples**: Use `num_samples=10` for faster inference
4. **Close other applications**: Free up as much RAM as possible

## Troubleshooting

### Out of Memory Errors

```python
# Use a smaller model
forecaster = ChronosSktimeForecaster(
    model_name="amazon/chronos-t5-tiny",
    num_samples=10,
    context_length=256,
)
```

### Slow First Prediction

The first prediction downloads and loads the model (~100-500MB). Subsequent predictions are much faster.

### Import Errors

```bash
# Make sure chronos-forecasting is installed correctly
pip uninstall chronos-forecasting
pip install chronos-forecasting --no-cache-dir
```

## License

This wrapper is provided under the MIT License. The Chronos models are subject to Amazon's licensing terms.

## References

- [Chronos Paper](https://arxiv.org/abs/2403.07815) - "Chronos: Learning the Language of Time Series"
- [Amazon Chronos GitHub](https://github.com/amazon-science/chronos-forecasting)
- [sktime Documentation](https://www.sktime.net/)

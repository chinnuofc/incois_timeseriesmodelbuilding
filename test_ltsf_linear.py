"""
Test Suite for LTSF-Linear Forecaster (DLinear/NLinear)
=======================================================

This script tests the custom LTSF-Linear implementation and generates
a visualization of the forecast performance on the Airline dataset.
"""

import sys
import io
import gc
import time
import traceback

# Force UTF-8 for Windows console
try:
    sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8')
    sys.stderr = io.TextIOWrapper(sys.stderr.buffer, encoding='utf-8')
except:
    pass

import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
import torch

# Test result tracking
test_results = []

def log_test(name: str, passed: bool, message: str = "", duration: float = 0):
    """Log test result with status indicator."""
    status = "[PASS]" if passed else "[FAIL]"
    test_results.append((name, passed, message))
    time_str = f" ({duration:.2f}s)" if duration > 0 else ""
    print(f"{status}: {name}{time_str}")
    if message and not passed:
        print(f"       └─ {message}")

def run_test(test_func):
    """Decorator to run a test function with error handling."""
    def wrapper(*args, **kwargs):
        start_time = time.time()
        try:
            result, message = test_func(*args, **kwargs)
            duration = time.time() - start_time
            log_test(test_func.__name__.replace("test_", "").replace("_", " ").title(), 
                    result, message, duration)
            return result
        except Exception as e:
            duration = time.time() - start_time
            log_test(test_func.__name__.replace("test_", "").replace("_", " ").title(), 
                    False, f"Exception: {str(e)}", duration)
            traceback.print_exc()
            return False
    return wrapper

# =============================================================================
# TEST 1: Import Verification
# =============================================================================
@run_test
def test_imports():
    try:
        from ltsf_linear_forecaster import LTSFLinearForecaster
        return True, "Imported successfully"
    except ImportError as e:
        return False, str(e)

# =============================================================================
# TEST 2: Model Instantiation
# =============================================================================
@run_test
def test_instantiation():
    from ltsf_linear_forecaster import LTSFLinearForecaster
    model = LTSFLinearForecaster(seq_len=48, model_type="DLinear")
    assert model.seq_len == 48
    assert model.model_type == "DLinear"
    assert model.get_tag("requires-fh-in-fit") == True
    return True, "Instantiated correctly"

# =============================================================================
# TEST 3: Fit and Predict (DLinear)
# =============================================================================
@run_test
def test_fit_predict_dlinear():
    from ltsf_linear_forecaster import LTSFLinearForecaster
    from sktime.datasets import load_airline
    from sktime.forecasting.model_selection import temporal_train_test_split
    
    y = load_airline()
    y_train, y_test = temporal_train_test_split(y, test_size=12)
    
    # Create model
    forecaster = LTSFLinearForecaster(
        seq_len=24,
        model_type="DLinear",
        num_epochs=20,     # Low epochs for fast testing
        learning_rate=0.01
    )
    
    # Must provide fh in fit
    fh = list(range(1, 13))
    forecaster.fit(y_train, fh=fh)
    
    # Verify internal state
    assert forecaster.model_ is not None, "PyTorch model not initialized"
    assert forecaster.pred_len_ == 12, "Prediction length not set correctly"
    
    # Predict
    y_pred = forecaster.predict(fh=fh)
    
    # Check shape/type
    assert len(y_pred) == 12
    assert isinstance(y_pred, pd.Series)
    assert not y_pred.isna().any(), "Predictions contain NaNs"
    
    return True, "Fit and Predict successful"

# =============================================================================
# TEST 4: Visualization Generation
# =============================================================================
@run_test
def test_visualization():
    from ltsf_linear_forecaster import LTSFLinearForecaster
    from sktime.datasets import load_airline
    from sktime.forecasting.model_selection import temporal_train_test_split
    from sktime.performance_metrics.forecasting import mean_absolute_percentage_error
    
    y = load_airline()
    y_train, y_test = temporal_train_test_split(y, test_size=12)
    fh = list(range(1, 13))
    
    # Train DLinear
    dlinear = LTSFLinearForecaster(seq_len=36, model_type="DLinear", num_epochs=50)
    dlinear.fit(y_train, fh=fh)
    y_dlinear = dlinear.predict(fh=fh)
    mape_d = mean_absolute_percentage_error(y_test, y_dlinear)
    
    # Train NLinear
    nlinear = LTSFLinearForecaster(seq_len=36, model_type="NLinear", num_epochs=50)
    nlinear.fit(y_train, fh=fh)
    y_nlinear = nlinear.predict(fh=fh)
    mape_n = mean_absolute_percentage_error(y_test, y_nlinear)
    
    # Plot
    plt.figure(figsize=(12, 6))
    
    # Plot history (zoom in a bit)
    y_train[-60:].plot(label="History", color="black", alpha=0.5)
    y_test.plot(label="Actual", color="green", linewidth=2)
    
    y_dlinear.plot(label=f"DLinear (MAPE: {mape_d:.2%})", color="blue", linestyle="--")
    y_nlinear.plot(label=f"NLinear (MAPE: {mape_n:.2%})", color="red", linestyle="--")
    
    plt.title("LTSF-Linear Models on Airline Data")
    plt.legend()
    plt.grid(True, alpha=0.3)
    
    output_path = "ltsf_linear_forecast_example.png"
    plt.savefig(output_path, dpi=150)
    plt.close()
    
    return True, f"Plot saved to {output_path} (DLinear MAPE: {mape_d:.2%})"

# =============================================================================
# MAIN
# =============================================================================
if __name__ == "__main__":
    print("="*60)
    print("TEST SUITE: LTSF-Linear Forecaster")
    print("="*60)
    
    test_imports()
    test_instantiation()
    test_fit_predict_dlinear()
    test_visualization()
    
    print("\n" + "="*60)
    failed = [r for r in test_results if not r[1]]
    if not failed:
        print("ALL TESTS PASSED!")
    else:
        print(f"{len(failed)} TESTS FAILED!")
        sys.exit(1)

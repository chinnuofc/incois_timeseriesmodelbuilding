"""
Test Suite for ChronosSktimeForecaster
======================================

This script provides comprehensive testing for the Chronos-sktime integration.
Run this after installation to verify everything works correctly.

Usage:
    python test_chronos_forecaster.py

Tests Included:
1. Import verification
2. Model instantiation
3. Fit/Predict workflow
4. Prediction intervals
5. sktime compatibility checks
6. Memory usage monitoring
7. Performance metrics evaluation
"""

import sys
import gc
import time
import traceback
from typing import Tuple, Optional

# Test result tracking
test_results = []


def log_test(name: str, passed: bool, message: str = "", duration: float = 0):
    """Log test result with status indicator."""
    status = "✅ PASS" if passed else "❌ FAIL"
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
def test_imports() -> Tuple[bool, str]:
    """Verify all required packages can be imported."""
    missing = []
    
    try:
        import numpy
    except ImportError:
        missing.append("numpy")
    
    try:
        import pandas
    except ImportError:
        missing.append("pandas")
    
    try:
        import torch
    except ImportError:
        missing.append("torch")
    
    try:
        import sktime
    except ImportError:
        missing.append("sktime")
    
    try:
        from chronos import ChronosPipeline
    except ImportError:
        missing.append("chronos-forecasting")
    
    try:
        from chronos_sktime_forecaster import ChronosSktimeForecaster
    except ImportError:
        missing.append("chronos_sktime_forecaster (local module)")
    
    if missing:
        return False, f"Missing packages: {', '.join(missing)}"
    return True, "All packages imported successfully"


# =============================================================================
# TEST 2: Model Instantiation
# =============================================================================
@run_test
def test_model_instantiation() -> Tuple[bool, str]:
    """Test that the forecaster can be instantiated without errors."""
    import torch
    from chronos_sktime_forecaster import ChronosSktimeForecaster
    
    # Test default parameters
    forecaster = ChronosSktimeForecaster()
    assert forecaster.model_name == "amazon/chronos-t5-small"
    assert forecaster.device == "cpu"
    assert forecaster.pipeline_ is None  # Model not loaded yet
    
    # Test custom parameters
    forecaster2 = ChronosSktimeForecaster(
        model_name="amazon/chronos-t5-tiny",
        device="cpu",
        torch_dtype=torch.float32,
        num_samples=10,
    )
    assert forecaster2.model_name == "amazon/chronos-t5-tiny"
    assert forecaster2.num_samples == 10
    
    return True, "Forecaster instantiated correctly"


# =============================================================================
# TEST 3: sktime Tag Verification
# =============================================================================
@run_test
def test_sktime_tags() -> Tuple[bool, str]:
    """Verify sktime tags are properly configured."""
    from chronos_sktime_forecaster import ChronosSktimeForecaster
    
    forecaster = ChronosSktimeForecaster()
    
    # Check essential tags
    assert forecaster.get_tag("scitype:y") == "univariate"
    assert forecaster.get_tag("y_inner_mtype") == "pd.Series"
    assert forecaster.get_tag("requires-fh-in-fit") == False
    assert forecaster.get_tag("capability:pred_int") == True
    
    return True, "All sktime tags configured correctly"


# =============================================================================
# TEST 4: Fit and Predict Workflow (Using Tiny Model for Speed)
# =============================================================================
@run_test
def test_fit_predict_workflow() -> Tuple[bool, str]:
    """Test the complete fit/predict workflow with a small dataset."""
    import pandas as pd
    import numpy as np
    import torch
    from chronos_sktime_forecaster import ChronosSktimeForecaster
    from sktime.datasets import load_airline
    
    # Load small dataset
    y = load_airline()
    
    # Use only last 48 points for faster testing
    y_train = y.iloc[-48:-12]
    y_test = y.iloc[-12:]
    
    # Use TINY model for faster testing
    print("       └─ Loading chronos-t5-tiny model (this may take a minute)...")
    forecaster = ChronosSktimeForecaster(
        model_name="amazon/chronos-t5-tiny",  # Fastest model
        device="cpu",
        torch_dtype=torch.float32,
        num_samples=5,  # Fewer samples for speed
    )
    
    # Fit
    forecaster.fit(y_train)
    assert forecaster.pipeline_ is not None, "Model should be loaded after fit"
    assert forecaster._y_train is not None, "Training data should be stored"
    
    # Predict
    fh = list(range(1, 13))
    y_pred = forecaster.predict(fh=fh)
    
    # Verify output
    assert isinstance(y_pred, pd.Series), "Output should be a pandas Series"
    assert len(y_pred) == 12, f"Should have 12 predictions, got {len(y_pred)}"
    assert not y_pred.isna().any(), "Predictions should not contain NaN"
    
    # Cleanup
    del forecaster
    gc.collect()
    
    return True, f"Predictions generated successfully (shape: {y_pred.shape})"


# =============================================================================
# TEST 5: Prediction Intervals
# =============================================================================
@run_test
def test_prediction_intervals() -> Tuple[bool, str]:
    """Test prediction interval generation."""
    import pandas as pd
    import torch
    from chronos_sktime_forecaster import ChronosSktimeForecaster
    from sktime.datasets import load_airline
    
    y = load_airline().iloc[-48:]
    y_train = y.iloc[:-12]
    
    forecaster = ChronosSktimeForecaster(
        model_name="amazon/chronos-t5-tiny",
        num_samples=10,
    )
    forecaster.fit(y_train)
    
    fh = list(range(1, 7))  # 6 steps
    pred_int = forecaster.predict_interval(fh=fh, coverage=[0.9])
    
    # Verify structure
    assert isinstance(pred_int, pd.DataFrame), "Should return DataFrame"
    assert len(pred_int) == 6, "Should have 6 rows"
    
    # Check that lower < upper for all intervals
    var_name = y.name if y.name else "y"
    lower = pred_int[(var_name, 0.9, 'lower')].values
    upper = pred_int[(var_name, 0.9, 'upper')].values
    assert all(lower < upper), "Lower bound should be less than upper bound"
    
    del forecaster
    gc.collect()
    
    return True, "Prediction intervals generated correctly"


# =============================================================================
# TEST 6: Memory Usage Check
# =============================================================================
@run_test
def test_memory_usage() -> Tuple[bool, str]:
    """Check approximate memory usage of the model."""
    import psutil
    import torch
    from chronos_sktime_forecaster import ChronosSktimeForecaster
    from sktime.datasets import load_airline
    
    process = psutil.Process()
    
    # Baseline memory
    gc.collect()
    mem_before = process.memory_info().rss / (1024 ** 2)  # MB
    
    # Load model
    forecaster = ChronosSktimeForecaster(
        model_name="amazon/chronos-t5-tiny",
        num_samples=5,
    )
    y = load_airline().iloc[-36:]
    forecaster.fit(y)
    
    mem_after_fit = process.memory_info().rss / (1024 ** 2)
    
    # Make prediction
    forecaster.predict(fh=[1, 2, 3])
    mem_after_predict = process.memory_info().rss / (1024 ** 2)
    
    # Cleanup
    del forecaster
    gc.collect()
    mem_after_cleanup = process.memory_info().rss / (1024 ** 2)
    
    model_memory = mem_after_fit - mem_before
    peak_memory = max(mem_after_fit, mem_after_predict) - mem_before
    
    # Check memory is reasonable (< 2GB for tiny model)
    passed = peak_memory < 2000
    
    return passed, f"Model: ~{model_memory:.0f}MB, Peak: ~{peak_memory:.0f}MB"


# =============================================================================
# TEST 7: Performance Metrics
# =============================================================================
@run_test
def test_performance_metrics() -> Tuple[bool, str]:
    """Test integration with sktime performance metrics."""
    import torch
    from chronos_sktime_forecaster import ChronosSktimeForecaster
    from sktime.datasets import load_airline
    from sktime.forecasting.model_selection import temporal_train_test_split
    from sktime.performance_metrics.forecasting import (
        mean_absolute_percentage_error,
        mean_absolute_error,
    )
    
    y = load_airline()
    y_train, y_test = temporal_train_test_split(y, test_size=12)
    
    forecaster = ChronosSktimeForecaster(
        model_name="amazon/chronos-t5-tiny",
        num_samples=10,
    )
    forecaster.fit(y_train)
    
    fh = list(range(1, 13))
    y_pred = forecaster.predict(fh=fh)
    
    # Calculate metrics
    mape = mean_absolute_percentage_error(y_test, y_pred)
    mae = mean_absolute_error(y_test, y_pred)
    
    del forecaster
    gc.collect()
    
    # MAPE should be reasonable (< 50% for a zero-shot model on airline data)
    passed = mape < 0.5
    
    return passed, f"MAPE: {mape:.2%}, MAE: {mae:.2f}"


# =============================================================================
# TEST 8: Quantile Predictions
# =============================================================================
@run_test
def test_quantile_predictions() -> Tuple[bool, str]:
    """Test quantile prediction functionality."""
    import pandas as pd
    import torch
    from chronos_sktime_forecaster import ChronosSktimeForecaster
    from sktime.datasets import load_airline
    
    y = load_airline().iloc[-48:]
    y_train = y.iloc[:-12]
    
    forecaster = ChronosSktimeForecaster(
        model_name="amazon/chronos-t5-tiny",
        num_samples=20,
    )
    forecaster.fit(y_train)
    
    fh = list(range(1, 7))
    alpha = [0.1, 0.5, 0.9]
    pred_q = forecaster.predict_quantiles(fh=fh, alpha=alpha)
    
    # Verify structure
    assert isinstance(pred_q, pd.DataFrame)
    assert len(pred_q) == 6
    
    # Check quantile ordering (q10 < q50 < q90)
    var_name = y.name if y.name else "y"
    q10 = pred_q[(var_name, 0.1)].values
    q50 = pred_q[(var_name, 0.5)].values
    q90 = pred_q[(var_name, 0.9)].values
    
    # Allow some tolerance for edge cases
    assert all(q10 <= q50 + 1), "q10 should be <= q50"
    assert all(q50 <= q90 + 1), "q50 should be <= q90"
    
    del forecaster
    gc.collect()
    
    return True, "Quantile predictions ordered correctly"


# =============================================================================
# MAIN TEST RUNNER
# =============================================================================
def main():
    """Run all tests and display summary."""
    print("=" * 70)
    print("ChronosSktimeForecaster Test Suite")
    print("=" * 70)
    print()
    
    # Run tests in order
    test_imports()
    
    # Only continue if imports pass
    if test_results[-1][1]:
        test_model_instantiation()
        test_sktime_tags()
        test_fit_predict_workflow()
        test_prediction_intervals()
        test_quantile_predictions()
        test_performance_metrics()
        
        # Memory test (requires psutil)
        try:
            import psutil
            test_memory_usage()
        except ImportError:
            print("⚠️  SKIP: Memory usage test (install psutil: pip install psutil)")
    
    # Summary
    print()
    print("=" * 70)
    print("TEST SUMMARY")
    print("=" * 70)
    
    passed = sum(1 for _, p, _ in test_results if p)
    total = len(test_results)
    
    print(f"Passed: {passed}/{total}")
    
    if passed == total:
        print("\n🎉 All tests passed! Your installation is working correctly.")
        return 0
    else:
        print("\n⚠️  Some tests failed. Check the output above for details.")
        failed_tests = [name for name, p, _ in test_results if not p]
        print(f"Failed tests: {', '.join(failed_tests)}")
        return 1


if __name__ == "__main__":
    sys.exit(main())

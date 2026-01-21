"""
Test Suite for LSTM Forecaster
==============================
"""

import sys
import io
import time
import traceback
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np

# Force UTF-8 for Windows console
try:
    sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8')
    sys.stderr = io.TextIOWrapper(sys.stderr.buffer, encoding='utf-8')
except:
    pass

test_results = []

def log_test(name: str, passed: bool, message: str = "", duration: float = 0):
    status = "[PASS]" if passed else "[FAIL]"
    test_results.append((name, passed, message))
    print(f"{status}: {name} ({duration:.2f}s)")
    if message and not passed:
        print(f"       => {message}")

def run_test(test_func):
    def wrapper(*args, **kwargs):
        start = time.time()
        try:
            res, msg = test_func(*args, **kwargs)
            dur = time.time() - start
            log_test(test_func.__name__, res, msg, dur)
            return res
        except Exception as e:
            dur = time.time() - start
            log_test(test_func.__name__, False, str(e), dur)
            traceback.print_exc()
            return False
    return wrapper

@run_test
def test_imports():
    try:
        from lstm_forecaster import LSTMForecaster
        return True, "Imported"
    except Exception as e:
        return False, str(e)

@run_test
def test_fit_predict_plot():
    from lstm_forecaster import LSTMForecaster
    from sktime.datasets import load_airline
    from sktime.forecasting.model_selection import temporal_train_test_split
    from sktime.performance_metrics.forecasting import mean_absolute_percentage_error
    
    y = load_airline()
    y_train, y_test = temporal_train_test_split(y, test_size=12)
    fh = list(range(1, 13))
    
    model = LSTMForecaster(
        seq_len=36,
        hidden_size=64,
        num_layers=2,
        num_epochs=50
    )
    
    model.fit(y_train, fh=fh)
    y_pred = model.predict(fh=fh)
    
    mape = mean_absolute_percentage_error(y_test, y_pred)
    
    # Plot
    plt.figure(figsize=(12, 6))
    y_train[-60:].plot(label="History", color="black", alpha=0.5)
    y_test.plot(label="Actual", color="green", linewidth=2)
    y_pred.plot(label=f"LSTM Forecast (MAPE: {mape:.2%})", color="purple", linestyle="--", linewidth=2)
    plt.title("LSTM Model Forecast on Airline Data")
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.savefig("lstm_forecast_example.png", dpi=150)
    plt.close()
    
    return True, f"MAPE: {mape:.2%}"

if __name__ == "__main__":
    print("LSTM TEST SUITE")
    print("===============")
    test_imports()
    test_fit_predict_plot()
    
    if all(r[1] for r in test_results):
        print("\nALL PASSED")
    else:
        print("\nSOME FAILED")
        sys.exit(1)

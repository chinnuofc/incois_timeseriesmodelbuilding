"""
Model Comparison Script
=======================

This script benchmarks all implemented time series models on the Airline dataset:
1. Chronos-T5-Small (Zero-shot Foundation Model)
2. LTSF-Linear (DLinear implementation)
3. LSTM (Recurrent Neural Network)
4. TCN (Temporal Convolutional Network)

It generates a combined visualization and a performance metrics table.
"""

import sys
import io
import time
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import warnings

# Force UTF-8 for Windows console
try:
    sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8')
    sys.stderr = io.TextIOWrapper(sys.stderr.buffer, encoding='utf-8')
except:
    pass

# Suppress warnings for cleaner output
warnings.filterwarnings("ignore")

from sktime.datasets import load_airline
from sktime.forecasting.model_selection import temporal_train_test_split
from sktime.performance_metrics.forecasting import (
    mean_absolute_percentage_error,
    mean_absolute_error,
    mean_squared_error
)

# Import our custom forecasters
try:
    # Append current directory to path just in case
    sys.path.append(".")
    from chronos_sktime_forecaster import ChronosSktimeForecaster
    from ltsf_linear_forecaster import LTSFLinearForecaster
    from lstm_forecaster import LSTMForecaster
    from tcn_forecaster import TCNForecaster
    print("All models imported successfully.")
except ImportError as e:
    print(f"Error importing models: {e}")
    sys.exit(1)

def main():
    print("\n" + "="*60)
    print("STARTING MODEL COMPARISON")
    print("="*60)

    # 1. Load Data
    print("Loading data...")
    y = load_airline()
    y_train, y_test = temporal_train_test_split(y, test_size=12)
    fh = list(range(1, 13))
    
    print(f"Train size: {len(y_train)}, Test size: {len(y_test)}")
    
    # 2. Define Models
    # Using slightly more robust settings for comparison
    models = {
        # Chronos: Zero-shot, pretrained
        "Chronos (Tiny)": ChronosSktimeForecaster(
            model_name="amazon/chronos-t5-tiny", # Use tiny for speed in demo
            device="cpu",
            num_samples=20
        ),
        
        # DLinear: Simple linear baseline
        "DLinear": LTSFLinearForecaster(
            seq_len=36,
            model_type="DLinear",
            num_epochs=100,
            learning_rate=0.005
        ),
        
        # LSTM: RNN baseline
        "LSTM": LSTMForecaster(
            seq_len=36,
            hidden_size=64,
            num_layers=2,
            num_epochs=100,
            learning_rate=0.005
        ),
        
        # TCN: CNN baseline
        "TCN": TCNForecaster(
            seq_len=36,
            num_channels=[32, 32],
            kernel_size=3,
            num_epochs=100,
            learning_rate=0.005
        )
    }
    
    results = {}
    preds = {}
    
    # 3. Train and Predict
    print("\nTraining and Evaluating Models:")
    print("-" * 60)
    
    for name, model in models.items():
        start_time = time.time()
        print(f"Running {name}...", end="", flush=True)
        
        try:
            # Fit
            model.fit(y_train, fh=fh)
            
            # Predict
            y_pred = model.predict(fh=fh)
            preds[name] = y_pred
            
            # Metrics
            mape = mean_absolute_percentage_error(y_test, y_pred)
            mae = mean_absolute_error(y_test, y_pred)
            rmse = mean_squared_error(y_test, y_pred, square_root=True)
            
            duration = time.time() - start_time
            print(f" Done ({duration:.2f}s) | MAPE: {mape:.2%}")
            
            results[name] = {
                "MAPE": mape,
                "MAE": mae,
                "RMSE": rmse,
                "Time (s)": duration
            }
            
        except Exception as e:
            print(f" Failed! Error: {e}")
            import traceback
            traceback.print_exc()

    # 4. Visualization
    print("\nGenerating Comparison Plot...")
    plt.figure(figsize=(14, 7))
    
    # Plot history (last 5 years)
    y_train_plot = y_train[-60:]
    plt.plot(y_train_plot.index.to_timestamp(), y_train_plot.values, 
             label="History", color="black", alpha=0.4, linewidth=1.5)
    
    # Plot Actual
    plt.plot(y_test.index.to_timestamp(), y_test.values, 
             label="Actual", color="black", linewidth=2.5)
    
    # Plot Predictions
    colors = {
        "Chronos (Tiny)": "blue",
        "DLinear": "red",
        "LSTM": "green",
        "TCN": "orange"
    }
    
    for name, y_pred in preds.items():
        mape = results[name]["MAPE"]
        plt.plot(y_pred.index.to_timestamp(), y_pred.values, 
                 label=f"{name} (MAPE: {mape:.1%})", 
                 color=colors.get(name, "gray"), 
                 linestyle="--", 
                 linewidth=2,
                 marker='o',
                 markersize=4)
    
    plt.title("Time Series Model Comparison - Airline Dataset", fontsize=16)
    plt.xlabel("Date", fontsize=12)
    plt.ylabel("Passengers", fontsize=12)
    plt.grid(True, alpha=0.3)
    plt.legend(fontsize=10, loc="upper left")
    
    output_path = "all_models_comparison.png"
    plt.savefig(output_path, dpi=150, bbox_inches="tight")
    print(f"Plot saved to: {output_path}")
    plt.close()
    
    # 5. Summary Table
    print("\n" + "="*60)
    print(f"{'Model':<20} | {'MAPE':<10} | {'MAE':<10} | {'RMSE':<10} | {'Time (s)':<10}")
    print("-" * 70)
    
    # Sort by MAPE
    sorted_results = sorted(results.items(), key=lambda x: x[1]["MAPE"])
    
    for name, metrics in sorted_results:
        print(f"{name:<20} | {metrics['MAPE']:.2%}      | {metrics['MAE']:.2f}      | {metrics['RMSE']:.2f}      | {metrics['Time (s)']:.2f}")
    print("="*60)

if __name__ == "__main__":
    main()

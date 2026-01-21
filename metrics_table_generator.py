
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

# Suppress warnings
warnings.filterwarnings("ignore")

from sktime.datasets import load_airline
from sktime.forecasting.model_selection import temporal_train_test_split
from sktime.performance_metrics.forecasting import (
    mean_absolute_percentage_error,
    mean_absolute_error,
    mean_squared_error
)

# Import custom forecasters
try:
    sys.path.append(".")
    from chronos_sktime_forecaster import ChronosSktimeForecaster
    from ltsf_linear_forecaster import LTSFLinearForecaster
    from lstm_forecaster import LSTMForecaster
    from tcn_forecaster import TCNForecaster
    print("Models imported successfully.")
except ImportError as e:
    print(f"Error importing models: {e}")
    sys.exit(1)

def run_comparison():
    print("Loading data...")
    y = load_airline()
    y_train, y_test = temporal_train_test_split(y, test_size=12)
    fh = list(range(1, 13))

    # Define Models
    models = {
        "Chronos (Tiny)": ChronosSktimeForecaster(
            model_name="amazon/chronos-t5-tiny",
            device="cpu",
            num_samples=20
        ),
        "DLinear": LTSFLinearForecaster(
            seq_len=36,
            model_type="DLinear",
            num_epochs=100,
            learning_rate=0.005
        ),
        "LSTM": LSTMForecaster(
            seq_len=36,
            hidden_size=64,
            num_layers=2,
            num_epochs=100,
            learning_rate=0.005
        ),
        "TCN": TCNForecaster(
            seq_len=36,
            num_channels=[32, 32],
            kernel_size=3,
            num_epochs=100,
            learning_rate=0.005
        )
    }

    results = []

    print("Evaluating models...")
    for name, model in models.items():
        print(f"Running {name}...")
        try:
            start_time = time.time()
            model.fit(y_train, fh=fh)
            y_pred = model.predict(fh=fh)
            duration = time.time() - start_time

            mape = mean_absolute_percentage_error(y_test, y_pred)
            mae = mean_absolute_error(y_test, y_pred)
            rmse = mean_squared_error(y_test, y_pred, square_root=True)

            results.append({
                "Model": name,
                "MAPE": mape,
                "MAE": mae,
                "RMSE": rmse,
                "Time (s)": duration
            })
        except Exception as e:
            print(f"Error evaluating {name}: {e}")

    return pd.DataFrame(results)

def save_table_image(df, filename="model_metrics_table.png"):
    if df.empty:
        print("No results to save.")
        return

    # Sort data for better presentation (e.g., by MAPE)
    df_sorted = df.sort_values(by="MAPE")
    
    # Format numbers
    # Create a display copy to avoid modifying the original data logic if we were to return it
    df_display = df_sorted.copy()
    df_display["MAPE"] = df_display["MAPE"].apply(lambda x: f"{x:.2%}")
    df_display["MAE"] = df_display["MAE"].apply(lambda x: f"{x:.2f}")
    df_display["RMSE"] = df_display["RMSE"].apply(lambda x: f"{x:.2f}")
    df_display["Time (s)"] = df_display["Time (s)"].apply(lambda x: f"{x:.2f}")

    # Set up plot
    fig, ax = plt.subplots(figsize=(10, 4))
    ax.axis('off')
    ax.axis('tight')

    # Create table
    table = ax.table(
        cellText=df_display.values,
        colLabels=df_display.columns,
        cellLoc='center',
        loc='center',
        colColours=["#e6e6e6"] * len(df_display.columns) # Light gray header
    )

    # Styling
    table.auto_set_font_size(False)
    table.set_fontsize(11)
    table.scale(1.2, 1.5) # Scale width and height

    # Colorize rows based on rank (optional, usually best to just keep clean)
    # But let's verify we just want a clean table.
    
    plt.title("Model Performance Metrics Comparison", fontsize=14, pad=20)
    
    plt.savefig(filename, dpi=300, bbox_inches='tight', pad_inches=0.1)
    print(f"Metrics table saved to {filename}")
    plt.close()

if __name__ == "__main__":
    df_results = run_comparison()
    print("\nResults DataFrame:")
    print(df_results)
    save_table_image(df_results)

"""
LTSF-Linear Forecaster: A lightweight, efficient linear model for time series forecasting
========================================================================================

This module implements the LTSF-Linear (Long-Term Time Series Forecasting Linear) family of models.
Despite their simplicity, these models (Linear, DLinear, NLinear) often outperform complex 
Transformers on long-term forecasting tasks while being ORDERS OF MAGNITUDE faster and lighter.

Key Features:
- **DLinear**: Decomposes series into trend and seasonality before linear mapping
- **NLinear**: Normalizes the input (subtracts last value) to handle non-stationarity
- **Fast Training**: Trains in seconds/minutes on CPU
- **Interpretability**: Simple linear weights
- **sktime Compatible**: Fully integrated with sktime API

Reference: "Are Transformers Effective for Time Series Forecasting?" (AAAI 2023)
https://arxiv.org/abs/2205.13504
"""

import gc
import warnings
from typing import Optional, Union, List

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset

# sktime imports
from sktime.forecasting.base import BaseForecaster, ForecastingHorizon

class LTSFLinearModel(nn.Module):
    """
    Core PyTorch implementation of LTSF-Linear models.
    Supports Linear, DLinear, and NLinear architectures.
    """
    def __init__(self, seq_len: int, pred_len: int, distinct_linear_for_each_var: bool = True, model_type: str = "DLinear"):
        super().__init__()
        self.seq_len = seq_len
        self.pred_len = pred_len
        self.model_type = model_type
        
        # In this simplified univariate implementation, we treat the input as 1 variable.
        # Ideally, for multivariate, we would use independent linear layers (channel independence).
        
        if self.model_type == 'Linear':
            self.Linear = nn.Linear(self.seq_len, self.pred_len)
        elif self.model_type == 'NLinear':
            self.Linear = nn.Linear(self.seq_len, self.pred_len)
        elif self.model_type == 'DLinear':
            # Moving Average for decomposition
            kernel_size = 25
            self.decompsition = SeriesDecomp(kernel_size)
            self.Linear_Seasonal = nn.Linear(self.seq_len, self.pred_len)
            self.Linear_Trend = nn.Linear(self.seq_len, self.pred_len)
        else:
            raise ValueError(f"Unknown model_type: {model_type}")

    def forward(self, x):
        # x shape: [Batch, Input Length]
        
        if self.model_type == 'Linear':
            return self.Linear(x)
            
        elif self.model_type == 'NLinear':
            seq_last = x[:, -1:].detach()
            x = x - seq_last
            x = self.Linear(x)
            x = x + seq_last
            return x
            
        elif self.model_type == 'DLinear':
            # Decompose
            seasonal_init, trend_init = self.decompsition(x)
            
            # Application of Linear layers
            seasonal_output = self.Linear_Seasonal(seasonal_init)
            trend_output = self.Linear_Trend(trend_init)
            
            x = seasonal_output + trend_output
            return x
            
        return None

class SeriesDecomp(nn.Module):
    """
    Series decomposition block using moving average.
    """
    def __init__(self, kernel_size):
        super().__init__()
        self.moving_avg = Moving_Avg(kernel_size, stride=1)

    def forward(self, x):
        moving_mean = self.moving_avg(x)
        res = x - moving_mean
        return res, moving_mean

class Moving_Avg(nn.Module):
    """
    Moving average block to highlight the trend of time series.
    """
    def __init__(self, kernel_size, stride):
        super().__init__()
        self.kernel_size = kernel_size
        self.avg = nn.AvgPool1d(kernel_size=kernel_size, stride=stride, padding=0)

    def forward(self, x):
        # x: [Batch, Seq_Len]
        # Padding on the both ends of time series
        # Front padding: repeat the first element
        front = x[:, 0:1].repeat(1, (self.kernel_size - 1) // 2)
        # End padding: repeat the last element
        end = x[:, -1:].repeat(1, (self.kernel_size - 1) // 2)
        
        x_pad = torch.cat([front, x, end], dim=1)
        
        # AvgPool1d requires [Batch, Channels, Seq_Len], but we rely on simple linear
        # So we can just treat it as 1 channel
        x_pad = x_pad.unsqueeze(1)
        x = self.avg(x_pad)
        x = x.squeeze(1)
        
        return x


class LTSFLinearForecaster(BaseForecaster):
    """
    LTSF-Linear: A fast, efficient linear forecaster (DLinear/NLinear).
    
    This forecaster implements the DLinear/NLinear architecture which is:
    1. Extremely fast to train
    2. Very lightweight (few parameters)
    3. Often more accurate than Transformers for numerical time series
    
    Parameters
    ----------
    seq_len : int, default=96
        Lookback window size (input sequence length).
    
    model_type : str, default="DLinear"
        Type of linear model: "Linear", "DLinear", "NLinear".
        - "DLinear": Best for data with trend and seasonality (Decomposition Linear)
        - "NLinear": Best for non-stationary data (Normalization Linear)
        - "Linear": Vanilla single-layer linear model
    
    num_epochs : int, default=50
        Number of training epochs.
        
    batch_size : int, default=32
        Training batch size.
        
    learning_rate : float, default=0.005
        Learning rate for optimizer.
        
    device : str, default="cpu"
        Device to run on ("cpu" or "cuda").
    """
    
    _tags = {
        "scitype:y": "univariate",
        "y_inner_mtype": "pd.Series",
        "X_inner_mtype": "pd.DataFrame", 
        "ignores-exogeneous-X": True,
        "requires-fh-in-fit": True,       # Need to know horizon size to build model
        "handles-missing-data": False,
        "capability:insample": False,
        "capability:pred_int": False,     # Only point forecasts for now
        "python_dependencies": ["torch"]
    }

    def __init__(
        self,
        seq_len: int = 96,
        model_type: str = "DLinear",
        num_epochs: int = 50,
        batch_size: int = 32,
        learning_rate: float = 0.005,
        device: str = "cpu"
    ):
        self.seq_len = seq_len
        self.model_type = model_type
        self.num_epochs = num_epochs
        self.batch_size = batch_size
        self.learning_rate = learning_rate
        self.device = device
        
        self._y_train = None
        self.model_ = None
        self.pred_len_ = None
        
        super().__init__()

    def _fit(self, y, X=None, fh=None):
        """
        Fit the LTSF-Linear model.
        """
        # Validate FH
        if fh is None:
            raise ValueError("fh (ForecastingHorizon) is required for fit in LTSF-Linear to determine output size.")
            
        # Determine prediction length from fh
        fh_relative = fh.to_relative(self.cutoff)
        
        # Use to_pandas() to get the relative steps safely
        if hasattr(fh_relative, "to_pandas"):
            steps = fh_relative.to_pandas()
        else:
             # Fallback for older sktime
            steps = fh_relative

        self.pred_len_ = int(max(steps))
        
        # Prepare training data (sliding window)
        y_values = y.values.astype(np.float32)
        
        # Scale data (StandardScaler) - important for Neural Nets
        self.mean_ = np.mean(y_values)
        self.std_ = np.std(y_values) + 1e-5
        y_scaled = (y_values - self.mean_) / self.std_
        
        X_train, Y_train = [], []
        
        # Create sliding windows
        # Need at least seq_len + pred_len data points
        if len(y_scaled) <= self.seq_len + self.pred_len_:
             # Fallback if data is too short: Repeat data or warn?
             # For now, just raise warning and try to use what we can
             warnings.warn(f"Data length ({len(y_scaled)}) is small for seq_len={self.seq_len} + pred_len={self.pred_len_}")
        
        for i in range(len(y_scaled) - self.seq_len - self.pred_len_ + 1):
            X_train.append(y_scaled[i : i + self.seq_len])
            Y_train.append(y_scaled[i + self.seq_len : i + self.seq_len + self.pred_len_])
            
        if not X_train:
            raise ValueError(
                f"Not enough data to create a single training sample. "
                f"Need > seq_len({self.seq_len}) + pred_len({self.pred_len_}) points. "
                f"Got {len(y_scaled)}."
            )
            
        X_train = torch.tensor(np.array(X_train), dtype=torch.float32)
        Y_train = torch.tensor(np.array(Y_train), dtype=torch.float32)
        
        # Initialize model
        self.model_ = LTSFLinearModel(
            seq_len=self.seq_len,
            pred_len=self.pred_len_,
            model_type=self.model_type
        ).to(self.device)
        
        # Train
        optimizer = optim.Adam(self.model_.parameters(), lr=self.learning_rate)
        criterion = nn.MSELoss()
        
        dataset = TensorDataset(X_train, Y_train)
        dataloader = DataLoader(dataset, batch_size=self.batch_size, shuffle=True)
        
        self.model_.train()
        for epoch in range(self.num_epochs):
            total_loss = 0
            for batch_x, batch_y in dataloader:
                batch_x = batch_x.to(self.device)
                batch_y = batch_y.to(self.device)
                
                optimizer.zero_grad()
                outputs = self.model_(batch_x)
                loss = criterion(outputs, batch_y)
                loss.backward()
                optimizer.step()
                
                total_loss += loss.item()
            
            # if (epoch + 1) % 10 == 0:
            #     print(f"Epoch [{epoch+1}/{self.num_epochs}], Loss: {total_loss/len(dataloader):.4f}")
                
        self._y_train = y
        return self

    def _predict(self, fh, X=None):
        """
        Generate forecasts.
        """
        # Prepare input: last seq_len points from training data
        y_values = self._y_train.values.astype(np.float32)
        
        # Normalize
        y_scaled = (y_values - self.mean_) / self.std_
        
        # Get last seq_len points
        if len(y_scaled) < self.seq_len:
            # Padding if history is too short
            padding = np.zeros(self.seq_len - len(y_scaled))
            input_seq = np.concatenate([padding, y_scaled])
        else:
            input_seq = y_scaled[-self.seq_len:]
            
        input_tensor = torch.tensor(input_seq, dtype=torch.float32).unsqueeze(0).to(self.device)
        
        # Inference
        self.model_.eval()
        with torch.no_grad():
            output_tensor = self.model_(input_tensor)
            
        # Denormalize
        pred_scaled = output_tensor.cpu().numpy().squeeze()
        pred_values = pred_scaled * self.std_ + self.mean_
        
        # Map to fh
        fh_relative = fh.to_relative(self.cutoff)
        
        if hasattr(fh_relative, "to_pandas"):
            steps = fh_relative.to_pandas()
        else:
            steps = fh_relative
            
        indices = [int(i) - 1 for i in steps]
        
        # Check if requested horizon is within our trained prediction length
        if max(indices) >= self.pred_len_:
            warnings.warn(
                f"Requested horizon {max(indices)+1} exceeds trained prediction length {self.pred_len_}. "
                "Returning NaN for out-of-bound steps."
            )
            
        output = np.full(len(fh), np.nan)
        
        for i, idx in enumerate(indices):
            if 0 <= idx < self.pred_len_:
                output[i] = pred_values[idx]
        
        # Create proper index
        fh_abs = fh.to_absolute(self.cutoff)
        if hasattr(fh_abs, "to_pandas"):
            out_index = fh_abs.to_pandas()
        else:
            out_index = fh_abs
                
        return pd.Series(output, index=out_index, name=self._y_train.name)

# =============================================================================
# USAGE EXAMPLE
# =============================================================================
if __name__ == "__main__":
    from sktime.datasets import load_airline
    from sktime.forecasting.model_selection import temporal_train_test_split
    from sktime.performance_metrics.forecasting import mean_absolute_percentage_error
    import matplotlib.pyplot as plt

    print("Running LTSF-Linear (DLinear) Example...")
    
    y = load_airline()
    y_train, y_test = temporal_train_test_split(y, test_size=12)
    
    forecaster = LTSFLinearForecaster(
        seq_len=24,       # Look back 2 years
        model_type="DLinear",
        num_epochs=100
    )
    
    # Needs fh in fit
    fh = list(range(1, 13))
    forecaster.fit(y_train, fh=fh)
    
    y_pred = forecaster.predict(fh=fh)
    
    mape = mean_absolute_percentage_error(y_test, y_pred)
    print(f"MAPE: {mape:.4f}")
    
    # Plot
    plt.figure(figsize=(10, 6))
    y_train.plot(label="Train")
    y_test.plot(label="Test")
    y_pred.plot(label="Forecast")
    plt.title(f"DLinear Forecast (MAPE: {mape:.2%})")
    plt.legend()
    # plt.show()
    print("Example finished.")

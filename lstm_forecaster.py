"""
LSTM Forecaster: Standard Long Short-Term Memory model for Time Series
======================================================================

This module implements a standard LSTM-based forecaster.
It uses an Encoder-Process-Decoder architecture where the LSTM processes
the input sequence and a linear layer maps the final hidden state to the forecast.

Key Features:
- **LSTM**: Captures long-term dependencies in sequential data
- **Simple Architecture**: Single or multi-layer LSTM + Linear Head
- **sktime Compatible**: Fully integrated with sktime API
"""

import gc
import warnings
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset

# sktime imports
from sktime.forecasting.base import BaseForecaster

class LSTMModel(nn.Module):
    """
    Standard LSTM model for time series forecasting.
    """
    def __init__(self, input_size, hidden_size, output_size, num_layers=1):
        super().__init__()
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        
        # LSTM Layer
        # batch_first=True means input shape is (batch, seq, feature)
        self.lstm = nn.LSTM(
            input_size=input_size,
            hidden_size=hidden_size,
            num_layers=num_layers,
            batch_first=True
        )
        
        # Linear Head
        self.fc = nn.Linear(hidden_size, output_size)

    def forward(self, x):
        # x shape: (batch, seq_len, input_size)
        
        # LSTM output shape: (batch, seq_len, hidden_size)
        # We only care about the output of the last time step
        lstm_out, _ = self.lstm(x)
        
        # Take the last time step output: (batch, hidden_size)
        last_step_out = lstm_out[:, -1, :]
        
        # Map to forecast horizon: (batch, output_size)
        out = self.fc(last_step_out)
        return out


class LSTMForecaster(BaseForecaster):
    """
    LSTMForecaster: PyTorch-based LSTM model compatible with sktime.
    
    Parameters
    ----------
    seq_len : int, default=48
        Input sequence length (lookback window).
        
    hidden_size : int, default=64
        Number of features in the hidden state of the LSTM.
        
    num_layers : int, default=1
        Number of stacked LSTM layers.
        
    num_epochs : int, default=50
        Number of training epochs.
        
    batch_size : int, default=32
        Training batch size.
        
    learning_rate : float, default=0.001
        Learning rate for Adam optimizer.
        
    device : str, default="cpu"
        Device to run on ("cpu" or "cuda").
    """
    
    _tags = {
        "scitype:y": "univariate",
        "y_inner_mtype": "pd.Series",
        "X_inner_mtype": "pd.DataFrame", 
        "ignores-exogeneous-X": True,
        "requires-fh-in-fit": True,
        "handles-missing-data": False,
        "capability:insample": False,
        "capability:pred_int": False,
        "python_dependencies": ["torch"]
    }

    def __init__(
        self,
        seq_len: int = 48,
        hidden_size: int = 64,
        num_layers: int = 1,
        num_epochs: int = 50,
        batch_size: int = 32,
        learning_rate: float = 0.001,
        device: str = "cpu"
    ):
        self.seq_len = seq_len
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.num_epochs = num_epochs
        self.batch_size = batch_size
        self.learning_rate = learning_rate
        self.device = device
        
        self._y_train = None
        self.model_ = None
        self.pred_len_ = None
        
        super().__init__()

    def _fit(self, y, X=None, fh=None):
        if fh is None:
            raise ValueError("fh (ForecastingHorizon) is required for fit to determine output size.")
            
        fh_relative = fh.to_relative(self.cutoff)
        
        # Safe conversion to pandas/list
        if hasattr(fh_relative, "to_pandas"):
            steps = fh_relative.to_pandas()
        else:
            steps = fh_relative
            
        self.pred_len_ = int(max(steps))
        
        # Prepare data
        y_values = y.values.astype(np.float32)
        
        self.mean_ = np.mean(y_values)
        self.std_ = np.std(y_values) + 1e-5
        y_scaled = (y_values - self.mean_) / self.std_
        
        X_train, Y_train = [], []
        
        if len(y_scaled) <= self.seq_len + self.pred_len_:
             warnings.warn(f"Data length ({len(y_scaled)}) is small for seq_len={self.seq_len} + pred_len={self.pred_len_}")
        
        for i in range(len(y_scaled) - self.seq_len - self.pred_len_ + 1):
            X_train.append(y_scaled[i : i + self.seq_len])
            Y_train.append(y_scaled[i + self.seq_len : i + self.seq_len + self.pred_len_])
            
        if not X_train:
            raise ValueError("Not enough data to train.")
            
        # Reshape X for LSTM: (Batch, Seq, Feature=1)
        X_train = np.array(X_train)[..., np.newaxis]
        
        X_tensor = torch.tensor(X_train, dtype=torch.float32)
        Y_tensor = torch.tensor(np.array(Y_train), dtype=torch.float32)
        
        # Init model
        self.model_ = LSTMModel(
            input_size=1,
            hidden_size=self.hidden_size,
            output_size=self.pred_len_,
            num_layers=self.num_layers
        ).to(self.device)
        
        optimizer = optim.Adam(self.model_.parameters(), lr=self.learning_rate)
        criterion = nn.MSELoss()
        
        dataset = TensorDataset(X_tensor, Y_tensor)
        dataloader = DataLoader(dataset, batch_size=self.batch_size, shuffle=True)
        
        self.model_.train()
        for epoch in range(self.num_epochs):
            for batch_x, batch_y in dataloader:
                batch_x = batch_x.to(self.device)
                batch_y = batch_y.to(self.device)
                
                optimizer.zero_grad()
                outputs = self.model_(batch_x)
                loss = criterion(outputs, batch_y)
                loss.backward()
                optimizer.step()
                
        self._y_train = y
        return self

    def _predict(self, fh, X=None):
        y_values = self._y_train.values.astype(np.float32)
        y_scaled = (y_values - self.mean_) / self.std_
        
        if len(y_scaled) < self.seq_len:
            padding = np.zeros(self.seq_len - len(y_scaled))
            input_seq = np.concatenate([padding, y_scaled])
        else:
            input_seq = y_scaled[-self.seq_len:]
            
        # Shape: (1, Seq, 1)
        input_tensor = torch.tensor(input_seq, dtype=torch.float32).unsqueeze(0).unsqueeze(2).to(self.device)
        
        self.model_.eval()
        with torch.no_grad():
            output_tensor = self.model_(input_tensor)
            
        pred_scaled = output_tensor.cpu().numpy().squeeze()
        pred_values = pred_scaled * self.std_ + self.mean_
        
        # Map to fh
        fh_relative = fh.to_relative(self.cutoff)
        if hasattr(fh_relative, "to_pandas"):
            steps = fh_relative.to_pandas()
        else:
            steps = fh_relative
            
        indices = [int(i) - 1 for i in steps]
        
        output = np.full(len(fh), np.nan)
        for i, idx in enumerate(indices):
            if 0 <= idx < self.pred_len_:
                output[i] = pred_values[idx]
        
        fh_abs = fh.to_absolute(self.cutoff)
        if hasattr(fh_abs, "to_pandas"):
            out_index = fh_abs.to_pandas()
        else:
            out_index = fh_abs
                
        return pd.Series(output, index=out_index, name=self._y_train.name)

# Usage Example
if __name__ == "__main__":
    from sktime.datasets import load_airline
    from sktime.forecasting.model_selection import temporal_train_test_split
    from sktime.performance_metrics.forecasting import mean_absolute_percentage_error
    
    print("Running LSTM Example...")
    y = load_airline()
    y_train, y_test = temporal_train_test_split(y, test_size=12)
    
    model = LSTMForecaster(seq_len=24, hidden_size=50, num_epochs=100)
    fh = list(range(1, 13))
    model.fit(y_train, fh=fh)
    y_pred = model.predict(fh=fh)
    
    print(f"MAPE: {mean_absolute_percentage_error(y_test, y_pred):.4f}")

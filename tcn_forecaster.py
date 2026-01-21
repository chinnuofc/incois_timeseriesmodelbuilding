"""
TCN Forecaster: Temporal Convolutional Network for Time Series
==============================================================

This module implements a TCN (Temporal Convolutional Network) based on
dilated causal convolutions. TCNs often match or exceed RNN performance
while observing long history and being parallelizable.

Key Features:
- **Dilated Convolutions**: Exponentially increasing receptive field
- **Causal**: No future leakage (past predicts future)
- **Residual Blocks**: Stable training of deep networks
"""

import warnings
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
from sktime.forecasting.base import BaseForecaster

class Chomp1d(nn.Module):
    """
    Removes the last elements of a sequence to ensure causality after padding.
    """
    def __init__(self, chomp_size):
        super(Chomp1d, self).__init__()
        self.chomp_size = chomp_size

    def forwarded(self, x):
        return x[:, :, :-self.chomp_size]

    def forward(self, x):
        return x[:, :, :-self.chomp_size].contiguous()

class TemporalBlock(nn.Module):
    """
    Standard TCN Residual Block:
    - Dilated Causal Conv -> WeightNorm -> ReLU -> Dropout
    - Dilated Causal Conv -> WeightNorm -> ReLU -> Dropout
    - Residual connection
    """
    def __init__(self, n_inputs, n_outputs, kernel_size, stride, dilation, padding, dropout=0.2):
        super(TemporalBlock, self).__init__()
        
        # Conv 1
        self.conv1 = nn.utils.weight_norm(nn.Conv1d(n_inputs, n_outputs, kernel_size,
                                           stride=stride, padding=padding, dilation=dilation))
        self.chomp1 = Chomp1d(padding)
        self.relu1 = nn.ReLU()
        self.dropout1 = nn.Dropout(dropout)

        # Conv 2
        self.conv2 = nn.utils.weight_norm(nn.Conv1d(n_outputs, n_outputs, kernel_size,
                                           stride=stride, padding=padding, dilation=dilation))
        self.chomp2 = Chomp1d(padding)
        self.relu2 = nn.ReLU()
        self.dropout2 = nn.Dropout(dropout)

        self.net = nn.Sequential(self.conv1, self.chomp1, self.relu1, self.dropout1,
                                 self.conv2, self.chomp2, self.relu2, self.dropout2)
                                 
        self.downsample = nn.Conv1d(n_inputs, n_outputs, 1) if n_inputs != n_outputs else None
        self.relu = nn.ReLU()
        self.init_weights()

    def init_weights(self):
        self.conv1.weight.data.normal_(0, 0.01)
        self.conv2.weight.data.normal_(0, 0.01)
        if self.downsample is not None:
            self.downsample.weight.data.normal_(0, 0.01)

    def forward(self, x):
        out = self.net(x)
        res = x if self.downsample is None else self.downsample(x)
        return self.relu(out + res)

class TCNModel(nn.Module):
    def __init__(self, num_inputs, num_channels, kernel_size=2, dropout=0.2, output_size=1):
        super(TCNModel, self).__init__()
        layers = []
        num_levels = len(num_channels)
        
        for i in range(num_levels):
            dilation_size = 2 ** i
            in_channels = num_inputs if i == 0 else num_channels[i-1]
            out_channels = num_channels[i]
            
            layers += [TemporalBlock(in_channels, out_channels, kernel_size, stride=1, dilation=dilation_size,
                                     padding=(kernel_size-1) * dilation_size, dropout=dropout)]

        self.network = nn.Sequential(*layers)
        self.linear = nn.Linear(num_channels[-1], output_size)

    def forward(self, x):
        # x shape: (Batch, Seq_Len, Channels/Features)
        # Permute to (Batch, Channels, Seq_Len) for Conv1d
        x = x.permute(0, 2, 1)
        
        y = self.network(x)
        
        # Take last time step from the last channel: (Batch, Hidden_Dim, Seq_Len)
        # We want the features at the last time step
        y = y[:, :, -1]
        
        # Map to output
        return self.linear(y)

class TCNForecaster(BaseForecaster):
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
        num_channels: list = [32, 32, 32],
        kernel_size: int = 2,
        dropout: float = 0.2,
        num_epochs: int = 50,
        batch_size: int = 32,
        learning_rate: float = 0.002,
        device: str = "cpu"
    ):
        self.seq_len = seq_len
        self.num_channels = num_channels
        self.kernel_size = kernel_size
        self.dropout = dropout
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
             raise ValueError("fh required")
             
        fh_rel = fh.to_relative(self.cutoff)
        steps = fh_rel.to_pandas() if hasattr(fh_rel, "to_pandas") else fh_rel
        self.pred_len_ = int(max(steps))
        
        # Data prep
        y_val = y.values.astype(np.float32)
        self.mean_ = np.mean(y_val)
        self.std_ = np.std(y_val) + 1e-5
        y_scaled = (y_val - self.mean_) / self.std_
        
        X_train, Y_train = [], []
        
        if len(y_scaled) <= self.seq_len + self.pred_len_:
            warnings.warn("Short data")
            
        for i in range(len(y_scaled) - self.seq_len - self.pred_len_ + 1):
            X_train.append(y_scaled[i : i+self.seq_len])
            Y_train.append(y_scaled[i+self.seq_len : i+self.seq_len+self.pred_len_])
            
        if not X_train:
             raise ValueError("Not enough data")
             
        # Reshape to (Batch, Seq, Feature=1)
        X_train = np.array(X_train)[..., np.newaxis]
        Y_train = np.array(Y_train)
        
        X_t = torch.tensor(X_train, dtype=torch.float32)
        Y_t = torch.tensor(Y_train, dtype=torch.float32)
        
        self.model_ = TCNModel(
            num_inputs=1, 
            num_channels=self.num_channels,
            kernel_size=self.kernel_size,
            dropout=self.dropout,
            output_size=self.pred_len_
        ).to(self.device)
        
        opt = optim.Adam(self.model_.parameters(), lr=self.learning_rate)
        crit = nn.MSELoss()
        
        ds = TensorDataset(X_t, Y_t)
        dl = DataLoader(ds, batch_size=self.batch_size, shuffle=True)
        
        self.model_.train()
        for e in range(self.num_epochs):
            for bx, by in dl:
                bx, by = bx.to(self.device), by.to(self.device)
                opt.zero_grad()
                out = self.model_(bx)
                loss = crit(out, by)
                loss.backward()
                opt.step()
                
        self._y_train = y
        return self

    def _predict(self, fh, X=None):
        y_val = self._y_train.values.astype(np.float32)
        y_scaled = (y_val - self.mean_) / self.std_
        
        if len(y_scaled) < self.seq_len:
            pad = np.zeros(self.seq_len - len(y_scaled))
            inp = np.concatenate([pad, y_scaled])
        else:
            inp = y_scaled[-self.seq_len:]
            
        inp_t = torch.tensor(inp, dtype=torch.float32).unsqueeze(0).unsqueeze(2).to(self.device)
        
        self.model_.eval()
        with torch.no_grad():
            out_t = self.model_(inp_t)
            
        pred = out_t.cpu().numpy().squeeze()
        pred = pred * self.std_ + self.mean_
        
        # Horizon mapping
        fh_rel = fh.to_relative(self.cutoff)
        steps = fh_rel.to_pandas() if hasattr(fh_rel, "to_pandas") else fh_rel
        indices = [int(x)-1 for x in steps]
        
        out = np.full(len(fh), np.nan)
        for i, idx in enumerate(indices):
            if 0 <= idx < self.pred_len_:
                out[i] = pred[idx]
                
        fh_abs = fh.to_absolute(self.cutoff)
        idx_abs = fh_abs.to_pandas() if hasattr(fh_abs, "to_pandas") else fh_abs
        
        return pd.Series(out, index=idx_abs, name=self._y_train.name)

if __name__ == "__main__":
    from sktime.datasets import load_airline
    from sktime.forecasting.model_selection import temporal_train_test_split
    from sktime.performance_metrics.forecasting import mean_absolute_percentage_error
    
    print("Running TCN Example...")
    y = load_airline()
    y_tr, y_te = temporal_train_test_split(y, test_size=12)
    
    m = TCNForecaster(seq_len=36, num_channels=[30, 30], num_epochs=100)
    fh = list(range(1, 13))
    m.fit(y_tr, fh=fh)
    p = m.predict(fh=fh)
    print(f"MAPE: {mean_absolute_percentage_error(y_te, p):.4f}")

"""
ChronosSktimeForecaster: A custom sktime-compatible wrapper for Chronos-T5-Small
================================================================================

This module implements a production-ready forecaster that wraps the Amazon Chronos
foundation model for time series forecasting within the sktime framework.

Hardware Optimization:
- Designed for CPU inference (no GPU required)
- Memory-efficient with garbage collection after predictions
- Uses float32 precision for CPU compatibility

Installation Commands:
----------------------
pip install sktime torch transformers accelerate chronos-forecasting

Or for a specific chronos version:
pip install git+https://github.com/amazon-science/chronos-forecasting.git

Author: Custom Implementation for Resource-Constrained Environments
"""

import gc
import warnings
from typing import Optional, Union

import numpy as np
import pandas as pd
import torch

# sktime imports
from sktime.forecasting.base import BaseForecaster, ForecastingHorizon

# Chronos imports - using the official chronos-forecasting package
try:
    from chronos import ChronosPipeline
except ImportError:
    raise ImportError(
        "chronos-forecasting package not found. Install it with:\n"
        "pip install chronos-forecasting\n"
        "or: pip install git+https://github.com/amazon-science/chronos-forecasting.git"
    )


class ChronosSktimeForecaster(BaseForecaster):
    """
    A sktime-compatible wrapper for the Amazon Chronos-T5 foundation model.
    
    This forecaster uses the pre-trained Chronos-T5 model for zero-shot time series
    forecasting. Since Chronos is a foundation model, no training is performed -
    the _fit method only loads the model into memory.
    
    Parameters
    ----------
    model_name : str, default="amazon/chronos-t5-small"
        The Hugging Face model identifier for the Chronos model.
        Available options:
        - "amazon/chronos-t5-tiny"   (~8M parameters, fastest)
        - "amazon/chronos-t5-mini"   (~20M parameters)
        - "amazon/chronos-t5-small"  (~46M parameters, recommended for 8GB RAM)
        - "amazon/chronos-t5-base"   (~200M parameters)
        - "amazon/chronos-t5-large"  (~710M parameters)
    
    device : str, default="cpu"
        Device to run inference on. Use "cpu" for CPU inference,
        "cuda" for GPU, or "mps" for Apple Silicon.
    
    torch_dtype : torch.dtype, default=torch.float32
        The torch dtype for model weights. Use torch.float32 for CPU
        inference or torch.bfloat16 if your CPU supports it.
    
    num_samples : int, default=20
        Number of sample trajectories to generate for probabilistic forecasts.
        Lower values use less memory but give less accurate prediction intervals.
    
    temperature : float, default=1.0
        Sampling temperature for the model. Higher values increase diversity
        of samples. Use 1.0 for default behavior.
    
    top_k : int, default=50
        Top-k sampling parameter. Only the top k tokens are considered for
        sampling at each step.
    
    top_p : float, default=1.0
        Top-p (nucleus) sampling parameter. Tokens with cumulative probability
        below this threshold are considered.
    
    context_length : int, optional, default=None
        Maximum context length to use. If None, uses the model's default.
        Reducing this can save memory for very long time series.
    
    Attributes
    ----------
    pipeline_ : ChronosPipeline
        The loaded Chronos model pipeline (set after fit).
    
    _y_train : pd.Series
        The training time series data stored for prediction context.
    
    Examples
    --------
    >>> from sktime.datasets import load_airline
    >>> from sktime.forecasting.model_selection import temporal_train_test_split
    >>> 
    >>> # Load data
    >>> y = load_airline()
    >>> y_train, y_test = temporal_train_test_split(y, test_size=12)
    >>> 
    >>> # Create and fit forecaster
    >>> forecaster = ChronosSktimeForecaster(model_name="amazon/chronos-t5-small")
    >>> forecaster.fit(y_train)
    >>> 
    >>> # Make predictions
    >>> fh = list(range(1, 13))  # Forecast 12 steps ahead
    >>> y_pred = forecaster.predict(fh=fh)
    
    Notes
    -----
    - This is a zero-shot forecaster: no training is performed on your data.
    - The model is loaded lazily during fit() to save memory.
    - Memory is cleaned up after each prediction to prevent OOM errors.
    - For resource-constrained systems (< 8GB RAM), use chronos-t5-tiny or mini.
    """
    
    # sktime tag configuration
    _tags = {
        # Forecaster type tags
        "scitype:y": "univariate",          # Supports univariate time series
        "y_inner_mtype": "pd.Series",       # Internal data type
        "X_inner_mtype": "pd.DataFrame",    # Exogenous data type (not used)
        "ignores-exogeneous-X": True,       # Chronos doesn't use exogenous vars
        "requires-fh-in-fit": False,        # Horizon not needed during fit
        "handles-missing-data": False,      # Does not handle NaN values
        "capability:insample": False,       # Cannot predict in-sample values
        "capability:pred_int": True,        # Can produce prediction intervals
        "capability:pred_var": False,       # Cannot produce variance forecasts
        "python_version": ">=3.8",          # Python version requirement
        "python_dependencies": [            # Required packages
            "torch",
            "chronos-forecasting",
        ],
    }
    
    def __init__(
        self,
        model_name: str = "amazon/chronos-t5-small",
        device: str = "cpu",
        torch_dtype: torch.dtype = torch.float32,
        num_samples: int = 20,
        temperature: float = 1.0,
        top_k: int = 50,
        top_p: float = 1.0,
        context_length: Optional[int] = None,
    ):
        """
        Initialize the ChronosSktimeForecaster.
        
        Parameters are stored as instance attributes for sklearn compatibility.
        The model is NOT loaded here to save memory - loading happens in _fit().
        """
        # Store all hyperparameters as instance attributes
        self.model_name = model_name
        self.device = device
        self.torch_dtype = torch_dtype
        self.num_samples = num_samples
        self.temperature = temperature
        self.top_k = top_k
        self.top_p = top_p
        self.context_length = context_length
        
        # Initialize internal state (set during fit)
        self.pipeline_ = None
        self._y_train = None
        
        # Call parent constructor - MUST be done after setting attributes
        super().__init__()
    
    def _fit(self, y, X=None, fh=None):
        """
        Fit the forecaster to the training data.
        
        For Chronos (a pre-trained foundation model), "fitting" means:
        1. Loading the pre-trained model weights into memory
        2. Storing the training data for use as context during prediction
        
        NO TRAINING/FINE-TUNING is performed - this is a zero-shot model.
        
        Parameters
        ----------
        y : pd.Series
            The target time series to forecast.
            Must have a proper DatetimeIndex or PeriodIndex.
        
        X : pd.DataFrame, optional
            Exogenous variables (ignored by Chronos).
        
        fh : ForecastingHorizon, optional
            The forecasting horizon (not required for fit).
        
        Returns
        -------
        self : ChronosSktimeForecaster
            The fitted forecaster instance.
        """
        # Store the training data - this will be used as context for predictions
        # Chronos uses the historical data to generate forecasts
        self._y_train = y.copy()
        
        # Load the Chronos model if not already loaded
        # This is done lazily to save memory during object instantiation
        if self.pipeline_ is None:
            print(f"Loading Chronos model: {self.model_name}")
            print(f"Device: {self.device}, Dtype: {self.torch_dtype}")
            
            # Load the pretrained Chronos pipeline
            # device_map="cpu" ensures CPU inference
            # dtype=float32 for CPU compatibility (bfloat16 may not work on all CPUs)
            self.pipeline_ = ChronosPipeline.from_pretrained(
                self.model_name,
                device_map=self.device,
                dtype=self.torch_dtype,  # Use 'dtype' (torch_dtype is deprecated)
            )
            
            print(f"Model loaded successfully!")
        
        return self
    
    def _predict(self, fh, X=None):
        """
        Generate point forecasts for the given forecast horizon.
        
        This method:
        1. Converts the stored training data to a PyTorch tensor
        2. Runs inference through the Chronos model
        3. Converts the output back to a pandas Series with proper index
        
        Parameters
        ----------
        fh : ForecastingHorizon
            The forecast horizon - specifies which future time points to predict.
            Can be relative (e.g., [1, 2, 3]) or absolute datetime indices.
        
        X : pd.DataFrame, optional
            Exogenous variables (ignored by Chronos).
        
        Returns
        -------
        y_pred : pd.Series
            Point forecasts for the specified horizon.
            Index matches the forecasting horizon's datetime index.
        
        Notes
        -----
        The forecasting horizon (fh) in sktime can be:
        - Relative: [1, 2, 3] means 1, 2, 3 steps ahead from the end of training data
        - Absolute: Actual datetime values for the predictions
        
        This method handles both cases by using sktime's ForecastingHorizon utilities.
        """
        # Convert ForecastingHorizon to integer steps
        # get_step_by_step returns relative integer steps
        if hasattr(fh, 'to_relative'):
            fh_relative = fh.to_relative(self.cutoff)
            prediction_length = int(max(fh_relative._values))
        else:
            # Handle simple integer list
            prediction_length = int(max(fh))
        
        # Prepare the context (historical data) for Chronos
        # Chronos expects a 2D tensor: (batch_size, context_length)
        # For univariate forecasting, batch_size = 1
        context_values = self._y_train.values.astype(np.float32)
        
        # Limit context length if specified (saves memory for very long series)
        if self.context_length is not None and len(context_values) > self.context_length:
            context_values = context_values[-self.context_length:]
            warnings.warn(
                f"Context truncated to last {self.context_length} observations "
                f"to save memory."
            )
        
        # Convert to PyTorch tensor - shape: (1, context_length)
        # Note: Chronos expects the context WITHOUT batch dimension in predict()
        context_tensor = torch.tensor(context_values, dtype=self.torch_dtype)
        
        # Generate predictions using Chronos
        # The output is a tensor of shape (num_samples, prediction_length)
        # containing multiple sample trajectories
        with torch.inference_mode():
            # Chronos predict() returns samples from the predictive distribution
            # Note: The API uses 'inputs' (not 'context') for the input tensor
            forecast_samples = self.pipeline_.predict(
                inputs=context_tensor,
                prediction_length=prediction_length,
                num_samples=self.num_samples,
                temperature=self.temperature,
                top_k=self.top_k,
                top_p=self.top_p,
            )
        
        # Convert forecast to numpy and compute point forecast (median)
        # Chronos output shape: (batch_size, num_samples, prediction_length)
        # For univariate: (1, num_samples, prediction_length)
        # We need to squeeze batch dim and take median across samples
        forecast_np = forecast_samples.numpy()
        
        # Squeeze batch dimension if present: (1, num_samples, pred_len) -> (num_samples, pred_len)
        if forecast_np.ndim == 3:
            forecast_np = forecast_np.squeeze(0)  # Now shape: (num_samples, prediction_length)
        
        # Use median as point forecast (more robust than mean for skewed distributions)
        # This gives shape: (prediction_length,)
        point_forecast = np.median(forecast_np, axis=0)
        
        # Create the forecast index
        # sktime's cutoff is the last timestamp in the training data
        # We need to create index values for the forecast horizon
        fh_absolute = fh.to_absolute(self.cutoff)
        
        # Select only the requested forecast steps
        # (in case fh doesn't start at 1 or has gaps)
        if hasattr(fh, 'to_relative'):
            fh_rel = fh.to_relative(self.cutoff)
            # Convert to 0-indexed positions in the forecast array
            fh_indices = [int(h) - 1 for h in fh_rel._values]
        else:
            fh_indices = [int(h) - 1 for h in fh]
        
        # Extract the forecasts for the requested horizon
        selected_forecasts = point_forecast[fh_indices]
        
        # Create the output Series with proper index
        y_pred = pd.Series(
            data=selected_forecasts,
            index=fh_absolute._values if hasattr(fh_absolute, '_values') else fh_absolute,
            name=self._y_train.name,
        )
        
        # Memory cleanup - important for resource-constrained systems!
        del context_tensor, forecast_samples, forecast_np
        gc.collect()
        
        # Clear CUDA cache if using GPU (not applicable for CPU but good practice)
        if self.device == "cuda":
            torch.cuda.empty_cache()
        
        return y_pred
    
    def _predict_interval(self, fh, X=None, coverage=None):
        """
        Generate prediction intervals for the forecast horizon.
        
        Prediction intervals quantify the uncertainty in forecasts by
        providing lower and upper bounds at specified coverage levels.
        
        Parameters
        ----------
        fh : ForecastingHorizon
            The forecast horizon.
        
        X : pd.DataFrame, optional
            Exogenous variables (ignored).
        
        coverage : list of float, default=[0.9]
            Coverage levels for the prediction intervals.
            E.g., 0.9 means 90% of observations should fall within the interval.
        
        Returns
        -------
        pred_int : pd.DataFrame
            DataFrame with MultiIndex columns: (variable_name, coverage, "lower"/"upper")
            Each row corresponds to a forecast horizon step.
        """
        if coverage is None:
            coverage = [0.9]
        
        # Ensure coverage is a list
        if not isinstance(coverage, list):
            coverage = [coverage]
        
        # Get forecast horizon details
        if hasattr(fh, 'to_relative'):
            fh_relative = fh.to_relative(self.cutoff)
            prediction_length = int(max(fh_relative._values))
        else:
            prediction_length = int(max(fh))
        
        # Prepare context
        context_values = self._y_train.values.astype(np.float32)
        if self.context_length is not None and len(context_values) > self.context_length:
            context_values = context_values[-self.context_length:]
        
        context_tensor = torch.tensor(context_values, dtype=self.torch_dtype)
        
        # Generate forecasts with multiple samples for interval estimation
        with torch.inference_mode():
            forecast_samples = self.pipeline_.predict(
                inputs=context_tensor,
                prediction_length=prediction_length,
                num_samples=self.num_samples,
                temperature=self.temperature,
                top_k=self.top_k,
                top_p=self.top_p,
            )
        
        # Handle output shape: (batch_size, num_samples, prediction_length)
        forecast_np = forecast_samples.numpy()
        if forecast_np.ndim == 3:
            forecast_np = forecast_np.squeeze(0)  # (num_samples, prediction_length)
        
        # Get forecast horizon indices
        if hasattr(fh, 'to_relative'):
            fh_rel = fh.to_relative(self.cutoff)
            fh_indices = [int(h) - 1 for h in fh_rel._values]
        else:
            fh_indices = [int(h) - 1 for h in fh]
        
        # Create absolute index for output
        fh_absolute = fh.to_absolute(self.cutoff)
        out_index = fh_absolute._values if hasattr(fh_absolute, '_values') else fh_absolute
        
        # Build the prediction interval DataFrame
        # sktime expects MultiIndex columns: (var_name, coverage, "lower"/"upper")
        var_name = self._y_train.name if self._y_train.name else "y"
        
        columns = pd.MultiIndex.from_product(
            [[var_name], coverage, ["lower", "upper"]],
            names=["variable", "coverage", "bound"]
        )
        
        pred_int_data = []
        for idx in fh_indices:
            row_data = []
            for cov in coverage:
                alpha = 1 - cov
                lower = np.percentile(forecast_np[:, idx], alpha / 2 * 100)
                upper = np.percentile(forecast_np[:, idx], (1 - alpha / 2) * 100)
                row_data.extend([lower, upper])
            pred_int_data.append(row_data)
        
        pred_int = pd.DataFrame(
            data=pred_int_data,
            index=out_index,
            columns=columns,
        )
        
        # Memory cleanup
        del context_tensor, forecast_samples, forecast_np
        gc.collect()
        
        return pred_int
    
    def _predict_quantiles(self, fh, X=None, alpha=None):
        """
        Generate quantile forecasts for the forecast horizon.
        
        Parameters
        ----------
        fh : ForecastingHorizon
            The forecast horizon.
        
        X : pd.DataFrame, optional
            Exogenous variables (ignored).
        
        alpha : list of float
            Quantile levels to predict. E.g., [0.1, 0.5, 0.9] for
            10th, 50th, and 90th percentiles.
        
        Returns
        -------
        pred_q : pd.DataFrame
            DataFrame with quantile forecasts.
        """
        if alpha is None:
            alpha = [0.1, 0.5, 0.9]
        
        if not isinstance(alpha, list):
            alpha = [alpha]
        
        # Get forecast horizon details
        if hasattr(fh, 'to_relative'):
            fh_relative = fh.to_relative(self.cutoff)
            prediction_length = int(max(fh_relative._values))
        else:
            prediction_length = int(max(fh))
        
        # Prepare context
        context_values = self._y_train.values.astype(np.float32)
        if self.context_length is not None and len(context_values) > self.context_length:
            context_values = context_values[-self.context_length:]
        
        context_tensor = torch.tensor(context_values, dtype=self.torch_dtype)
        
        # Generate forecasts
        with torch.inference_mode():
            forecast_samples = self.pipeline_.predict(
                inputs=context_tensor,
                prediction_length=prediction_length,
                num_samples=self.num_samples,
                temperature=self.temperature,
                top_k=self.top_k,
                top_p=self.top_p,
            )
        
        # Handle output shape: (batch_size, num_samples, prediction_length)
        forecast_np = forecast_samples.numpy()
        if forecast_np.ndim == 3:
            forecast_np = forecast_np.squeeze(0)  # (num_samples, prediction_length)
        
        # Get forecast horizon indices
        if hasattr(fh, 'to_relative'):
            fh_rel = fh.to_relative(self.cutoff)
            fh_indices = [int(h) - 1 for h in fh_rel._values]
        else:
            fh_indices = [int(h) - 1 for h in fh]
        
        # Create absolute index for output
        fh_absolute = fh.to_absolute(self.cutoff)
        out_index = fh_absolute._values if hasattr(fh_absolute, '_values') else fh_absolute
        
        # Build quantile DataFrame
        var_name = self._y_train.name if self._y_train.name else "y"
        
        columns = pd.MultiIndex.from_product(
            [[var_name], alpha],
            names=["variable", "quantile"]
        )
        
        quantile_data = []
        for idx in fh_indices:
            row_data = [np.percentile(forecast_np[:, idx], q * 100) for q in alpha]
            quantile_data.append(row_data)
        
        pred_q = pd.DataFrame(
            data=quantile_data,
            index=out_index,
            columns=columns,
        )
        
        # Memory cleanup
        del context_tensor, forecast_samples, forecast_np
        gc.collect()
        
        return pred_q
    
    def get_fitted_params(self):
        """
        Get fitted parameters of the forecaster.
        
        For Chronos, this returns information about the loaded model
        since no actual parameters are fitted.
        
        Returns
        -------
        fitted_params : dict
            Dictionary containing model information.
        """
        return {
            "model_name": self.model_name,
            "device": self.device,
            "torch_dtype": str(self.torch_dtype),
            "is_fitted": self.pipeline_ is not None,
        }
    
    @classmethod
    def get_test_params(cls, parameter_set="default"):
        """
        Return testing parameter settings for the estimator.
        
        Used by sktime's testing framework.
        
        Parameters
        ----------
        parameter_set : str, default="default"
            The parameter set to return.
        
        Returns
        -------
        params : dict or list of dict
            Parameters to create testing instances.
        """
        # Use the smallest model for testing
        return {
            "model_name": "amazon/chronos-t5-tiny",
            "device": "cpu",
            "torch_dtype": torch.float32,
            "num_samples": 5,  # Fewer samples for faster testing
        }


# =============================================================================
# USAGE EXAMPLE AND VALIDATION
# =============================================================================

if __name__ == "__main__":
    """
    Demonstration of ChronosSktimeForecaster with the Airline dataset.
    
    This example shows:
    1. Loading a standard sktime dataset
    2. Train/test splitting using temporal_train_test_split
    3. Creating and fitting the Chronos forecaster
    4. Making predictions
    5. Evaluating with mean_absolute_percentage_error
    """
    import matplotlib.pyplot as plt
    
    from sktime.datasets import load_airline
    from sktime.forecasting.model_selection import temporal_train_test_split
    from sktime.performance_metrics.forecasting import (
        mean_absolute_percentage_error,
        mean_absolute_error,
        mean_squared_error,
    )
    
    print("=" * 70)
    print("ChronosSktimeForecaster - Usage Example")
    print("=" * 70)
    
    # =========================================================================
    # Step 1: Load the dataset
    # =========================================================================
    print("\n[Step 1] Loading the Airline Passengers dataset...")
    y = load_airline()
    print(f"Dataset shape: {y.shape}")
    print(f"Date range: {y.index[0]} to {y.index[-1]}")
    print(f"Sample data:\n{y.head()}")
    
    # =========================================================================
    # Step 2: Train/Test Split
    # =========================================================================
    print("\n[Step 2] Splitting data into train and test sets...")
    # Hold out last 12 months for testing
    y_train, y_test = temporal_train_test_split(y, test_size=12)
    print(f"Training set: {len(y_train)} observations ({y_train.index[0]} to {y_train.index[-1]})")
    print(f"Test set: {len(y_test)} observations ({y_test.index[0]} to {y_test.index[-1]})")
    
    # =========================================================================
    # Step 3: Create and Fit the Forecaster
    # =========================================================================
    print("\n[Step 3] Creating and fitting the ChronosSktimeForecaster...")
    
    # Create the forecaster with CPU-optimized settings
    forecaster = ChronosSktimeForecaster(
        model_name="amazon/chronos-t5-small",  # Good balance of speed and accuracy
        device="cpu",                           # CPU inference
        torch_dtype=torch.float32,              # Float32 for CPU compatibility
        num_samples=20,                         # Number of forecast samples
        temperature=1.0,                        # Default sampling temperature
    )
    
    # Fit the forecaster (loads the model and stores training data)
    print("Fitting the forecaster...")
    forecaster.fit(y_train)
    print("Forecaster fitted successfully!")
    
    # =========================================================================
    # Step 4: Make Predictions
    # =========================================================================
    print("\n[Step 4] Generating forecasts...")
    
    # Define the forecast horizon - 12 steps ahead
    fh = list(range(1, 13))  # [1, 2, 3, ..., 12]
    
    # Generate point forecasts
    y_pred = forecaster.predict(fh=fh)
    print(f"Predictions generated for {len(y_pred)} time steps")
    print(f"Predicted values:\n{y_pred}")
    
    # =========================================================================
    # Step 5: Evaluate Performance
    # =========================================================================
    print("\n[Step 5] Evaluating forecast performance...")
    
    # Calculate various error metrics
    mape = mean_absolute_percentage_error(y_test, y_pred)
    mae = mean_absolute_error(y_test, y_pred)
    rmse = mean_squared_error(y_test, y_pred, square_root=True)
    
    print(f"\nPerformance Metrics:")
    print(f"  Mean Absolute Percentage Error (MAPE): {mape:.4f} ({mape * 100:.2f}%)")
    print(f"  Mean Absolute Error (MAE): {mae:.2f}")
    print(f"  Root Mean Squared Error (RMSE): {rmse:.2f}")
    
    # =========================================================================
    # Step 6: Generate Prediction Intervals (Optional)
    # =========================================================================
    print("\n[Step 6] Generating prediction intervals...")
    
    # Generate 90% prediction intervals
    pred_intervals = forecaster.predict_interval(fh=fh, coverage=[0.9])
    print("90% Prediction Intervals:")
    print(pred_intervals)
    
    # =========================================================================
    # Step 7: Visualization
    # =========================================================================
    print("\n[Step 7] Creating visualization...", flush=True)
    
    # Helper to convert PeriodIndex to Timestamp for plotting
    def to_plot_idx(idx):
        return idx.to_timestamp() if hasattr(idx, "to_timestamp") else idx
    
    # Create a figure with the forecast
    fig, ax = plt.subplots(figsize=(12, 6))
    
    # Plot training data
    ax.plot(to_plot_idx(y_train.index), y_train.values, 'b-', label='Training Data', linewidth=1.5)
    
    # Plot test data (actual values)
    ax.plot(to_plot_idx(y_test.index), y_test.values, 'g-', label='Actual (Test)', linewidth=2)
    
    # Plot predictions
    ax.plot(to_plot_idx(y_pred.index), y_pred.values, 'r--', label='Chronos Forecast', linewidth=2)
    
    # Plot prediction intervals if available
    var_name = y.name if y.name else "Number of airline passengers"
    if pred_intervals is not None:
        lower = pred_intervals[(var_name, 0.9, 'lower')].values
        upper = pred_intervals[(var_name, 0.9, 'upper')].values
        ax.fill_between(
            to_plot_idx(y_pred.index), lower, upper,
            alpha=0.2, color='red', label='90% Prediction Interval'
        )
    
    ax.set_xlabel('Date', fontsize=12)
    ax.set_ylabel('Number of Passengers', fontsize=12)
    ax.set_title('Airline Passenger Forecast using Chronos-T5-Small', fontsize=14)
    ax.legend(loc='upper left')
    ax.grid(True, alpha=0.3)
    
    plt.tight_layout()
    
    # Save the figure
    output_path = "chronos_forecast_example.png"
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    print(f"Plot saved to: {output_path}")
    
    # Show the plot (comment out if running in non-interactive mode)
    # plt.show()
    
    # =========================================================================
    # Step 8: Memory Cleanup
    # =========================================================================
    print("\n[Step 8] Cleaning up...")
    
    # Force garbage collection to free memory
    del forecaster
    gc.collect()
    
    print("\n" + "=" * 70)
    print("Example completed successfully!")
    print("=" * 70)

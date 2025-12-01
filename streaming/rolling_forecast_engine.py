"""
Rolling Forecast Engine - Incremental LSTM predictions
"""
import numpy as np
import pandas as pd
from typing import Optional, List
from tensorflow.keras.models import load_model
import pickle
import sys
import os

# Add parent directory to path for imports
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import config
from mlad_anomaly_detection import create_features
from streaming.streaming_config import *


class RollingForecastEngine:
    """
    Generates LSTM forecasts incrementally as new data arrives.
    Maintains sliding window of features for predictions.
    """

    def __init__(self):
        """Initialize the forecast engine."""
        self.model = None
        self.scaler = None
        self.feature_cols = None
        self.feature_window = []  # Last 168 hours of features
        self.forecast_history = []  # Store forecasts for drift calculation
        self.actual_history = []  # Store actuals for drift calculation
        self.initialized = False

    def load_models(self):
        """Load pre-trained LSTM model and scaler."""
        print("Loading LSTM model and scaler...")

        # Load model
        if not os.path.exists(config.FORECASTER_MODEL_PATH):
            raise FileNotFoundError(f"Model not found: {config.FORECASTER_MODEL_PATH}")

        self.model = load_model(config.FORECASTER_MODEL_PATH)
        print(f"Model loaded from {config.FORECASTER_MODEL_PATH}")

        # Load scaler
        if not os.path.exists(config.SCALER_PATH):
            raise FileNotFoundError(f"Scaler not found: {config.SCALER_PATH}")

        with open(config.SCALER_PATH, 'rb') as f:
            self.scaler = pickle.load(f)
        print(f"Scaler loaded from {config.SCALER_PATH}")

        # Feature columns (we'll determine from model input shape)
        # For now, use standard feature columns
        self.feature_cols = [
            'load_lag_1', 'load_lag_24', 'load_lag_168',
            'day_of_week', 'day_of_year', 'month', 'is_weekend',
            'hour_sin', 'hour_cos', 'month_sin', 'month_cos'
        ]

    def initialize(self, initial_data: pd.DataFrame):
        """
        Initialize with initial data to build feature window.

        Args:
            initial_data: DataFrame with at least FORECAST_WINDOW_HOURS of data
        """
        if self.model is None:
            self.load_models()

        print(f"Initializing forecast engine with {len(initial_data)} hours of data...")

        # Create features for initial data
        df_features = create_features(initial_data)

        # Extract feature values
        feature_values = df_features[self.feature_cols].values

        # Scale features
        feature_values_scaled = self.scaler.transform(feature_values)

        # Store in window (keep last FORECAST_WINDOW_HOURS)
        self.feature_window = feature_values_scaled[-FORECAST_WINDOW_HOURS:].tolist()

        # Store actuals for drift calculation
        self.actual_history = initial_data['load'].values[-FORECAST_WINDOW_HOURS:].tolist()

        self.initialized = True
        print(f"Forecast engine initialized with {len(self.feature_window)} hours in window")

    def predict_next_hour(self, current_data_point: pd.Series) -> float:
        """
        Predict load for the next hour.

        Args:
            current_data_point: Current hour's data (with timestamp and load)

        Returns:
            Forecasted load value
        """
        if not self.initialized:
            raise RuntimeError("Forecast engine not initialized. Call initialize() first.")

        # Get timestamp
        timestamp = current_data_point['timestamp']
        if isinstance(timestamp, str):
            timestamp = pd.to_datetime(timestamp)

        # Create features manually (simplified - in production, use create_features)
        features = self._create_features_for_timestamp(
            timestamp,
            current_data_point['load'],
            self.actual_history[-1] if self.actual_history else current_data_point['load'],
            self.actual_history[-24] if len(self.actual_history) >= 24 else current_data_point['load'],
            self.actual_history[-168] if len(self.actual_history) >= 168 else current_data_point['load']
        )

        # Scale features
        features_scaled = self.scaler.transform([features])

        # Reshape for LSTM [samples, timesteps, features]
        features_reshaped = features_scaled.reshape((1, 1, len(features)))

        # Predict
        forecast = self.model.predict(features_reshaped, verbose=0)[0, 0]

        # Update feature window (keep last FORECAST_WINDOW_HOURS)
        self.feature_window.append(features_scaled[0].tolist())
        if len(self.feature_window) > FORECAST_WINDOW_HOURS:
            self.feature_window.pop(0)

        # Update actual history
        self.actual_history.append(current_data_point['load'])
        if len(self.actual_history) > FORECAST_WINDOW_HOURS:
            self.actual_history.pop(0)

        # Store forecast for drift calculation
        self.forecast_history.append(forecast)
        if len(self.forecast_history) > RETENTION_HOURS:
            self.forecast_history.pop(0)

        return forecast

    def _create_features_for_timestamp(self, timestamp: pd.Timestamp,
                                      current_load: float,
                                      lag_1: float, lag_24: float, lag_168: float) -> List[float]:
        """Create feature vector for a timestamp."""
        hour = timestamp.hour
        day_of_week = timestamp.dayofweek
        day_of_year = timestamp.dayofyear
        month = timestamp.month
        is_weekend = 1 if day_of_week >= 5 else 0

        # Cyclical encodings
        hour_sin = np.sin(2 * np.pi * hour / 24)
        hour_cos = np.cos(2 * np.pi * hour / 24)
        month_sin = np.sin(2 * np.pi * month / 12)
        month_cos = np.cos(2 * np.pi * month / 12)

        return [
            lag_1, lag_24, lag_168,
            day_of_week, day_of_year, month, is_weekend,
            hour_sin, hour_cos, month_sin, month_cos
        ]

    def calculate_forecast_drift(self, window_hours: int = 24) -> dict:
        """
        Calculate forecast accuracy drift over time.

        Args:
            window_hours: Number of recent hours to analyze

        Returns:
            Dictionary with drift metrics
        """
        if len(self.forecast_history) < window_hours or len(self.actual_history) < window_hours:
            return {
                'mae': None,
                'mape': None,
                'drift_percentage': None,
                'sufficient_data': False
            }

        # Get recent forecasts and actuals
        recent_forecasts = np.array(self.forecast_history[-window_hours:])
        recent_actuals = np.array(self.actual_history[-window_hours:])

        # Calculate metrics
        mae = np.mean(np.abs(recent_forecasts - recent_actuals))
        mape = np.mean(np.abs((recent_forecasts - recent_actuals) / (recent_actuals + 1e-6))) * 100

        # Compare to baseline (first window_hours)
        if len(self.forecast_history) >= window_hours * 2:
            baseline_forecasts = np.array(self.forecast_history[:window_hours])
            baseline_actuals = np.array(self.actual_history[:window_hours])
            baseline_mae = np.mean(np.abs(baseline_forecasts - baseline_actuals))

            drift_percentage = ((mae - baseline_mae) / (baseline_mae + 1e-6)) * 100
        else:
            drift_percentage = None

        return {
            'mae': mae,
            'mape': mape,
            'drift_percentage': drift_percentage,
            'sufficient_data': True,
            'window_hours': window_hours
        }


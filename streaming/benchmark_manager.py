"""
Benchmark Manager - K-means benchmark retrieval
"""
import numpy as np
import pandas as pd
import pickle
from typing import Optional
import sys
import os

# Add parent directory to path for imports
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import config
from streaming.streaming_config import *


class BenchmarkManager:
    """
    Manages K-means benchmark model for retrieving normal load patterns.
    """

    def __init__(self):
        """Initialize the benchmark manager."""
        self.kmeans_model = None
        self.loaded = False

    def load_model(self):
        """Load pre-trained K-means model."""
        print("Loading K-means benchmark model...")

        if not os.path.exists(config.KMEANS_MODEL_PATH):
            raise FileNotFoundError(f"K-means model not found: {config.KMEANS_MODEL_PATH}")

        with open(config.KMEANS_MODEL_PATH, 'rb') as f:
            self.kmeans_model = pickle.load(f)

        self.loaded = True
        print(f"K-means model loaded from {config.KMEANS_MODEL_PATH}")
        print(f"Number of clusters: {len(self.kmeans_model.cluster_centers_)}")

    def get_benchmark_for_hour(self, forecast_values: np.ndarray, hour_of_day: int) -> float:
        """
        Get benchmark value for a specific hour.

        Args:
            forecast_values: Array of forecast values (last 24 hours)
            hour_of_day: Hour of day (0-23)

        Returns:
            Benchmark value for that hour
        """
        if not self.loaded:
            self.load_model()

        # Ensure we have at least 24 hours
        if len(forecast_values) < 24:
            # Pad with last value
            padded = np.pad(forecast_values, (24 - len(forecast_values), 0),
                          mode='edge')
            daily_pattern = padded[-24:].reshape(1, -1).astype(np.float32)
        else:
            daily_pattern = forecast_values[-24:].reshape(1, -1).astype(np.float32)

        # Find closest cluster
        cluster_label = self.kmeans_model.predict(daily_pattern)[0]
        benchmark_pattern = self.kmeans_model.cluster_centers_[cluster_label]

        # Return benchmark for specific hour
        return float(benchmark_pattern[hour_of_day])

    def get_benchmark_for_period(self, forecast_values: np.ndarray) -> np.ndarray:
        """
        Get benchmark values for a period matching forecast length.

        Args:
            forecast_values: Array of forecast values

        Returns:
            Array of benchmark values
        """
        if not self.loaded:
            self.load_model()

        n_hours = len(forecast_values)
        benchmark = np.zeros(n_hours)

        # Process in 24-hour windows
        for i in range(0, n_hours, 24):
            end_idx = min(i + 24, n_hours)
            window_size = end_idx - i

            if window_size == 24:
                # Full 24-hour window
                daily_pattern = forecast_values[i:end_idx].reshape(1, -1).astype(np.float32)
                cluster_label = self.kmeans_model.predict(daily_pattern)[0]
                benchmark[i:end_idx] = self.kmeans_model.cluster_centers_[cluster_label]
            else:
                # Partial window at the end
                if i > 0:
                    prev_pattern = forecast_values[i-24:i].reshape(1, -1).astype(np.float32)
                    cluster_label = self.kmeans_model.predict(prev_pattern)[0]
                    benchmark[i:end_idx] = self.kmeans_model.cluster_centers_[cluster_label][:window_size]
                else:
                    # Use first cluster as default
                    benchmark[i:end_idx] = self.kmeans_model.cluster_centers_[0][:window_size]

        return benchmark


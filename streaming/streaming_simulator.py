"""
Data Stream Simulator - Replays historical data as hourly stream
"""
import pandas as pd
import numpy as np
import time
from datetime import datetime, timedelta
from typing import Optional, Callable, Iterator
import sys
import os

# Add parent directory to path for imports
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import config
from mlad_anomaly_detection import load_and_preprocess_data
from streaming.streaming_config import *


class DataStreamSimulator:
    """
    Simulates real-time data arrival by replaying historical data.
    """

    def __init__(self, dataset_dir: str = None, start_index: int = 0, speed: float = 1.0):
        """
        Initialize the simulator.

        Args:
            dataset_dir: Path to dataset directory (default: config.DATASET_DIR)
            start_index: Starting index in dataset (default: 0)
            speed: Simulation speed multiplier (1.0 = real-time)
        """
        self.dataset_dir = dataset_dir or config.DATASET_DIR
        self.start_index = start_index
        self.speed = speed
        self.current_index = start_index
        self.data = None
        self.training_split_idx = None
        self.is_running = False
        self.start_time = None

    def load_data(self):
        """Load and prepare historical data."""
        print("Loading historical data...")
        df = load_and_preprocess_data(self.dataset_dir)

        # Use training portion (first 80%) for normal operations
        if USE_TRAINING_DATA:
            split_idx = int(len(df) * config.TRAIN_TEST_SPLIT_RATIO)
            self.data = df.iloc[:split_idx].copy()
            self.training_split_idx = split_idx
            print(f"Using training data: {len(self.data)} hours")
        else:
            self.data = df
            print(f"Using full dataset: {len(self.data)} hours")

        print(f"Data range: {self.data.index[0]} to {self.data.index[-1]}")
        return self

    def start(self, start_index: Optional[int] = None):
        """Start the simulation."""
        if self.data is None:
            self.load_data()

        if start_index is not None:
            self.current_index = start_index

        if self.current_index >= len(self.data):
            raise ValueError(f"Start index {self.current_index} exceeds data length {len(self.data)}")

        self.is_running = True
        self.start_time = datetime.now()
        print(f"Simulation started at index {self.current_index}")

    def stop(self):
        """Stop the simulation."""
        self.is_running = False
        print("Simulation stopped")

    def get_next_hour(self) -> Optional[pd.Series]:
        """
        Get the next hour of data.

        Returns:
            Series with load data and timestamp, or None if end reached
        """
        if not self.is_running:
            return None

        if self.current_index >= len(self.data):
            self.stop()
            return None

        # Get current hour data
        row = self.data.iloc[self.current_index]
        timestamp = self.data.index[self.current_index]

        # Create data point
        data_point = pd.Series({
            'timestamp': timestamp,
            'load': row['load'],
            'index': self.current_index
        })

        self.current_index += 1

        # Simulate real-time delay (if speed = 1.0, wait 1 hour of real time)
        if self.speed == 1.0:
            # For demo, we'll use faster speeds, but this shows the concept
            time.sleep(1.0 / self.speed)  # Adjust for demo

        return data_point

    def stream(self) -> Iterator[pd.Series]:
        """
        Generator that yields hourly data points.

        Yields:
            Series with load data and timestamp
        """
        self.start()
        while self.is_running:
            data_point = self.get_next_hour()
            if data_point is None:
                break
            yield data_point

    def get_current_time(self) -> Optional[datetime]:
        """Get current simulation time."""
        if not self.is_running or self.current_index == 0:
            return None
        if self.current_index > len(self.data):
            return self.data.index[-1]
        return self.data.index[self.current_index - 1]

    def get_progress(self) -> dict:
        """Get simulation progress."""
        if self.data is None:
            return {'progress': 0, 'current_index': 0, 'total': 0}

        return {
            'progress': self.current_index / len(self.data) * 100,
            'current_index': self.current_index,
            'total': len(self.data),
            'current_time': str(self.get_current_time()) if self.get_current_time() else None
        }


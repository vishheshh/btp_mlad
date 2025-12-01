"""
Data Store - In-memory storage with 30-day retention
"""
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from typing import List, Dict, Optional
from collections import deque
from streaming.streaming_config import RETENTION_HOURS


class DataStore:
    """
    In-memory data store with automatic cleanup for old data.
    Maintains 30-day retention policy.
    """

    def __init__(self):
        """Initialize the data store."""
        self.forecasts = deque(maxlen=RETENTION_HOURS)
        self.actuals = deque(maxlen=RETENTION_HOURS)
        self.benchmarks = deque(maxlen=RETENTION_HOURS)
        self.scaling_ratios = deque(maxlen=RETENTION_HOURS)
        self.timestamps = deque(maxlen=RETENTION_HOURS)
        self.detections = deque(maxlen=RETENTION_HOURS)
        self.alerts = deque(maxlen=100)  # Keep last 100 alerts

    def add_hourly_data(self, timestamp: pd.Timestamp, forecast: float,
                       actual: float, benchmark: float, scaling_ratio: float):
        """
        Add hourly data point.

        Args:
            timestamp: Timestamp
            forecast: Forecast value
            actual: Actual load value
            benchmark: Benchmark value
            scaling_ratio: Scaling ratio
        """
        self.timestamps.append(timestamp)
        self.forecasts.append(forecast)
        self.actuals.append(actual)
        self.benchmarks.append(benchmark)
        self.scaling_ratios.append(scaling_ratio)

    def add_detection(self, detection: Dict):
        """
        Add detection result.

        Args:
            detection: Detection dictionary
        """
        self.detections.append({
            **detection,
            'stored_at': datetime.now()
        })

    def add_alert(self, alert: Dict):
        """
        Add alert.

        Args:
            alert: Alert dictionary
        """
        self.alerts.append({
            **alert,
            'stored_at': datetime.now()
        })

    def get_recent_data(self, hours: int = 24) -> pd.DataFrame:
        """
        Get recent data as DataFrame.

        Args:
            hours: Number of recent hours to return

        Returns:
            DataFrame with recent data
        """
        if len(self.timestamps) == 0:
            return pd.DataFrame()

        recent_slice = slice(-hours, None) if hours > 0 else slice(None)

        return pd.DataFrame({
            'timestamp': list(self.timestamps)[recent_slice],
            'forecast': list(self.forecasts)[recent_slice],
            'actual': list(self.actuals)[recent_slice],
            'benchmark': list(self.benchmarks)[recent_slice],
            'scaling_ratio': list(self.scaling_ratios)[recent_slice]
        })

    def get_detections(self, limit: int = 100) -> List[Dict]:
        """
        Get recent detections.

        Args:
            limit: Maximum number of detections to return

        Returns:
            List of detection dictionaries
        """
        return list(self.detections)[-limit:]

    def get_alerts(self, limit: int = 100) -> List[Dict]:
        """
        Get recent alerts.

        Args:
            limit: Maximum number of alerts to return

        Returns:
            List of alert dictionaries
        """
        return list(self.alerts)[-limit:]
    
    def reset(self):
        """Reset all data - clear all stored data."""
        self.forecasts.clear()
        self.actuals.clear()
        self.benchmarks.clear()
        self.scaling_ratios.clear()
        self.timestamps.clear()
        self.detections.clear()
        self.alerts.clear()

    def get_statistics(self) -> Dict:
        """
        Get data store statistics.

        Returns:
            Dictionary with statistics
        """
        return {
            'total_hours': len(self.timestamps),
            'total_detections': len(self.detections),
            'total_alerts': len(self.alerts),
            'retention_hours': RETENTION_HOURS,
            'oldest_timestamp': str(self.timestamps[0]) if len(self.timestamps) > 0 else None,
            'newest_timestamp': str(self.timestamps[-1]) if len(self.timestamps) > 0 else None
        }


# Phase 1: Core Streaming Engine

## 📋 Overview

**Duration**: 1 week  
**Dependencies**: None  
**Objective**: Build the foundational streaming infrastructure that simulates real-time data arrival and processes it through the MLAD pipeline.

## ✅ Deliverables

By the end of this phase, you will have:

1. ✅ Streaming data simulator (replays historical data)
2. ✅ Rolling forecast engine (incremental LSTM predictions)
3. ✅ Benchmark manager (K-means benchmark retrieval)
4. ✅ Basic detection pipeline (scaling ratio calculation)
5. ✅ Performance monitor (basic metrics tracking)

---

## 🎯 Step-by-Step Implementation

### STEP 1: Create Directory Structure

**Action**: Create the streaming module structure

```bash
# In power_grid_protection/
mkdir -p streaming
cd streaming
```

**Files to create**:

- `__init__.py`
- `streaming_simulator.py`
- `rolling_forecast_engine.py`
- `benchmark_manager.py`
- `realtime_detector.py` (basic version)
- `performance_monitor.py`
- `streaming_config.py`

**Validation Checkpoint 1.1**: ✅

- [ ] Directory `streaming/` exists
- [ ] All files created (empty for now)

---

### STEP 2: Create Configuration File

**File**: `streaming/streaming_config.py`

**Code**:

```python
"""
Configuration for real-time streaming simulation
"""
import os

# Base directory
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

# Simulation settings
SIMULATION_SPEED = 1.0  # 1.0 = real-time, 10.0 = 10x speed, 100.0 = 100x speed

# Window sizes
FORECAST_WINDOW_HOURS = 168  # For lag features (1 week)
DETECTION_WINDOW_HOURS = 500  # For DP algorithm
STATISTICAL_BASELINE_HOURS = 50  # Minimum for statistical tests

# Update frequencies
BENCHMARK_UPDATE_DAYS = 7  # How often to retrain K-means (not used in Phase 1)
FORECAST_DRIFT_CHECK_HOURS = 24  # Check drift every N hours

# Performance targets
TARGET_LATENCY_SECONDS = 60  # < 1 minute
TARGET_THROUGHPUT_HOURS_PER_SEC = 100
MAX_MEMORY_GB = 4

# Data retention
RETENTION_DAYS = 30  # Keep last N days in memory
RETENTION_HOURS = RETENTION_DAYS * 24

# Data source
USE_TRAINING_DATA = True  # Use training set (first 80%) for normal operations
```

**Validation Checkpoint 1.2**: ✅

- [ ] File created
- [ ] Can import: `from streaming.streaming_config import *`
- [ ] No syntax errors

---

### STEP 3: Implement Streaming Simulator

**File**: `streaming/streaming_simulator.py`

**Purpose**: Replay historical data as hourly stream

**Code**:

```python
"""
Data Stream Simulator - Replays historical data as hourly stream
"""
import pandas as pd
import numpy as np
import time
from datetime import datetime, timedelta
from typing import Optional, Callable, Iterator
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
```

**Validation Checkpoint 1.3**: ✅

- [ ] File created
- [ ] Can import: `from streaming.streaming_simulator import DataStreamSimulator`
- [ ] Test basic functionality:
  ```python
  simulator = DataStreamSimulator(speed=100.0)  # Fast for testing
  simulator.load_data()
  simulator.start()
  for i, data_point in enumerate(simulator.stream()):
      print(f"Hour {i}: {data_point['timestamp']}, Load: {data_point['load']:.2f}")
      if i >= 10:  # Test first 10 hours
          break
  ```
- [ ] No errors, data loads correctly

---

### STEP 4: Implement Rolling Forecast Engine

**File**: `streaming/rolling_forecast_engine.py`

**Purpose**: Generate LSTM forecasts incrementally

**Code**:

```python
"""
Rolling Forecast Engine - Incremental LSTM predictions
"""
import numpy as np
import pandas as pd
from typing import Optional, List
from tensorflow.keras.models import load_model
import pickle
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

        # Create a temporary DataFrame with current data point
        # We need to maintain the full DataFrame structure for create_features
        # For now, we'll use a simpler approach: manually create features

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
```

**Validation Checkpoint 1.4**: ✅

- [ ] File created
- [ ] Can import: `from streaming.rolling_forecast_engine import RollingForecastEngine`
- [ ] Test initialization:

  ```python
  # Load some initial data
  from mlad_anomaly_detection import load_and_preprocess_data
  df = load_and_preprocess_data(config.DATASET_DIR)
  split_idx = int(len(df) * config.TRAIN_TEST_SPLIT_RATIO)
  initial_data = df.iloc[:split_idx+200].copy()  # Get 200 hours for initialization

  engine = RollingForecastEngine()
  engine.initialize(initial_data.iloc[:200])

  # Test prediction
  test_point = initial_data.iloc[200]
  forecast = engine.predict_next_hour(pd.Series({
      'timestamp': test_point.name,
      'load': test_point['load']
  }))
  print(f"Forecast: {forecast:.2f}, Actual: {test_point['load']:.2f}")
  ```

- [ ] No errors, forecast generated

---

### STEP 5: Implement Benchmark Manager

**File**: `streaming/benchmark_manager.py`

**Purpose**: Provide K-means benchmark values

**Code**:

```python
"""
Benchmark Manager - K-means benchmark retrieval
"""
import numpy as np
import pandas as pd
import pickle
from typing import Optional
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
```

**Validation Checkpoint 1.5**: ✅

- [ ] File created
- [ ] Can import: `from streaming.benchmark_manager import BenchmarkManager`
- [ ] Test benchmark retrieval:

  ```python
  manager = BenchmarkManager()
  manager.load_model()

  # Create sample forecast (24 hours)
  sample_forecast = np.random.rand(24) * 10000 + 10000

  benchmark = manager.get_benchmark_for_period(sample_forecast)
  print(f"Forecast shape: {sample_forecast.shape}")
  print(f"Benchmark shape: {benchmark.shape}")
  print(f"Sample values - Forecast: {sample_forecast[0]:.2f}, Benchmark: {benchmark[0]:.2f}")
  ```

- [ ] No errors, benchmark values retrieved

---

### STEP 6: Implement Basic Detection Pipeline

**File**: `streaming/realtime_detector.py` (Basic Version)

**Purpose**: Calculate scaling ratios (full detection in Phase 2)

**Code**:

```python
"""
Real-Time Detector - Basic version for Phase 1
Full hybrid detection will be added in Phase 2
"""
import numpy as np
import pandas as pd
from typing import Optional, List
from streaming.streaming_config import *


class RealTimeDetector:
    """
    Basic real-time detector for Phase 1.
    Calculates scaling ratios and basic anomaly indicators.
    Full hybrid detection will be implemented in Phase 2.
    """

    def __init__(self):
        """Initialize the detector."""
        self.scaling_history = []  # Store scaling ratios
        self.detection_window = []  # Sliding window for detection
        self.max_window_size = DETECTION_WINDOW_HOURS

    def process_hourly_data(self, forecast: float, benchmark: float,
                           actual_load: float, timestamp: pd.Timestamp) -> dict:
        """
        Process one hour of data.

        Args:
            forecast: LSTM forecast value
            benchmark: K-means benchmark value
            actual_load: Actual load value
            timestamp: Timestamp

        Returns:
            Dictionary with scaling ratio and basic indicators
        """
        # Calculate scaling ratio (forecast/benchmark)
        scaling_ratio = forecast / (benchmark + 1e-6)

        # Calculate deviation
        deviation = abs(scaling_ratio - 1.0)

        # Store in history
        data_point = {
            'timestamp': timestamp,
            'forecast': forecast,
            'benchmark': benchmark,
            'actual_load': actual_load,
            'scaling_ratio': scaling_ratio,
            'deviation': deviation
        }

        self.scaling_history.append(data_point)
        self.detection_window.append(data_point)

        # Maintain window size
        if len(self.detection_window) > self.max_window_size:
            self.detection_window.pop(0)

        # Clean old history (keep last RETENTION_HOURS)
        if len(self.scaling_history) > RETENTION_HOURS:
            self.scaling_history.pop(0)

        # Basic anomaly indicator (simple threshold check)
        is_anomalous = deviation > 0.09  # 9% threshold from config

        return {
            'scaling_ratio': scaling_ratio,
            'deviation': deviation,
            'is_anomalous': is_anomalous,
            'timestamp': timestamp
        }

    def get_recent_scaling_data(self, hours: int = 24) -> np.ndarray:
        """
        Get recent scaling ratios for analysis.

        Args:
            hours: Number of recent hours to return

        Returns:
            Array of scaling ratios
        """
        if len(self.scaling_history) == 0:
            return np.array([])

        recent = self.scaling_history[-hours:]
        return np.array([d['scaling_ratio'] for d in recent])

    def get_window_size(self) -> int:
        """Get current window size."""
        return len(self.detection_window)
```

**Validation Checkpoint 1.6**: ✅

- [ ] File created
- [ ] Can import: `from streaming.realtime_detector import RealTimeDetector`
- [ ] Test basic processing:

  ```python
  detector = RealTimeDetector()

  result = detector.process_hourly_data(
      forecast=15000.0,
      benchmark=14000.0,
      actual_load=15000.0,
      timestamp=pd.Timestamp('2024-01-15 14:00:00')
  )

  print(f"Scaling ratio: {result['scaling_ratio']:.4f}")
  print(f"Deviation: {result['deviation']:.4f}")
  print(f"Is anomalous: {result['is_anomalous']}")
  ```

- [ ] No errors, scaling ratio calculated correctly

---

### STEP 7: Implement Performance Monitor

**File**: `streaming/performance_monitor.py`

**Purpose**: Track system performance metrics

**Code**:

```python
"""
Performance Monitor - Track system performance metrics
"""
import time
import psutil
import os
from datetime import datetime
from typing import Dict, List
from collections import deque


class PerformanceMonitor:
    """
    Monitors system performance metrics.
    """

    def __init__(self):
        """Initialize the performance monitor."""
        self.process = psutil.Process(os.getpid())
        self.start_time = None
        self.hours_processed = 0
        self.latency_history = deque(maxlen=1000)  # Last 1000 measurements
        self.memory_history = deque(maxlen=100)  # Last 100 measurements

    def start(self):
        """Start monitoring."""
        self.start_time = time.time()
        self.hours_processed = 0
        print("Performance monitoring started")

    def record_processing(self, processing_time: float):
        """
        Record processing time for one hour.

        Args:
            processing_time: Time taken to process one hour (seconds)
        """
        self.latency_history.append(processing_time)
        self.hours_processed += 1

        # Record memory usage periodically
        if self.hours_processed % 10 == 0:
            memory_mb = self.process.memory_info().rss / 1024 / 1024
            self.memory_history.append({
                'timestamp': datetime.now(),
                'memory_mb': memory_mb,
                'hours_processed': self.hours_processed
            })

    def get_latency_stats(self) -> Dict:
        """Get latency statistics."""
        if len(self.latency_history) == 0:
            return {
                'mean': None,
                'p50': None,
                'p95': None,
                'p99': None,
                'max': None,
                'count': 0
            }

        latencies = list(self.latency_history)
        latencies_sorted = sorted(latencies)

        return {
            'mean': sum(latencies) / len(latencies),
            'p50': latencies_sorted[len(latencies_sorted) // 2],
            'p95': latencies_sorted[int(len(latencies_sorted) * 0.95)],
            'p99': latencies_sorted[int(len(latencies_sorted) * 0.99)],
            'max': max(latencies),
            'count': len(latencies)
        }

    def get_throughput(self) -> float:
        """Get throughput (hours per second)."""
        if self.start_time is None:
            return 0.0

        elapsed = time.time() - self.start_time
        if elapsed == 0:
            return 0.0

        return self.hours_processed / elapsed

    def get_memory_usage(self) -> Dict:
        """Get current memory usage."""
        memory_info = self.process.memory_info()
        memory_mb = memory_info.rss / 1024 / 1024
        memory_gb = memory_mb / 1024

        return {
            'memory_mb': memory_mb,
            'memory_gb': memory_gb,
            'memory_percent': self.process.memory_percent(),
            'within_limit': memory_gb < MAX_MEMORY_GB
        }

    def get_all_metrics(self) -> Dict:
        """Get all performance metrics."""
        return {
            'latency': self.get_latency_stats(),
            'throughput_hours_per_sec': self.get_throughput(),
            'memory': self.get_memory_usage(),
            'hours_processed': self.hours_processed,
            'uptime_seconds': time.time() - self.start_time if self.start_time else 0,
            'targets': {
                'target_latency_seconds': TARGET_LATENCY_SECONDS,
                'target_throughput': TARGET_THROUGHPUT_HOURS_PER_SEC,
                'max_memory_gb': MAX_MEMORY_GB
            }
        }
```

**Note**: Requires `psutil` package. Add to requirements.txt:

```
psutil>=5.9.0
```

**Validation Checkpoint 1.7**: ✅

- [ ] File created
- [ ] Install psutil: `pip install psutil`
- [ ] Can import: `from streaming.performance_monitor import PerformanceMonitor`
- [ ] Test monitoring:

  ```python
  monitor = PerformanceMonitor()
  monitor.start()

  # Simulate processing
  import time
  for i in range(5):
      start = time.time()
      time.sleep(0.1)  # Simulate processing
      monitor.record_processing(time.time() - start)

  metrics = monitor.get_all_metrics()
  print(f"Throughput: {metrics['throughput_hours_per_sec']:.2f} hours/sec")
  print(f"Memory: {metrics['memory']['memory_mb']:.2f} MB")
  ```

- [ ] No errors, metrics tracked

---

### STEP 8: Integration Test

**File**: `streaming/test_integration.py`

**Purpose**: Test all components working together

**Code**:

```python
"""
Integration test for Phase 1 - Test all components together
"""
import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import pandas as pd
import time
from streaming.streaming_simulator import DataStreamSimulator
from streaming.rolling_forecast_engine import RollingForecastEngine
from streaming.benchmark_manager import BenchmarkManager
from streaming.realtime_detector import RealTimeDetector
from streaming.performance_monitor import PerformanceMonitor
import config
from mlad_anomaly_detection import load_and_preprocess_data


def test_integration():
    """Test all components integrated."""
    print("="*60)
    print("PHASE 1 INTEGRATION TEST")
    print("="*60)

    # Initialize components
    print("\n1. Initializing components...")
    simulator = DataStreamSimulator(speed=100.0)  # Fast for testing
    forecast_engine = RollingForecastEngine()
    benchmark_manager = BenchmarkManager()
    detector = RealTimeDetector()
    monitor = PerformanceMonitor()

    # Load data
    print("\n2. Loading data...")
    simulator.load_data()

    # Initialize forecast engine with initial data
    print("\n3. Initializing forecast engine...")
    df = load_and_preprocess_data(config.DATASET_DIR)
    split_idx = int(len(df) * config.TRAIN_TEST_SPLIT_RATIO)
    initial_data = df.iloc[:split_idx+200].copy()
    forecast_engine.initialize(initial_data.iloc[:200])

    # Start monitoring
    monitor.start()

    # Process first 24 hours
    print("\n4. Processing 24 hours of data...")
    simulator.start()

    results = []
    for i in range(24):
        # Get next hour
        data_point = simulator.get_next_hour()
        if data_point is None:
            break

        # Start timing
        start_time = time.time()

        # Generate forecast
        forecast = forecast_engine.predict_next_hour(data_point)

        # Get benchmark (need recent forecasts for pattern matching)
        recent_forecasts = np.array(forecast_engine.forecast_history[-24:])
        if len(recent_forecasts) < 24:
            # Pad if needed
            recent_forecasts = np.pad(recent_forecasts, (24 - len(recent_forecasts), 0),
                                    mode='edge')
        benchmark = benchmark_manager.get_benchmark_for_hour(
            recent_forecasts,
            data_point['timestamp'].hour
        )

        # Process detection
        detection_result = detector.process_hourly_data(
            forecast=forecast,
            benchmark=benchmark,
            actual_load=data_point['load'],
            timestamp=data_point['timestamp']
        )

        # Record processing time
        processing_time = time.time() - start_time
        monitor.record_processing(processing_time)

        # Store result
        results.append({
            'timestamp': data_point['timestamp'],
            'actual': data_point['load'],
            'forecast': forecast,
            'benchmark': benchmark,
            'scaling_ratio': detection_result['scaling_ratio'],
            'deviation': detection_result['deviation'],
            'is_anomalous': detection_result['is_anomalous'],
            'processing_time': processing_time
        })

        if (i + 1) % 5 == 0:
            print(f"  Processed {i + 1}/24 hours...")

    # Print results
    print("\n5. Results Summary:")
    print("-" * 60)
    df_results = pd.DataFrame(results)
    print(f"Total hours processed: {len(df_results)}")
    print(f"Mean forecast: {df_results['forecast'].mean():.2f} MWh")
    print(f"Mean actual: {df_results['actual'].mean():.2f} MWh")
    print(f"Mean scaling ratio: {df_results['scaling_ratio'].mean():.4f}")
    print(f"Anomalous hours: {df_results['is_anomalous'].sum()}")

    # Performance metrics
    print("\n6. Performance Metrics:")
    print("-" * 60)
    metrics = monitor.get_all_metrics()
    print(f"Throughput: {metrics['throughput_hours_per_sec']:.2f} hours/sec")
    print(f"Mean latency: {metrics['latency']['mean']:.4f} seconds")
    print(f"P95 latency: {metrics['latency']['p95']:.4f} seconds")
    print(f"Memory usage: {metrics['memory']['memory_mb']:.2f} MB")
    print(f"Memory within limit: {metrics['memory']['within_limit']}")

    # Forecast drift
    print("\n7. Forecast Drift:")
    print("-" * 60)
    drift = forecast_engine.calculate_forecast_drift(window_hours=24)
    if drift['sufficient_data']:
        print(f"MAE: {drift['mae']:.2f} MWh")
        print(f"MAPE: {drift['mape']:.2f}%")
        if drift['drift_percentage'] is not None:
            print(f"Drift: {drift['drift_percentage']:.2f}%")
    else:
        print("Insufficient data for drift calculation")

    print("\n" + "="*60)
    print("INTEGRATION TEST COMPLETE")
    print("="*60)

    return results, metrics


if __name__ == "__main__":
    results, metrics = test_integration()
```

**Validation Checkpoint 1.8**: ✅

- [ ] File created
- [ ] Run test: `python streaming/test_integration.py`
- [ ] All components work together
- [ ] No errors
- [ ] Performance metrics reasonable
- [ ] Data flows correctly

---

## 🧪 End-of-Phase Testing

### Test 1: Component Isolation

Test each component independently:

- [ ] Streaming simulator loads and streams data
- [ ] Forecast engine generates predictions
- [ ] Benchmark manager retrieves benchmarks
- [ ] Detector calculates scaling ratios
- [ ] Monitor tracks metrics

### Test 2: Integration

Test components working together:

- [ ] End-to-end flow works
- [ ] Data flows correctly
- [ ] No memory leaks (run for 100+ hours)
- [ ] Performance acceptable

### Test 3: Edge Cases

- [ ] Handles end of data gracefully
- [ ] Handles missing data
- [ ] Handles very small/large values
- [ ] Handles rapid speed changes

---

## ✅ Success Criteria

Phase 1 is complete when:

1. ✅ All 5 components implemented and working
2. ✅ Integration test passes
3. ✅ Can process 24+ hours without errors
4. ✅ Performance metrics within targets:
   - Latency < 1 second per hour
   - Memory < 1GB for 24 hours
   - Throughput > 10 hours/sec
5. ✅ No crashes or memory leaks

---

## 🐛 Common Issues & Solutions

### Issue 1: Model files not found

**Error**: `FileNotFoundError: Model not found`
**Solution**:

- Ensure models are trained: Run `python mlad_anomaly_detection.py` first
- Check paths in `config.py`

### Issue 2: Feature mismatch

**Error**: `ValueError: Feature shape mismatch`
**Solution**:

- Ensure feature columns match training time
- Check `create_features()` function

### Issue 3: Memory issues

**Error**: Memory usage too high
**Solution**:

- Reduce `RETENTION_HOURS` in config
- Clear history more frequently
- Use data types (float32 instead of float64)

### Issue 4: Slow performance

**Error**: Processing too slow
**Solution**:

- Increase simulation speed for testing
- Optimize feature creation
- Use numpy operations instead of loops

---

## 📝 Next Steps

Once Phase 1 is complete and validated:

1. ✅ Update `VALIDATION_CHECKLIST.md` - Mark Phase 1 complete
2. ✅ Tag me with: `@PHASE2_DETECTION_LOGIC.md` to proceed
3. ✅ Save any notes/issues in the checklist

---

## 📚 Reference

- Main plan: `REAL_TIME_SIMULATION_PLAN.md`
- FAQ: `REAL_TIME_FAQ_AND_CLARIFICATIONS.md`
- Config: `config.py`
- Main detection: `mlad_anomaly_detection.py`

---

**Ready to start?** Begin with STEP 1! 🚀

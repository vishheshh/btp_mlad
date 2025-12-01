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


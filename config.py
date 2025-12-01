"""
Configuration file for MLAD Anomaly Detection System
All paths and parameters are defined here to avoid hardcoding.
"""

import os

# Directory paths
ROOT_DIR = os.path.dirname(os.path.abspath(__file__))
DATASET_DIR = os.path.join(ROOT_DIR, 'dataset')
MODELS_DIR = os.path.join(ROOT_DIR, 'models')

# Model file paths
FORECASTER_MODEL_PATH = os.path.join(MODELS_DIR, 'load_forecaster.h5')
SCALER_PATH = os.path.join(MODELS_DIR, 'scaler.pkl')
KMEANS_MODEL_PATH = os.path.join(MODELS_DIR, 'kmeans_model.pkl')

# Data parameters
CSV_SKIP_ROWS = 7  # Number of metadata/header rows to skip
TRAIN_TEST_SPLIT_RATIO = 0.8  # 80% train, 20% test

# Feature engineering parameters
LAG_FEATURES = [1, 24, 168]  # 1 hour, 1 day, 1 week

# Neural network parameters
LSTM_UNITS = 64
DENSE_UNITS = 32
EPOCHS = 50
BATCH_SIZE = 32
LEARNING_RATE = 0.001

# K-means parameters
N_CLUSTERS = 100
HOURS_PER_DAY = 24

# Anomaly detection parameters - TWO-TIER SYSTEM
# Normal Mode: Balanced detection (avoid false alarms)
MAGNITUDE_THRESHOLD = 0.09  # Normal detection: 9% deviation threshold (lowered from 10% - still above 8% floor)
MIN_ANOMALY_DURATION = 2    # Default minimum duration (for medium/strong attacks)

# Scaling ratio configuration
SCALING_REFERENCE = 'forecast'  # Options: 'forecast', 'benchmark'
SCALING_EPSILON = 1e-6          # Small constant to avoid division by zero

# Instant alert configuration (Phase 2)
INSTANT_ALERT_THRESHOLD = 0.35           # deviation level for instant trigger (~35%)
INSTANT_ALERT_CONFIRMATION_HOURS = 2     # consecutive hours required before alerting
EMERGENCY_COOLDOWN_HOURS = 6             # minimum hours between emergency alerts

# Deviation buffer to reduce noise-triggered DP scoring
DEVIATION_BUFFER = 0.02  # require ~2% additional deviation beyond threshold for scoring

# Magnitude-aware duration requirements (Phase 2 optimization)
MIN_DURATION_WEAK = 4       # Weak attacks (<25%): require moderate duration
MIN_DURATION_MEDIUM = 3     # Medium attacks (25-40%): standard duration
MIN_DURATION_STRONG = 1     # Strong attacks (>40%): can detect very quickly

# Emergency Mode: INSTANT detection for catastrophic spikes
EMERGENCY_THRESHOLD = 0.50  # Emergency: 50%+ deviation = INSTANT ALERT
EMERGENCY_MIN_DURATION = 1  # Emergency: Alert after just 1 hour

# Scoring parameters
LAMBDA_SCORE = 2.0  # Super-additive score function parameter (improved for better score accumulation)
MIN_ANOMALY_SCORE = 0.30  # Minimum score threshold to accept detection (default for medium durations)

# Duration-aware score thresholds (Phase 2 optimization)
MIN_ANOMALY_SCORE_SHORT = 0.12   # For durations 1-5 hours
MIN_ANOMALY_SCORE_MEDIUM = 0.28  # For durations 6-11 hours
MIN_ANOMALY_SCORE_LONG = 0.22    # For durations 12+ hours (longer attacks accumulate more evidence)

# Segmentation parameters (Phase 1 improvement)
SEGMENT_GAP_HOURS = 3  # Consecutive normal hours needed to break segments
MIN_SEGMENT_SCORE = 0.08  # Minimum score for a segment to be valid (fine-tuned from 0.10)
MIN_SEGMENT_DURATION_FOR_SPLIT = 10  # Only segment intervals longer than this

# PHASE 4: Statistical Hypothesis Testing Parameters
STATISTICAL_ALPHA = 0.01  # Significance level for statistical tests (1% false positive rate)
COHENS_D_THRESHOLD = 0.5  # Minimum effect size (Cohen's d) for practical significance
USE_HYBRID_DETECTION = True  # Enable hybrid detection (DP + Statistical)

# PHASE 4: Window Size Optimization (to be tuned)
OPTIMAL_WINDOW_SIZE = 500  # Will be updated after window size experiments 
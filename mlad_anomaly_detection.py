"""
Machine Learning-Based Anomaly Detection (MLAD) for Power Grid Load Data

This script implements a simplified MLAD methodology to detect the presence 
and timing of anomalies in power grid load data without classifying attack types.

Main Components:
1. Data loading and preprocessing
2. Neural network-based load forecasting (LSTM)
3. K-means clustering for benchmark profiles
4. Dynamic programming for anomaly timing detection
"""

import os
import glob
import numpy as np
import pandas as pd
import pickle
from sklearn.preprocessing import MinMaxScaler
from sklearn.cluster import KMeans
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras.models import Sequential, load_model
from tensorflow.keras.layers import LSTM, Dense, Dropout
from tensorflow.keras.optimizers import Adam
import scipy.stats
import warnings
warnings.filterwarnings('ignore')

# Import configuration
import config


def load_and_preprocess_data(folder_path):
    """
    Load and preprocess all CSV files from the dataset directory.
    
    This function:
    - Scans the specified directory for all CSV files
    - Skips the first 7 metadata/header rows in each file
    - Combines all files into a single DataFrame
    - Cleans and formats the data with proper datetime indexing
    
    Args:
        folder_path (str): Path to the directory containing CSV files
        
    Returns:
        pd.DataFrame: Preprocessed DataFrame with datetime index and load data
    """
    print(f"Loading data from: {folder_path}")
    
    # Get all CSV files in the directory
    csv_files = glob.glob(os.path.join(folder_path, '*.csv'))
    print(f"Found {len(csv_files)} CSV files")
    
    if len(csv_files) == 0:
        raise FileNotFoundError(f"No CSV files found in {folder_path}")
    
    # List to store individual DataFrames
    dataframes = []
    
    # Load each CSV file
    for file in csv_files:
        try:
            # Skip first 6 rows (metadata and headers)
            # Data rows start with "D" prefix which becomes the first column
            df = pd.read_csv(file, skiprows=6, header=None)
            # Filter to keep only data rows (those that start with "D")
            df = df[df.iloc[:, 0] == 'D']
            # Drop the first column (the "D" indicator)
            df = df.iloc[:, 1:]
            # Reset index after filtering
            df = df.reset_index(drop=True)
            dataframes.append(df)
        except Exception as e:
            print(f"Error loading {file}: {e}")
            continue
    
    # Combine all DataFrames
    master_df = pd.concat(dataframes, ignore_index=True)
    print(f"Combined data shape: {master_df.shape}")
    
    # Rename columns for consistency
    master_df.columns = ['date', 'hour', 'load']
    
    # Convert hour and load columns to numeric
    master_df['hour'] = pd.to_numeric(master_df['hour'], errors='coerce')
    master_df['load'] = pd.to_numeric(master_df['load'], errors='coerce')
    
    # Create datetime column by combining date and hour
    # Hour is typically 1-24, so we need to subtract 1 to get 0-23 for proper datetime
    master_df['datetime'] = pd.to_datetime(master_df['date']) + pd.to_timedelta(master_df['hour'] - 1, unit='h')
    
    # Remove any rows with missing load values
    master_df = master_df.dropna(subset=['load'])
    
    # Set datetime as index
    master_df.set_index('datetime', inplace=True)
    
    # Sort by datetime to ensure chronological order
    master_df.sort_index(inplace=True)
    
    # Remove duplicate timestamps (keep first occurrence)
    master_df = master_df[~master_df.index.duplicated(keep='first')]
    
    print(f"Final preprocessed data shape: {master_df.shape}")
    print(f"Date range: {master_df.index.min()} to {master_df.index.max()}")
    
    return master_df


def create_features(df, target_col='load'):
    """
    Create time-series features for the neural network.
    
    This function adds:
    - Lag features (1 hour ago, 24 hours ago, 1 week ago)
    - Time-based features (hour, day of week, day of year, month)
    
    Args:
        df (pd.DataFrame): Input DataFrame with datetime index
        target_col (str): Name of the target column
        
    Returns:
        pd.DataFrame: DataFrame with additional features
    """
    df_features = df.copy()
    
    # Lag features
    for lag in config.LAG_FEATURES:
        df_features[f'load_lag_{lag}'] = df_features[target_col].shift(lag)
    
    # Time-based features
    df_features['hour'] = df_features.index.hour
    df_features['day_of_week'] = df_features.index.dayofweek
    df_features['day_of_year'] = df_features.index.dayofyear
    df_features['month'] = df_features.index.month
    df_features['is_weekend'] = (df_features['day_of_week'] >= 5).astype(int)
    
    # Cyclical encoding for hour (to capture circularity)
    df_features['hour_sin'] = np.sin(2 * np.pi * df_features['hour'] / 24)
    df_features['hour_cos'] = np.cos(2 * np.pi * df_features['hour'] / 24)
    
    # Cyclical encoding for month
    df_features['month_sin'] = np.sin(2 * np.pi * df_features['month'] / 12)
    df_features['month_cos'] = np.cos(2 * np.pi * df_features['month'] / 12)
    
    # Drop rows with NaN values (from lag features)
    df_features = df_features.dropna()
    
    return df_features


def build_lstm_model(input_shape):
    """
    Build LSTM neural network model for load forecasting.
    
    Args:
        input_shape (tuple): Shape of input features (timesteps, features)
        
    Returns:
        keras.Model: Compiled LSTM model
    """
    model = Sequential([
        LSTM(config.LSTM_UNITS, activation='relu', return_sequences=True, 
             input_shape=input_shape),
        Dropout(0.2),
        LSTM(config.DENSE_UNITS, activation='relu'),
        Dropout(0.2),
        Dense(1)
    ])
    
    optimizer = Adam(learning_rate=config.LEARNING_RATE)
    model.compile(optimizer=optimizer, loss='mean_squared_error', 
                  metrics=['mean_absolute_error'])
    
    return model


def train_forecaster(df):
    """
    Train the LSTM neural network for load forecasting.
    
    Args:
        df (pd.DataFrame): Preprocessed DataFrame with load data
        
    Returns:
        tuple: (trained_model, scaler, feature_columns)
    """
    print("\n" + "="*60)
    print("STEP 2: Training Neural Network Forecaster")
    print("="*60)
    
    # Create features
    df_features = create_features(df)
    
    # Define feature columns (exclude target and original features)
    feature_cols = [col for col in df_features.columns 
                   if col not in ['load', 'date', 'hour']]
    
    X = df_features[feature_cols].values
    y = df_features['load'].values
    
    # Split data (80% train, 20% test)
    split_idx = int(len(X) * config.TRAIN_TEST_SPLIT_RATIO)
    X_train, X_test = X[:split_idx], X[split_idx:]
    y_train, y_test = y[:split_idx], y[split_idx:]
    
    print(f"Training samples: {len(X_train)}, Testing samples: {len(X_test)}")
    
    # Scale the features
    scaler = MinMaxScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)
    
    # Reshape for LSTM [samples, timesteps, features]
    X_train_reshaped = X_train_scaled.reshape((X_train_scaled.shape[0], 1, X_train_scaled.shape[1]))
    X_test_reshaped = X_test_scaled.reshape((X_test_scaled.shape[0], 1, X_test_scaled.shape[1]))
    
    # Build model
    model = build_lstm_model(input_shape=(1, X_train_scaled.shape[1]))
    
    print("\nModel Architecture:")
    model.summary()
    
    # Train model
    print(f"\nTraining for {config.EPOCHS} epochs...")
    history = model.fit(
        X_train_reshaped, y_train,
        epochs=config.EPOCHS,
        batch_size=config.BATCH_SIZE,
        validation_data=(X_test_reshaped, y_test),
        verbose=1
    )
    
    # Evaluate model
    train_loss, train_mae = model.evaluate(X_train_reshaped, y_train, verbose=0)
    test_loss, test_mae = model.evaluate(X_test_reshaped, y_test, verbose=0)
    
    print(f"\nTraining MAE: {train_mae:.2f} MWh")
    print(f"Testing MAE: {test_mae:.2f} MWh")
    
    # Create models directory if it doesn't exist
    os.makedirs(config.MODELS_DIR, exist_ok=True)
    
    # Save model and scaler
    model.save(config.FORECASTER_MODEL_PATH)
    with open(config.SCALER_PATH, 'wb') as f:
        pickle.dump(scaler, f)
    
    print(f"\nModel saved to: {config.FORECASTER_MODEL_PATH}")
    print(f"Scaler saved to: {config.SCALER_PATH}")
    
    return model, scaler, feature_cols


def train_kmeans_benchmark(df):
    """
    Train K-means clustering model on normal data to establish benchmark profiles.
    
    This function reshapes the data so each row represents a 24-hour day pattern,
    then clusters these daily patterns to find typical load profiles.
    
    Args:
        df (pd.DataFrame): Preprocessed DataFrame with load data
        
    Returns:
        KMeans: Trained K-means model
    """
    print("\n" + "="*60)
    print("STEP 3: Training K-Means Benchmark Model")
    print("="*60)
    
    # Use training portion (assuming first 80% is normal data)
    split_idx = int(len(df) * config.TRAIN_TEST_SPLIT_RATIO)
    train_data = df.iloc[:split_idx].copy()
    
    # Reshape data into 24-hour daily patterns
    # Each row will represent one day with 24 load values
    load_values = train_data['load'].values
    
    # Calculate number of complete days
    n_complete_days = len(load_values) // config.HOURS_PER_DAY
    
    # Truncate to complete days only
    load_values_truncated = load_values[:n_complete_days * config.HOURS_PER_DAY]
    
    # Reshape into (n_days, 24) array and convert to float32
    daily_patterns = load_values_truncated.reshape(-1, config.HOURS_PER_DAY).astype(np.float32)
    
    print(f"Number of complete days: {n_complete_days}")
    print(f"Daily patterns shape: {daily_patterns.shape}")
    
    # Train K-means model
    print(f"\nTraining K-means with {config.N_CLUSTERS} clusters...")
    kmeans = KMeans(n_clusters=config.N_CLUSTERS, random_state=42, n_init=10)
    kmeans.fit(daily_patterns)
    
    print(f"K-means training complete")
    print(f"Inertia: {kmeans.inertia_:.2f}")
    
    # Save model
    with open(config.KMEANS_MODEL_PATH, 'wb') as f:
        pickle.dump(kmeans, f)
    
    print(f"K-means model saved to: {config.KMEANS_MODEL_PATH}")
    
    return kmeans


def find_consecutive_true(boolean_array):
    """
    Find all consecutive True regions in boolean array.
    
    PHASE 4 ADDITION: Helper function for statistical detection.
    
    Args:
        boolean_array (np.array): Boolean array to search
        
    Returns:
        list: List of (start_idx, end_idx) tuples for consecutive True regions
    """
    intervals = []
    in_interval = False
    start = None
    
    for i in range(len(boolean_array)):
        if boolean_array[i] and not in_interval:
            start = i
            in_interval = True
        elif not boolean_array[i] and in_interval:
            intervals.append((start, i-1))
            in_interval = False
    
    if in_interval:
        intervals.append((start, len(boolean_array)-1))
    
    return intervals


def statistical_anomaly_detection(scaling_data, alpha=None):
    """
    PHASE 4: Statistical hypothesis testing for anomaly detection.
    
    This function uses statistical rigor to distinguish weak attacks from natural
    fluctuations. Instead of just finding high-magnitude intervals, it tests
    whether intervals are statistically significantly different from normal baseline.
    
    Key Advantages:
    - Weak attacks that are statistically significant get detected even if magnitude is low
    - Natural spikes that aren't statistically significant get filtered out
    - Bonferroni correction prevents false positives from multiple testing
    - Cohen's d ensures practical significance (not just statistical)
    
    Approach:
    1. Model normal distribution from baseline hours (below threshold)
    2. Find all candidate intervals (consecutive hours above threshold)
    3. Test each interval with Wilcoxon signed-rank test (non-parametric, robust)
    4. Apply Bonferroni correction for multiple testing
    5. Filter by effect size (Cohen's d) for practical significance
    
    Args:
        scaling_data (np.array): Array of forecast/benchmark ratios
        alpha (float): Significance level (default from config)
        
    Returns:
        list: List of (start, end, score, p_value, cohens_d) tuples for significant intervals
    """
    if alpha is None:
        alpha = config.STATISTICAL_ALPHA
    
    deviation = np.abs(scaling_data - 1.0)
    n = len(deviation)
    
    # Step 1: Model normal distribution
    # Use hours below threshold as "normal" baseline
    baseline_mask = deviation <= config.MAGNITUDE_THRESHOLD
    normal_samples = deviation[baseline_mask]
    
    if len(normal_samples) < 50:
        # Not enough data for reliable statistics - return empty
        # Will fall back to DP detection in hybrid mode
        return []
    
    # Fit normal distribution
    normal_mean = np.mean(normal_samples)
    normal_std = np.std(normal_samples)
    
    # Step 2: Find candidate intervals
    # Any consecutive hours above threshold
    above_threshold = deviation > config.MAGNITUDE_THRESHOLD
    intervals = find_consecutive_true(above_threshold)
    
    if len(intervals) == 0:
        return []
    
    # Step 3: Test each interval
    results = []
    bonferroni_alpha = alpha / len(intervals)  # Multiple testing correction
    
    for start, end in intervals:
        duration = end - start + 1
        interval_data = deviation[start:end+1]
        
        # Skip very short intervals (< 1 hour)
        if duration < 1:
            continue
        
        # Statistical test: Is this interval significantly different from normal?
        # Use Wilcoxon signed-rank test (non-parametric, more robust than t-test)
        try:
            if duration == 1:
                # Single observation - use Z-score test
                z_score = (interval_data[0] - normal_mean) / (normal_std + 1e-6)
                # One-sided p-value
                p_value = 1.0 - scipy.stats.norm.cdf(z_score)
            else:
                # Multiple observations - use Wilcoxon
                differences = interval_data - normal_mean
                
                # Check if all differences are zero (edge case)
                if np.all(differences == 0):
                    p_value = 1.0
                else:
                    stat, p_value = scipy.stats.wilcoxon(
                        differences,
                        alternative='greater'
                    )
        except Exception as e:
            # If test fails (e.g., all zeros), skip this interval
            continue
        
        # Calculate effect size (Cohen's d)
        cohens_d = (np.mean(interval_data) - normal_mean) / (normal_std + 1e-6)
        
        # Accept if statistically significant AND practically significant
        if p_value < bonferroni_alpha and cohens_d > config.COHENS_D_THRESHOLD:
            # Check minimum duration requirements based on magnitude
            max_dev = np.max(interval_data)
            if max_dev < 0.25:
                min_dur = 1  # Weak: require at least 1 hour
            elif max_dev < 0.40:
                min_dur = 1  # Medium: 1 hour
            else:
                min_dur = 1  # Strong: 1 hour
            
            if duration >= min_dur:
                # Score based on statistical significance (negative log p-value)
                # Higher = more significant
                score = -np.log10(p_value + 1e-300)  # Add tiny epsilon to avoid log(0)
                results.append((start, end, score, p_value, cohens_d))
    
    # Sort by p-value (most significant first)
    results.sort(key=lambda x: x[3])  # Sort by p_value (ascending)
    
    return results


def intervals_overlap(interval1, interval2):
    """
    Check if two intervals overlap.
    
    PHASE 4 ADDITION: Helper for hybrid detection.
    
    Args:
        interval1, interval2: Tuples of (start, end)
        
    Returns:
        bool: True if intervals overlap
    """
    start1, end1 = interval1
    start2, end2 = interval2
    return not (end1 < start2 or end2 < start1)


def hybrid_detection(scaling_data, timestamps=None, max_detections=10):
    """
    PHASE 4: Hybrid detection combining DP and statistical methods.
    
    This function gets candidates from BOTH methods and intelligently merges them:
    - DP method: Good at finding sustained high-magnitude intervals
    - Statistical method: Good at finding weak but statistically significant intervals
    
    Prioritization:
    1. BOTH methods agree (highest confidence) 
    2. Statistical only (rigorous, but DP missed it - likely weak attack)
    3. DP only (magnitude-based, but not statistically significant)
    
    This hybrid approach should improve weak attack detection while maintaining
    strong attack detection and low false positive rate.
    
    Args:
        scaling_data (np.array): Array of forecast/benchmark ratios
        timestamps (pd.DatetimeIndex): Corresponding timestamps (optional)
        max_detections (int): Maximum number of anomalies to return
        
    Returns:
        list: List of detection dicts with start, end, score, method, etc.
    """
    # Get candidates from BOTH methods
    dp_detections = detect_anomaly_timing(scaling_data, timestamps, max_detections=max_detections)
    stat_detections = statistical_anomaly_detection(scaling_data)
    
    # Merge and re-rank
    all_detections = []
    
    # Add DP detections with their scores
    if dp_detections and len(dp_detections) > 0:
        for det in dp_detections:
            if isinstance(det, tuple) and len(det) >= 3:
                start, end, score = det[:3]
                all_detections.append({
                    'start': start,
                    'end': end,
                    'dp_score': score,
                    'method': 'DP'
                })
    
    # Add statistical detections
    for start, end, score, p_value, cohens_d in stat_detections:
        # Check if this overlaps with DP detection
        overlap = False
        for det in all_detections:
            if intervals_overlap((start, end), (det['start'], det['end'])):
                # Enhance existing detection with statistical evidence
                det['stat_score'] = score
                det['p_value'] = p_value
                det['cohens_d'] = cohens_d
                det['method'] = 'BOTH'
                overlap = True
                break
        
        if not overlap:
            all_detections.append({
                'start': start,
                'end': end,
                'stat_score': score,
                'p_value': p_value,
                'cohens_d': cohens_d,
                'method': 'STATISTICAL'
            })
    
    # Rank detections: prioritize those found by BOTH methods
    def detection_priority(det):
        if det['method'] == 'BOTH':
            return 3  # Highest priority
        elif det['method'] == 'STATISTICAL':
            return 2  # Medium priority (statistical is more rigorous)
        else:
            return 1  # DP only
    
    all_detections.sort(key=lambda d: (detection_priority(d), 
                                        d.get('stat_score', d.get('dp_score', 0))), 
                        reverse=True)
    
    # Limit to max_detections
    return all_detections[:max_detections]


def detect_anomaly_timing(scaling_data, timestamps=None, max_detections=1):
    """
    Detect anomaly timing using TWO-TIER detection system.
    
    TWO-TIER SYSTEM:
    1. EMERGENCY MODE: Detects massive spikes (>50%) INSTANTLY (1 hour)
       - Critical for grid-destroying attacks (500%+ spikes)
       - Immediate alert for grid operators
    
    2. NORMAL MODE: Detects moderate anomalies (15-50%) after 3+ hours
       - Uses dynamic programming for accuracy
       - Avoids false alarms from brief fluctuations
    
    The magnitude rule (R_mag) checks if scaling_data deviates from 1.0 by
    more than the magnitude threshold (Tr_mag).
    
    PHASE 3 ENHANCEMENT: Multi-Detection Mode
    - When max_detections > 1, returns multiple anomaly candidates
    - Solves the "global maximum problem" where natural spikes dominate weak attacks
    
    Args:
        scaling_data (np.array): Array of forecast/benchmark ratios
        timestamps (pd.DatetimeIndex): Corresponding timestamps (optional)
        max_detections (int): Maximum number of anomalies to return (1 = legacy single detection)
        
    Returns:
        If max_detections == 1: tuple (start_idx, end_idx, max_score) or (None, None, 0)
        If max_detections > 1: list of (start_idx, end_idx, score) tuples (empty list if none found)
    """
    n = len(scaling_data)
    
    # EMERGENCY MODE CHECK - Detect MASSIVE spikes IMMEDIATELY
    deviation = np.abs(scaling_data - 1.0)
    emergency_points = np.where(deviation > config.EMERGENCY_THRESHOLD)[0]
    
    if len(emergency_points) > 0:
        # INSTANT ALERT - Critical grid threat detected!
        start_idx = emergency_points[0]
        end_idx = emergency_points[-1]
        max_deviation = np.max(deviation)
        score = max_deviation * 10  # High emergency score
        
        print(f"\n🚨 EMERGENCY ALERT: Extreme spike detected!")
        print(f"   Deviation: {max_deviation:.2%} (Threshold: {config.EMERGENCY_THRESHOLD:.0%})")
        print(f"   Response: IMMEDIATE ACTION REQUIRED")
        
        # Return format depends on max_detections parameter
        if max_detections > 1:
            return [(start_idx, end_idx, score)]
        else:
            return start_idx, end_idx, score
    
    # NORMAL MODE - Requires sustained anomaly
    if n < config.MIN_ANOMALY_DURATION:
        if max_detections > 1:
            return []
        else:
            return None, None, 0
    
    # Initialize dynamic programming tables
    # dp[i] = maximum score of interval ending at position i
    dp = np.zeros(n)
    # start[i] = start index of the best interval ending at position i
    start_indices = np.arange(n)
    
    # Phase 2 Optimization: Differential detection to filter natural baseline fluctuations
    # Calculate local baseline (rolling median over 48-hour window)
    window_size = min(48, n)
    local_baseline = np.zeros(n)
    for i in range(n):
        start = max(0, i - window_size // 2)
        end = min(n, i + window_size // 2 + 1)
        local_baseline[i] = np.median(deviation[start:end])
    
    # Differential deviation: how much does this hour exceed its local context?
    differential_deviation = deviation - local_baseline
    differential_deviation = np.maximum(differential_deviation, 0)  # Only positive differences
    
    # Calculate deviation scores based on differential (change from local baseline)
    base_scores = np.where(differential_deviation > config.MAGNITUDE_THRESHOLD, 
                           differential_deviation - config.MAGNITUDE_THRESHOLD, 
                           0)
    
    # Apply magnitude-aware scoring with exponential amplification
    # Use DIFFERENTIAL deviation to classify attack strength
    weak_mask = (differential_deviation >= config.MAGNITUDE_THRESHOLD) & (differential_deviation < 0.30)
    medium_strong_mask = differential_deviation >= 0.30
    
    scores = np.zeros_like(base_scores)
    
    # WEAK ATTACKS (10-30% above local baseline): Exponential amplification
    # This detects CHANGES from baseline, filtering sustained natural fluctuations
    scores[weak_mask] = base_scores[weak_mask] * np.exp(differential_deviation[weak_mask] / 0.06) * 5.0
    
    # MEDIUM/STRONG ATTACKS (>30% above baseline): Standard super-additive function
    scores[medium_strong_mask] = base_scores[medium_strong_mask] * (1 + config.LAMBDA_SCORE * base_scores[medium_strong_mask])
    
    # Dynamic programming to find maximum scoring interval
    for i in range(n):
        if i == 0:
            dp[i] = scores[i]
            start_indices[i] = 0
        else:
            # Option 1: Start new interval at i
            score_new = scores[i]
            
            # Option 2: Extend interval from i-1
            score_extend = dp[i-1] + scores[i]
            
            if score_extend > score_new:
                dp[i] = score_extend
                start_indices[i] = start_indices[i-1]
            else:
                dp[i] = score_new
                start_indices[i] = i
    
    # Phase 2 & 3 Optimization: Multi-peak detection with edge filtering
    # Find ALL local maxima above threshold, not just global maximum
    # PHASE 3: More aggressive peak finding to capture weak attacks
    local_maxima = []
    edge_zone_start = int(n * 0.75)  # Last 25% of data is considered edge zone
    
    # PHASE 3: Find peaks using a more sensitive approach
    # Look for ANY point where DP score is locally high
    for i in range(n):
        start = start_indices[i]
        duration = i - start + 1
        score = dp[i]
        
        # PHASE 3: More lenient peak detection
        # A peak is any point where:
        # 1. Score drops significantly after (30% drop), OR
        # 2. Score is locally maximum within a small window, OR
        # 3. It's the end of the data
        is_peak = False
        
        if i == n-1:
            is_peak = True  # Always consider the last point
        elif i < n-1 and dp[i+1] < dp[i] * 0.7:
            is_peak = True  # 30% drop indicates new interval
        elif i >= 2 and i < n-2:
            # Check if this is a local maximum within ±2 window
            local_window = dp[max(0, i-2):min(n, i+3)]
            if score >= np.max(local_window):
                is_peak = True
        
        # PHASE 3: Include ALL peaks with any positive score and sufficient deviation
        if is_peak and score > 0:
            max_dev = np.max(deviation[start:i+1])
            # Only include if deviation is above threshold (filter out pure noise)
            if max_dev > config.MAGNITUDE_THRESHOLD:
                is_in_edge_zone = start >= edge_zone_start
                local_maxima.append((start, i, score, max_dev, is_in_edge_zone))
    
    # If no peaks found, use global maximum
    if len(local_maxima) == 0:
        max_score = np.max(dp)
        end_idx = np.argmax(dp)
        start_idx = start_indices[end_idx]
        local_maxima = [(start_idx, end_idx, max_score, np.max(deviation[start_idx:end_idx+1]), False)]
    
    # Sort peaks: prioritize non-edge detections, then highest deviation, then score
    local_maxima.sort(key=lambda x: (not x[4], x[3], x[2]), reverse=True)
    
    # PHASE 3: Multi-detection mode
    if max_detections > 1:
        # Return multiple detections after filtering
        # In multi-detection mode, use MORE LENIENT filtering to capture weak attacks
        valid_detections = []
        
        for start_idx, end_idx, score, max_dev, is_edge in local_maxima[:max_detections * 5]:  # Check many candidates
            duration = end_idx - start_idx + 1
            
            # PHASE 3: Relaxed duration requirements for multi-detection mode
            # Goal: Allow weak attacks to be detected even if short
            if max_dev < 0.25:  # Weak attack
                min_required_duration = 1  # Allow 1-hour weak attacks (relaxed from 3)
            elif max_dev < 0.40:  # Medium attack
                min_required_duration = 1  # Allow 1-hour medium attacks (relaxed from 2)
            else:  # Strong attack
                min_required_duration = config.MIN_DURATION_STRONG
            
            if duration < min_required_duration:
                continue
            
            # PHASE 3: Relaxed score thresholds for multi-detection mode
            # Goal: Lower bar to capture more candidates
            if duration <= 5:
                score_threshold = config.MIN_ANOMALY_SCORE_SHORT * 0.5  # 50% lower
            elif duration <= 11:
                score_threshold = config.MIN_ANOMALY_SCORE_MEDIUM * 0.5
            else:
                score_threshold = config.MIN_ANOMALY_SCORE_LONG * 0.5
            
            if score < score_threshold:
                continue
            
            valid_detections.append((start_idx, end_idx, score))
            
            # Stop if we have enough valid detections
            if len(valid_detections) >= max_detections:
                break
        
        return valid_detections
    
    # LEGACY MODE: Single detection (max_detections == 1)
    start_idx, end_idx, max_score, max_deviation, _ = local_maxima[0]
    
    # Check if the anomaly meets minimum duration and score threshold
    duration = end_idx - start_idx + 1
    
    # Apply magnitude-specific duration requirements
    if max_deviation < 0.25:  # Weak attack
        min_required_duration = config.MIN_DURATION_WEAK
    elif max_deviation < 0.40:  # Medium attack
        min_required_duration = config.MIN_DURATION_MEDIUM
    else:  # Strong attack
        min_required_duration = config.MIN_DURATION_STRONG
    
    if duration < min_required_duration:
        return None, None, 0
    
    # Duration-aware score thresholds
    if duration <= 5:
        score_threshold = config.MIN_ANOMALY_SCORE_SHORT
    elif duration <= 11:
        score_threshold = config.MIN_ANOMALY_SCORE_MEDIUM
    else:
        score_threshold = config.MIN_ANOMALY_SCORE_LONG
    
    if max_score < score_threshold:
        return None, None, 0
    
    return start_idx, end_idx, max_score


def segment_anomaly_interval(scaling_data, start_idx, end_idx, min_gap=None):
    """
    Break a long detected interval into distinct anomaly segments.
    
    This addresses the "Hour 46 Problem" where the DP algorithm creates
    one giant interval spanning multiple separate attacks. Uses gap detection
    to split intervals where there are consecutive hours below threshold.
    
    Phase 2 Optimization: Adaptive gap threshold based on attack magnitude
    to prevent over-segmentation of weak attacks.
    
    Args:
        scaling_data (np.array): Array of scaling ratios
        start_idx (int): Start of detected interval
        end_idx (int): End of detected interval
        min_gap (int): Minimum consecutive normal hours to create a break
                      (default: adaptive based on magnitude)
        
    Returns:
        list: List of (segment_start, segment_end, segment_score) tuples
    """
    if min_gap is None:
        # Phase 2: Adaptive gap based on attack magnitude
        max_dev = np.max(np.abs(scaling_data[start_idx:end_idx+1] - 1.0))
        if max_dev < 0.25:  # Weak attack detected
            min_gap = 5  # Require longer break to segment (keeps weak attacks together)
        else:
            min_gap = config.SEGMENT_GAP_HOURS
    
    deviation = np.abs(scaling_data - 1.0)
    
    # Find hours above threshold in the interval
    above_threshold = deviation[start_idx:end_idx+1] > config.MAGNITUDE_THRESHOLD
    
    segments = []
    segment_start = None
    gap_count = 0
    
    for i in range(start_idx, end_idx + 1):
        if above_threshold[i - start_idx]:
            # Hour is anomalous
            if segment_start is None:
                segment_start = i  # Start new segment
            gap_count = 0  # Reset gap counter
        else:
            # Hour is normal
            gap_count += 1
            
            # If we have min_gap consecutive normal hours, close segment
            if segment_start is not None and gap_count >= min_gap:
                segment_end = i - min_gap
                
                # Calculate segment score
                seg_deviation = deviation[segment_start:segment_end+1]
                seg_base_scores = np.where(seg_deviation > config.MAGNITUDE_THRESHOLD,
                                          seg_deviation - config.MAGNITUDE_THRESHOLD,
                                          0)
                seg_scores = seg_base_scores * (1 + config.LAMBDA_SCORE * seg_base_scores)
                segment_score = np.sum(seg_scores)
                
                # Only keep segments with sufficient score and duration
                if segment_score >= config.MIN_SEGMENT_SCORE and (segment_end - segment_start + 1) >= config.MIN_ANOMALY_DURATION:
                    segments.append((segment_start, segment_end, segment_score))
                
                segment_start = None
                gap_count = 0
    
    # Close final segment if still open
    if segment_start is not None:
        segment_end = end_idx
        seg_deviation = deviation[segment_start:segment_end+1]
        seg_base_scores = np.where(seg_deviation > config.MAGNITUDE_THRESHOLD,
                                   seg_deviation - config.MAGNITUDE_THRESHOLD,
                                   0)
        seg_scores = seg_base_scores * (1 + config.LAMBDA_SCORE * seg_base_scores)
        segment_score = np.sum(seg_scores)
        
        if segment_score >= config.MIN_SEGMENT_SCORE and (segment_end - segment_start + 1) >= config.MIN_ANOMALY_DURATION:
            segments.append((segment_start, segment_end, segment_score))
    
    return segments


def detect_anomaly_with_segmentation(scaling_data, timestamps=None, max_detections=1):
    """
    Enhanced anomaly detection with interval segmentation.
    
    This wrapper function:
    1. Uses the original DP algorithm to find anomalous intervals
    2. For long intervals (>10 hours), applies segmentation to break them
    3. Returns the best segment (highest score) or multiple detections
    
    This fixes the precision issues where one giant interval spans multiple
    separate attacks.
    
    PHASE 3 ENHANCEMENT: Multi-Detection Support
    - When max_detections > 1, returns multiple anomaly candidates
    - Enables detection of weak attacks that don't win global maximum competition
    
    Args:
        scaling_data (np.array): Array of forecast/benchmark ratios
        timestamps (pd.DatetimeIndex): Corresponding timestamps (optional)
        max_detections (int): Maximum number of anomalies to return (1 = legacy single detection)
        
    Returns:
        If max_detections == 1: tuple (start_idx, end_idx, max_score) or (None, None, 0)
        If max_detections > 1: list of (start_idx, end_idx, score) tuples (empty list if none found)
    """
    # Call DP algorithm with multi-detection support
    result = detect_anomaly_timing(scaling_data, timestamps, max_detections)
    
    # PHASE 3: Multi-detection mode
    if max_detections > 1:
        if not result or len(result) == 0:
            return []
        
        # Process each detection: apply segmentation if needed
        all_detections = []
        for start_idx, end_idx, score in result:
            duration = end_idx - start_idx + 1
            
            # If interval is short, keep as-is
            if duration < config.MIN_SEGMENT_DURATION_FOR_SPLIT:
                all_detections.append((start_idx, end_idx, score))
            else:
                # For long intervals, segment them
                segments = segment_anomaly_interval(scaling_data, start_idx, end_idx)
                
                if len(segments) > 0:
                    # Add the best segment from this interval
                    best_segment = max(segments, key=lambda x: x[2])
                    all_detections.append(best_segment)
        
        return all_detections
    
    # LEGACY MODE: Single detection (max_detections == 1)
    start_idx, end_idx, max_score = result
    
    if start_idx is None:
        return None, None, 0
    
    # If interval is short, return as-is (likely a precise detection)
    duration = end_idx - start_idx + 1
    if duration < config.MIN_SEGMENT_DURATION_FOR_SPLIT:
        return start_idx, end_idx, max_score
    
    # For long intervals, segment them
    segments = segment_anomaly_interval(scaling_data, start_idx, end_idx)
    
    if len(segments) == 0:
        # No valid segments found - return None
        return None, None, 0
    
    # Return the segment with highest score
    # (This gives the most confident detection)
    best_segment = max(segments, key=lambda x: x[2])
    
    return best_segment[0], best_segment[1], best_segment[2]


def get_benchmark_for_period(kmeans_model, forecast_values):
    """
    Get benchmark values for a forecast period using the trained K-means model.
    
    For each 24-hour window in the forecast, finds the nearest cluster centroid
    as the benchmark profile.
    
    Args:
        kmeans_model (KMeans): Trained K-means model
        forecast_values (np.array): Forecasted load values
        
    Returns:
        np.array: Benchmark values matching the forecast period
    """
    n_hours = len(forecast_values)
    benchmark = np.zeros(n_hours)
    
    # Process in 24-hour windows
    for i in range(0, n_hours, config.HOURS_PER_DAY):
        end_idx = min(i + config.HOURS_PER_DAY, n_hours)
        window_size = end_idx - i
        
        if window_size == config.HOURS_PER_DAY:
            # Full 24-hour window
            daily_pattern = forecast_values[i:end_idx].reshape(1, -1).astype(np.float32)
            cluster_label = kmeans_model.predict(daily_pattern)[0]
            benchmark[i:end_idx] = kmeans_model.cluster_centers_[cluster_label]
        else:
            # Partial window at the end
            # Use the last complete window's cluster
            if i > 0:
                prev_pattern = forecast_values[i-config.HOURS_PER_DAY:i].reshape(1, -1).astype(np.float32)
                cluster_label = kmeans_model.predict(prev_pattern)[0]
                benchmark[i:end_idx] = kmeans_model.cluster_centers_[cluster_label][:window_size]
    
    return benchmark


def demonstrate_anomaly_detection(df, model, scaler, feature_cols, kmeans_model):
    """
    Demonstrate the complete anomaly detection pipeline with an injected anomaly.
    
    Args:
        df (pd.DataFrame): Original preprocessed data
        model: Trained LSTM model
        scaler: Trained MinMaxScaler
        feature_cols: List of feature column names
        kmeans_model: Trained K-means model
    """
    print("\n" + "="*60)
    print("STEP 5: Anomaly Detection Demonstration")
    print("="*60)
    
    # Use test data for demonstration
    df_features = create_features(df)
    
    split_idx = int(len(df_features) * config.TRAIN_TEST_SPLIT_RATIO)
    test_data = df_features.iloc[split_idx:].copy()
    
    # Select one week of test data (168 hours)
    demo_period = test_data.iloc[:168].copy()
    
    print(f"\nDemo period: {demo_period.index[0]} to {demo_period.index[-1]}")
    print(f"Duration: {len(demo_period)} hours")
    
    # Generate forecast
    X_demo = demo_period[feature_cols].values
    X_demo_scaled = scaler.transform(X_demo)
    X_demo_reshaped = X_demo_scaled.reshape((X_demo_scaled.shape[0], 1, X_demo_scaled.shape[1]))
    
    forecast = model.predict(X_demo_reshaped, verbose=0).flatten()
    
    print(f"\nOriginal forecast - Mean: {np.mean(forecast):.2f} MWh, Std: {np.std(forecast):.2f} MWh")
    
    # INJECT ANOMALY: Multiply load by 1.2 for a 10-hour window
    anomaly_start = 50
    anomaly_end = 60
    forecast_with_anomaly = forecast.copy()
    forecast_with_anomaly[anomaly_start:anomaly_end] *= 1.2
    
    print(f"\n*** INJECTING ANOMALY ***")
    print(f"Anomaly injected from hour {anomaly_start} to {anomaly_end} (10 hours)")
    print(f"Injected at timestamps: {demo_period.index[anomaly_start]} to {demo_period.index[anomaly_end-1]}")
    print(f"Multiplier: 1.2x")
    
    # Get benchmark for the period
    benchmark = get_benchmark_for_period(kmeans_model, forecast_with_anomaly)
    
    print(f"\nBenchmark - Mean: {np.mean(benchmark):.2f} MWh, Std: {np.std(benchmark):.2f} MWh")
    
    # Calculate scaling data
    # Add small epsilon to avoid division by zero
    scaling_data = forecast_with_anomaly / (benchmark + 1e-6)
    
    print(f"Scaling data - Mean: {np.mean(scaling_data):.4f}, Std: {np.std(scaling_data):.4f}")
    
    # Detect anomaly timing (with segmentation for improved precision)
    print("\n" + "-"*60)
    print("Running Dynamic Programming Anomaly Detection with Segmentation...")
    print("-"*60)
    
    detected_start, detected_end, score = detect_anomaly_with_segmentation(scaling_data, demo_period.index)
    
    if detected_start is not None:
        print(f"\n*** ANOMALY DETECTED ***")
        print(f"Detected interval: hour {detected_start} to {detected_end}")
        print(f"Detection score: {score:.4f}")
        print(f"Duration: {detected_end - detected_start + 1} hours")
        print(f"\nActual timestamps:")
        print(f"  Start: {demo_period.index[detected_start]}")
        print(f"  End:   {demo_period.index[detected_end]}")
        
        # Compare with injected anomaly
        print(f"\nComparison with injected anomaly:")
        print(f"  Injected: hours {anomaly_start}-{anomaly_end}")
        print(f"  Detected: hours {detected_start}-{detected_end}")
        
        overlap_start = max(anomaly_start, detected_start)
        overlap_end = min(anomaly_end, detected_end)
        if overlap_start <= overlap_end:
            overlap_hours = overlap_end - overlap_start
            print(f"  Overlap: {overlap_hours} hours")
            accuracy = (overlap_hours / (anomaly_end - anomaly_start)) * 100
            print(f"  Detection accuracy: {accuracy:.1f}%")
    else:
        print("\n*** NO ANOMALY DETECTED ***")
        print("The algorithm did not detect any significant anomalous interval.")
        print(f"(Note: Injected anomaly was at hours {anomaly_start}-{anomaly_end})")


def main():
    """
    Main execution block that orchestrates the entire MLAD pipeline.
    """
    print("="*60)
    print("MACHINE LEARNING-BASED ANOMALY DETECTION (MLAD)")
    print("Power Grid Load Data Analysis")
    print("="*60)
    
    # STEP 1: Load and preprocess data
    print("\n" + "="*60)
    print("STEP 1: Loading and Preprocessing Data")
    print("="*60)
    
    df = load_and_preprocess_data(config.DATASET_DIR)
    
    # STEP 2: Train neural network forecaster
    model, scaler, feature_cols = train_forecaster(df)
    
    # STEP 3: Train K-means benchmark model
    kmeans_model = train_kmeans_benchmark(df)
    
    # STEP 4 & 5: Demonstrate anomaly detection
    demonstrate_anomaly_detection(df, model, scaler, feature_cols, kmeans_model)
    
    print("\n" + "="*60)
    print("MLAD Pipeline Complete!")
    print("="*60)
    print(f"\nTrained models saved in: {config.MODELS_DIR}")
    print("You can now use these models to detect anomalies in new data.")


if __name__ == "__main__":
    main() 
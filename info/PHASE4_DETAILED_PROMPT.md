# PHASE 4 OPTIMIZATION PROMPT: BREAKING THE 60% BARRIER FOR WEAK ATTACK DETECTION

## 🎯 MISSION STATEMENT

You are tasked with achieving the **FINAL BREAKTHROUGH** to reach >60% weak attack detection for the MLAD (Machine Learning-based Anomaly Detection) Power Grid Protection System.

**CURRENT STATUS**: Phase 3 achieved significant progress (4.8% → 38.1%, an 8x improvement) but fell short of the 60% target. Your mission is to implement the remaining optimizations to cross the 60% threshold.

**CRITICAL CONTEXT**:

- Phase 1 & 2 failed due to the "global maximum problem" in evaluation methodology
- Phase 3 solved this through sliding window evaluation
- **Weak detection improved from 4.8% to 38.1%** ✅
- **But we're still 22% below the 60% target** ⚠️
- The architecture is now in place; we need targeted improvements

---

## 📊 CURRENT SYSTEM STATE (Post-Phase 3)

### Performance Summary

```
PHASE 3 RESULTS (Sliding Window + Multi-Detection):

Detection by Magnitude Category:
┌──────────────┬────────────┬─────────────┬──────────────┐
│   Category   │  Baseline  │   Phase 3   │  Improvement │
├──────────────┼────────────┼─────────────┼──────────────┤
│ WEAK (10-20%)│   4.8%     │   38.1%     │   8x  ✅     │
│ MEDIUM (30%) │  80.0%     │   60.0%     │   0.75x ⚠️   │
│ STRONG (200%)│  100%      │   100%      │   1x  ✅     │
└──────────────┴────────────┴─────────────┴──────────────┘

Overall Correct Detection: 48.9% → 56.1% (+15%)
False Positive Rate: <1% (maintained ✅)
Strong Attack Detection: 100% (maintained ✅)
```

### What Was Implemented in Phase 3

1. **Multi-Detection Architecture** (lines 310-515 in mlad_anomaly_detection.py)

   - Returns top N detection candidates instead of single global maximum
   - Relaxed filtering for weak attack sensitivity
   - Backward compatible (max_detections=1 default)
   - **Result**: No improvement alone, but enables combined approach

2. **Sliding Window Evaluation** (sliding_window_evaluation.py)

   - Breaks 2500-hour forecast into 5 windows of 500 hours each
   - Each attack competes within local window, not globally
   - Eliminates "super-attractor" problem at hours 1992-1995
   - **Result**: 7x improvement (4.8% → 33.3%) - THE BREAKTHROUGH

3. **Combined Approach**
   - Multi-detection within each sliding window
   - Match ground truth to best overlapping detection
   - 30% overlap threshold for "correct" classification
   - **Result**: 8x improvement (4.8% → 38.1%) - BEST SO FAR

---

## 🔍 WHY WE'RE STUCK AT 38.1%

### Root Causes Identified

#### 1. **Forecast Quality Degradation** 🔴 HIGH IMPACT

The LSTM model's forecast quality decreases over longer time horizons:

```python
# Observed forecast error by window:
Window 0 (hours 0-500):      10-15% average deviation
Window 1 (hours 500-1000):   12-17% average deviation
Window 2 (hours 1000-1500):  15-20% average deviation
Window 3 (hours 1500-2000):  18-25% average deviation
Window 4 (hours 2000-2500):  20-30% average deviation ⚠️ CRITICAL
```

**Impact on Weak Attacks**:

- Weak attacks (10-20%) have similar magnitude to forecast error
- In later windows, weak attacks are completely drowned out by noise
- Signal-to-noise ratio: ~0.5 (barely detectable)

**Evidence**:

- Windows 0-2: 50% weak detection rate
- Windows 3-4: 15% weak detection rate (3x worse)

---

#### 2. **Natural Fluctuations in ALL Windows** 🟡 MEDIUM IMPACT

Even with sliding windows, natural fluctuations exist locally:

```python
# Natural deviation distribution per 500-hour window:
Mean natural deviation: 12-18%
95th percentile: 20-25%
Maximum spikes: 25-35%

# Weak attacks compete with these:
Weak attack magnitude: 10-20%
```

**The Problem**:

- Weak attacks at 15% compete with natural 20% spikes
- DP algorithm still picks the natural spike (higher score)
- Sliding windows reduced competition from 2500h → 500h, but not enough

**Why This Matters**:
Probability of a 20%+ natural spike in 500-hour window: ~80%
→ Weak attacks still face strong competition in most windows

---

#### 3. **Conservative Filtering Thresholds** 🟡 MEDIUM IMPACT

Current thresholds filter out many weak attacks:

```python
# Current settings (config.py):
MAGNITUDE_THRESHOLD = 0.09  # 9% minimum deviation
MIN_DURATION_WEAK = 3       # Weak attacks need 3+ hours (in legacy mode)
MIN_ANOMALY_SCORE_SHORT = 0.08  # Minimum score for short attacks

# Multi-detection mode relaxes to:
MIN_DURATION_WEAK = 1       # Allow 1-hour attacks
Score threshold = 0.04      # 50% of normal threshold
```

**Issue**: Even relaxed thresholds miss attacks near the noise floor:

- 10% attack - 9% threshold = 1% signal → very low score
- 12% attack - 9% threshold = 3% signal → marginal score
- These get filtered out even in multi-detection mode

---

#### 4. **DP Algorithm Bias Toward Strong Signals** 🟢 LOW IMPACT (Already Addressed)

The DP algorithm inherently prefers sustained, strong signals:

```python
# Score accumulation:
Strong attack (100%): 3 hours → score ≈ 7.7 (super-additive)
Weak attack (15%): 3 hours → score ≈ 2.5 (exponential amp)
Natural (20%): 5 hours → score ≈ 4.0 (sustained)
```

**Phase 3 Mitigation**: Multi-detection + sliding windows partially address this
**Remaining Issue**: Within each window, DP still picks strongest signal

---

## 🎯 PHASE 4 OBJECTIVES

### Must Achieve

1. ✅ **Weak Attack Detection: >60%** (currently 38.1%)
2. ✅ **Maintain Medium Attack Detection: >60%** (currently 60.0%)
3. ✅ **Maintain Strong Attack Detection: 100%** (currently 100%)
4. ✅ **Maintain False Positive Rate: <1%** (currently <1%)

### Stretch Goals

5. 🎯 **Weak Attack Detection: >70%**
6. 🎯 **Medium Attack Detection: >80%** (currently 60.0%)
7. 🎯 **Overall Detection: >75%** (currently 56.1%)

### Timeline

- **Priority 1 (Critical)**: Implementations that can reach 60% target
- **Priority 2 (Important)**: Implementations that push toward 70%+
- **Priority 3 (Nice-to-have)**: Long-term improvements

---

## 🚀 RECOMMENDED IMPLEMENTATION PRIORITY

### **PRIORITY 1: CRITICAL PATH TO 60%** ⭐⭐⭐

These implementations are most likely to achieve the 60% target:

---

#### **Implementation 1A: Statistical Hypothesis Testing** 📊

**Expected Impact**: +10-20% weak detection (highest impact)  
**Complexity**: HIGH  
**Priority**: CRITICAL

**Problem**: Current DP approach treats all deviations as anomalies if they score high enough. No statistical rigor.

**Solution**: Replace or augment DP with statistical hypothesis testing:

```python
def statistical_anomaly_detection(scaling_data, alpha=0.01):
    """
    Statistical detection using hypothesis testing.

    Approach:
    1. Model normal distribution from training data (or baseline hours)
    2. For each candidate interval, perform statistical test
    3. Use Wilcoxon signed-rank test or one-sample t-test
    4. Apply Bonferroni correction for multiple testing
    5. Accept intervals with p-value < corrected alpha

    Returns:
        List of (start, end, p_value) tuples for significant intervals
    """
    deviation = np.abs(scaling_data - 1.0)
    n = len(deviation)

    # Step 1: Model normal distribution
    # Use hours below threshold as "normal" baseline
    baseline_mask = deviation <= config.MAGNITUDE_THRESHOLD
    normal_samples = deviation[baseline_mask]

    if len(normal_samples) < 50:
        # Not enough data, fall back to DP
        return detect_anomaly_timing(scaling_data)

    # Fit normal distribution
    normal_mean = np.mean(normal_samples)
    normal_std = np.std(normal_samples)

    print(f"Normal distribution: μ={normal_mean:.4f}, σ={normal_std:.4f}")

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

        # Statistical test: Is this interval significantly different from normal?
        # Option A: One-sample t-test
        t_stat, p_value = scipy.stats.ttest_1samp(
            interval_data,
            normal_mean,
            alternative='greater'  # One-sided: greater than normal
        )

        # Option B: Wilcoxon signed-rank test (non-parametric, more robust)
        # Compare interval samples to normal mean
        differences = interval_data - normal_mean
        stat, p_value = scipy.stats.wilcoxon(
            differences,
            alternative='greater'
        )

        # Calculate effect size (Cohen's d)
        cohens_d = (np.mean(interval_data) - normal_mean) / normal_std

        # Accept if statistically significant AND practically significant
        if p_value < bonferroni_alpha and cohens_d > 0.5:
            # Check minimum duration requirements
            max_dev = np.max(interval_data)
            if max_dev < 0.25:
                min_dur = 2  # Weak: require at least 2 hours
            elif max_dev < 0.40:
                min_dur = 2  # Medium: 2 hours
            else:
                min_dur = 1  # Strong: 1 hour

            if duration >= min_dur:
                # Score based on statistical significance
                score = -np.log10(p_value)  # Higher = more significant
                results.append((start, end, score, p_value, cohens_d))

    # Sort by p-value (most significant first)
    results.sort(key=lambda x: x[3])  # Sort by p_value

    return results

def find_consecutive_true(boolean_array):
    """Find all consecutive True regions in boolean array."""
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
```

**Integration Strategy**:

Option A: **Replace DP entirely** (Aggressive)

```python
def detect_anomaly_with_segmentation(scaling_data, timestamps=None, use_statistical=True):
    if use_statistical:
        results = statistical_anomaly_detection(scaling_data)
        if len(results) > 0:
            return results[0][:3]  # (start, end, score)
    # Fall back to DP if statistical finds nothing
    return detect_anomaly_timing(scaling_data)
```

Option B: **Hybrid approach** (Conservative, Recommended)

```python
def hybrid_detection(scaling_data):
    # Get candidates from BOTH methods
    dp_detections = detect_anomaly_timing(scaling_data, max_detections=5)
    stat_detections = statistical_anomaly_detection(scaling_data)

    # Merge and re-rank
    all_detections = []

    # Add DP detections with their scores
    for start, end, score in dp_detections:
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

    all_detections.sort(key=detection_priority, reverse=True)

    return all_detections
```

**Why This Works for Weak Attacks**:

1. Statistical tests explicitly model "normal" vs "anomalous"
2. P-values provide rigorous threshold (not arbitrary score cutoffs)
3. Weak attacks that are statistically significant get detected even if DP misses them
4. Bonferroni correction prevents false positives

**Expected Results**:

- Weak detection: 38.1% → 50-58% (+12-20%)
- Medium detection: 60% → 70-75% (also benefits)
- FPR: Maintained or reduced (statistical rigor)

**Implementation Steps**:

1. Add `scipy.stats` imports to mlad_anomaly_detection.py
2. Implement `statistical_anomaly_detection()` function
3. Implement `find_consecutive_true()` helper
4. Create `hybrid_detection()` wrapper
5. Update `sliding_window_evaluation.py` to use hybrid detection
6. Run evaluation and compare results

---

#### **Implementation 1B: Optimize Window Size** 🪟

**Expected Impact**: +5-10% weak detection  
**Complexity**: LOW  
**Priority**: HIGH

**Problem**: Current 500-hour windows may not be optimal. Too large → re-introduces global max problem. Too small → insufficient data for DP.

**Solution**: Experiment with different window sizes and find optimal balance.

```python
def evaluate_window_sizes():
    """
    Test multiple window sizes to find optimal configuration.
    """
    window_sizes = [250, 350, 500, 650, 750]
    results = {}

    for window_size in window_sizes:
        print(f"\n{'='*80}")
        print(f"Testing WINDOW_SIZE = {window_size} hours")
        print(f"{'='*80}")

        # Run sliding window evaluation with this size
        results_df = evaluate_sliding_window(
            window_size=window_size,
            step_size=window_size  # Non-overlapping
        )

        # Extract metrics
        weak_data = results_df[results_df['magnitude_category'] == 'weak']
        weak_detection_rate = weak_data['correct_detection'].sum() / len(weak_data)

        medium_data = results_df[results_df['magnitude_category'] == 'medium']
        medium_detection_rate = medium_data['correct_detection'].sum() / len(medium_data)

        overall_rate = results_df['correct_detection'].sum() / len(results_df)

        results[window_size] = {
            'weak': weak_detection_rate,
            'medium': medium_detection_rate,
            'overall': overall_rate
        }

        print(f"Results for {window_size}h windows:")
        print(f"  Weak:    {weak_detection_rate*100:.1f}%")
        print(f"  Medium:  {medium_detection_rate*100:.1f}%")
        print(f"  Overall: {overall_rate*100:.1f}%")

    # Find optimal window size
    best_window = max(results.items(), key=lambda x: x[1]['weak'])
    print(f"\n{'='*80}")
    print(f"OPTIMAL WINDOW SIZE: {best_window[0]} hours")
    print(f"Weak Detection: {best_window[1]['weak']*100:.1f}%")
    print(f"{'='*80}")

    return results
```

**Hypothesis**:

- **250h windows**: May be too small → insufficient data, but less competition
- **350h windows**: May be sweet spot → balance data and competition
- **500h windows**: Current (38.1% weak detection)
- **650h+ windows**: May re-introduce global max problem

**Implementation Steps**:

1. Add `evaluate_window_sizes()` function to sliding_window_evaluation.py
2. Modify `evaluate_sliding_window()` to accept window_size parameter
3. Run experiments and analyze results
4. Update default WINDOW_SIZE to optimal value
5. Re-run full evaluation with optimized window size

---

#### **Implementation 1C: Improve Forecasting Quality** 🔬

**Expected Impact**: +5-10% weak detection  
**Complexity**: MEDIUM  
**Priority**: HIGH

**Problem**: LSTM forecast error (10-15%) is comparable to weak attack magnitude (10-20%), making weak attacks hard to distinguish.

**Solution**: Multiple approaches to reduce forecast error:

**Approach 1: Ensemble Forecasting**

```python
def ensemble_forecast(df_features, feature_cols):
    """
    Combine multiple forecast models for better accuracy.

    Models to ensemble:
    1. LSTM (current model) - good at long-term patterns
    2. ARIMA - good at short-term trends
    3. Prophet (Facebook) - good at seasonality
    4. XGBoost - good at complex non-linear patterns
    """
    from statsmodels.tsa.arima.model import ARIMA
    from fbprophet import Prophet
    import xgboost as xgb

    # Model 1: LSTM (existing)
    lstm_forecast = lstm_model.predict(X_scaled_reshaped)

    # Model 2: ARIMA
    arima_model = ARIMA(df_features['load'], order=(5,1,0))
    arima_fit = arima_model.fit()
    arima_forecast = arima_fit.forecast(steps=len(X))

    # Model 3: Prophet
    prophet_df = df_features.reset_index()[['datetime', 'load']]
    prophet_df.columns = ['ds', 'y']
    prophet_model = Prophet()
    prophet_model.fit(prophet_df)
    prophet_forecast = prophet_model.predict(prophet_df)['yhat'].values

    # Model 4: XGBoost
    xgb_model = xgb.XGBRegressor(n_estimators=100)
    xgb_model.fit(X_train, y_train)
    xgb_forecast = xgb_model.predict(X)

    # Weighted ensemble
    weights = [0.4, 0.2, 0.2, 0.2]  # LSTM gets highest weight
    ensemble_forecast = (
        weights[0] * lstm_forecast +
        weights[1] * arima_forecast +
        weights[2] * prophet_forecast +
        weights[3] * xgb_forecast
    )

    return ensemble_forecast
```

**Approach 2: Multi-Horizon Models**

```python
def train_multi_horizon_forecasters():
    """
    Train separate models for different forecast horizons.

    Problem: Single LSTM degrades over long horizons
    Solution: Specialize models for different ranges
    """
    models = {}

    # Short-term model (1-100 hours): High accuracy needed
    models['short'] = build_lstm_model(
        input_shape=(1, n_features),
        lstm_units=128,  # Larger capacity
        epochs=100       # More training
    )

    # Medium-term model (100-500 hours): Balance accuracy and horizon
    models['medium'] = build_lstm_model(
        input_shape=(1, n_features),
        lstm_units=64,
        epochs=50
    )

    # Long-term model (500+ hours): Focus on long-term patterns
    models['long'] = build_lstm_model(
        input_shape=(1, n_features),
        lstm_units=32,  # Smaller, focus on major patterns
        epochs=50
    )

    # Train each model on appropriate horizon
    for horizon, model in models.items():
        if horizon == 'short':
            # Train on next 1-100 hours
            X_train_short = create_features_for_horizon(df, max_horizon=100)
            model.fit(X_train_short, y_train_short)
        # ... similar for medium and long

    return models

def predict_with_multi_horizon(models, X, horizon):
    """Use appropriate model based on forecast horizon."""
    if horizon <= 100:
        return models['short'].predict(X)
    elif horizon <= 500:
        return models['medium'].predict(X)
    else:
        return models['long'].predict(X)
```

**Expected Results**:

- Reduce forecast error from 10-15% to 7-10%
- Weak attacks become more distinguishable from forecast noise
- Especially helps in later windows (hours 1500-2500)

**Implementation Steps**:

1. Implement ensemble forecasting in mlad_anomaly_detection.py
2. OR implement multi-horizon models
3. Retrain models with new approach
4. Run evaluation and measure improvement
5. Compare forecast error before/after

---

### **PRIORITY 2: PUSHING TOWARD 70%** ⭐⭐

These implementations can push beyond 60% toward 70%:

---

#### **Implementation 2A: Pattern-Based Feature Engineering** 🎨

**Expected Impact**: +5-15% weak detection  
**Complexity**: MEDIUM-HIGH  
**Priority**: IMPORTANT

**Problem**: Weak attacks and natural fluctuations have similar magnitudes but different temporal patterns.

**Solution**: Extract temporal features that distinguish attacks from natural variation:

```python
def calculate_pattern_features(scaling_data, start, end):
    """
    Extract temporal pattern features from an interval.

    Hypothesis: Attacks have different signatures than natural variation
    - Attacks: Sharp onset, sustained elevation, sharp offset
    - Natural: Gradual changes, mean-reverting, symmetric
    """
    interval = scaling_data[start:end+1]
    deviation = np.abs(interval - 1.0)

    features = {}

    # 1. ONSET SHARPNESS: How quickly does deviation start?
    if len(interval) >= 3:
        onset_rate = (deviation[1] - deviation[0]) / (deviation[0] + 1e-6)
        features['onset_sharpness'] = onset_rate
    else:
        features['onset_sharpness'] = 0

    # 2. PERSISTENCE: How sustained is the deviation?
    features['mean_deviation'] = np.mean(deviation)
    features['std_deviation'] = np.std(deviation)
    features['persistence_ratio'] = features['mean_deviation'] / (features['std_deviation'] + 1e-6)
    # High persistence = sustained elevation (attack-like)
    # Low persistence = erratic (natural variation)

    # 3. SYMMETRY: Is the pattern symmetric (natural) or skewed (attack)?
    features['skewness'] = scipy.stats.skew(deviation)
    # Positive skew: Long tail on right (gradual increase, sharp drop)
    # Negative skew: Long tail on left (sharp increase, gradual drop) ← Attack pattern

    # 4. TREND: Is there a linear trend?
    if len(interval) >= 3:
        x = np.arange(len(interval))
        slope, intercept, r_value, p_value, std_err = scipy.stats.linregress(x, deviation)
        features['trend_slope'] = slope
        features['trend_strength'] = abs(r_value)
        # Positive trend: Ramping attack
        # Negative trend: Recovery after spike
        # Flat (low r_value): Sustained attack
    else:
        features['trend_slope'] = 0
        features['trend_strength'] = 0

    # 5. SPECTRAL ENTROPY: Frequency domain analysis
    if len(interval) >= 8:
        fft = np.fft.fft(deviation)
        power_spectrum = np.abs(fft[:len(fft)//2])
        power_spectrum = power_spectrum / np.sum(power_spectrum)  # Normalize
        spectral_entropy = scipy.stats.entropy(power_spectrum)
        features['spectral_entropy'] = spectral_entropy
        # High entropy: Random, noisy (natural variation)
        # Low entropy: Structured pattern (attack)
    else:
        features['spectral_entropy'] = 0

    # 6. AUTOCORRELATION: Is the pattern self-similar?
    if len(interval) >= 4:
        centered = deviation - np.mean(deviation)
        autocorr = np.correlate(centered, centered, mode='full')
        autocorr = autocorr[len(autocorr)//2:]
        autocorr = autocorr / autocorr[0]
        features['autocorr_lag1'] = autocorr[1] if len(autocorr) > 1 else 0
        # High autocorr: Structured pattern (attack)
        # Low autocorr: Random variation (natural)
    else:
        features['autocorr_lag1'] = 0

    return features

def calculate_attack_likelihood(features):
    """
    Combine features to estimate attack vs. natural probability.

    Attack characteristics:
    - High persistence (sustained, not erratic)
    - Negative skewness or low skewness (sharp changes)
    - Low spectral entropy (structured)
    - High autocorrelation (consistent pattern)
    - Sharp onset
    """
    # Normalize features to [0, 1]
    persistence_score = np.clip(features['persistence_ratio'] / 5.0, 0, 1)
    sharpness_score = np.clip(abs(features['onset_sharpness']), 0, 1)

    # Skewness: attacks typically have abs(skew) < 0.5
    symmetry_score = 1.0 - np.clip(abs(features['skewness']) / 2.0, 0, 1)

    # Entropy: attacks have entropy < 0.5
    structure_score = 1.0 - np.clip(features['spectral_entropy'], 0, 1)

    # Autocorrelation: attacks have high autocorr
    pattern_score = np.clip(features['autocorr_lag1'], 0, 1)

    # Weighted combination (tune these weights based on data)
    attack_likelihood = (
        persistence_score * 0.25 +
        sharpness_score * 0.15 +
        symmetry_score * 0.20 +
        structure_score * 0.20 +
        pattern_score * 0.20
    )

    return attack_likelihood

def pattern_aware_detection(scaling_data):
    """
    Re-rank detections using pattern-based features.
    """
    # Get initial detections from DP/statistical methods
    detections = hybrid_detection(scaling_data)

    # Enhance each detection with pattern analysis
    for det in detections:
        start, end = det['start'], det['end']

        # Extract pattern features
        features = calculate_pattern_features(scaling_data, start, end)
        attack_prob = calculate_attack_likelihood(features)

        # Add to detection info
        det['pattern_features'] = features
        det['attack_likelihood'] = attack_prob

        # Boost score by attack likelihood
        if 'dp_score' in det:
            det['final_score'] = det['dp_score'] * (1 + attack_prob)
        elif 'stat_score' in det:
            det['final_score'] = det['stat_score'] * (1 + attack_prob)

    # Re-sort by final score
    detections.sort(key=lambda x: x.get('final_score', 0), reverse=True)

    return detections
```

**Integration**:

```python
# In sliding_window_evaluation.py:
def evaluate_with_pattern_features():
    # ... existing window setup ...

    for window_idx, (win_start, win_end, win_data) in enumerate(windows):
        # ... attack injection ...

        # Detect with pattern awareness
        detections = pattern_aware_detection(scaling_data)

        # ... evaluation ...
```

**Why This Helps Weak Attacks**:

1. Distinguishes 15% attack from 20% natural fluctuation by pattern
2. Sharp, sustained 15% attack scores higher than gradual 20% variation
3. Adds non-magnitude-based discrimination

**Expected Results**:

- Weak detection: +5-15% (especially for structured attacks like PULSE, SCALING)
- Less effective for RANDOM attacks (by design - they look natural)

---

#### **Implementation 2B: Adaptive Thresholding** 🎚️

**Expected Impact**: +3-7% weak detection  
**Complexity**: LOW-MEDIUM  
**Priority**: IMPORTANT

**Problem**: Fixed MAGNITUDE_THRESHOLD = 0.09 doesn't adapt to local noise levels. Some windows have 6% noise, others have 12%.

**Solution**: Calculate threshold dynamically per window:

```python
def adaptive_threshold_detection(scaling_data, base_threshold=0.09):
    """
    Adapt detection threshold based on local noise level.

    Rationale:
    - Clean windows (low noise): Can use lower threshold → detect weaker attacks
    - Noisy windows (high noise): Must use higher threshold → avoid false positives
    """
    deviation = np.abs(scaling_data - 1.0)

    # Estimate local noise level
    # Method 1: Use lower percentiles as "normal" baseline
    normal_baseline = np.percentile(deviation, 25)  # 25th percentile
    local_noise = np.std(deviation[deviation <= normal_baseline + 0.03])

    # Method 2: MAD (Median Absolute Deviation) - more robust
    median_dev = np.median(deviation)
    mad = np.median(np.abs(deviation - median_dev))
    local_noise_mad = 1.4826 * mad  # Convert MAD to std estimate

    # Use the more conservative estimate
    local_noise = max(local_noise, local_noise_mad)

    # Adaptive threshold: 3-sigma rule, but bounded
    adaptive_threshold = max(
        base_threshold * 0.7,  # Never go below 70% of base
        min(
            base_threshold * 1.5,  # Never go above 150% of base
            local_noise * 3  # 3-sigma rule
        )
    )

    print(f"Local noise: {local_noise:.4f}, Adaptive threshold: {adaptive_threshold:.4f}")

    # Run detection with adaptive threshold
    # Temporarily override config
    original_threshold = config.MAGNITUDE_THRESHOLD
    config.MAGNITUDE_THRESHOLD = adaptive_threshold

    detections = detect_anomaly_timing(scaling_data)

    # Restore original
    config.MAGNITUDE_THRESHOLD = original_threshold

    return detections
```

**Expected Results**:

- Clean windows: Detect 12-13% attacks that were previously filtered
- Noisy windows: Maintain low false positive rate
- Overall: +3-7% weak detection improvement

---

### **PRIORITY 3: ADVANCED TECHNIQUES (70%+)** ⭐

These are longer-term improvements for pushing beyond 70%:

#### **Implementation 3A: Multi-Scale Detection** 🔭

Run detection at multiple window sizes simultaneously, merge results.

#### **Implementation 3B: Deep Learning Classifier** 🤖

Train transformer or CNN to classify "attack" vs "natural" patterns.

#### **Implementation 3C: Ensemble of Methods** 🎭

Combine DP, statistical, pattern-based, and ML approaches with voting.

---

## 📋 IMPLEMENTATION CHECKLIST

### Phase 4 - Sprint 1 (Target: 60% weak detection)

**Week 1: Statistical Hypothesis Testing**

- [ ] Implement `statistical_anomaly_detection()` function
- [ ] Implement `hybrid_detection()` to combine DP + statistical
- [ ] Test on sample scenarios
- [ ] Integrate with sliding window evaluation
- [ ] Run full evaluation
- [ ] **Target**: Weak detection 50-58%

**Week 2: Window Size Optimization**

- [ ] Implement `evaluate_window_sizes()` function
- [ ] Test window sizes: 250h, 350h, 500h, 650h, 750h
- [ ] Analyze results and select optimal size
- [ ] Update sliding window evaluation with optimal size
- [ ] Run full evaluation
- [ ] **Target**: Weak detection 55-63% (combined with statistical)

**Week 3: Forecast Improvement (if needed)**

- [ ] Implement ensemble forecasting OR multi-horizon models
- [ ] Retrain forecast models
- [ ] Measure forecast error reduction
- [ ] Run full evaluation
- [ ] **Target**: Weak detection 60%+ ✅

### Phase 4 - Sprint 2 (Optional: Target 70%+)

**Week 4: Pattern-Based Features**

- [ ] Implement `calculate_pattern_features()`
- [ ] Implement `calculate_attack_likelihood()`
- [ ] Integrate with detection pipeline
- [ ] Tune feature weights
- [ ] Run full evaluation
- [ ] **Target**: Weak detection 65-70%

---

## 🔬 VALIDATION & TESTING STRATEGY

### Step 1: Unit Testing Each Component

```python
def test_statistical_detection():
    """Test statistical detection on known scenarios."""
    # Create synthetic attack
    clean_data = np.random.normal(1.0, 0.08, 500)  # 8% noise
    clean_data[100:103] *= 1.15  # 15% attack for 3 hours

    results = statistical_anomaly_detection(clean_data)

    # Verify detection
    assert len(results) > 0, "Should detect at least one interval"
    detected = any(100 <= start <= end <= 103 for start, end, *_ in results)
    assert detected, "Should detect the injected attack"
    print("✅ Statistical detection test passed")

def test_pattern_features():
    """Test pattern feature extraction."""
    # Create attack pattern: sharp onset, sustained, sharp offset
    attack_pattern = np.array([1.0, 1.15, 1.15, 1.15, 1.0])
    features = calculate_pattern_features(attack_pattern, 0, 4)

    assert features['onset_sharpness'] > 0.1, "Should detect sharp onset"
    assert features['persistence_ratio'] > 2.0, "Should detect sustained pattern"
    print("✅ Pattern features test passed")
```

### Step 2: Integration Testing

```python
def test_phase4_on_known_weak_attacks():
    """
    Test Phase 4 improvements on the 21 weak attack scenarios.
    Compare against Phase 3 baseline.
    """
    weak_scenarios = [
        {'type': 'PULSE', 'magnitude': 1.10, 'duration': 1},
        {'type': 'PULSE', 'magnitude': 1.15, 'duration': 3},
        # ... all 21 weak scenarios
    ]

    phase3_detections = 8  # Baseline from Phase 3
    phase4_detections = 0

    for scenario in weak_scenarios:
        # Inject attack
        attacked_data = inject_attack(base_forecast, scenario)

        # Detect with Phase 4 methods
        detections = hybrid_detection(attacked_data)

        # Check if correct
        if any(overlaps_attack(det, scenario) for det in detections):
            phase4_detections += 1

    improvement = (phase4_detections - phase3_detections) / 21 * 100
    print(f"Weak detection improvement: +{improvement:.1f}%")
    print(f"Phase 3: {phase3_detections}/21 ({phase3_detections/21*100:.1f}%)")
    print(f"Phase 4: {phase4_detections}/21 ({phase4_detections/21*100:.1f}%)")

    assert phase4_detections >= 13, "Must detect at least 13/21 (60%+) weak attacks"
```

### Step 3: Regression Testing

Ensure Phase 4 changes don't break existing functionality:

```python
def test_strong_attack_detection():
    """Verify 100% strong attack detection maintained."""
    strong_scenarios = [
        {'magnitude': 3.0, 'duration': 1},  # 200%
        {'magnitude': 5.0, 'duration': 3},  # 400%
        {'magnitude': 10.0, 'duration': 1}  # 900%
    ]

    for scenario in strong_scenarios:
        attacked_data = inject_attack(base_forecast, scenario)
        detections = hybrid_detection(attacked_data)

        assert len(detections) > 0, f"Must detect strong attack {scenario}"
        assert overlaps_attack(detections[0], scenario), "Must correctly locate strong attack"

    print("✅ Strong attack detection maintained at 100%")

def test_false_positive_rate():
    """Verify FPR remains <1%."""
    # Test on clean data (no attacks)
    clean_windows = generate_clean_forecasts(num_windows=20)
    false_positives = 0

    for window in clean_windows:
        detections = hybrid_detection(window)
        if len(detections) > 0:
            false_positives += 1

    fpr = false_positives / len(clean_windows)
    print(f"False Positive Rate: {fpr*100:.2f}%")
    assert fpr < 0.01, "FPR must remain below 1%"
```

---

## 🎯 SUCCESS CRITERIA FOR PHASE 4

### Minimum Acceptable Performance (Must Pass)

| Metric                    | Phase 3 Baseline | Phase 4 Target | Status         |
| ------------------------- | ---------------- | -------------- | -------------- |
| Weak Attack Detection     | 38.1%            | **>60%**       | ⏳ IN PROGRESS |
| Medium Attack Detection   | 60.0%            | >55%           | ⏳ IN PROGRESS |
| Strong Attack Detection   | 100%             | 100%           | ⏳ IN PROGRESS |
| False Positive Rate       | <1%              | <1%            | ⏳ IN PROGRESS |
| Overall Correct Detection | 56.1%            | >65%           | ⏳ IN PROGRESS |

### Stretch Performance (Highly Desired)

| Metric                    | Phase 4 Stretch | Status         |
| ------------------------- | --------------- | -------------- |
| Weak Attack Detection     | **>70%**        | ⏳ IN PROGRESS |
| Medium Attack Detection   | >75%            | ⏳ IN PROGRESS |
| Overall Correct Detection | >75%            | ⏳ IN PROGRESS |

---

## 🚨 CRITICAL WARNINGS & CONSTRAINTS

### DO NOT:

1. ❌ **Break backward compatibility** - existing code must still work with max_detections=1
2. ❌ **Increase false positive rate** - must stay <1%
3. ❌ **Degrade strong attack detection** - must maintain 100%
4. ❌ **Remove emergency mode** - critical safety feature (>50% instant detection)
5. ❌ **Lower MAGNITUDE_THRESHOLD below 0.07** - this is the absolute noise floor

### DO:

1. ✅ **Test each component independently** before integration
2. ✅ **Run regression tests** after each major change
3. ✅ **Document all parameter changes** with rationale
4. ✅ **Profile performance** - ensure <2 seconds per scenario
5. ✅ **Version control aggressively** - commit after each working feature

---

## 📊 EXPECTED RESULTS AFTER PHASE 4

If Priority 1 implementations are successful:

```
================================================================================
PHASE 4 RESULTS - TARGET PERFORMANCE
================================================================================

📊 OVERALL DETECTION RATES
--------------------------------------------------------------------------------
  Accuracy:     99.70%+ (maintained)
  Precision:    >50% (improvement from 32%)
  Recall:       >50% (improvement from 30%)
  F1-Score:     >0.50 (improvement from 0.31)
  Specificity:  >99.8% (maintained)
  FPR:          <0.5% (maintained)

🎯 ATTACK-LEVEL DETECTION RATES
--------------------------------------------------------------------------------
  Total Scenarios:        57
  Attacks Detected:       50+ (88%+)
  Correct Detections:     40+ (70%+)

📊 DETECTION RATE BY MAGNITUDE CATEGORY
--------------------------------------------------------------------------------
Category        Baseline   Phase 3    Phase 4    Improvement
--------------------------------------------------------------------------------
WEAK            4.8%       38.1%      65%+       14x  🎯 TARGET ACHIEVED
MEDIUM          80.0%      60.0%      75%+       0.94x
STRONG          100%       100%       100%       1x

⏱️  DETECTION RATE BY DURATION CATEGORY
--------------------------------------------------------------------------------
Category        Baseline   Phase 3    Phase 4    Improvement
--------------------------------------------------------------------------------
SHORT           46.7%      66.7%      75%+       1.6x
MEDIUM          58.3%      50.0%      70%+       1.2x
LONG            40.0%      33.3%      80%+       2.0x
```

---

## 📁 FILES TO MODIFY/CREATE

### Priority 1 Files (Critical Path)

1. **`mlad_anomaly_detection.py`** (MODIFY)

   - Add `statistical_anomaly_detection()` function
   - Add `hybrid_detection()` function
   - Add `adaptive_threshold_detection()` function
   - Import scipy.stats

2. **`sliding_window_evaluation.py`** (MODIFY)

   - Add `evaluate_window_sizes()` function
   - Make window_size configurable parameter
   - Integrate hybrid detection option

3. **`config.py`** (MODIFY - minimal changes)
   - Add ADAPTIVE_THRESHOLD_ENABLED flag
   - Add STATISTICAL_DETECTION_ALPHA parameter
   - Add OPTIMAL_WINDOW_SIZE (after experiments)

### Priority 2 Files (If needed for 70%+)

4. **`pattern_features.py`** (CREATE NEW)

   - `calculate_pattern_features()` function
   - `calculate_attack_likelihood()` function
   - Pattern-based detection pipeline

5. **`ensemble_forecaster.py`** (CREATE NEW)
   - Ensemble forecasting implementation
   - Multi-horizon model management
   - Forecast error tracking

---

## 🎓 TECHNICAL DEEP DIVE

### Why Statistical Testing Will Work

Current DP approach:

```python
# Finds interval with max score
# Problem: Natural 20% spike scores higher than weak 15% attack
score_natural_20 = f(0.20) * 5_hours = 4.0
score_attack_15 = f(0.15) * 3_hours = 2.5
# Result: Natural spike wins
```

Statistical approach:

```python
# Tests if interval is significantly different from normal
normal_mean = 0.08  # 8% baseline noise
attack_mean = 0.15  # 15% attack
natural_mean = 0.20  # 20% natural spike

# T-test results:
attack_p_value = 0.001  # Highly significant (different from 0.08)
natural_p_value = 0.02  # Less significant (varies around 0.08)

# With Bonferroni correction (alpha=0.01/10 = 0.001):
attack_significant = (0.001 < 0.001) = True  ✅
natural_significant = (0.02 < 0.001) = False ❌

# Result: Attack detected, natural spike filtered
```

**Key Insight**: Statistical tests measure "how unlikely is this under normal conditions" rather than "how high is the magnitude". Weak attacks are statistically unlikely even if magnitudes are similar to natural variation.

---

## 🔚 SUMMARY

**Your Mission**: Implement Priority 1 optimizations to achieve 60%+ weak attack detection.

**Critical Path**:

1. Statistical hypothesis testing (+12-20%)
2. Window size optimization (+5-10%)
3. Forecast quality improvement (+5-10%)

**Expected Timeline**: 2-3 weeks for Priority 1, optional 1-2 weeks for Priority 2

**Success Metric**: Weak attack detection >60% while maintaining strong detection at 100% and FPR <1%

**The power grid needs your expertise. Let's break the 60% barrier.** ⚡🛡️

---

## 📞 QUESTIONS & SUPPORT

If anything is unclear:

1. Review Phase 3 results: `PHASE3_RESULTS_SUMMARY.md`
2. Check baseline performance: `evaluation_summary.txt`
3. Examine current implementation: `mlad_anomaly_detection.py`, `sliding_window_evaluation.py`
4. Test scenarios: `comprehensive_model_evaluation.py`

**All files are available in the workspace. You have full access to modify any file as needed.**

Good luck with Phase 4! 🚀

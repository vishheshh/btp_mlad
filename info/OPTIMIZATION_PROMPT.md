# COMPREHENSIVE PROMPT FOR MLAD POWER GRID PROTECTION SYSTEM OPTIMIZATION

## 🎯 MISSION

You are tasked with optimizing a Machine Learning-based Anomaly Detection (MLAD) system for power grid protection. The system currently has **CRITICAL VULNERABILITIES** in detecting weak/subtle attacks while maintaining excellent specificity. Your goal is to improve detection rates to **production-ready levels (>80% correct detection)** while preserving the low false positive rate.

---

## 📊 CURRENT PERFORMANCE METRICS (BASELINE)

### **Overall Hour-Level Classification Metrics**

```
Accuracy:      99.69% ✅ (Excellent - but misleading due to class imbalance)
Precision:     34.48% ❌ (Poor - only 1 in 3 alerts is correct)
Recall:        27.78% ❌ (Critical - missing 72% of attack hours)
F1-Score:      0.3077 ❌ (Very Poor)
Specificity:   99.87% ✅ (Excellent - very few false alarms)
FPR:           0.13%  ✅ (Excellent - low false positive rate)
```

### **Confusion Matrix (117,500 total hours across 47 scenarios)**

```
                    Predicted Normal    Predicted Attack
Actual Normal            117,060              152 (False Positives)
Actual Attack               208               80 (True Positives)
```

### **Attack-Level Detection Performance**

```
Total Scenarios:        47
Attacks Detected:       47 (100.0%) ✅ Every attack triggered SOMETHING
Correct Detections:     22 (46.8%)  ❌ But only half were correctly identified
```

### **⚠️ CRITICAL VULNERABILITY: Detection by Magnitude**

```
WEAK Attacks:     4.8% correct  ❌❌❌ CATASTROPHIC - 1 out of 21 detected correctly
MEDIUM Attacks:   75.0% correct ⚠️  Acceptable but needs improvement
STRONG Attacks:   100% correct  ✅ Perfect for severe attacks
```

### **Detection by Duration**

```
SHORT (1-3h):     46.7% correct
MEDIUM (6-12h):   50.0% correct
LONG (24h):       40.0% correct  ⚠️ Counterintuitive - should be EASIER to detect
```

### **Detection by Attack Type**

```
PULSE Attacks:    50.0% correct (9 out of 18)
SCALING Attacks:  44.8% correct (13 out of 29)
```

---

## 🔧 SYSTEM ARCHITECTURE & DETECTION METHODOLOGY

### **Three-Stage Pipeline**

#### **Stage 1: Load Forecasting (LSTM Neural Network)**

- **Model**: Sequential LSTM with 64 units → Dropout → 32 units → Dropout → Dense(1)
- **Input Features**:
  - Lag features: 1 hour, 24 hours, 168 hours (1 week)
  - Time features: hour, day_of_week, day_of_year, month, is_weekend
  - Cyclical encoding: hour_sin/cos, month_sin/cos
- **Training**: 50 epochs, batch size 32, Adam optimizer (lr=0.001)
- **Purpose**: Predicts expected load for each hour
- **Performance**: Training MAE ~70 MWh, Testing MAE ~85 MWh

#### **Stage 2: Benchmark Profiling (K-Means Clustering)**

- **Model**: K-Means with 100 clusters
- **Input**: 24-hour daily load patterns (reshaped to [n_days, 24])
- **Purpose**: Establishes "normal" behavior profiles from historical data
- **Output**: For each forecast 24-hour window, finds nearest cluster centroid as benchmark

#### **Stage 3: Anomaly Detection (Two-Tier Dynamic Programming)**

**A. Data Preparation**

```python
scaling_data = forecast / (benchmark + 1e-6)  # Ratio of forecast to expected
deviation = |scaling_data - 1.0|              # Distance from normal (1.0)
```

**B. TWO-TIER DETECTION SYSTEM**

**EMERGENCY MODE** (Instant detection for catastrophic attacks):

```python
EMERGENCY_THRESHOLD = 0.50       # 50% deviation
EMERGENCY_MIN_DURATION = 1       # Alert after just 1 hour
```

- If `deviation > 0.50`, INSTANT ALERT
- No dynamic programming needed
- Used for grid-destroying attacks (500%+ spikes)
- **Working perfectly** - detected all strong attacks

**NORMAL MODE** (Balanced detection for subtle attacks):

```python
MAGNITUDE_THRESHOLD = 0.10       # 10% deviation (was 0.08, tuned from calculated 8%)
MIN_ANOMALY_DURATION = 3         # Requires 3 consecutive hours
LAMBDA_SCORE = 2.0               # Super-additive scoring parameter
MIN_ANOMALY_SCORE = 0.30         # Minimum cumulative score threshold
```

**Dynamic Programming Algorithm**:

```python
# Calculate base scores for each hour
base_scores = where(deviation > MAGNITUDE_THRESHOLD,
                    deviation - MAGNITUDE_THRESHOLD,
                    0)

# Apply super-additive function (rewards consecutive anomalies)
scores = base_scores * (1 + LAMBDA_SCORE * base_scores)

# DP: Find maximum scoring interval
for i in range(n):
    option1 = scores[i]                    # Start new interval
    option2 = dp[i-1] + scores[i]          # Extend interval
    dp[i] = max(option1, option2)

# Accept detection if:
# - Duration >= MIN_ANOMALY_DURATION (3 hours)
# - Score >= MIN_ANOMALY_SCORE (0.30)
```

**C. INTERVAL SEGMENTATION** (Phase 1 Improvement)

```python
SEGMENT_GAP_HOURS = 3              # Break segments after 3 normal hours
MIN_SEGMENT_SCORE = 0.08           # Minimum score for segment validity
MIN_SEGMENT_DURATION_FOR_SPLIT = 10  # Only segment intervals >10 hours
```

- For long detected intervals (>10h), breaks them into segments
- Prevents single giant interval spanning multiple attacks
- Returns highest-scoring segment
- **Improved precision from ~20% to 34%**, but still insufficient

---

## 🧪 TEST SCENARIOS (47 Comprehensive Attack Patterns)

### **Magnitude Definitions**

```python
'weak':   [1.10, 1.15, 1.20]    # 10%, 15%, 20% increase
'medium': [1.30, 1.50, 2.0]     # 30%, 50%, 100% increase
'strong': [3.0, 5.0, 10.0]      # 200%, 400%, 900% increase
```

### **Duration Definitions**

```python
'short':  [1, 3] hours
'medium': [6, 12] hours
'long':   [24] hours
```

### **Attack Types**

1. **PULSE**: Sharp instantaneous spikes (`data *= magnitude`)
2. **SCALING**: Constant multiplication (`data *= scale_factor`)
3. **RAMPING**: Linear ramp from 1.0 to max (`data *= linspace(1.0, max, duration)`)
4. **RANDOM**: Erratic noise (`data *= uniform(1-noise, 1+noise)`)
5. **SMOOTH-CURVE**: Sinusoidal deviation (`data *= 1 + (amp-1) * sin(...)`)

### **Attack Injection Strategy**

- Base forecast: 2500 hours of clean test data
- Attacks injected starting at hour 100, spaced 50 hours apart
- 47 scenarios spanning all combinations of type/magnitude/duration
- Each scenario independently evaluated with ground truth labels

---

## ❌ ROOT CAUSE ANALYSIS: WHY WEAK ATTACK DETECTION FAILS

### **Problem 1: Threshold Too Conservative**

Current weak attacks have deviations of 10-20%, but:

```python
MAGNITUDE_THRESHOLD = 0.10  # Only catches deviations >10%
```

- A 10% attack (`scaling_data = 1.10`) has `deviation = 0.10`
- This is **EXACTLY** at threshold, giving `base_score = 0.0` (no detection)
- A 15% attack (`scaling_data = 1.15`) has `deviation = 0.15`
- This gives `base_score = 0.05` (very small)
- After 3 hours: `score ≈ 0.15` (below `MIN_ANOMALY_SCORE = 0.30`)

**Example from Results (Scenario 1: WEAK PULSE 10% for 1h)**:

```
max_deviation: 0.1518 (15.18%)
detected: True
correct_detection: False  ❌ (detected hours 1992-1995 instead of hour 100)
```

The model DID detect something, but at the **wrong location** (hour 1992 instead of 100).

### **Problem 2: Score Accumulation Insufficient**

```python
# For a 15% deviation (0.15):
base_score = 0.15 - 0.10 = 0.05
score = 0.05 * (1 + 2.0 * 0.05) = 0.055

# Over 3 hours:
total_score = 3 * 0.055 = 0.165  < 0.30 threshold ❌
```

The super-additive function helps, but not enough for weak attacks.

### **Problem 3: Duration Requirement Conflict**

```python
MIN_ANOMALY_DURATION = 3  # Requires 3 consecutive hours
```

- Many test attacks are 1 hour long
- Even if deviation is high, 1-hour attacks might not accumulate enough score
- But we can't lower this too much or false positives explode

### **Problem 4: Segmentation Over-Aggressive**

```python
SEGMENT_GAP_HOURS = 3  # Breaks after 3 normal hours
MIN_SEGMENT_SCORE = 0.08
```

- Long weak attacks (24h with 10% deviation) get broken into tiny segments
- Each segment scores below threshold, entire attack missed
- See Scenario 23: WEAK SCALING 10% for 24h → 0% correct detection

---

## 🚫 CONSTRAINTS & REQUIREMENTS

### **HARD CONSTRAINTS**

1. **Threshold Floor**: `MAGNITUDE_THRESHOLD >= 0.08`

   - **JUSTIFICATION**: The threshold of 0.08 (8%) was **calculated from historical data analysis**
   - Normal fluctuations in power grid data have standard deviation ≈8%
   - Going below 0.08 will cause **EXPLOSION of false positives**
   - This is a **DATA-DRIVEN constraint**, not arbitrary
   - **DO NOT suggest lowering threshold below 0.08** - this is NON-NEGOTIABLE

2. **False Positive Rate**: Must maintain FPR < 1% (currently 0.13% ✅)

   - Critical for operator trust in production
   - Aim to keep specificity > 99%

3. **Strong Attack Detection**: Must remain 100% (currently ✅)
   - Cannot sacrifice detection of catastrophic attacks
   - Emergency mode must remain intact

### **SOFT CONSTRAINTS**

1. **Computational Cost**: Keep inference time <1 second per scenario
2. **Model Complexity**: Prefer parameter tuning over architecture changes
3. **Code Maintainability**: Solutions should be explainable to grid operators

---

## 🎯 OPTIMIZATION OBJECTIVES (PRIORITIZED)

### **PRIMARY GOAL: Weak Attack Detection**

- **Target**: Increase weak attack correct detection from 4.8% to >60%
- **Critical**: This is a **SECURITY VULNERABILITY** - sophisticated attackers use subtle methods
- **Acceptable Tradeoff**: Can increase FPR slightly (up to 0.5%) if needed

### **SECONDARY GOAL: Improve Precision**

- **Target**: Increase precision from 34.48% to >60%
- **Benefit**: Reduces alert fatigue, increases operator confidence
- **Approach**: Better localization of attack boundaries

### **TERTIARY GOAL: Improve Recall**

- **Target**: Increase recall from 27.78% to >50%
- **Benefit**: Catch more attack hours, better damage assessment
- **Note**: Less critical than correct detection (detecting THAT attack exists matters more than catching every hour)

### **QUATERNARY GOAL: Long Attack Detection**

- **Target**: Improve long attack detection from 40% to >70%
- **Benefit**: Sustained attacks should be easier to detect
- **Issue**: Currently worse than short attacks (counterintuitive)

---

## 💡 SUGGESTED INVESTIGATION AREAS

### **Area 1: Scoring Function Redesign**

Current scoring is linear-ish. Consider:

- **Exponential scoring** for weak deviations to amplify small signals
- **Adaptive thresholds** based on attack duration (lower for longer attacks)
- **Cumulative deviation** scoring instead of individual hour scores
- **Percentile-based scoring** relative to historical deviation distribution

Example Alternative:

```python
# Exponential amplification for weak signals
if deviation < 0.20:  # Weak/medium range
    amplification_factor = exp(deviation / 0.10)  # Grows exponentially
    score = base_score * amplification_factor * (1 + LAMBDA * base_score)
```

### **Area 2: Multi-Scale Detection**

Run detection at multiple parameter scales:

- **Sensitive mode**: Lower thresholds for longer durations
  - If `potential_duration >= 12h`, use `threshold = 0.08`
  - If `potential_duration >= 6h`, use `threshold = 0.09`
  - If `potential_duration < 6h`, use `threshold = 0.10`
- **Aggregate results** from multiple passes

### **Area 3: Statistical Envelope Detection**

Instead of fixed threshold, use rolling statistics:

```python
# Calculate rolling mean and std of deviation
window_mean = rolling_mean(deviation, window=24)
window_std = rolling_std(deviation, window=24)

# Score based on how many standard deviations above mean
z_score = (deviation - window_mean) / (window_std + 1e-6)
anomaly_score = where(z_score > 2.0, z_score, 0)  # 2-sigma rule
```

### **Area 4: Duration-Aware Scoring**

Penalize short detections, reward sustained anomalies:

```python
# After finding interval [start, end]:
duration = end - start + 1
duration_bonus = log(1 + duration / MIN_ANOMALY_DURATION)
final_score = base_score * duration_bonus
```

### **Area 5: Segmentation Refinement**

Current segmentation breaks too aggressively:

- **Adaptive gap threshold**: Use larger gap for weak attacks
  ```python
  if max(deviation) < 0.20:  # Weak attack suspected
      min_gap = 5  # Require longer break
  else:
      min_gap = 3
  ```
- **Score-weighted segmentation**: Keep segments connected if combined score > threshold
- **Hierarchical segmentation**: Find all segments, then merge nearby ones

### **Area 6: Feature Engineering for Forecaster**

Improve forecast accuracy to reduce false deviations:

- Add **seasonal decomposition** features (trend, seasonal, residual)
- Include **weather correlation** features if available
- Use **attention mechanism** in LSTM to focus on relevant time steps
- Implement **ensemble forecasting** (average multiple models)

### **Area 7: Benchmark Model Enhancement**

Improve benchmark to be more representative:

- Increase `N_CLUSTERS` from 100 to 200-500 (finer-grained profiles)
- Use **hierarchical clustering** to capture nested patterns
- Implement **time-weighted clustering** (recent data matters more)
- Use **Gaussian Mixture Models** instead of hard K-Means

### **Area 8: Post-Processing Filters**

After detection, validate with secondary checks:

- **Trend analysis**: Does attack show consistent pattern?
- **Magnitude consistency**: Are all hours above threshold?
- **Temporal coherence**: Are attack hours temporally close?
- **Score distribution**: Is score concentrated or spread?

---

## 📁 CODEBASE STRUCTURE

```
power_grid_protection/
├── config.py                              # All parameters defined here
├── mlad_anomaly_detection.py             # Core detection logic
├── comprehensive_model_evaluation.py      # Evaluation script (47 scenarios)
├── models/
│   ├── load_forecaster.h5                # Trained LSTM model
│   ├── scaler.pkl                        # MinMaxScaler for features
│   └── kmeans_model.pkl                  # K-Means benchmark model
├── dataset/                              # 97 CSV files (34,702 hours total)
├── evaluation_results.csv                # Detailed per-scenario results
└── evaluation_summary.txt                # Aggregate metrics
```

**Key Functions to Modify**:

1. `detect_anomaly_timing()` in `mlad_anomaly_detection.py` (lines 310-403)
   - Main DP algorithm for normal mode detection
2. `segment_anomaly_interval()` in `mlad_anomaly_detection.py` (lines 406-478)
   - Interval segmentation logic
3. `detect_anomaly_with_segmentation()` in `mlad_anomaly_detection.py` (lines 481-522)
   - Wrapper combining detection and segmentation

**Key Parameters in `config.py`**:

```python
# Line 38: Normal detection threshold (CANNOT GO BELOW 0.08)
MAGNITUDE_THRESHOLD = 0.10

# Line 39: Minimum consecutive hours for detection
MIN_ANOMALY_DURATION = 3

# Line 46: Super-additive scoring parameter
LAMBDA_SCORE = 2.0

# Line 47: Minimum cumulative score to accept detection
MIN_ANOMALY_SCORE = 0.30

# Line 50: Gap hours to break segments
SEGMENT_GAP_HOURS = 3

# Line 51: Minimum segment score
MIN_SEGMENT_SCORE = 0.08

# Line 52: Minimum duration to trigger segmentation
MIN_SEGMENT_DURATION_FOR_SPLIT = 10
```

---

## 🔬 EVALUATION METHODOLOGY

Run comprehensive evaluation after each change:

```bash
python comprehensive_model_evaluation.py
```

This will:

1. Load trained models (forecaster, scaler, kmeans)
2. Generate base forecast on 2500-hour test period
3. Inject 47 attack scenarios with known ground truth
4. Run detection on each scenario
5. Calculate hour-level metrics (TP, TN, FP, FN, precision, recall, F1)
6. Calculate attack-level detection rates (detected, correct)
7. Export `evaluation_results.csv` and `evaluation_summary.txt`

**Evaluation Runtime**: ~30-60 seconds total

**Success Metrics to Monitor**:

- Weak attack correct detection rate (PRIMARY)
- Overall precision (watch for degradation)
- FPR / Specificity (must stay < 1% / > 99%)
- Strong attack detection (must stay 100%)

---

## 📊 EXPECTED RESULTS AFTER OPTIMIZATION

### **Target Performance (Production-Ready)**

```
Overall Metrics:
- Precision:    >60%  (vs 34.48% baseline)
- Recall:       >50%  (vs 27.78% baseline)
- F1-Score:     >0.55 (vs 0.31 baseline)
- Specificity:  >99%  (vs 99.87% baseline) ✅ maintain
- FPR:          <1%   (vs 0.13% baseline) ⚠️ allow slight increase

Attack Detection:
- Correct Detection: >80% (vs 46.8% baseline)

By Magnitude:
- WEAK:   >60% (vs 4.8% baseline)  ❗PRIMARY GOAL
- MEDIUM: >85% (vs 75% baseline)
- STRONG: 100% (vs 100% baseline) ✅ maintain

By Duration:
- SHORT:  >70% (vs 46.7% baseline)
- MEDIUM: >80% (vs 50% baseline)
- LONG:   >75% (vs 40% baseline)
```

---

## 💻 YOUR TASK

1. **Analyze** the current detection methodology and identify the root causes of weak attack detection failure (beyond what's listed here)

2. **Propose** specific, actionable modifications to the codebase:

   - Parameter changes in `config.py`
   - Algorithm changes in detection functions
   - New features or post-processing steps

3. **Implement** changes iteratively:

   - Start with lowest-risk, highest-impact changes
   - Test after each change using `comprehensive_model_evaluation.py`
   - Document results and rationale

4. **Optimize** for the objectives in priority order:

   - PRIMARY: Weak attack detection >60%
   - SECONDARY: Precision >60%
   - TERTIARY: Recall >50%
   - Maintain FPR <1% and strong attack detection 100%

5. **Validate** that solutions:

   - Respect the MAGNITUDE_THRESHOLD >= 0.08 constraint
   - Are explainable to non-technical grid operators
   - Don't significantly increase computational cost
   - Generalize across attack types and durations

6. **Document** all changes with:
   - Clear rationale for each modification
   - Before/after metrics comparison
   - Trade-offs and limitations
   - Recommendations for production deployment

---

## 🚨 CRITICAL REMINDERS

1. **DO NOT lower MAGNITUDE_THRESHOLD below 0.08** - this was calculated from data analysis and represents real noise floor

2. **Prioritize weak attack detection** - this is a security vulnerability, not just a metrics problem

3. **Maintain low FPR** - false alarms destroy operator trust in production systems

4. **Test frequently** - run evaluation after every change to catch regressions early

5. **Think like an attacker** - sophisticated adversaries use subtle, prolonged attacks to avoid detection

6. **Consider operational context** - power grid operators need explainable, trustworthy alerts

---

## 📎 ATTACHED FILES CONTEXT

The evaluation output shows:

- ✅ All 47 scenarios were detected (something triggered)
- ❌ Only 22 were correctly classified
- ❌ Weak attacks almost completely missed (1/21)
- ✅ Emergency mode working (12 emergency alerts for strong attacks)
- ⚠️ Many false positives around hour 1992-1995 (suspicious - investigate why)

**Start your optimization work now. Good luck!** 🚀

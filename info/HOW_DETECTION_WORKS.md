# How the Model Knows What's "Normal" - Complete Explanation

## Your Question 🤔

**"When we test attacks, how does the model know what is normal? What does it compare to?"**

Great question! Let me show you the COMPLETE flow:

---

## The Two Sources of "Normal" 📊

### 1. **LSTM Forecaster** = "What SHOULD happen"

Trained on historical data to predict:

- "Monday 3 PM usually has 14,000 MWh"
- "Summer weekdays peak at 16,000 MWh"
- "Winter nights drop to 11,000 MWh"

### 2. **K-Means Benchmark** = "What's TYPICAL"

Trained on historical 24-hour patterns:

- 100 clusters of "normal" daily profiles
- Each cluster = a typical day pattern
- Example: Cluster 23 = "Hot summer weekday"

---

## During Testing: Step-by-Step 🔍

### Step 1: Generate FORECAST (Prediction)

```python
# test_attack_types.py, line 193
base_forecast = forecaster.predict(X_test_reshaped).flatten()

# This creates predictions like:
# Hour 0: 12,000 MWh (predicted)
# Hour 1: 13,000 MWh (predicted)
# Hour 2: 14,500 MWh (predicted)
# ...
```

**What is this?** The LSTM's prediction of what SHOULD happen based on:

- Recent history (lag features)
- Time of day/week/year
- Seasonal patterns

---

### Step 2: INJECT Attack (Simulation)

```python
# test_attack_types.py, line 257
attacked_forecast = simulator.pulse_attack(base_forecast, start=20, magnitude=5.0)

# This modifies the forecast:
# Hour 0: 12,000 MWh (unchanged)
# ...
# Hour 20: 14,000 → 70,000 MWh (ATTACKED! 5x multiplier)
# Hour 21: 13,500 MWh (back to normal)
```

**Important:** We're NOT attacking the real data! We're attacking the FORECAST to simulate "what if an attack happened?"

---

### Step 3: Get BENCHMARK (Normal Pattern)

```python
# test_attack_types.py, line 263
benchmark = get_benchmark_for_period(kmeans, attacked_forecast)

# This retrieves the "normal" pattern:
# Hour 0: 12,500 MWh (typical for this hour)
# Hour 1: 13,200 MWh (typical)
# Hour 2: 14,000 MWh (typical)
# ...
# Hour 20: 15,000 MWh (typical) ← THIS is what's "normal"
# Hour 21: 14,800 MWh (typical)
```

**What is this?** K-means finds the closest "normal" 24-hour pattern from the 100 clusters it learned during training.

---

### Step 4: COMPARE (Detection)

```python
# test_attack_types.py, line 264
scaling_data = attacked_forecast / benchmark

# Hour 20 comparison:
# attacked_forecast[20] = 70,000 MWh (with attack)
# benchmark[20] = 15,000 MWh (normal)
# scaling_data[20] = 70,000 / 15,000 = 4.67
# Deviation = |4.67 - 1.0| = 3.67 = 367%! 🚨

# The system sees:
# "This should be 15,000 MWh (normal), but it's 70,000!"
# "That's 367% higher than normal!"
# "EMERGENCY ALERT!" ⚡
```

---

## Visual Example 📈

### Training Phase (Learning "Normal"):

```
Historical Data (2021-2025):
Monday 3 PM → Usually 14,000-15,000 MWh
Tuesday 3 PM → Usually 14,200-15,200 MWh
Wednesday 3 PM → Usually 13,800-15,400 MWh

LSTM learns: "3 PM on weekdays ≈ 14,500 MWh"
K-means learns: "Typical weekday pattern has 3 PM at 14,800 MWh"
```

### Testing Phase (Detecting Attacks):

```
Test Day: Wednesday 3 PM

1. LSTM Forecast: "I predict 14,500 MWh"
2. We INJECT attack: 14,500 → 70,000 MWh (simulated)
3. K-means Benchmark: "Normal for Wednesday 3 PM is 14,800 MWh"
4. Comparison:
   - Attacked forecast: 70,000 MWh
   - Normal benchmark: 14,800 MWh
   - Ratio: 70,000 / 14,800 = 4.73
   - Deviation: 373%! 🚨
5. ALERT: "This is NOT normal!"
```

---

## Why TWO Sources of Normal? 🤔

### LSTM Forecast (Dynamic)

- **Purpose**: Adapts to recent trends
- **Strength**: Considers recent history, weather patterns, etc.
- **Example**: "Yesterday was hot, so today's load will be higher"

### K-Means Benchmark (Static)

- **Purpose**: Represents long-term typical patterns
- **Strength**: Stable baseline that doesn't change with short-term fluctuations
- **Example**: "Wednesday 3 PM typically looks like THIS pattern"

### Why Compare Them?

```
Normal situation:
  Forecast ≈ Benchmark (both around 14,500 MWh)
  Ratio ≈ 1.0 ✅

Attack situation:
  Forecast = 70,000 MWh (attacked)
  Benchmark = 14,800 MWh (still normal)
  Ratio = 4.73 (way off!) 🚨
```

---

## Code Walkthrough: Where "Normal" Comes From

### Training Phase (mlad_anomaly_detection.py):

**Training LSTM (lines 174-248):**

```python
def train_forecaster(df):
    # Historical data from 2021-2025
    df_features = create_features(df)

    # 80% for training
    split_idx = int(len(X) * 0.8)
    X_train = X[:split_idx]
    y_train = y[:split_idx]  # This is the "normal" load

    # LSTM learns patterns from this normal data
    model.fit(X_train, y_train)

    # Now LSTM knows: "For these features, expect THIS load"
```

**Training K-means (lines 251-302):**

```python
def train_kmeans_benchmark(df):
    # Use training data (assumed normal)
    train_data = df.iloc[:split_idx]

    # Reshape into 24-hour daily patterns
    daily_patterns = load_values.reshape(-1, 24)
    # Example patterns:
    # Day 1: [12000, 11500, 11000, ..., 15000, 14500, ...]
    # Day 2: [12200, 11700, 11200, ..., 15200, 14700, ...]
    # ...

    # K-means clusters these into 100 groups
    kmeans.fit(daily_patterns)

    # Now K-means knows: "These are the 100 typical day patterns"
```

---

### Testing Phase (test_attack_types.py):

**Generate Forecast:**

```python
# Line 193
base_forecast = forecaster.predict(X_test_reshaped).flatten()

# LSTM uses its trained knowledge:
# Input features: [lag_1=14000, lag_24=14200, hour=15, day_of_week=2, ...]
# Output prediction: 14,500 MWh
#
# This is what the LSTM thinks SHOULD happen based on:
# - Recent load (14,000 MWh last hour)
# - Yesterday same time (14,200 MWh)
# - Time patterns (3 PM on Wednesday)
```

**Inject Attack:**

```python
# Line 257
attacked_forecast = simulator.pulse_attack(base_forecast, start=20, magnitude=5.0)

# We're simulating:
# "What if at hour 20, an attacker multiplied the load by 5?"
#
# Original forecast[20] = 14,500 MWh
# Attacked forecast[20] = 14,500 * 5 = 72,500 MWh
```

**Get Benchmark:**

```python
# Line 263
benchmark = get_benchmark_for_period(kmeans, attacked_forecast)

# K-means looks at the forecast pattern and says:
# "This looks most like Cluster #47: Typical Wednesday pattern"
# Cluster #47 has: hour 20 = 14,800 MWh
#
# So benchmark[20] = 14,800 MWh (what's NORMAL for hour 20)
```

**Compare:**

```python
# Line 264
scaling_data = attacked_forecast / benchmark

# scaling_data[20] = 72,500 / 14,800 = 4.90
# deviation = |4.90 - 1.0| = 3.90 = 390%
#
# System: "390% deviation! Way beyond 50% emergency threshold!"
# Result: 🚨 EMERGENCY ALERT!
```

---

## The Key Insight 💡

**"Normal" = What K-means learned from 4 years of historical data**

When you run tests:

1. ❌ You're NOT comparing to the current real-time data
2. ❌ You're NOT comparing to what actually happened
3. ✅ You're comparing to what the **trained K-means clusters say is typical**

**Example:**

```
Real data on 2024-12-17 at 3 PM: 14,200 MWh (doesn't matter for test)
LSTM forecast for test: 14,500 MWh (predicted)
We inject attack: 14,500 → 72,500 MWh (simulated attack)
K-means benchmark: 14,800 MWh (learned from 2021-2025 history)

Comparison: 72,500 / 14,800 = 4.90 (490% of normal!)
```

---

## Why This Works 🎯

### The K-Means "Memory" of Normal:

During training, K-means saw patterns like:

```
Cluster 23 (Summer weekdays):
Hour 0:  11,500 MWh
Hour 1:  11,200 MWh
...
Hour 15: 16,200 MWh  ← "Normal" for 3 PM in summer
...
Hour 23: 13,000 MWh

Cluster 47 (Winter weekdays):
Hour 0:  12,800 MWh
Hour 1:  12,500 MWh
...
Hour 15: 14,800 MWh  ← "Normal" for 3 PM in winter
...
Hour 23: 13,500 MWh
```

During testing, when you test Wednesday 3 PM in December:

- K-means says: "This looks like Cluster 47 (Winter weekday)"
- Benchmark = Cluster 47's pattern
- Hour 15 benchmark = 14,800 MWh ← **This is what's "normal"**

---

## Real-World Deployment Difference 🌍

### In Testing (Simulation):

```python
# We inject attacks into FORECAST
attacked_forecast = forecast * 5.0  # Simulated attack

# Compare to learned normal
deviation = attacked_forecast / benchmark
```

### In Production (Real Grid):

```python
# Real-time reading from SCADA
current_load = scada.get_reading()  # Actual current load

# Get expected from LSTM
expected_load = forecaster.predict(features)

# Get normal from K-means
benchmark = kmeans.get_benchmark(current_time)

# Compare
deviation = current_load / benchmark

# If attacker manipulates SCADA:
# current_load = 70,000 MWh (real attack!)
# benchmark = 14,800 MWh (normal)
# deviation = 4.73 → ALERT!
```

---

## Summary: The Three "Loads" 📊

| Type          | Source           | Purpose                | Example    |
| ------------- | ---------------- | ---------------------- | ---------- |
| **FORECAST**  | LSTM prediction  | What SHOULD happen now | 14,500 MWh |
| **BENCHMARK** | K-means clusters | What's TYPICAL/NORMAL  | 14,800 MWh |
| **ATTACKED**  | Injected anomaly | Simulated attack       | 72,500 MWh |

**Detection Logic:**

```python
if (attacked / benchmark) > threshold:
    ALERT!  # Not normal!
```

---

## Your Question Answered ✅

**Q: "How does the model know what is normal?"**

**A:** The K-means clusters (trained on 4 years of data) remember 100 different "normal" daily patterns. During testing, it finds the closest matching pattern and uses that as the baseline for "normal."

**Q: "What does it compare to?"**

**A:** It compares:

- **What we're testing** (forecast with injected attack)
- **Against** what K-means says is normal (learned from history)

**Key:** The "normal" is **frozen in time** from training. It doesn't change during testing - that's how we can detect when things deviate!

---

## Analogy 🏠

Think of it like a home security system:

**Training:**

- Camera learns: "Normal = 2 people in house, dog walking around"

**Testing/Deployment:**

- New video shows: "5 people in house!" (attack injected/real intrusion)
- Compare: 5 people vs. normal 2 people
- Alert: "Intrusion detected!" 🚨

The system doesn't need to know if the current situation is "truly normal" - it just compares to what it learned was normal during training!

---

**Bottom line:** K-means is your "memory of normal" - it remembers what 2021-2025 looked like and compares everything against that! 🎯

# Where is Time Information During Testing? 🕐

## Your Question 🤔

**"Where are we telling time in the testing file while injecting attacks?"**

Great observation! The time information is **implicit**, not explicit. Let me show you where it's hidden:

---

## The Answer: Time is in the FEATURES! ⭐

### Step-by-Step Time Flow:

```python
# test_attack_types.py, Line 184
test_data = df_features.iloc[split_idx:split_idx+168].copy()
```

**What's in test_data?**

```
test_data has these columns:
├─ load              (actual load values)
├─ load_lag_1        (load 1 hour ago)
├─ load_lag_24       (load 24 hours ago)
├─ load_lag_168      (load 1 week ago)
├─ hour              (0-23, actual hour)
├─ day_of_week       (0-6, Monday=0)
├─ day_of_year       (1-365)
├─ month             (1-12)
├─ is_weekend        (0 or 1)
├─ hour_sin          (sin encoding of hour) ⭐
├─ hour_cos          (cos encoding of hour) ⭐
├─ month_sin         (sin encoding of month) ⭐
└─ month_cos         (cos encoding of month) ⭐

PLUS: datetime index (e.g., 2024-03-15 14:00:00)
```

**Key insight:** The test_data already knows what time each row represents!

---

## How Time Travels Through the System 🚀

### Stage 1: Test Data Has Timestamps

```python
# test_attack_types.py, line 184
test_data = df_features.iloc[split_idx:split_idx+168].copy()

# Example of what test_data looks like:
#
# datetime            | load  | hour | day_of_week | hour_sin | hour_cos | ...
# --------------------|-------|------|-------------|----------|----------|----
# 2024-03-15 00:00:00 | 12500 |  0   |     4       |  0.000   |  1.000   | ...
# 2024-03-15 01:00:00 | 12200 |  1   |     4       |  0.259   |  0.966   | ...
# 2024-03-15 02:00:00 | 11800 |  2   |     4       |  0.500   |  0.866   | ...
# ...
# 2024-03-15 20:00:00 | 14500 | 20   |     4       | -0.866   | -0.500   | ...  ← Hour 20!
```

---

### Stage 2: LSTM Gets Time Features

```python
# test_attack_types.py, lines 186-193
feature_cols = [col for col in test_data.columns
               if col not in ['load', 'date', 'hour']]

X_test = test_data[feature_cols].values
# X_test contains: [load_lag_1, load_lag_24, load_lag_168,
#                   day_of_week, day_of_year, month, is_weekend,
#                   hour_sin, hour_cos, month_sin, month_cos]

base_forecast = forecaster.predict(X_test_reshaped).flatten()
```

**What happens here:**

```
LSTM receives for hour 20:
  Input: [load_lag_1=14200, load_lag_24=14000, ...,
          hour_sin=-0.866, hour_cos=-0.500, ...]  ← TIME INFO HERE!

  LSTM thinks: "Oh, this is hour 20 (8 PM) based on sin/cos values"
  LSTM predicts: 14,500 MWh (what's normal for 8 PM)
```

**The forecast VALUES already encode time information!**

---

### Stage 3: Inject Attack (No Time Needed!)

```python
# test_attack_types.py, line 257
attacked_forecast = simulator.pulse_attack(base_forecast, start=20, magnitude=5.0)

# What this does:
# attacked_forecast[20] = base_forecast[20] * 5.0
# attacked_forecast[20] = 14,500 * 5 = 72,500 MWh
```

**Key:** We're just modifying array index 20, which corresponds to:

- The 20th hour of the test window
- Which happens to be 2024-03-15 20:00:00 (from test_data index)
- But we don't need to pass the timestamp explicitly!

---

### Stage 4: K-Means Gets Pattern (Time is in the Pattern!)

```python
# test_attack_types.py, line 263
benchmark = get_benchmark_for_period(kmeans, attacked_forecast)

# Inside get_benchmark_for_period (mlad_anomaly_detection.py, line 430):
daily_pattern = forecast_values[0:24].reshape(1, -1)  # First 24 hours
cluster_label = kmeans_model.predict(daily_pattern)[0]
benchmark[0:24] = kmeans_model.cluster_centers_[cluster_label]
```

**What K-means does:**

```
Input: 24-hour pattern = [12500, 12200, 11800, ..., 14500, ...]
                          hour 0  hour 1  hour 2      hour 20

K-means: "This pattern looks like Cluster #47 (typical weekday)"
Output: Cluster #47's pattern = [12800, 12400, 12000, ..., 14800, ...]
                                 hour 0  hour 1  hour 2      hour 20

benchmark[20] = 14,800 MWh  ← THIS is "normal" for hour 20!
```

**K-means doesn't need explicit time - it just matches the PATTERN!**

---

## The Hidden Truth 💡

**Time is NEVER explicitly passed during injection/benchmarking!**

Instead:

1. ✅ Time is in the **datetime index** of test_data
2. ✅ Time is in the **features** (hour_sin, hour_cos, day_of_week, etc.)
3. ✅ LSTM learns "hour 20 typically has X load"
4. ✅ Forecast VALUES encode the time information
5. ✅ K-means matches the PATTERN (not the timestamp)

---

## Visual Flow 📊

```
┌─────────────────────────────────────────────────────────────┐
│ test_data (has datetime index + time features)             │
│ 2024-03-15 20:00:00 | hour=20 | hour_sin=-0.866 | ...     │
└────────────────────┬────────────────────────────────────────┘
                     ↓
              ┌──────────────┐
              │ LSTM Model   │ ← Receives time features
              └──────┬───────┘
                     ↓
           base_forecast[20] = 14,500 MWh
           (LSTM used hour_sin/cos to know it's 8 PM)
                     ↓
              ┌──────────────┐
              │ INJECT ATTACK│ ← Just modifies index 20
              └──────┬───────┘
                     ↓
           attacked_forecast[20] = 72,500 MWh
                     ↓
              ┌──────────────┐
              │ K-Means      │ ← Matches 24-hour PATTERN
              └──────┬───────┘
                     ↓
           benchmark[20] = 14,800 MWh
           (Cluster #47's value for position 20)
                     ↓
              ┌──────────────┐
              │ COMPARE      │
              └──────┬───────┘
                     ↓
           72,500 / 14,800 = 4.90 → 390% deviation!
```

---

## Why This Works 🎯

### Time is Implicit in Position:

```python
# When we say "start=20" in the attack:
attacked_forecast[20]  # This is the 20th element

# But test_data knows this is:
test_data.index[20]  # datetime: 2024-03-15 20:00:00

# And LSTM knows this because:
test_data.iloc[20]['hour_sin']  # -0.866 (unique to hour 20)
test_data.iloc[20]['hour_cos']  # -0.500 (unique to hour 20)
```

### The Array Index IS the Time Reference!

```
Array index 0  → test_data.index[0]  → 2024-03-15 00:00:00
Array index 1  → test_data.index[1]  → 2024-03-15 01:00:00
...
Array index 20 → test_data.index[20] → 2024-03-15 20:00:00
```

---

## Example with Real Values 📈

Let's trace hour 20 specifically:

### 1. Test Data (Line 184):

```python
test_data.iloc[20]:
  datetime: 2024-03-15 20:00:00
  load: 14,300
  hour: 20
  hour_sin: -0.866
  hour_cos: -0.500
  day_of_week: 4 (Friday)
  month: 3 (March)
```

### 2. LSTM Forecast (Line 193):

```python
# LSTM receives features for position 20:
features[20] = [load_lag_1=14200, load_lag_24=14000, ...,
                hour_sin=-0.866, hour_cos=-0.500, ...]

# LSTM output:
base_forecast[20] = 14,500 MWh
# (LSTM learned: "When hour_sin=-0.866 and hour_cos=-0.500,
#  that's 8 PM, which typically has ~14,500 MWh")
```

### 3. Attack Injection (Line 257):

```python
# Multiply position 20 by 5.0:
attacked_forecast[20] = 14,500 * 5.0 = 72,500 MWh

# We don't need to say "8 PM" - position 20 IS 8 PM!
```

### 4. K-Means Benchmark (Line 263):

```python
# K-means gets 24-hour pattern:
pattern = [12500, 12200, ..., 72500, ...]  # Position 20 = 72,500
          hour 0  hour 1      hour 20

# K-means finds matching cluster:
# "This looks like Cluster #47 (typical Friday)"
# Cluster #47 says: hour 20 typically = 14,800 MWh

benchmark[20] = 14,800 MWh  # Position 20 = "normal" for 8 PM
```

### 5. Comparison:

```python
scaling_data[20] = attacked_forecast[20] / benchmark[20]
                 = 72,500 / 14,800
                 = 4.90

deviation = |4.90 - 1.0| = 3.90 = 390%!
```

---

## Key Takeaways ✅

1. **Time is in the features** (hour_sin, hour_cos, day_of_week, etc.)
2. **Array position = time reference** (index 20 = hour 20)
3. **LSTM encodes time** in its predictions (learned from time features)
4. **K-means doesn't need time** (it just matches 24-hour patterns)
5. **test_data keeps the actual timestamps** (in its datetime index)

---

## If You Want to See Timestamps During Testing 🔍

You can modify the test script to print them:

```python
# Add this after line 267 in test_attack_types.py:

print(f"\nTime information:")
print(f"  Hour {start} corresponds to: {test_data.index[start]}")
print(f"  Hour {start+duration} corresponds to: {test_data.index[start+duration]}")
print(f"  Actual timestamp of attack: {test_data.index[start]} to {test_data.index[start+duration]}")
```

This would show:

```
Time information:
  Hour 20 corresponds to: 2024-03-15 20:00:00
  Hour 21 corresponds to: 2024-03-15 21:00:00
  Actual timestamp of attack: 2024-03-15 20:00:00 to 2024-03-15 21:00:00
```

---

**Bottom line:** Time is **implicit** in the array positions and feature values - we don't need to pass timestamps explicitly because the position in the array already corresponds to a specific time that the LSTM and K-means understand! 🎯

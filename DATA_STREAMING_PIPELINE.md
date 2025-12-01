# MLAD Data Streaming Pipeline Documentation

## Overview

This document describes the complete data streaming pipeline for the **Machine Learning Anomaly Detection (MLAD)** system. The system simulates real-time power grid monitoring by replaying historical load data through an ML-based anomaly detection pipeline.

---

## Architecture Diagram

```
┌─────────────────────────────────────────────────────────────────────────────────┐
│                           MLAD DATA STREAMING PIPELINE                          │
└─────────────────────────────────────────────────────────────────────────────────┘

                              ┌──────────────────────┐
                              │   Historical Data    │
                              │  (CSV - 4 years of   │
                              │   hourly load data)  │
                              └──────────┬───────────┘
                                         │
                                         ▼
┌─────────────────────────────────────────────────────────────────────────────────┐
│                         1. DATA STREAM SIMULATOR                                │
│  streaming/streaming_simulator.py :: DataStreamSimulator                        │
│  ─────────────────────────────────────────────────────────────────────────────  │
│  • Replays historical data hour-by-hour                                         │
│  • Configurable speed: 1x (real-time) to 1000x                                  │
│  • Uses training set (first 80%) for normal operations                          │
│  • Outputs: {timestamp, load, index} per hour                                   │
└─────────────────────────────────────────────────────────────────────────────────┘
                                         │
                                         ▼
┌─────────────────────────────────────────────────────────────────────────────────┐
│                         2. SIMULATION CONTROLLER                                │
│  api/simulation_controller.py :: SimulationController                           │
│  ─────────────────────────────────────────────────────────────────────────────  │
│  • Orchestrates entire simulation lifecycle                                     │
│  • Manages start/stop/speed control                                             │
│  • Handles attack injection (PULSE, SCALING, RAMPING, RANDOM, etc.)             │
│  • Coordinates all pipeline components                                          │
│  • Runs simulation loop in background thread                                    │
└─────────────────────────────────────────────────────────────────────────────────┘
                                         │
            ┌────────────────────────────┼────────────────────────────┐
            │                            │                            │
            ▼                            ▼                            ▼
┌───────────────────────┐  ┌───────────────────────┐  ┌───────────────────────┐
│ 3. ROLLING FORECAST   │  │ 4. BENCHMARK MANAGER  │  │  ATTACK INJECTION     │
│       ENGINE          │  │                       │  │     (Optional)        │
│ ────────────────────  │  │ ────────────────────  │  │ ────────────────────  │
│ rolling_forecast_     │  │ benchmark_manager.py  │  │ Applied by controller │
│ engine.py             │  │                       │  │                       │
│                       │  │ • K-means clustering  │  │ Attack Types:         │
│ • Pre-trained LSTM    │  │ • 100 cluster centers │  │ • PULSE (spike)       │
│ • Feature engineering │  │ • Pattern matching    │  │ • SCALING (constant)  │
│ • 168-hour lag window │  │ • Returns benchmark   │  │ • RAMPING (linear)    │
│ • Drift calculation   │  │   for current hour    │  │ • RANDOM (noise)      │
│                       │  │                       │  │ • SMOOTH-CURVE (sine) │
│ Output: forecast      │  │ Output: benchmark     │  │ • POINT-BURST         │
└───────────────────────┘  └───────────────────────┘  │ • CONTEXTUAL-SEASONAL │
            │                            │            │ • RAMPING-TYPE2       │
            │                            │            └───────────────────────┘
            │                            │                        │
            └────────────────────────────┼────────────────────────┘
                                         │
                                         ▼
┌─────────────────────────────────────────────────────────────────────────────────┐
│                         5. REAL-TIME DETECTOR                                   │
│  streaming/realtime_detector.py :: RealTimeDetector                             │
│  ─────────────────────────────────────────────────────────────────────────────  │
│  INPUTS:                                                                        │
│    • forecast (from LSTM)                                                       │
│    • benchmark (from K-means)                                                   │
│    • actual_load (potentially attacked)                                         │
│    • timestamp                                                                  │
│                                                                                 │
│  PROCESSING:                                                                    │
│    1. Calculate scaling_ratio = forecast / benchmark                            │
│    2. Calculate deviation = |scaling_ratio - 1.0|                               │
│    3. Maintain sliding window (500 hours)                                       │
│    4. Run hybrid detection (DP + Statistical)                                   │
│    5. Generate alerts based on severity                                         │
│                                                                                 │
│  SEVERITY THRESHOLDS (from config.py):                                          │
│    • NORMAL: < 9% deviation                                                     │
│    • LOW: 9-25% deviation                                                       │
│    • MEDIUM: 25-40% deviation                                                   │
│    • HIGH: 40-50% deviation                                                     │
│    • EMERGENCY: ≥ 50% deviation (INSTANT ALERT)                                 │
│                                                                                 │
│  DETECTION METHODS:                                                             │
│    • DP (Dynamic Programming): Finds optimal anomaly intervals                  │
│    • STATISTICAL: Hypothesis testing with p-value and Cohen's d                 │
│    • BOTH: Hybrid fusion when both methods agree                                │
│                                                                                 │
│  OUTPUTS:                                                                       │
│    • scaling_ratio, deviation, is_anomalous                                     │
│    • detections[] (intervals with scores)                                       │
│    • alerts[] (with severity, method, timestamps)                               │
└─────────────────────────────────────────────────────────────────────────────────┘
                                         │
                                         ▼
┌─────────────────────────────────────────────────────────────────────────────────┐
│                         6. DATA STORE                                           │
│  api/data_store.py :: DataStore                                                 │
│  ─────────────────────────────────────────────────────────────────────────────  │
│  • In-memory storage with 30-day retention (720 hours)                          │
│  • Stores: timestamps, forecasts, actuals, benchmarks, scaling_ratios           │
│  • Stores: detections (last 720), alerts (last 100)                             │
│  • Uses deque for automatic cleanup                                             │
│  • Provides get_recent_data(hours) for API queries                              │
└─────────────────────────────────────────────────────────────────────────────────┘
                                         │
                    ┌────────────────────┴────────────────────┐
                    │                                         │
                    ▼                                         ▼
┌───────────────────────────────────┐       ┌───────────────────────────────────┐
│         7. REST API               │       │      8. WEBSOCKET HANDLER         │
│  api/api_server.py                │       │  api/websocket_handler.py         │
│  ───────────────────────────────  │       │  ───────────────────────────────  │
│  Flask + Flask-CORS               │       │  Flask-SocketIO                   │
│                                   │       │                                   │
│  ENDPOINTS:                       │       │  EVENTS:                          │
│  GET  /api/status                 │       │  • connect/disconnect             │
│  GET  /api/forecast?hours=N       │       │  • subscribe                      │
│  GET  /api/detections?limit=N     │       │  • update (broadcast)             │
│  GET  /api/alerts?limit=N         │       │  • alert (broadcast)              │
│  GET  /api/metrics                │       │                                   │
│  POST /api/simulation/start       │       │  Real-time push to all clients    │
│  POST /api/simulation/stop        │       │  on every hour processed          │
│  POST /api/simulation/speed       │       │                                   │
│  POST /api/simulation/inject_attack│      │                                   │
│  POST /api/simulation/reset       │       │                                   │
│  GET  /api/simulation/active_attacks│     │                                   │
│  GET  /api/simulation/injected_attacks│   │                                   │
└───────────────────────────────────┘       └───────────────────────────────────┘
                    │                                         │
                    └────────────────────┬────────────────────┘
                                         │
                                         ▼
┌─────────────────────────────────────────────────────────────────────────────────┐
│                         9. STREAMLIT DASHBOARD                                  │
│  frontend/streamlit_dashboard.py                                                │
│  ─────────────────────────────────────────────────────────────────────────────  │
│  TWO MODES:                                                                     │
│                                                                                 │
│  LIVE MODE (during simulation):                                                 │
│    • Fast HTTP polling (1-second refresh)                                       │
│    • Minimal UI for performance                                                 │
│    • Model Reaction Monitor panel                                               │
│    • Real-time load chart with severity-colored anomaly regions                 │
│    • Attack injection controls                                                  │
│                                                                                 │
│  ANALYSIS MODE (after simulation):                                              │
│    • Full historical charts                                                     │
│    • Attack Injection Analysis with per-attack metrics                          │
│    • Time range slider for exploration                                          │
│    • Detailed ML model response analysis                                        │
│    • Dotted boundaries for injected attack regions                              │
└─────────────────────────────────────────────────────────────────────────────────┘
```

---

## Component Details

### 1. DataStreamSimulator (`streaming/streaming_simulator.py`)

**Purpose**: Replays historical power grid load data as an hourly stream.

**Key Attributes**:
- `dataset_dir`: Path to CSV data files
- `speed`: Simulation speed multiplier (1.0 = real-time, 100.0 = 100x faster)
- `current_index`: Current position in dataset
- `data`: Loaded pandas DataFrame with datetime index

**Key Methods**:
```python
load_data()           # Load CSV and preprocess
start() / stop()      # Control simulation
get_next_hour()       # Returns pd.Series{timestamp, load, index}
stream()              # Generator yielding hourly data points
get_progress()        # Returns progress statistics
```

**Data Source**: Uses training set (first 80% of 4-year historical data) representing normal grid operations.

---

### 2. SimulationController (`api/simulation_controller.py`)

**Purpose**: Central orchestrator managing the entire simulation lifecycle.

**Key Attributes**:
- `simulator`: DataStreamSimulator instance
- `forecast_engine`: RollingForecastEngine instance
- `benchmark_manager`: BenchmarkManager instance
- `detector`: RealTimeDetector instance
- `monitor`: PerformanceMonitor instance
- `active_attacks`: List of currently active attack injections
- `all_injected_attacks`: Historical record of all attacks
- `callbacks`: List of functions to notify on updates

**Main Loop (`_run_simulation()`)**:
```python
while is_running:
    1. data_point = simulator.get_next_hour()
    2. forecast = forecast_engine.predict_next_hour(data_point)
    3. benchmark = benchmark_manager.get_benchmark_for_hour(...)
    4. actual_load = apply_attacks(data_point['load'])  # If attacks active
    5. detection_result = detector.process_hourly_data(forecast, benchmark, actual_load, timestamp)
    6. data_store.add_hourly_data(...)
    7. data_store.add_detection(...) / add_alert(...)
    8. notify_callbacks(update_data)
    9. sleep(1.0 / speed)
```

**Attack Types Supported**:
| Type | Description |
|------|-------------|
| PULSE | Sharp spike, constant magnitude |
| SCALING | Constant scaling factor |
| RAMPING | Linear ramp from 0 to magnitude |
| RANDOM | Random noise within ±magnitude |
| SMOOTH-CURVE | Sine wave pattern |
| POINT-BURST | Single spike at midpoint |
| CONTEXTUAL-SEASONAL | 24-hour seasonal pattern |
| RAMPING-TYPE2 | Ramp up then down |

---

### 3. RollingForecastEngine (`streaming/rolling_forecast_engine.py`)

**Purpose**: Generates LSTM forecasts incrementally as new data arrives.

**Key Attributes**:
- `model`: Pre-trained Keras LSTM model
- `scaler`: StandardScaler for feature normalization
- `feature_window`: Last 168 hours of features (1 week)
- `forecast_history`: Historical forecasts for drift calculation

**Features Used**:
```python
['load_lag_1', 'load_lag_24', 'load_lag_168',  # Lag features
 'day_of_week', 'day_of_year', 'month', 'is_weekend',  # Calendar
 'hour_sin', 'hour_cos', 'month_sin', 'month_cos']  # Cyclical encodings
```

**Key Methods**:
```python
load_models()                    # Load LSTM and scaler from disk
initialize(initial_data)         # Bootstrap with 168+ hours of data
predict_next_hour(data_point)    # Returns forecast for next hour
calculate_forecast_drift()       # Monitors model accuracy over time
```

---

### 4. BenchmarkManager (`streaming/benchmark_manager.py`)

**Purpose**: Provides K-means cluster-based benchmark values representing "normal" load patterns.

**Key Attributes**:
- `kmeans_model`: Pre-trained K-means model with 100 clusters
- Each cluster center represents a typical 24-hour load pattern

**Key Methods**:
```python
load_model()                              # Load K-means from disk
get_benchmark_for_hour(forecasts, hour)   # Returns benchmark for specific hour
get_benchmark_for_period(forecasts)       # Returns benchmarks for entire period
```

**Logic**: Finds closest cluster to current 24-hour forecast pattern, returns that cluster's value for the specific hour.

---

### 5. RealTimeDetector (`streaming/realtime_detector.py`)

**Purpose**: Detects anomalies using hybrid DP + Statistical methods.

**Key Attributes**:
- `scaling_history`: All scaling ratios (for RETENTION_HOURS)
- `detection_window`: Sliding window (500 hours) for detection
- `statistical_baseline`: Normal deviation baseline (50 hours)
- `alert_queue`: Last 100 alerts
- `active_alerts`: Currently active alerts

**Detection Flow**:
```python
process_hourly_data(forecast, benchmark, actual_load, timestamp):
    1. scaling_ratio = forecast / (benchmark + 1e-6)
    2. deviation = |scaling_ratio - 1.0|
    3. Update sliding windows
    4. if enough_data:
           detections = run_detection()  # Hybrid DP + Statistical
           alerts = generate_alerts(detections)
    5. Return {scaling_ratio, deviation, is_anomalous, detections, alerts}
```

**Severity Determination**:
```python
if max_deviation >= 0.50:  return EMERGENCY  # 50%+ deviation
if method == 'BOTH' and p_value < 0.001:  return HIGH
if method == 'BOTH':  return MEDIUM
if method == 'STATISTICAL' and p_value < 0.001:  return MEDIUM
if method == 'DP' and dp_score > 1.0:  return HIGH
# ... etc
```

---

### 6. DataStore (`api/data_store.py`)

**Purpose**: In-memory storage with automatic cleanup.

**Storage (all using `deque` with `maxlen`)**:
| Data | Max Length |
|------|------------|
| timestamps, forecasts, actuals, benchmarks, scaling_ratios | 720 hours (30 days) |
| detections | 720 |
| alerts | 100 |

**Key Methods**:
```python
add_hourly_data(timestamp, forecast, actual, benchmark, scaling_ratio)
add_detection(detection_dict)
add_alert(alert_dict)
get_recent_data(hours=24)  # Returns DataFrame
get_detections(limit=100)
get_alerts(limit=100)
get_statistics()  # Returns summary dict
reset()  # Clear all data
```

---

### 7. API Server (`api/api_server.py`)

**Purpose**: Flask REST API + WebSocket server.

**REST Endpoints**:
| Endpoint | Method | Description |
|----------|--------|-------------|
| `/api/status` | GET | Simulation status + data store stats |
| `/api/forecast` | GET | Recent forecast data (param: hours) |
| `/api/detections` | GET | Detection results (param: limit) |
| `/api/alerts` | GET | Alerts (params: limit, severity, acknowledged) |
| `/api/metrics` | GET | Performance metrics |
| `/api/simulation/start` | POST | Start simulation (body: speed) |
| `/api/simulation/stop` | POST | Stop simulation |
| `/api/simulation/speed` | POST | Change speed (body: speed) |
| `/api/simulation/inject_attack` | POST | Inject attack |
| `/api/simulation/reset` | POST | Reset all state |
| `/api/simulation/active_attacks` | GET | Currently active attacks |
| `/api/simulation/injected_attacks` | GET | All injected attacks with timestamps |

---

### 8. WebSocketHandler (`api/websocket_handler.py`)

**Purpose**: Real-time push updates to connected clients.

**Events**:
| Event | Direction | Description |
|-------|-----------|-------------|
| `connect` | Client→Server | Client connects |
| `connected` | Server→Client | Connection confirmed |
| `disconnect` | Client→Server | Client disconnects |
| `subscribe` | Client→Server | Subscribe to channels |
| `update` | Server→Client | Hourly data update broadcast |
| `alert` | Server→Client | Alert broadcast |

**Update Payload** (broadcast every simulation hour):
```json
{
  "timestamp": "2020-01-01 12:00:00",
  "forecast": 1234.56,
  "actual": 1250.00,
  "benchmark": 1200.00,
  "scaling_ratio": 1.028,
  "alerts": [...]
}
```

---

### 9. Streamlit Dashboard (`frontend/streamlit_dashboard.py`)

**Purpose**: Interactive visualization and control interface.

**Live Mode Features**:
- 1-second HTTP polling refresh
- Minimal UI for performance
- Model Reaction Monitor (detection status, attack status, method)
- Real-time load chart with severity-colored anomaly regions
- Attack injection controls (type, magnitude 0-10, duration, start hour)

**Analysis Mode Features**:
- Full historical data visualization
- Time range slider
- Attack Injection Analysis cards with:
  - Max/Avg Deviation
  - Hours Detected / Duration
  - Detection Method
  - Severity classification
- Dotted boundaries for attack regions
- Alert history table

---

## Configuration (`config.py`)

**Key Parameters**:
```python
# Detection Thresholds
MAGNITUDE_THRESHOLD = 0.09      # 9% deviation triggers detection
EMERGENCY_THRESHOLD = 0.50      # 50% deviation = EMERGENCY

# Duration Requirements
MIN_DURATION_WEAK = 3           # Weak attacks (<25%): 3 hours
MIN_DURATION_MEDIUM = 2         # Medium attacks (25-40%): 2 hours
MIN_DURATION_STRONG = 1         # Strong attacks (>40%): 1 hour

# Detection Windows
DETECTION_WINDOW_HOURS = 500    # Sliding window for DP
STATISTICAL_BASELINE_HOURS = 50 # Baseline for statistical tests

# Retention
RETENTION_HOURS = 720           # 30 days of data
```

---

## Data Flow Summary

```
Historical CSV → DataStreamSimulator → [hour-by-hour]
                          ↓
                 SimulationController
                    ↓         ↓         ↓
            Forecast    Benchmark    Attack
            (LSTM)      (K-means)    Injection
                    ↓         ↓         ↓
                 RealTimeDetector
                 (DP + Statistical)
                          ↓
              scaling_ratio, detections, alerts
                          ↓
                     DataStore
                    ↓         ↓
              REST API    WebSocket
                    ↓         ↓
              Streamlit Dashboard
              (HTTP Polling / WS)
```

---

## Running the System

```bash
# Terminal 1: Start API Server
cd btp_mlad
python api/api_server.py
# Server runs on http://localhost:5000

# Terminal 2: Start Streamlit Dashboard
cd btp_mlad
streamlit run frontend/streamlit_dashboard.py
# Dashboard runs on http://localhost:8501
```

**Dashboard Usage**:
1. Click "Start Simulation" with desired speed (e.g., 100x)
2. Inject attacks via sidebar controls
3. Monitor real-time detection in Live Mode
4. Stop simulation to enter Analysis Mode for detailed review

---

## Key Concepts

### Scaling Ratio
```
scaling_ratio = LSTM_forecast / K-means_benchmark
```
- `= 1.0`: Normal operation (forecast matches expected pattern)
- `> 1.0`: Load higher than expected
- `< 1.0`: Load lower than expected

### Deviation
```
deviation = |scaling_ratio - 1.0|
```
Represents percentage deviation from normal. Used for severity classification.

### Hybrid Detection
Combines two methods:
1. **DP (Dynamic Programming)**: Finds optimal contiguous anomaly intervals by maximizing score
2. **Statistical**: Hypothesis testing with t-tests, p-values, and Cohen's d effect size

When both methods agree, confidence is higher (severity escalates).

---

*Document generated for LLM context. Last updated: December 2024*


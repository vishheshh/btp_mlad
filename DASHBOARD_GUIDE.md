# Power Grid Protection Dashboard - User Guide

## 📊 Dashboard Components Explained

### 1. **Control Panel (Left Sidebar)**

#### Simulation Control

- **Status Indicator**: Shows if simulation is running (🟢) or stopped (⚪)
- **Start/Stop Buttons**: Control the simulation lifecycle
- **Speed Control**: Adjust simulation speed (1x = real-time, 100x = 100x faster)

#### Attack Injection

- **Attack Type**: Select from 8 attack types (PULSE, SCALING, RAMPING, etc.)
- **Start Hour**: When to inject attack (offset from current position)
- **Duration**: How long the attack lasts (in hours)
- **Magnitude**: Attack strength (0.1 = 10% increase, 1.0 = 100% increase)
- **Inject Button**: Inject the configured attack

### 2. **Main Dashboard Area**

#### Status Metrics (Top Row)

- **Status**: Current simulation state
- **Data Hours**: Total hours of data collected
- **Active Alerts**: Number of detected anomalies
- **Throughput**: Processing speed (hours per second)

#### Real-Time Load Chart

**Three Lines:**

- **Actual Load** (Blue solid): Real measured power grid load from historical data
- **Forecast** (Green dashed): LSTM model prediction of expected load
- **Benchmark** (Orange dotted): Normal load pattern from K-means clustering

**Shaded Regions:**

- **Purple regions**: Injected attacks (manually injected for testing)
- **Red regions**: Detected anomalies (system detected potential attacks)

**Controls:**

- **Time Range Slider**: Adjust how many hours to display (1-168 hours)

#### Alerts Panel

- **Filter by Severity**: Select which alert severities to show
- **Show Acknowledged**: Toggle to show/hide acknowledged alerts
- **Alert List**: Shows detected anomalies with:
  - Severity level (color-coded)
  - Time range
  - Detection method (DP, STATISTICAL, or BOTH)
  - Acknowledge button

#### Performance Metrics

- **Mean Latency**: Average processing time per hour
- **Throughput**: Hours processed per second
- **Memory Usage**: Current memory consumption
- **Hours Processed**: Total hours processed
- **Latency Distribution**: Bar chart showing latency percentiles

### 3. **Auto-Refresh**

- **Checkbox**: Enable/disable auto-refresh
- **Refresh Interval**: Select refresh rate (5, 10, 15, or 30 seconds)
- **Smart Behavior**: Only refreshes when simulation is running

---

## 🔧 How It Works

### Data Flow:

1. **Simulation** → Processes historical data hour by hour
2. **Forecast Engine** → Generates LSTM predictions
3. **Benchmark Manager** → Provides normal load patterns
4. **Detector** → Compares forecast vs benchmark (scaling ratio)
5. **Alert System** → Flags anomalies when scaling ratio deviates >9%
6. **Dashboard** → Displays everything in real-time

### Attack Injection Flow:

1. User configures attack in sidebar
2. Attack is scheduled for future hour
3. When that hour arrives, attack modifies actual load
4. System detects the anomaly (red region)
5. Alert is generated and shown in alerts panel

---

## 🎯 Understanding the Chart

### Normal Operation:

- All three lines follow similar patterns
- No red or purple regions
- Scaling ratio ≈ 1.0

### Attack Detected:

- Red shaded region appears
- Lines diverge significantly
- Alert appears in alerts panel

### Attack Injected:

- Purple shaded region shows where you injected attack
- Actual load line shows modified values
- System should detect it (red region may overlap)

---

## ⚠️ Troubleshooting

### Performance Metrics Empty?

- Metrics only populate when simulation is running
- Wait a few hours of processing for metrics to appear
- Check that simulation is actually running

### Too Many API Requests?

- Auto-refresh is disabled by default now
- Only enable when needed
- Increase refresh interval to reduce requests

### Can't See Injected Attacks?

- Check the "Active Attacks" info below chart
- Purple regions show injected attacks
- Attacks are scheduled relative to current simulation position

---

## 📝 Tips

1. **Start simulation** before injecting attacks
2. **Use lower speeds** (10x-50x) for better visualization
3. **Check alerts panel** for detailed detection information
4. **Use time range slider** to zoom in on specific periods
5. **Purple = Injected**, **Red = Detected**

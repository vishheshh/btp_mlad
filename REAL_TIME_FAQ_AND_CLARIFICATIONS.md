# Real-Time Simulation System - FAQ & Clarifications

## Your Questions Answered

---

## 1. **Normal Operations Input Load Wave - What Data Will We Use?**

### Answer:

**For Demo/Simulation Mode:**

- We will use **historical normal load data** from your existing dataset
- Specifically, we'll replay data from the **training set** (first 80% of your 4-year dataset)
- This represents **normal, non-attacked** power grid operations
- The data will be streamed hour-by-hour, simulating real-time SCADA readings

### Data Flow:

```
Historical CSV Files (Normal Operations)
    ↓
Data Stream Simulator
    ↓
Replays as hourly stream (simulating real-time)
    ↓
Rolling Forecast Engine → Real-Time Detector
```

### Why This Approach?

1. **Realistic**: Uses actual historical grid load patterns
2. **Reproducible**: Same data every time for consistent demos
3. **Flexible**: Can inject attacks at any point
4. **Validated**: Data is from your training set (known to be normal)

### Alternative Approaches (Not Recommended for Demo):

- ❌ **Synthetic data**: Less realistic, doesn't match real grid patterns
- ❌ **Test set data**: May contain patterns the model hasn't seen
- ✅ **Training set data**: Best choice - represents normal operations the model learned from

### Implementation:

```python
# streaming_simulator.py will:
1. Load historical data from dataset/ folder
2. Use training portion (first 80%)
3. Stream hour-by-hour with timestamps
4. Simulate real-time arrival (configurable speed)
```

---

## 2. **Frontend Attack Injection Options**

### Answer: **YES!** ✅

The frontend will have a **comprehensive attack injection interface** with full control.

### Frontend Attack Injection Panel:

```
┌─────────────────────────────────────────────────┐
│  ATTACK INJECTION CONTROL PANEL                 │
├─────────────────────────────────────────────────┤
│                                                  │
│  Attack Type: [Dropdown ▼]                      │
│  ├─ PULSE                                       │
│  ├─ SCALING                                     │
│  ├─ RAMPING (Type I)                            │
│  ├─ RAMPING (Type II)                           │
│  ├─ RANDOM                                      │
│  ├─ SMOOTH-CURVE                                │
│  ├─ POINT-BURST                                 │
│  └─ CONTEXTUAL-SEASONAL                         │
│                                                  │
│  Start Time: [Date/Time Picker]                │
│  Duration: [Slider: 1-48 hours]                 │
│                                                  │
│  Magnitude: [Slider: 10% - 900%]               │
│  Current: 150%                                  │
│                                                  │
│  [Preview Attack] [Inject Attack] [Cancel]      │
│                                                  │
│  ┌──────────────────────────────────────────┐   │
│  │ Preview Graph:                          │   │
│  │ [Shows how load will look with attack]  │   │
│  └──────────────────────────────────────────┘   │
└─────────────────────────────────────────────────┘
```

### Features:

1. **Attack Type Selection**: Dropdown with all 8 attack types
2. **Parameter Controls**:
   - Start time (when to inject)
   - Duration (1-48 hours)
   - Magnitude (10% to 900% deviation)
   - Attack-specific parameters (e.g., ramp rate, noise level)
3. **Preview**: Shows how the attack will look before injection
4. **Real-time Injection**: Inject attacks while simulation is running
5. **Attack History**: List of all injected attacks with status

### API Endpoint:

```python
POST /api/simulation/inject_attack
Body: {
    "attack_type": "PULSE",
    "start_time": "2024-01-15T14:00:00",
    "duration_hours": 5,
    "magnitude": 2.5,  # 150% increase
    "parameters": {
        "pulse_magnitude": 3.0,
        # ... attack-specific params
    }
}
```

### Frontend Component:

```jsx
// AttackInjectionPanel.jsx
- Attack type selector
- Parameter sliders/inputs
- Preview chart
- Inject button
- Active attacks list
```

---

## 3. **Connecting to Real Grid Hardware - The Integration Layer**

### Answer: **Critical Production Deployment Layer**

This is the **Data Acquisition & Integration Layer** that connects software to actual grid hardware.

### Architecture for Real Grid Connection:

```
┌─────────────────────────────────────────────────────────────┐
│                    REAL POWER GRID                           │
│  ┌──────────────┐  ┌──────────────┐  ┌──────────────┐     │
│  │   SCADA      │  │     PMU       │  │   RTU        │     │
│  │  Systems     │  │  (Phasor      │  │  (Remote     │     │
│  │              │  │   Meas. Unit) │  │   Terminal)  │     │
│  └──────┬───────┘  └──────┬───────┘  └──────┬───────┘     │
│         │                  │                  │              │
│         └──────────────────┼──────────────────┘             │
│                            │                                │
│                    ┌───────▼────────┐                       │
│                    │  Grid Network  │                       │
│                    │  (IEC 61850,   │                       │
│                    │   DNP3, Modbus)│                       │
│                    └───────┬────────┘                       │
└────────────────────────────┼────────────────────────────────┘
                             │
                    ┌────────▼──────────────────────────────┐
                    │   DATA ACQUISITION LAYER              │
                    │   (Integration Gateway)               │
                    ├──────────────────────────────────────┤
                    │  ┌────────────────────────────────┐ │
                    │  │ Protocol Adapters              │ │
                    │  │ - IEC 61850 Client             │ │
                    │  │ - DNP3 Client                   │ │
                    │  │ - Modbus TCP/RTU               │ │
                    │  │ - OPC-UA Client                │ │
                    │  └────────────────────────────────┘ │
                    │           │                          │
                    │  ┌────────▼──────────────────────┐ │
                    │  │ Data Normalization            │ │
                    │  │ - Convert to standard format  │ │
                    │  │ - Timestamp alignment         │ │
                    │  │ - Unit conversion            │ │
                    │  └────────┬──────────────────────┘ │
                    │           │                          │
                    │  ┌────────▼──────────────────────┐ │
                    │  │ Data Validation               │ │
                    │  │ - Range checks                 │ │
                    │  │ - Quality flags               │ │
                    │  │ - Missing data handling       │ │
                    │  └────────┬──────────────────────┘ │
                    │           │                          │
                    │  ┌────────▼──────────────────────┐ │
                    │  │ Message Queue/Buffer          │ │
                    │  │ - Kafka / RabbitMQ / Redis    │ │
                    │  │ - Handles backpressure        │ │
                    │  │ - Ensures no data loss        │ │
                    │  └────────┬──────────────────────┘ │
                    └───────────┼────────────────────────┘
                                 │
                    ┌────────────▼──────────────────────────┐
                    │   MLAD REAL-TIME ENGINE               │
                    │   (Your Current System)                │
                    │   - Rolling Forecast Engine            │
                    │   - Real-Time Detector                 │
                    │   - Alert System                       │
                    └────────────────────────────────────────┘
```

### Key Components:

#### **A. Protocol Adapters**

**Purpose**: Connect to different grid communication protocols

**Common Protocols**:

1. **IEC 61850** (Modern standard)
   - Substation automation
   - GOOSE messages (fast, < 4ms)
   - MMS for data exchange
2. **DNP3** (Widely used)

   - Distributed Network Protocol
   - Master-slave architecture
   - Secure authentication (DNP3-SA)

3. **Modbus** (Legacy, still common)

   - Modbus TCP/IP
   - Modbus RTU (serial)
   - Simple, widely supported

4. **OPC-UA** (Unified Architecture)
   - Modern, secure
   - Platform-independent
   - Information modeling

#### **B. Data Normalization Layer**

**Purpose**: Convert all protocols to standard internal format

```python
# Standardized data format
@dataclass
class GridDataPoint:
    timestamp: pd.Timestamp
    load_mw: float
    voltage_kv: float
    frequency_hz: float
    quality_flag: str  # 'GOOD', 'BAD', 'UNCERTAIN'
    source: str  # 'SCADA', 'PMU', 'RTU'
    location: str  # Substation/bus ID
```

#### **C. Data Validation**

**Purpose**: Ensure data quality before processing

- Range checks (load within expected bounds)
- Quality flags (data validity)
- Missing data handling (interpolation or alert)
- Timestamp validation (no duplicates, no gaps)

#### **D. Message Queue/Buffer**

**Purpose**: Handle data flow, prevent loss, manage backpressure

**Options**:

- **Apache Kafka**: High throughput, distributed
- **RabbitMQ**: Reliable, easy to use
- **Redis Streams**: Fast, lightweight
- **In-memory buffer**: Simple, for low-volume

### Implementation Example:

```python
# grid_integration_layer.py

class GridDataAcquisition:
    """Connects to real grid hardware"""

    def __init__(self, protocol='IEC61850'):
        self.protocol_adapter = self._get_adapter(protocol)
        self.normalizer = DataNormalizer()
        self.validator = DataValidator()
        self.message_queue = MessageQueue()

    def connect_to_grid(self, grid_endpoint):
        """Connect to SCADA/PMU system"""
        self.protocol_adapter.connect(grid_endpoint)

    def start_streaming(self):
        """Start receiving real-time data"""
        while True:
            # Receive raw data from grid
            raw_data = self.protocol_adapter.read()

            # Normalize to standard format
            normalized = self.normalizer.normalize(raw_data)

            # Validate
            if self.validator.is_valid(normalized):
                # Send to MLAD system
                self.message_queue.publish(normalized)
            else:
                # Log invalid data, alert
                self.handle_invalid_data(normalized)
```

### For Your Professor's Question:

**"How do we connect to a real grid?"**

**Answer Structure**:

1. **Data Acquisition Layer** (What we just described)

   - Protocol adapters for SCADA/PMU systems
   - Handles IEC 61850, DNP3, Modbus, OPC-UA

2. **Data Pipeline** (Real-time processing)

   - Message queue for reliable delivery
   - Data normalization and validation
   - Timestamp synchronization

3. **Integration Points**:

   - **SCADA Systems**: Primary data source
   - **PMU (Phasor Measurement Units)**: High-frequency data (30-60 samples/sec)
   - **RTU (Remote Terminal Units)**: Remote substation data

4. **Security Considerations**:

   - Encrypted communication (TLS/SSL)
   - Authentication (certificates, keys)
   - Firewall rules
   - Network segmentation

5. **Deployment Architecture**:
   ```
   Grid Network (Isolated) → Firewall → Integration Gateway → MLAD System
   ```

### Implementation Priority:

- **Phase 1 (Current)**: Simulation mode (replay historical data)
- **Phase 2 (Future)**: Add integration layer for real grid connection
- **Phase 3 (Production)**: Full production deployment with grid hardware

---

## 4. **Live Attack Detection & Frontend Display**

### Answer: **Real-Time Detection with Progressive Visualization**

### How It Works:

#### **A. Detection Process (Real-Time)**:

```
Hour 0: Normal load (12,000 MWh)
Hour 1: Attack starts (load = 18,000 MWh, 50% increase)
        ↓
        Real-Time Detector processes
        ↓
        Detection triggered! (within 1 minute)
        ↓
        Alert generated
        ↓
        Frontend updates immediately
```

#### **B. Multi-Hour Attack Display**:

**Scenario**: Attack spans 5-10 hours

**Frontend Visualization**:

```
┌─────────────────────────────────────────────────────────┐
│  REAL-TIME LOAD CHART                                    │
├─────────────────────────────────────────────────────────┤
│                                                          │
│  Load (MWh)                                             │
│  20k ┤                                    ╭─╮           │
│      │                          ╭─────────╯ ╰─────────  │
│  15k ┤              ╭──────────╯                       │
│      │    ╭─────────╯                                   │
│  10k ┤────╯                                             │
│      │                                                  │
│   0  └───────────────────────────────────────────────   │
│      0h   2h   4h   6h   8h   10h  12h  14h  16h      │
│                                                          │
│  ┌──────────────────────────────────────────────────┐  │
│  │ 🚨 ACTIVE ATTACK DETECTED                        │  │
│  │ Start: 14:00 | Duration: 5h | Severity: HIGH     │  │
│  │ [View Details] [Acknowledge]                     │  │
│  └──────────────────────────────────────────────────┘  │
│                                                          │
│  Legend:                                                │
│  ─── Actual Load                                        │
│  ─── Forecast                                           │
│  ─── Benchmark                                          │
│  ███ Attack Region (highlighted)                         │
└─────────────────────────────────────────────────────────┘
```

### Progressive Display Features:

1. **Real-Time Updates**:

   - Chart updates every hour (or configurable interval)
   - Attack region highlights as it progresses
   - Alert badge shows "ACTIVE" while attack continues

2. **Attack Timeline View**:

   ```
   ┌─────────────────────────────────────────────────┐
   │  ATTACK TIMELINE                                 │
   ├─────────────────────────────────────────────────┤
   │                                                  │
   │  [14:00] Attack Started                         │
   │    ↓                                            │
   │  [14:05] Detected (5 min latency)              │
   │    ↓                                            │
   │  [15:00] Still Active (1h duration)            │
   │    ↓                                            │
   │  [16:00] Still Active (2h duration)            │
   │    ↓                                            │
   │  [19:00] Attack Ended                           │
   │    ↓                                            │
   │  [19:05] Post-Attack Analysis                   │
   │                                                  │
   └─────────────────────────────────────────────────┘
   ```

3. **Alert Panel** (Shows Active Attacks):

   ```
   ┌─────────────────────────────────────────────────┐
   │  ACTIVE ALERTS                                   │
   ├─────────────────────────────────────────────────┤
   │                                                  │
   │  🚨 HIGH SEVERITY                               │
   │  Attack Type: SCALING                            │
   │  Started: 14:00 | Duration: 5h 23m              │
   │  Magnitude: 150% | Status: ONGOING               │
   │  [View Details] [Acknowledge]                   │
   │                                                  │
   │  ────────────────────────────────────────────  │
   │                                                  │
   │  ⚠️  MEDIUM SEVERITY                            │
   │  Attack Type: RAMPING                            │
   │  Started: 10:00 | Duration: 8h 15m               │
   │  Magnitude: 30% | Status: ONGOING                │
   │  [View Details] [Acknowledge]                   │
   │                                                  │
   └─────────────────────────────────────────────────┘
   ```

4. **Progressive Highlighting**:
   - As attack progresses, the chart region is highlighted
   - Color intensity increases with attack severity
   - Animated indicator shows "LIVE" detection

### Implementation Details:

#### **Backend (Real-Time Detection)**:

```python
# realtime_detector.py

class RealTimeDetector:
    def process_hourly_data(self, data_point):
        """Called every hour with new data"""

        # Add to sliding window
        self.sliding_window.append(data_point)

        # Run detection on window
        detections = self.hybrid_detection(self.sliding_window)

        # Check for new detections
        for detection in detections:
            if self.is_new_detection(detection):
                # Generate alert
                alert = self.create_alert(detection)

                # Send to frontend via WebSocket
                self.websocket.broadcast({
                    'type': 'NEW_ATTACK',
                    'alert': alert,
                    'detection': detection
                })

        # Update ongoing attacks
        self.update_ongoing_attacks(detections)

        # Broadcast update
        self.websocket.broadcast({
            'type': 'ATTACK_UPDATE',
            'ongoing_attacks': self.get_ongoing_attacks()
        })
```

#### **Frontend (Progressive Display)**:

```jsx
// AttackVisualization.jsx

function AttackVisualization({ attacks, currentTime }) {
  return (
    <div>
      {/* Main Chart */}
      <LoadChart data={loadData} attacks={attacks} highlightRegions={true} />

      {/* Active Attacks Panel */}
      <ActiveAttacksPanel
        attacks={attacks.filter((a) => a.status === "ONGOING")}
      />

      {/* Attack Timeline */}
      <AttackTimeline attacks={attacks} currentTime={currentTime} />
    </div>
  );
}
```

### Key Features:

1. **Immediate Detection**: Attack detected within 1-2 hours of start
2. **Progressive Display**: Chart updates as attack continues
3. **Status Tracking**: Shows "ONGOING" vs "ENDED" attacks
4. **Duration Tracking**: Real-time duration counter
5. **Multi-Attack Support**: Can show multiple simultaneous attacks
6. **Historical View**: Can scroll back to see full attack timeline

### Example Flow (5-Hour Attack):

```
Hour 0 (14:00): Attack starts
    ↓
Hour 1 (15:00):
    - Detection triggered
    - Alert appears in frontend
    - Chart highlights hour 0-1 region
    - Status: "ONGOING - 1h duration"

Hour 2 (16:00):
    - Attack continues
    - Chart highlights hour 0-2 region (expanded)
    - Status: "ONGOING - 2h duration"

Hour 3 (17:00):
    - Attack continues
    - Chart highlights hour 0-3 region
    - Status: "ONGOING - 3h duration"

Hour 4 (18:00):
    - Attack continues
    - Chart highlights hour 0-4 region
    - Status: "ONGOING - 4h duration"

Hour 5 (19:00):
    - Attack ends
    - Chart shows full highlighted region (0-5 hours)
    - Status: "ENDED - 5h total duration"
    - Post-attack analysis available
```

---

## Summary

1. **Normal Operations Data**: Historical training data replayed as stream ✅
2. **Frontend Attack Injection**: Full UI with all attack types and parameters ✅
3. **Grid Integration Layer**: Protocol adapters + normalization + validation ✅
4. **Live Attack Display**: Progressive visualization with real-time updates ✅

All features will be implemented in the system!

# Real-Time Simulation System - Detailed Execution Plan

## Executive Summary

This document outlines a comprehensive plan to build a **real-time streaming simulation system** for the MLAD power grid protection system. The solution will be implemented in software first, with architecture designed for future hardware acceleration.

**Timeline**: 3-4 weeks  
**Status**: Feasible ✅ (with some considerations noted below)

---

## 1. FEASIBILITY ASSESSMENT

### ✅ **What IS Feasible:**

1. **Streaming Data Simulator**: ✅ Fully feasible

   - Replay historical data as hourly stream
   - Simulate real-time data arrival
   - Inject attacks at runtime

2. **Online Forecasting**: ✅ Feasible with current LSTM

   - Use pre-trained model incrementally
   - Generate forecasts for each new hour
   - Maintain sliding window of features

3. **Real-Time Detection**: ✅ Feasible

   - Hybrid detection (DP + Statistical) can run incrementally
   - Sliding window approach for online algorithm
   - Alert generation and queuing

4. **Performance Metrics**: ✅ Fully feasible

   - Latency measurement
   - Memory profiling
   - Throughput calculation

5. **Frontend Dashboard**: ✅ Fully feasible
   - Real-time visualization
   - Alert display
   - Performance monitoring

### ⚠️ **Challenges & Considerations:**

1. **K-Means Dynamic Updates**: ⚠️ **Moderate Challenge**

   - **Current**: K-means is static (trained once)
   - **Challenge**: Updating clusters in real-time is computationally expensive
   - **Solution**: Use rolling window approach (update every N days, not every hour)
   - **Alternative**: Keep static K-means, measure drift separately

2. **LSTM Forecast Drift**: ⚠️ **Expected Behavior**

   - **Reality**: LSTM will drift over time without retraining
   - **Solution**: Measure drift, implement retraining triggers
   - **Timeline**: Retrain weekly/monthly (not real-time)

3. **Statistical Detection Baseline**: ⚠️ **Requires Buffer**

   - **Challenge**: Needs 50+ hours of baseline data
   - **Solution**: Maintain rolling baseline buffer
   - **Impact**: First 50 hours may have limited statistical detection

4. **Memory Management**: ⚠️ **Needs Attention**
   - **Challenge**: Long-running simulation accumulates data
   - **Solution**: Implement data retention policies (keep last N days)

### ❌ **What is NOT Feasible (or Not Recommended):**

1. **Real-Time Model Retraining**: ❌ Not feasible in real-time

   - Retraining LSTM takes hours
   - **Solution**: Scheduled retraining (weekly/monthly)

2. **Hardware Acceleration (Initial Phase)**: ⏸️ Deferred
   - Software implementation first
   - Architecture designed for future FPGA/GPU acceleration

---

## 2. SYSTEM ARCHITECTURE

### 2.1 High-Level Architecture

```
┌─────────────────────────────────────────────────────────────────┐
│                    FRONTEND DASHBOARD (React/Flask)             │
│  - Real-time charts, alerts, performance metrics               │
└────────────────────────────┬────────────────────────────────────┘
                             │ WebSocket/HTTP API
┌────────────────────────────▼────────────────────────────────────┐
│                    API LAYER (Flask/FastAPI)                    │
│  - REST endpoints for queries                                   │
│  - WebSocket for real-time updates                              │
└────────────────────────────┬────────────────────────────────────┘
                             │
┌────────────────────────────▼────────────────────────────────────┐
│              REAL-TIME SIMULATION ENGINE                         │
│  ┌──────────────────────────────────────────────────────────┐  │
│  │  Data Stream Simulator                                    │  │
│  │  - Replays historical data as hourly stream             │  │
│  │  - Simulates data arrival timing                        │  │
│  │  - Optional attack injection                            │  │
│  └──────────────────────────────────────────────────────────┘  │
│                             │                                    │
│  ┌──────────────────────────▼──────────────────────────────┐  │
│  │  Rolling Forecast Engine                                 │  │
│  │  - Incremental LSTM predictions                         │  │
│  │  - Feature window management                            │  │
│  │  - Forecast drift tracking                               │  │
│  └──────────────────────────┬──────────────────────────────┘  │
│                             │                                    │
│  ┌──────────────────────────▼──────────────────────────────┐  │
│  │  Benchmark Manager                                       │  │
│  │  - K-means benchmark retrieval                           │  │
│  │  - Rolling window updates (periodic)                    │  │
│  └──────────────────────────┬──────────────────────────────┘  │
│                             │                                    │
│  ┌──────────────────────────▼──────────────────────────────┐  │
│  │  Real-Time Detection Pipeline                           │  │
│  │  - Hybrid detection (DP + Statistical)                   │  │
│  │  - Sliding window processing                            │  │
│  │  - Alert generation                                     │  │
│  └──────────────────────────┬──────────────────────────────┘  │
│                             │                                    │
│  ┌──────────────────────────▼──────────────────────────────┐  │
│  │  Performance Monitor                                     │  │
│  │  - Latency tracking                                     │  │
│  │  - Memory usage                                         │  │
│  │  - Throughput measurement                               │  │
│  └──────────────────────────────────────────────────────────┘  │
└─────────────────────────────────────────────────────────────────┘
```

### 2.2 Component Details

#### **A. Data Stream Simulator**

- **Purpose**: Simulate hourly data arrival
- **Input**: Historical CSV data from **training set** (normal operations, first 80% of dataset)
- **Output**: Hourly data points with timestamps
- **Features**:
  - Configurable speed (real-time, 10x, 100x)
  - Optional attack injection at runtime
  - Data gap simulation (network delays)
- **Data Source**: Replays historical normal load data to simulate real-time SCADA readings

#### **B. Rolling Forecast Engine**

- **Purpose**: Generate LSTM forecasts incrementally
- **State Management**:
  - Maintain last 168 hours for lag features
  - Update feature window as new data arrives
  - Track forecast accuracy over time
- **Output**: Forecast for current hour + drift metrics

#### **C. Benchmark Manager**

- **Purpose**: Provide K-means benchmark values
- **Approach**:
  - **Option 1 (Recommended)**: Static K-means with rolling pattern matching
  - **Option 2**: Periodic K-means retraining (every 7 days)
- **Output**: Benchmark values for current hour

#### **D. Real-Time Detection Pipeline**

- **Purpose**: Detect anomalies as data streams
- **Algorithm**: Hybrid detection (DP + Statistical)
- **State Management**:
  - Sliding window (last 500 hours recommended)
  - Rolling baseline for statistical tests (50+ hours)
  - Alert queue
- **Output**: Detection results + alerts

#### **E. Performance Monitor**

- **Purpose**: Track system performance
- **Metrics**:
  - Operational latency (data arrival → detection)
  - Forecast drift (MAE over time)
  - Memory usage
  - Throughput (hours/second)
  - Detection delay (attack start → alert)

---

## 3. IMPLEMENTATION PHASES

### **PHASE 1: Core Streaming Engine (Week 1)**

**Goal**: Build basic streaming simulation infrastructure

**Tasks**:

1. Create `streaming_simulator.py`:

   - Data stream generator (replay historical data)
   - Configurable speed control
   - Timestamp management

2. Create `rolling_forecast_engine.py`:

   - Incremental LSTM prediction
   - Feature window management (last 168 hours)
   - Forecast storage and drift calculation

3. Create `benchmark_manager.py`:

   - Static K-means benchmark retrieval
   - Pattern matching for current hour
   - (Future: periodic retraining logic)

4. Create `realtime_detector.py`:

   - Sliding window management
   - Incremental hybrid detection
   - Alert generation

5. Create `performance_monitor.py`:
   - Latency tracking
   - Memory profiling
   - Throughput calculation

**Deliverables**:

- ✅ Streaming data flow working
- ✅ Incremental forecasts generated
- ✅ Basic detection pipeline operational

---

### **PHASE 2: Real-Time Detection Logic (Week 2)**

**Goal**: Implement online detection algorithms

**Tasks**:

1. **Sliding Window DP Detection**:

   - Maintain rolling window (500 hours)
   - Run DP algorithm on window
   - Update as new data arrives

2. **Online Statistical Detection**:

   - Maintain rolling baseline (50+ hours)
   - Update statistical tests incrementally
   - Handle baseline buffer initialization

3. **Hybrid Fusion**:

   - Merge DP + Statistical results
   - Priority ranking
   - Alert generation

4. **Alert System**:
   - Alert queue management
   - Alert severity levels
   - Alert persistence

**Deliverables**:

- ✅ Online detection working
- ✅ Alerts generated in real-time
- ✅ Detection latency < 1 minute

---

### **PHASE 3: API & Backend Services (Week 2-3)**

**Goal**: Build API layer for frontend integration

**Tasks**:

1. Create `api_server.py` (Flask/FastAPI):

   - REST endpoints:
     - `/api/status` - System status
     - `/api/forecast` - Get current forecast
     - `/api/detections` - Get recent detections
     - `/api/alerts` - Get active alerts
     - `/api/metrics` - Performance metrics
   - WebSocket endpoint for real-time updates

2. Create `data_store.py`:

   - In-memory data storage (Redis optional)
   - Recent history retention (last 30 days)
   - Alert persistence

3. Create `simulation_controller.py`:
   - Start/stop simulation
   - Speed control
   - Attack injection API

**Deliverables**:

- ✅ REST API functional
- ✅ WebSocket streaming working
- ✅ Simulation control interface

---

### **PHASE 4: Frontend Dashboard (Week 3-4)**

**Goal**: Build visualization dashboard

**Tasks**:

1. **Technology Stack**:

   - **Option A**: React + Chart.js/Recharts (Recommended)
   - **Option B**: Flask + Plotly Dash (Simpler, Python-based)
   - **Option C**: Streamlit (Fastest prototype)

2. **Dashboard Components**:

   - **Real-Time Load Chart**: Current load, forecast, benchmark
     - Progressive highlighting of ongoing attacks
     - Multi-hour attack visualization
     - Real-time updates via WebSocket
   - **Anomaly Detection View**: Detected intervals highlighted
     - Attack timeline visualization
     - Duration tracking for multi-hour attacks
     - Status indicators (ONGOING, ENDED)
   - **Attack Injection Panel**: Full UI for injecting attacks
     - Attack type selector (all 8 types)
     - Parameter controls (duration, magnitude, start time)
     - Preview before injection
     - Active attacks list
   - **Alert Panel**: Active alerts with severity
     - Real-time alert updates
     - Acknowledgment system
     - Alert history
   - **Performance Metrics**: Latency, drift, throughput
   - **System Status**: Health indicators

3. **Features**:
   - Auto-refresh (WebSocket updates)
   - Time range selection
   - Alert filtering
   - Export capabilities

**Deliverables**:

- ✅ Interactive dashboard
- ✅ Real-time updates
- ✅ Alert visualization

---

### **PHASE 5: Testing & Optimization (Week 4)**

**Goal**: Validate performance and optimize

**Tasks**:

1. **Performance Testing**:

   - Run 1-week simulation
   - Measure all metrics
   - Identify bottlenecks

2. **Optimization**:

   - Memory usage optimization
   - Latency reduction
   - Throughput improvement

3. **Documentation**:
   - API documentation
   - Deployment guide
   - User manual

**Deliverables**:

- ✅ Performance report
- ✅ Optimized system
- ✅ Complete documentation

---

## 4. TECHNICAL SPECIFICATIONS

### 4.1 Data Structures

```python
# Hourly Data Point
@dataclass
class HourlyDataPoint:
    timestamp: pd.Timestamp
    load: float
    forecast: float
    benchmark: float
    scaling_ratio: float
    is_attack: bool
    attack_type: Optional[str]

# Detection Result
@dataclass
class DetectionResult:
    start_idx: int
    end_idx: int
    start_time: pd.Timestamp
    end_time: pd.Timestamp
    score: float
    method: str  # 'DP', 'STATISTICAL', 'BOTH'
    severity: str  # 'LOW', 'MEDIUM', 'HIGH', 'EMERGENCY'
    detection_latency: float  # seconds

# Alert
@dataclass
class Alert:
    id: str
    timestamp: pd.Timestamp
    severity: str
    message: str
    detection: DetectionResult
    acknowledged: bool
```

### 4.2 Configuration

```python
# streaming_config.py
STREAMING_CONFIG = {
    # Simulation speed
    'simulation_speed': 1.0,  # 1.0 = real-time, 10.0 = 10x speed

    # Window sizes
    'forecast_window_hours': 168,  # For lag features
    'detection_window_hours': 500,  # For DP algorithm
    'statistical_baseline_hours': 50,  # Minimum for statistical tests

    # Update frequencies
    'benchmark_update_days': 7,  # How often to retrain K-means
    'forecast_drift_check_hours': 24,  # Check drift every N hours

    # Performance targets
    'target_latency_seconds': 60,  # < 1 minute
    'target_throughput_hours_per_sec': 100,
    'max_memory_gb': 4,

    # Data retention
    'retention_days': 30,  # Keep last N days in memory
}
```

### 4.3 API Endpoints

```
GET  /api/status
     Returns: System status, uptime, current simulation time

GET  /api/forecast?hours=24
     Returns: Forecast data for last N hours

GET  /api/detections?limit=10
     Returns: Recent detection results

GET  /api/alerts?severity=HIGH&limit=20
     Returns: Active alerts

GET  /api/metrics
     Returns: Performance metrics (latency, drift, memory, throughput)

POST /api/simulation/start
     Body: {speed: 1.0, start_time: "2024-01-01T00:00:00"}

POST /api/simulation/stop

POST /api/simulation/inject_attack
     Body: {
         attack_type: "PULSE",  # PULSE, SCALING, RAMPING-TYPE1, etc.
         start_time: "2024-01-15T14:00:00",  # ISO format
         duration_hours: 10,
         magnitude: 3.0,  # 200% increase
         parameters: {  # Attack-specific params
             pulse_magnitude: 3.0,
             # ... other params
         }
     }

WS   /ws/stream
     WebSocket for real-time updates
```

---

## 5. GRID INTEGRATION LAYER (Production Deployment)

### 5.1 Overview

For connecting to **real grid hardware**, we need a **Data Acquisition & Integration Layer** that interfaces between grid SCADA/PMU systems and the MLAD software.

### 5.2 Architecture

```
Real Grid Hardware (SCADA/PMU/RTU)
    ↓
Protocol Adapters (IEC 61850, DNP3, Modbus, OPC-UA)
    ↓
Data Normalization Layer
    ↓
Data Validation & Quality Checks
    ↓
Message Queue (Kafka/RabbitMQ/Redis)
    ↓
MLAD Real-Time Engine (Your Current System)
```

### 5.3 Key Components

#### **A. Protocol Adapters**

- **IEC 61850**: Modern substation automation standard
- **DNP3**: Distributed Network Protocol (widely used)
- **Modbus**: Legacy but common (TCP/IP, RTU)
- **OPC-UA**: Unified Architecture (modern, secure)

#### **B. Data Normalization**

Converts all protocols to standard internal format:

- Timestamp alignment
- Unit conversion (MW, MWh, kV, Hz)
- Quality flag mapping

#### **C. Data Validation**

- Range checks
- Quality flags
- Missing data handling
- Timestamp validation

#### **D. Message Queue**

- Handles data flow
- Prevents data loss
- Manages backpressure
- Options: Kafka, RabbitMQ, Redis Streams

### 5.4 Implementation Priority

- **Phase 1 (Current)**: Simulation mode (historical data replay)
- **Phase 2 (Future)**: Add integration layer for real grid connection
- **Phase 3 (Production)**: Full production deployment

### 5.5 Security Considerations

- Encrypted communication (TLS/SSL)
- Authentication (certificates, keys)
- Firewall rules
- Network segmentation

---

## 6. FUTURE HARDWARE CONSIDERATIONS

### 5.1 Architecture Design for Hardware Acceleration

**Current (Software)**:

- Python-based implementation
- CPU processing
- In-memory data structures

**Future (Hardware)**:

- **FPGA Acceleration**:

  - LSTM inference (low latency)
  - DP algorithm (parallel processing)
  - Matrix operations (K-means)

- **GPU Acceleration**:
  - Batch LSTM predictions
  - Statistical tests (parallel)
  - Data preprocessing

### 5.2 Abstraction Layer

Design software with hardware abstraction:

```python
# Abstract base class for hardware acceleration
class InferenceEngine:
    def predict(self, features):
        raise NotImplementedError

class CPUInferenceEngine(InferenceEngine):
    # Current implementation
    pass

class FPGAInferenceEngine(InferenceEngine):
    # Future: FPGA-based LSTM
    pass

class GPUInferenceEngine(InferenceEngine):
    # Future: GPU-accelerated batch processing
    pass
```

### 5.3 Migration Path

1. **Phase 1 (Current)**: Pure software
2. **Phase 2**: Add hardware abstraction layer
3. **Phase 3**: Implement FPGA/GPU backends
4. **Phase 4**: Hybrid CPU + Hardware acceleration

---

## 7. FILE STRUCTURE

```
power_grid_protection/
├── streaming/
│   ├── __init__.py
│   ├── streaming_simulator.py      # Data stream generator
│   ├── rolling_forecast_engine.py   # Incremental LSTM
│   ├── benchmark_manager.py         # K-means benchmark
│   ├── realtime_detector.py         # Online detection
│   ├── performance_monitor.py      # Metrics tracking
│   ├── alert_manager.py             # Alert system
│   └── streaming_config.py          # Configuration
│
├── api/
│   ├── __init__.py
│   ├── api_server.py                # Flask/FastAPI server
│   ├── data_store.py                # Data storage
│   ├── simulation_controller.py     # Simulation control
│   └── websocket_handler.py         # WebSocket updates
│
├── frontend/
│   ├── dashboard/                   # React dashboard (if Option A)
│   │   ├── src/
│   │   │   ├── components/
│   │   │   │   ├── LoadChart.jsx           # Real-time load visualization
│   │   │   │   ├── AttackInjectionPanel.jsx  # Attack injection UI
│   │   │   │   ├── AlertPanel.jsx         # Active alerts
│   │   │   │   ├── AttackTimeline.jsx      # Attack timeline view
│   │   │   │   ├── MetricsPanel.jsx       # Performance metrics
│   │   │   │   └── StatusPanel.jsx         # System status
│   │   │   └── App.jsx
│   │   └── package.json
│   │
│   └── streamlit_dashboard.py       # Streamlit version (if Option C)
│
├── integration/                     # Grid integration layer (Future)
│   ├── __init__.py
│   ├── protocol_adapters.py         # IEC 61850, DNP3, Modbus, OPC-UA
│   ├── data_normalizer.py           # Protocol → standard format
│   ├── data_validator.py            # Quality checks
│   └── message_queue.py             # Kafka/RabbitMQ/Redis
│
├── tests/
│   ├── test_streaming_simulator.py
│   ├── test_realtime_detector.py
│   └── test_performance.py
│
└── docs/
    ├── API_DOCUMENTATION.md
    ├── DEPLOYMENT_GUIDE.md
    └── USER_MANUAL.md
```

---

## 8. SUCCESS CRITERIA

### Performance Targets

- ✅ **Operational Latency**: < 1 minute (data arrival → detection)
- ✅ **Detection Delay**: P95 < 2 hours (attack start → alert)
- ✅ **Forecast Drift**: < 10% accuracy degradation over 1 week
- ✅ **Throughput**: > 100 hours/second (faster than real-time)
- ✅ **Memory Usage**: < 4GB for 30-day retention
- ✅ **Zero Data Loss**: No dropped detections

### Functional Requirements

- ✅ Real-time data streaming
- ✅ Incremental forecasting
- ✅ Online anomaly detection
- ✅ Alert generation
- ✅ Performance monitoring
- ✅ Frontend visualization
- ✅ API access

---

## 9. RISKS & MITIGATION

| Risk                       | Impact | Probability | Mitigation                                          |
| -------------------------- | ------ | ----------- | --------------------------------------------------- |
| Memory overflow            | High   | Medium      | Implement data retention policies, periodic cleanup |
| Detection latency too high | High   | Low         | Optimize algorithms, use caching                    |
| Forecast drift significant | Medium | High        | Implement retraining triggers, alert on drift       |
| K-means update bottleneck  | Medium | Medium      | Use periodic updates, not real-time                 |
| Frontend performance       | Low    | Low         | Use efficient charting libraries, limit data points |

---

## 10. QUESTIONS FOR DISCUSSION

Before implementation, please confirm:

1. **Frontend Technology Preference**:

   - [ ] React (more flexible, better for production)
   - [ ] Streamlit (faster to build, Python-based)
   - [ ] Flask + Plotly Dash (middle ground)

2. **K-Means Update Strategy**:

   - [ ] Static (no updates, measure drift separately) - **Recommended**
   - [ ] Periodic retraining (every 7 days)
   - [ ] Rolling window updates (more complex)

3. **Simulation Speed**:

   - [ ] Real-time only (1x speed)
   - [ ] Configurable (1x, 10x, 100x) - **Recommended**

4. **Data Retention**:

   - [ ] 7 days (lower memory)
   - [ ] 30 days (more history) - **Recommended**
   - [ ] Configurable

5. **Attack Injection**:
   - [ ] Pre-configured scenarios only
   - [ ] Runtime injection via API - **Recommended**

---

## 11. NEXT STEPS

1. **Review this plan** and provide feedback on questions above
2. **Confirm technology choices** (frontend, update strategies)
3. **Start Phase 1** implementation
4. **Iterative development** with regular testing

---

## 12. ESTIMATED TIMELINE

| Phase                    | Duration | Dependencies |
| ------------------------ | -------- | ------------ |
| Phase 1: Core Engine     | 1 week   | None         |
| Phase 2: Detection Logic | 1 week   | Phase 1      |
| Phase 3: API & Backend   | 1 week   | Phase 1, 2   |
| Phase 4: Frontend        | 1 week   | Phase 3      |
| Phase 5: Testing         | 3-4 days | All phases   |

**Total**: 3-4 weeks

---

## CONCLUSION

This real-time simulation system is **feasible and achievable** within the 3-4 week timeline. The architecture is designed to be:

- ✅ **Software-first** (immediate implementation)
- ✅ **Hardware-ready** (future acceleration possible)
- ✅ **Production-oriented** (scalable, maintainable)
- ✅ **User-friendly** (frontend dashboard)

The main challenges (K-means updates, forecast drift) have practical solutions that don't compromise the core functionality.

**Ready to proceed?** Please review and provide feedback on the questions in **Section 10** (QUESTIONS FOR DISCUSSION).

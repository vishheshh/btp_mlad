"""
Real-Time Detector - Phase 2 Enhanced Version
Implements online hybrid detection (DP + Statistical) with alert system
"""
import numpy as np
import pandas as pd
from typing import Optional, List, Dict
from datetime import datetime
from collections import deque
import sys
import os

# Add parent directory to path for imports
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import config
from mlad_anomaly_detection import (
    detect_anomaly_timing,
    statistical_anomaly_detection,
    hybrid_detection,
    intervals_overlap
)
from streaming.streaming_config import *


class Alert:
    """Represents an anomaly detection alert."""
    
    SEVERITY_LOW = "LOW"
    SEVERITY_MEDIUM = "MEDIUM"
    SEVERITY_HIGH = "HIGH"
    SEVERITY_EMERGENCY = "EMERGENCY"
    
    def __init__(self, start_idx: int, end_idx: int, start_time: pd.Timestamp,
                 end_time: pd.Timestamp, method: str, severity: str,
                 dp_score: float = None, stat_score: float = None,
                 p_value: float = None, cohens_d: float = None):
        """
        Initialize an alert.
        
        Args:
            start_idx: Start index in detection window
            end_idx: End index in detection window
            start_time: Start timestamp
            end_time: End timestamp
            method: Detection method ('DP', 'STATISTICAL', 'BOTH')
            severity: Alert severity level
            dp_score: DP detection score
            stat_score: Statistical detection score
            p_value: Statistical p-value
            cohens_d: Cohen's d effect size
        """
        self.start_idx = start_idx
        self.end_idx = end_idx
        self.start_time = start_time
        self.end_time = end_time
        self.method = method
        self.severity = severity
        self.dp_score = dp_score
        self.stat_score = stat_score
        self.p_value = p_value
        self.cohens_d = cohens_d
        self.created_at = datetime.now()
        self.acknowledged = False
        
    def to_dict(self) -> Dict:
        """Convert alert to dictionary."""
        return {
            'start_idx': self.start_idx,
            'end_idx': self.end_idx,
            'start_time': str(self.start_time),
            'end_time': str(self.end_time),
            'method': self.method,
            'severity': self.severity,
            'dp_score': self.dp_score,
            'stat_score': self.stat_score,
            'p_value': self.p_value,
            'cohens_d': self.cohens_d,
            'created_at': str(self.created_at),
            'acknowledged': self.acknowledged
        }


class RealTimeDetector:
    """
    Enhanced real-time detector with Phase 2 features:
    - Sliding window DP detection
    - Online statistical detection
    - Hybrid fusion
    - Alert system
    """

    def __init__(self):
        """Initialize the detector."""
        self.scaling_history = []  # Store scaling ratios
        self.detection_window = []  # Sliding window for detection (500 hours)
        self.timestamps_window = []  # Timestamps for detection window
        self.max_window_size = DETECTION_WINDOW_HOURS
        
        # Statistical baseline buffer
        self.statistical_baseline = deque(maxlen=STATISTICAL_BASELINE_HOURS)
        self.baseline_initialized = False
        
        # Alert system
        self.alert_queue = deque(maxlen=100)  # Keep last 100 alerts
        self.active_alerts = []  # Currently active alerts
        
        # Detection tracking
        self.last_detection_time = None
        self.detection_latency_history = deque(maxlen=100)
        self.instant_counter = 0
        self.instant_start_idx = None
        self.last_emergency_alert_time = None

    def process_hourly_data(self, forecast: float, benchmark: float,
                           actual_load: float, timestamp: pd.Timestamp) -> dict:
        """
        Process one hour of data and run detection.

        Args:
            forecast: LSTM forecast value
            benchmark: K-means benchmark value
            actual_load: Actual load value
            timestamp: Timestamp

        Returns:
            Dictionary with scaling ratio, detection results, and alerts
        """
        # Calculate scaling ratio using actual load against configured reference
        scaling_reference = getattr(config, 'SCALING_REFERENCE', 'forecast').lower()
        if scaling_reference == 'benchmark':
            reference_value = benchmark
        else:
            reference_value = forecast
        epsilon = getattr(config, 'SCALING_EPSILON', 1e-6)
        scaling_ratio = actual_load / (reference_value + epsilon)

        # Calculate deviation
        deviation = abs(scaling_ratio - 1.0)

        # Store in history
        data_point = {
            'timestamp': timestamp,
            'forecast': forecast,
            'benchmark': benchmark,
            'actual_load': actual_load,
            'scaling_ratio': scaling_ratio,
            'deviation': deviation
        }

        self.scaling_history.append(data_point)
        self.detection_window.append(data_point)
        self.timestamps_window.append(timestamp)

        # Maintain window size
        if len(self.detection_window) > self.max_window_size:
            self.detection_window.pop(0)
            self.timestamps_window.pop(0)

        # Clean old history (keep last RETENTION_HOURS)
        if len(self.scaling_history) > RETENTION_HOURS:
            self.scaling_history.pop(0)

        # Update statistical baseline
        if deviation <= config.MAGNITUDE_THRESHOLD:
            self.statistical_baseline.append(deviation)
            if len(self.statistical_baseline) >= STATISTICAL_BASELINE_HOURS:
                self.baseline_initialized = True

        # Run detection if we have enough data
        detections = []
        alerts = []
        
        instant_detection = self._check_instant_alert(deviation)
        if instant_detection:
            detections.append(instant_detection)
        
        if len(self.detection_window) >= config.MIN_ANOMALY_DURATION:
            detections.extend(self.run_detection())
            alerts = self.generate_alerts(detections)

        return {
            'scaling_ratio': scaling_ratio,
            'deviation': deviation,
            'is_anomalous': deviation > config.MAGNITUDE_THRESHOLD,
            'timestamp': timestamp,
            'detections': detections,
            'alerts': [alert.to_dict() for alert in alerts]
        }

    def run_detection(self) -> List[Dict]:
        """
        Run hybrid detection on current window.
        
        Returns:
            List of detection dictionaries
        """
        if len(self.detection_window) < config.MIN_ANOMALY_DURATION:
            return []

        # Extract scaling ratios and timestamps
        scaling_data = np.array([d['scaling_ratio'] for d in self.detection_window])
        timestamps = pd.DatetimeIndex(self.timestamps_window)

        # Run hybrid detection
        if config.USE_HYBRID_DETECTION:
            detections = hybrid_detection(
                scaling_data,
                timestamps=timestamps,
                max_detections=10
            )
        else:
            # Fallback to DP only
            dp_results = detect_anomaly_timing(
                scaling_data,
                timestamps=timestamps,
                max_detections=10
            )
            detections = []
            if dp_results:
                for det in dp_results:
                    if isinstance(det, tuple) and len(det) >= 3:
                        start, end, score = det[:3]
                        detections.append({
                            'start': start,
                            'end': end,
                            'dp_score': score,
                            'method': 'DP'
                        })

        # Convert to relative indices (from start of window)
        # Note: detections are already relative to the window
        return detections

    def _check_instant_alert(self, deviation: float) -> Optional[Dict]:
        """
        Check if deviation exceeds the instant alert threshold for the required duration.
        """
        threshold = getattr(config, 'INSTANT_ALERT_THRESHOLD', None)
        confirmation_hours = getattr(config, 'INSTANT_ALERT_CONFIRMATION_HOURS', 0)
        
        if threshold is None or confirmation_hours <= 0:
            return None
        
        current_idx = len(self.detection_window) - 1
        if current_idx < 0:
            return None
        
        if deviation >= threshold:
            if self.instant_counter == 0:
                self.instant_start_idx = current_idx
            self.instant_counter += 1
        else:
            self.instant_counter = 0
            self.instant_start_idx = None
            return None
        
        if self.instant_counter >= confirmation_hours and self.instant_start_idx is not None:
            start_idx = max(0, current_idx - confirmation_hours + 1)
            detection = {
                'start': start_idx,
                'end': current_idx,
                'method': 'INSTANT',
                'peak_deviation': deviation
            }
            self.instant_counter = 0
            self.instant_start_idx = None
            return detection
        
        return None

    def generate_alerts(self, detections: List[Dict]) -> List[Alert]:
        """
        Generate alerts from detections.
        
        Args:
            detections: List of detection dictionaries
            
        Returns:
            List of Alert objects
        """
        alerts = []
        
        for det in detections:
            start_idx = det['start']
            end_idx = det['end']
            method = det.get('method', 'DP')
            
            # Get timestamps
            if start_idx < len(self.timestamps_window) and end_idx < len(self.timestamps_window):
                start_time = self.timestamps_window[start_idx]
                end_time = self.timestamps_window[end_idx]
            else:
                continue  # Skip invalid detections
            
            # Determine severity based on method and scores
            severity = self._determine_severity(det)
            severity = self._apply_emergency_cooldown(severity, start_time)
            
            # Check if this overlaps with existing active alerts
            is_new = True
            for existing_alert in self.active_alerts:
                if intervals_overlap(
                    (start_idx, end_idx),
                    (existing_alert.start_idx, existing_alert.end_idx)
                ):
                    is_new = False
                    break
            
            if is_new:
                alert = Alert(
                    start_idx=start_idx,
                    end_idx=end_idx,
                    start_time=start_time,
                    end_time=end_time,
                    method=method,
                    severity=severity,
                    dp_score=det.get('dp_score'),
                    stat_score=det.get('stat_score'),
                    p_value=det.get('p_value'),
                    cohens_d=det.get('cohens_d')
                )
                
                alerts.append(alert)
                self.alert_queue.append(alert)
                self.active_alerts.append(alert)
                
                # Track detection latency
                if self.last_detection_time is not None:
                    latency = (start_time - self.last_detection_time).total_seconds() / 3600
                    self.detection_latency_history.append(latency)
                
                self.last_detection_time = start_time
        
        # Clean up old active alerts (older than window size)
        if len(self.detection_window) > 0:
            current_time = self.timestamps_window[-1]
            self.active_alerts = [
                alert for alert in self.active_alerts
                if (current_time - alert.end_time).total_seconds() / 3600 < self.max_window_size
            ]
        
        return alerts

    def _apply_emergency_cooldown(self, severity: str, timestamp: pd.Timestamp) -> str:
        """
        Prevent emergency alert spam by enforcing a cooldown window.
        """
        if severity != Alert.SEVERITY_EMERGENCY:
            return severity
        
        cooldown_hours = getattr(config, 'EMERGENCY_COOLDOWN_HOURS', 0)
        if cooldown_hours <= 0:
            self.last_emergency_alert_time = timestamp
            return severity
        
        if self.last_emergency_alert_time is None:
            self.last_emergency_alert_time = timestamp
            return severity
        
        elapsed_hours = (timestamp - self.last_emergency_alert_time).total_seconds() / 3600
        if elapsed_hours < cooldown_hours:
            return Alert.SEVERITY_HIGH
        
        self.last_emergency_alert_time = timestamp
        return severity

    def _determine_severity(self, detection: Dict) -> str:
        """
        Determine alert severity based on detection characteristics.
        
        Args:
            detection: Detection dictionary
            
        Returns:
            Severity level string
        """
        method = detection.get('method', 'DP')
        peak_deviation = detection.get('peak_deviation')
        
        if peak_deviation is None and len(self.detection_window) > 0:
            start_idx = detection['start']
            end_idx = detection['end']
            if start_idx < len(self.detection_window) and end_idx < len(self.detection_window):
                scaling_ratios = [
                    self.detection_window[i]['scaling_ratio']
                    for i in range(start_idx, min(end_idx + 1, len(self.detection_window)))
                ]
                peak_deviation = max([abs(r - 1.0) for r in scaling_ratios])
        
        if peak_deviation is not None and peak_deviation >= config.EMERGENCY_THRESHOLD:
            return Alert.SEVERITY_EMERGENCY
        
        if method == 'INSTANT':
            if peak_deviation is not None and peak_deviation >= config.MAGNITUDE_THRESHOLD:
                return Alert.SEVERITY_HIGH
            else:
                return Alert.SEVERITY_MEDIUM
        
        # Determine severity based on method and scores
        if method == 'BOTH':
            # Both methods agree - high confidence
            p_value = detection.get('p_value', 1.0)
            if p_value < 0.001:
                return Alert.SEVERITY_HIGH
            else:
                return Alert.SEVERITY_MEDIUM
        elif method == 'STATISTICAL':
            # Statistical only - rigorous but might be weak
            p_value = detection.get('p_value', 1.0)
            if p_value < 0.001:
                return Alert.SEVERITY_MEDIUM
            else:
                return Alert.SEVERITY_LOW
        else:  # DP only
            # DP only - magnitude-based
            dp_score = detection.get('dp_score', 0)
            if dp_score > 1.0:
                return Alert.SEVERITY_HIGH
            elif dp_score > 0.3:
                return Alert.SEVERITY_MEDIUM
            else:
                return Alert.SEVERITY_LOW

    def get_recent_scaling_data(self, hours: int = 24) -> np.ndarray:
        """
        Get recent scaling ratios for analysis.

        Args:
            hours: Number of recent hours to return

        Returns:
            Array of scaling ratios
        """
        if len(self.scaling_history) == 0:
            return np.array([])

        recent = self.scaling_history[-hours:]
        return np.array([d['scaling_ratio'] for d in recent])

    def get_window_size(self) -> int:
        """Get current window size."""
        return len(self.detection_window)

    def get_alerts(self, severity: str = None, acknowledged: bool = None) -> List[Alert]:
        """
        Get alerts from queue.
        
        Args:
            severity: Filter by severity (optional)
            acknowledged: Filter by acknowledgment status (optional)
            
        Returns:
            List of Alert objects
        """
        alerts = list(self.alert_queue)
        
        if severity:
            alerts = [a for a in alerts if a.severity == severity]
        
        if acknowledged is not None:
            alerts = [a for a in alerts if a.acknowledged == acknowledged]
        
        return alerts

    def acknowledge_alert(self, alert: Alert):
        """Mark an alert as acknowledged."""
        alert.acknowledged = True

    def get_detection_latency_stats(self) -> Dict:
        """Get detection latency statistics."""
        if len(self.detection_latency_history) == 0:
            return {
                'mean': None,
                'p50': None,
                'p95': None,
                'count': 0
            }
        
        latencies = sorted(list(self.detection_latency_history))
        
        return {
            'mean': sum(latencies) / len(latencies),
            'p50': latencies[len(latencies) // 2],
            'p95': latencies[int(len(latencies) * 0.95)] if len(latencies) > 0 else None,
            'count': len(latencies)
        }

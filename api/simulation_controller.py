"""
Simulation Controller - Manages simulation state and control
"""
import threading
import time
from datetime import datetime
from typing import Optional, Dict, List
import sys
import os

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from streaming.streaming_simulator import DataStreamSimulator
from streaming.rolling_forecast_engine import RollingForecastEngine
from streaming.benchmark_manager import BenchmarkManager
from streaming.realtime_detector import RealTimeDetector
from streaming.performance_monitor import PerformanceMonitor
from streaming.streaming_config import *
import config
from mlad_anomaly_detection import load_and_preprocess_data
import numpy as np


class SimulationController:
    """
    Controls the simulation lifecycle and attack injection.
    """

    def __init__(self, data_store):
        """
        Initialize the simulation controller.

        Args:
            data_store: DataStore instance for storing results
        """
        self.data_store = data_store
        self.simulator = None
        self.forecast_engine = None
        self.benchmark_manager = None
        self.detector = None
        self.monitor = None
        
        self.is_running = False
        self.speed = SIMULATION_SPEED
        self.simulation_thread = None
        self.callbacks = []  # Callbacks for real-time updates
        
        # Attack injection
        self.active_attacks = []  # List of active attack injections
        self.all_injected_attacks = []  # List of all injected attacks (including completed)
    
    def reset(self):
        """Reset simulation state - clear all data and attacks."""
        self.stop()
        self.data_store.reset()
        self.active_attacks.clear()
        self.all_injected_attacks.clear()
        # Reset detector state if needed
        if hasattr(self.detector, 'reset'):
            self.detector.reset()
        # Reset forecast engine history
        if hasattr(self.forecast_engine, 'forecast_history'):
            self.forecast_engine.forecast_history.clear()
        # Reset performance monitor
        if self.monitor:
            self.monitor.reset()

    def initialize(self):
        """Initialize all components."""
        print("Initializing simulation controller...")
        
        self.simulator = DataStreamSimulator(speed=self.speed)
        self.forecast_engine = RollingForecastEngine()
        self.benchmark_manager = BenchmarkManager()
        self.detector = RealTimeDetector()
        self.monitor = PerformanceMonitor()
        
        # Load data
        self.simulator.load_data()
        
        # Initialize forecast engine
        df = load_and_preprocess_data(config.DATASET_DIR)
        split_idx = int(len(df) * config.TRAIN_TEST_SPLIT_RATIO)
        initial_data = df.iloc[:split_idx+200].copy()
        self.forecast_engine.initialize(initial_data.iloc[:200])
        
        print("Simulation controller initialized")

    def start(self, speed: float = None):
        """
        Start the simulation.

        Args:
            speed: Simulation speed multiplier (optional)
        """
        if self.is_running:
            return {"status": "error", "message": "Simulation already running"}
        
        if speed is not None:
            self.speed = speed
            if self.simulator:
                self.simulator.speed = speed
        
        if self.simulator is None:
            self.initialize()
        
        self.is_running = True
        self.monitor.start()
        self.simulator.start()
        
        # Start simulation thread
        self.simulation_thread = threading.Thread(target=self._run_simulation, daemon=True)
        self.simulation_thread.start()
        
        return {"status": "success", "message": "Simulation started"}

    def stop(self):
        """Stop the simulation."""
        if not self.is_running:
            return {"status": "error", "message": "Simulation not running"}
        
        self.is_running = False
        if self.simulator:
            self.simulator.stop()
        
        return {"status": "success", "message": "Simulation stopped"}

    def set_speed(self, speed: float):
        """
        Set simulation speed.

        Args:
            speed: Speed multiplier (1.0 = real-time)
        """
        self.speed = speed
        if self.simulator:
            self.simulator.speed = speed
        return {"status": "success", "speed": speed}

    def inject_attack(self, attack_type: str, start_hour: int, duration: int,
                     magnitude: float, **kwargs) -> Dict:
        """
        Inject an attack into the simulation.

        Args:
            attack_type: Type of attack (PULSE, SCALING, RAMPING, etc.)
            start_hour: Hour offset from current position to inject
            duration: Duration in hours
            magnitude: Attack magnitude parameter
            **kwargs: Additional attack parameters

        Returns:
            Dictionary with injection result
        """
        if not self.is_running:
            return {"status": "error", "message": "Simulation not running"}
        
        # Get current simulation position
        current_hour = self.simulator.current_index - 1 if self.simulator else 0
        
        attack_info = {
            'type': attack_type,
            'start_hour': start_hour,
            'duration': duration,
            'magnitude': magnitude,
            'parameters': kwargs,
            'injected_at': datetime.now(),
            'injection_hour': current_hour  # Store when attack was injected
        }
        
        self.active_attacks.append(attack_info)
        self.all_injected_attacks.append(attack_info.copy())  # Keep record of all attacks
        
        return {
            "status": "success",
            "message": f"Attack {attack_type} scheduled",
            "attack": attack_info
        }

    def get_status(self) -> Dict:
        """
        Get simulation status.

        Returns:
            Dictionary with status information
        """
        status = {
            'is_running': self.is_running,
            'speed': self.speed,
            'active_attacks': len(self.active_attacks)
        }
        
        if self.simulator:
            status['progress'] = self.simulator.get_progress()
        
        if self.monitor:
            status['metrics'] = self.monitor.get_all_metrics()
        
        return status

    def register_callback(self, callback):
        """
        Register a callback for real-time updates.

        Args:
            callback: Callback function that receives update data
        """
        self.callbacks.append(callback)

    def _run_simulation(self):
        """Run simulation loop (runs in separate thread)."""
        while self.is_running:
            try:
                # Start timing for this hour
                hour_start_time = time.time()
                
                # Get next hour
                data_point = self.simulator.get_next_hour()
                if data_point is None:
                    self.is_running = False
                    break
                
                # Generate forecast
                forecast = self.forecast_engine.predict_next_hour(data_point)
                
                # Get benchmark
                recent_forecasts = np.array(self.forecast_engine.forecast_history[-24:])
                if len(recent_forecasts) < 24:
                    recent_forecasts = np.pad(recent_forecasts, (24 - len(recent_forecasts), 0),
                                            mode='edge')
                
                benchmark = self.benchmark_manager.get_benchmark_for_hour(
                    recent_forecasts,
                    data_point['timestamp'].hour
                )
                
                # Apply attack injection if needed
                actual_load = data_point['load']
                current_hour = self.simulator.current_index - 1
                
                for attack in self.active_attacks[:]:
                    # Calculate absolute attack start hour (relative to simulation start)
                    injection_hour = attack.get('injection_hour', 0)
                    attack_start_relative = attack['start_hour']  # Relative to injection point
                    attack_start_absolute = injection_hour + attack_start_relative
                    attack_end_absolute = attack_start_absolute + attack['duration']
                    
                    # Debug: print attack timing (remove after testing)
                    # print(f"Attack {attack['type']}: injection={injection_hour}, relative_start={attack_start_relative}, absolute_start={attack_start_absolute}, current={current_hour}")
                    
                    if attack_start_absolute <= current_hour < attack_end_absolute:
                        # Apply attack - hour_offset is position within attack duration
                        hour_offset = current_hour - attack_start_absolute
                        original_load = actual_load
                        actual_load = self._apply_attack(
                            actual_load, attack['type'], attack['magnitude'],
                            hour_offset, attack['duration'],
                            **attack['parameters']
                        )
                        # Debug: verify attack is being applied (remove after testing)
                        # if abs(actual_load - original_load) > 0.01:
                        #     print(f"Attack applied: {attack['type']} at hour {current_hour}, load changed from {original_load:.2f} to {actual_load:.2f}")
                    elif current_hour >= attack_end_absolute:
                        # Remove completed attack
                        self.active_attacks.remove(attack)
                
                # Process detection
                detection_result = self.detector.process_hourly_data(
                    forecast=forecast,
                    benchmark=benchmark,
                    actual_load=actual_load,
                    timestamp=data_point['timestamp']
                )
                
                # Store data
                self.data_store.add_hourly_data(
                    timestamp=data_point['timestamp'],
                    forecast=forecast,
                    actual=actual_load,
                    benchmark=benchmark,
                    scaling_ratio=detection_result['scaling_ratio']
                )
                
                # Store detections and alerts
                for det in detection_result.get('detections', []):
                    self.data_store.add_detection(det)
                
                # Store alerts
                for alert in detection_result.get('alerts', []):
                    self.data_store.add_alert(alert)
                
                # AUTO-STOP ON CRITICAL ALERTS - COMMENTED OUT
                # # Check for critical alerts before storing
                # critical_alert_detected = False
                # for alert in detection_result.get('alerts', []):
                #     alert_severity = alert.get('severity', 'LOW')
                #     if alert_severity in ['EMERGENCY', 'HIGH']:
                #         critical_alert_detected = True
                #     self.data_store.add_alert(alert)
                # 
                # # Auto-stop simulation on EMERGENCY or HIGH severity alerts
                # if critical_alert_detected:
                #     print(f"⚠️ CRITICAL ALERT DETECTED (EMERGENCY/HIGH) - Auto-stopping simulation")
                #     self.is_running = False
                #     break  # Exit loop to stop simulation
                
                # Record performance - measure processing time
                processing_time = time.time() - hour_start_time
                if self.monitor:
                    self.monitor.record_processing(processing_time)
                
                # Notify callbacks
                update_data = {
                    'timestamp': str(data_point['timestamp']),
                    'forecast': forecast,
                    'actual': actual_load,
                    'benchmark': benchmark,
                    'scaling_ratio': detection_result['scaling_ratio'],
                    'alerts': detection_result.get('alerts', [])
                }
                
                for callback in self.callbacks:
                    try:
                        callback(update_data)
                    except Exception as e:
                        print(f"Callback error: {e}")
                
                # Simulate delay based on speed
                if self.speed > 0:
                    time.sleep(1.0 / self.speed)
                    
            except Exception as e:
                print(f"Simulation error: {e}")
                import traceback
                traceback.print_exc()
                self.is_running = False
                break

    def _apply_attack(self, base_load: float, attack_type: str, magnitude: float,
                     hour_offset: int, duration: int, **kwargs) -> float:
        """
        Apply attack modification to load value.

        Args:
            base_load: Base load value
            attack_type: Type of attack
            magnitude: Attack magnitude (as multiplier, e.g., 1.7 = 70% increase)
            hour_offset: Current hour within attack duration (0-indexed)
            duration: Total attack duration
            **kwargs: Additional parameters

        Returns:
            Modified load value
        """
        if attack_type == 'PULSE':
            # Sharp spike - constant magnitude throughout
            return base_load * (1 + magnitude)
        
        elif attack_type == 'SCALING':
            # Constant scaling - same as PULSE
            return base_load * (1 + magnitude)
        
        elif attack_type == 'RAMPING':
            # Linear ramp from 0 to magnitude
            if duration > 0:
                progress = hour_offset / max(duration - 1, 1)  # Normalize to [0, 1]
                return base_load * (1 + magnitude * progress)
            else:
                return base_load * (1 + magnitude)
        
        elif attack_type == 'RANDOM':
            # Random noise - different random value each hour
            # Magnitude represents max deviation percentage (e.g., 1.7 = ±170% variation)
            # Use hour_offset + base_load as seed to ensure different values each hour
            # but reproducible for same hour_offset
            seed_value = int(hour_offset * 1000 + int(base_load)) % (2**32)
            rng = np.random.RandomState(seed_value)
            # Random factor between -magnitude and +magnitude
            noise_factor = rng.uniform(-magnitude, magnitude)
            # Apply as multiplicative factor: base_load * (1 + noise_factor)
            # With magnitude 1.7, noise_factor can be -1.7 to +1.7
            # So load becomes base_load * (1 + 1.7) = 2.7x or base_load * (1 - 1.7) = -0.7x
            # To make it more visible and reasonable, use magnitude as percentage
            # For magnitude 1.7, we want ±170% variation, so noise_factor should be ±1.7
            modified_load = base_load * (1 + noise_factor)
            # Ensure load doesn't go negative (minimum 10% of original)
            return max(modified_load, base_load * 0.1)
        
        elif attack_type == 'SMOOTH-CURVE':
            # Smooth curve attack (sine wave)
            if duration > 0:
                progress = hour_offset / max(duration - 1, 1)
                curve_factor = magnitude * np.sin(np.pi * progress)
                return base_load * (1 + curve_factor)
            else:
                return base_load
        
        elif attack_type == 'POINT-BURST':
            # Single point burst at midpoint
            if hour_offset == duration // 2:
                return base_load * (1 + magnitude)
            else:
                return base_load
        
        elif attack_type == 'CONTEXTUAL-SEASONAL':
            # Seasonal pattern attack
            seasonal_factor = magnitude * np.sin(2 * np.pi * hour_offset / 24)
            return base_load * (1 + seasonal_factor)
        
        elif attack_type == 'RAMPING-TYPE2':
            # Ramp up then down
            if duration > 0:
                progress = hour_offset / max(duration - 1, 1)
                if progress < 0.5:
                    ramp_factor = magnitude * (progress * 2)  # 0 to magnitude
                else:
                    ramp_factor = magnitude * (2 - progress * 2)  # magnitude to 0
                return base_load * (1 + ramp_factor)
            else:
                return base_load
        
        else:
            # Default: no modification
            return base_load


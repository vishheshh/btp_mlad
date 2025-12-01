"""
Integration test for Phase 1 & 2 - Test all components together
"""
import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import pandas as pd
import numpy as np
import time
from streaming.streaming_simulator import DataStreamSimulator
from streaming.rolling_forecast_engine import RollingForecastEngine
from streaming.benchmark_manager import BenchmarkManager
from streaming.realtime_detector import RealTimeDetector
from streaming.performance_monitor import PerformanceMonitor
import config
from mlad_anomaly_detection import load_and_preprocess_data


def test_integration():
    """Test all components integrated."""
    print("="*60)
    print("PHASE 1 & 2 INTEGRATION TEST")
    print("="*60)

    # Initialize components
    print("\n1. Initializing components...")
    simulator = DataStreamSimulator(speed=100.0)  # Fast for testing
    forecast_engine = RollingForecastEngine()
    benchmark_manager = BenchmarkManager()
    detector = RealTimeDetector()
    monitor = PerformanceMonitor()

    # Load data
    print("\n2. Loading data...")
    simulator.load_data()

    # Initialize forecast engine with initial data
    print("\n3. Initializing forecast engine...")
    df = load_and_preprocess_data(config.DATASET_DIR)
    split_idx = int(len(df) * config.TRAIN_TEST_SPLIT_RATIO)
    initial_data = df.iloc[:split_idx+200].copy()
    forecast_engine.initialize(initial_data.iloc[:200])

    # Start monitoring
    monitor.start()

    # Process first 100 hours (need more for detection window)
    print("\n4. Processing 100 hours of data...")
    simulator.start()

    results = []
    total_alerts = 0
    for i in range(100):
        # Get next hour
        data_point = simulator.get_next_hour()
        if data_point is None:
            break

        # Start timing
        start_time = time.time()

        # Generate forecast
        forecast = forecast_engine.predict_next_hour(data_point)

        # Get benchmark (need recent forecasts for pattern matching)
        recent_forecasts = np.array(forecast_engine.forecast_history[-24:])
        if len(recent_forecasts) < 24:
            # Pad if needed
            recent_forecasts = np.pad(recent_forecasts, (24 - len(recent_forecasts), 0),
                                    mode='edge')
        benchmark = benchmark_manager.get_benchmark_for_hour(
            recent_forecasts,
            data_point['timestamp'].hour
        )

        # Process detection
        detection_result = detector.process_hourly_data(
            forecast=forecast,
            benchmark=benchmark,
            actual_load=data_point['load'],
            timestamp=data_point['timestamp']
        )

        # Record processing time
        processing_time = time.time() - start_time
        monitor.record_processing(processing_time)

        # Store result
        results.append({
            'timestamp': data_point['timestamp'],
            'actual': data_point['load'],
            'forecast': forecast,
            'benchmark': benchmark,
            'scaling_ratio': detection_result['scaling_ratio'],
            'deviation': detection_result['deviation'],
            'is_anomalous': detection_result['is_anomalous'],
            'processing_time': processing_time,
            'detections': len(detection_result.get('detections', [])),
            'alerts': len(detection_result.get('alerts', []))
        })
        
        # Track alerts
        if detection_result.get('alerts'):
            total_alerts += len(detection_result['alerts'])

        if (i + 1) % 20 == 0:
            print(f"  Processed {i + 1}/100 hours...")

    # Print results
    print("\n5. Results Summary:")
    print("-" * 60)
    df_results = pd.DataFrame(results)
    print(f"Total hours processed: {len(df_results)}")
    print(f"Mean forecast: {df_results['forecast'].mean():.2f} MWh")
    print(f"Mean actual: {df_results['actual'].mean():.2f} MWh")
    print(f"Mean scaling ratio: {df_results['scaling_ratio'].mean():.4f}")
    print(f"Anomalous hours: {df_results['is_anomalous'].sum()}")
    print(f"Total alerts generated: {total_alerts}")
    
    # Phase 2: Detection and Alert Summary
    print("\n6. Phase 2 Detection Summary:")
    print("-" * 60)
    all_alerts = detector.get_alerts()
    print(f"Total alerts in queue: {len(all_alerts)}")
    
    if len(all_alerts) > 0:
        severity_counts = {}
        method_counts = {}
        for alert in all_alerts:
            severity_counts[alert.severity] = severity_counts.get(alert.severity, 0) + 1
            method_counts[alert.method] = method_counts.get(alert.method, 0) + 1
        
        print(f"Severity breakdown: {severity_counts}")
        print(f"Method breakdown: {method_counts}")
        
        # Show sample alerts
        print("\nSample alerts (first 3):")
        for alert in all_alerts[:3]:
            print(f"  - {alert.severity} alert ({alert.method}): "
                  f"{alert.start_time} to {alert.end_time}")
    
    # Detection latency
    latency_stats = detector.get_detection_latency_stats()
    if latency_stats['count'] > 0:
        print(f"\nDetection Latency:")
        print(f"  Mean: {latency_stats['mean']:.2f} hours")
        print(f"  P50: {latency_stats['p50']:.2f} hours")
        if latency_stats['p95']:
            print(f"  P95: {latency_stats['p95']:.2f} hours")

    # Performance metrics
    print("\n7. Performance Metrics:")
    print("-" * 60)
    metrics = monitor.get_all_metrics()
    print(f"Throughput: {metrics['throughput_hours_per_sec']:.2f} hours/sec")
    print(f"Mean latency: {metrics['latency']['mean']:.4f} seconds")
    print(f"P95 latency: {metrics['latency']['p95']:.4f} seconds")
    print(f"Memory usage: {metrics['memory']['memory_mb']:.2f} MB")
    print(f"Memory within limit: {metrics['memory']['within_limit']}")

    # Forecast drift
    print("\n8. Forecast Drift:")
    print("-" * 60)
    drift = forecast_engine.calculate_forecast_drift(window_hours=24)
    if drift['sufficient_data']:
        print(f"MAE: {drift['mae']:.2f} MWh")
        print(f"MAPE: {drift['mape']:.2f}%")
        if drift['drift_percentage'] is not None:
            print(f"Drift: {drift['drift_percentage']:.2f}%")
    else:
        print("Insufficient data for drift calculation")

    print("\n" + "="*60)
    print("INTEGRATION TEST COMPLETE")
    print("="*60)

    return results, metrics


if __name__ == "__main__":
    results, metrics = test_integration()


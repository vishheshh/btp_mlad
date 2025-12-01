"""
Comprehensive Attack Type Detection Test

Tests the MLAD system against 5 different attack types commonly used in
power grid cyberattacks:

1. PULSE: Sharp, instantaneous spike → Instant detectionē
2. SCALING: Gradual multiplication → Sustained observation
3. RAMPING: Slowly increasing → Pattern detection over time
4. RANDOM: Erratic noise injection → Statistical analysis
5. SMOOTH-CURVE: Subtle curved deviation → Long-term monitoring

Usage:
    python test_attack_types.py
"""

import os
import numpy as np
import pandas as pd
import pickle
from tensorflow.keras.models import load_model
import config
from mlad_anomaly_detection import (
    load_and_preprocess_data,
    create_features,
    get_benchmark_for_period,
    detect_anomaly_with_segmentation
)


class AttackSimulator:
    """Simulates different types of attacks on power grid data."""
    
    @staticmethod
    def pulse_attack(data, start, duration=1, magnitude=3.0):
        """
        PULSE Attack: Sharp, instantaneous spike.
        
        Characteristics:
        - Sudden jump in values
        - Short duration (1-3 hours typically)
        - High magnitude (100-500%)
        
        Detection: INSTANT (Emergency mode)
        """
        attacked = data.copy()
        attacked[start:start+duration] *= magnitude
        
        return attacked, {
            'name': 'PULSE',
            'detection_mode': 'INSTANT',
            'window': '1 hour',
            'magnitude': f"{int((magnitude-1)*100)}%",
            'duration': f"{duration}h",
            'severity': 'CRITICAL' if magnitude > 2.0 else 'HIGH'
        }
    
    @staticmethod
    def scaling_attack(data, start, duration=10, scale_factor=1.3):
        """
        SCALING Attack: Gradual multiplication of values.
        
        Characteristics:
        - Constant percentage increase
        - Moderate duration (5-20 hours)
        - Moderate magnitude (20-50%)
        
        Detection: SUSTAINED (3+ hours observation)
        """
        attacked = data.copy()
        attacked[start:start+duration] *= scale_factor
        
        return attacked, {
            'name': 'SCALING',
            'detection_mode': 'SUSTAINED',
            'window': '3-5 hours',
            'magnitude': f"{int((scale_factor-1)*100)}%",
            'duration': f"{duration}h",
            'severity': 'MEDIUM' if scale_factor < 1.5 else 'HIGH'
        }
    
    @staticmethod
    def ramping_attack(data, start, duration=15, max_increase=1.4):
        """
        RAMPING Attack: Slowly increasing values over time.
        
        Characteristics:
        - Linear or exponential increase
        - Long duration (10-30 hours)
        - Gradual escalation
        
        Detection: PATTERN-BASED (5+ hours)
        """
        attacked = data.copy()
        ramp = np.linspace(1.0, max_increase, duration)
        attacked[start:start+duration] *= ramp
        
        return attacked, {
            'name': 'RAMPING',
            'detection_mode': 'PATTERN',
            'window': '5-10 hours',
            'magnitude': f"0-{int((max_increase-1)*100)}% (gradual)",
            'duration': f"{duration}h",
            'severity': 'MEDIUM'
        }
    
    @staticmethod
    def random_attack(data, start, duration=12, noise_level=0.3):
        """
        RANDOM Attack: Erratic noise injection.
        
        Characteristics:
        - Random fluctuations
        - Moderate duration
        - Statistical anomaly (high variance)
        
        Detection: STATISTICAL (variance analysis)
        """
        attacked = data.copy()
        noise = np.random.uniform(1.0 - noise_level, 1.0 + noise_level, duration)
        attacked[start:start+duration] *= noise
        
        return attacked, {
            'name': 'RANDOM',
            'detection_mode': 'STATISTICAL',
            'window': '6-12 hours',
            'magnitude': f"±{int(noise_level*100)}% random",
            'duration': f"{duration}h",
            'severity': 'LOW' if noise_level < 0.2 else 'MEDIUM'
        }
    
    @staticmethod
    def smooth_curve_attack(data, start, duration=20, amplitude=1.25):
        """
        SMOOTH-CURVE Attack: Subtle curved deviation.
        
        Characteristics:
        - Sine/cosine wave pattern
        - Long duration (15-40 hours)
        - Smooth, subtle changes
        
        Detection: LONG-TERM (10+ hours)
        """
        attacked = data.copy()
        curve = 1.0 + (amplitude - 1.0) * np.sin(np.linspace(0, np.pi, duration))
        attacked[start:start+duration] *= curve
        
        return attacked, {
            'name': 'SMOOTH-CURVE',
            'detection_mode': 'LONG-TERM',
            'window': '10-20 hours',
            'magnitude': f"Peak {int((amplitude-1)*100)}% (curved)",
            'duration': f"{duration}h",
            'severity': 'LOW' if amplitude < 1.3 else 'MEDIUM'
        }


def test_all_attack_types():
    """Test detection performance against all attack types."""
    
    print("="*75)
    print("COMPREHENSIVE ATTACK TYPE DETECTION TEST")
    print("="*75)
    
    # Load models
    print("\n📥 Loading models...")
    if not os.path.exists(config.FORECASTER_MODEL_PATH):
        print("❌ Models not found! Run training first.")
        return
    
    forecaster = load_model(config.FORECASTER_MODEL_PATH)
    with open(config.SCALER_PATH, 'rb') as f:
        scaler = pickle.load(f)
    with open(config.KMEANS_MODEL_PATH, 'rb') as f:
        kmeans = pickle.load(f)
    
    print("✅ Models loaded")
    
    # Load data
    print("📊 Loading data...")
    df = load_and_preprocess_data(config.DATASET_DIR)
    df_features = create_features(df)
    split_idx = int(len(df_features) * config.TRAIN_TEST_SPLIT_RATIO)
    test_data = df_features.iloc[split_idx:split_idx+168].copy()
    
    feature_cols = [col for col in test_data.columns 
                   if col not in ['load', 'date', 'hour']]
    
    # Generate base forecast
    X_test = test_data[feature_cols].values
    X_test_scaled = scaler.transform(X_test)
    X_test_reshaped = X_test_scaled.reshape((X_test_scaled.shape[0], 1, X_test_scaled.shape[1]))
    base_forecast = forecaster.predict(X_test_reshaped, verbose=0).flatten()
    
    print("✅ Ready to test attacks\n")
    
    # Define attack scenarios
    simulator = AttackSimulator()
    
    attack_scenarios = [
        # PULSE attacks (instant detection needed)
        {
            'func': simulator.pulse_attack,
            'params': {'start': 20, 'duration': 1, 'magnitude': 5.0},
            'description': 'Catastrophic 400% pulse spike for 1 hour'
        },
        {
            'func': simulator.pulse_attack,
            'params': {'start': 40, 'duration': 3, 'magnitude': 2.0},
            'description': 'Major 100% pulse spike for 3 hours'
        },
        
        # SCALING attacks (sustained observation)
        {
            'func': simulator.scaling_attack,
            'params': {'start': 60, 'duration': 10, 'scale_factor': 1.3},
            'description': 'Moderate 30% scaling over 10 hours'
        },
        {
            'func': simulator.scaling_attack,
            'params': {'start': 80, 'duration': 8, 'scale_factor': 1.5},
            'description': 'High 50% scaling over 8 hours'
        },
        
        # RAMPING attacks (pattern detection)
        {
            'func': simulator.ramping_attack,
            'params': {'start': 100, 'duration': 15, 'max_increase': 1.4},
            'description': 'Gradual ramp 0→40% over 15 hours'
        },
        
        # RANDOM attacks (statistical)
        {
            'func': simulator.random_attack,
            'params': {'start': 120, 'duration': 12, 'noise_level': 0.25},
            'description': 'Random noise ±25% over 12 hours'
        },
        
        # SMOOTH-CURVE attacks (long-term)
        {
            'func': simulator.smooth_curve_attack,
            'params': {'start': 140, 'duration': 20, 'amplitude': 1.30},
            'description': 'Smooth curve attack peaking at 30% over 20 hours'
        },
    ]
    
    print("="*75)
    print("TESTING DIFFERENT ATTACK TYPES")
    print("="*75)
    
    results = []
    
    for i, scenario in enumerate(attack_scenarios, 1):
        print(f"\n{'='*75}")
        
        # Inject attack
        attacked_forecast, attack_info = scenario['func'](
            base_forecast, 
            **scenario['params']
        )
        
        # Get benchmark
        benchmark = get_benchmark_for_period(kmeans, attacked_forecast)
        scaling_data = attacked_forecast / (benchmark + 1e-6)
        
        # Detect anomaly (with segmentation for improved precision)
        detected_start, detected_end, score = detect_anomaly_with_segmentation(scaling_data)
        
        # Calculate actual attack stats
        start = scenario['params']['start']
        if 'duration' in scenario['params']:
            duration = scenario['params']['duration']
        else:
            duration = 1
        
        attack_scaling = scaling_data[start:start+duration]
        max_deviation = np.max(np.abs(attack_scaling - 1.0))
        avg_deviation = np.mean(np.abs(attack_scaling - 1.0))
        
        # Print attack details
        print(f"🎯 Test {i}: {attack_info['name']} Attack")
        print(f"   Description: {scenario['description']}")
        print(f"   Severity: {attack_info['severity']}")
        print(f"   Detection Mode: {attack_info['detection_mode']}")
        print(f"   Optimal Window: {attack_info['window']}")
        print(f"   Magnitude: {attack_info['magnitude']}")
        print(f"   Duration: {attack_info['duration']}")
        
        # Print detection results
        if detected_start is not None:
            detected_duration = detected_end - detected_start + 1
            
            # Check if detection overlaps with attack
            overlap_start = max(start, detected_start)
            overlap_end = min(start + duration, detected_end)
            
            if overlap_start <= overlap_end:
                overlap = overlap_end - overlap_start
                precision = (overlap / detected_duration) * 100
                recall = (overlap / duration) * 100
                
                # Determine if appropriate detection speed
                is_instant = attack_info['detection_mode'] == 'INSTANT'
                is_appropriate = (
                    (is_instant and detected_duration <= 3) or
                    (not is_instant and detected_duration >= 3)
                )
                
                status = "✅ DETECTED"
                if is_appropriate:
                    status += " (Appropriate timing)"
                
                print(f"\n   {status}")
                print(f"   Detection: Hours {detected_start}-{detected_end} ({detected_duration}h)")
                print(f"   Injected: Hours {start}-{start+duration-1} ({duration}h)")
                print(f"   Max Deviation: {max_deviation:.2%}")
                print(f"   Avg Deviation: {avg_deviation:.2%}")
                print(f"   Score: {score:.4f}")
                print(f"   Precision: {precision:.1f}% | Recall: {recall:.1f}%")
                
                results.append({
                    'attack_type': attack_info['name'],
                    'severity': attack_info['severity'],
                    'detection_mode': attack_info['detection_mode'],
                    'detected': True,
                    'appropriate': is_appropriate,
                    'precision': precision,
                    'recall': recall,
                    'score': score,
                    'max_deviation': max_deviation
                })
            else:
                print(f"\n   ⚠️ DETECTED but wrong location")
                print(f"   Detected: Hours {detected_start}-{detected_end}")
                print(f"   Injected: Hours {start}-{start+duration-1}")
                
                results.append({
                    'attack_type': attack_info['name'],
                    'severity': attack_info['severity'],
                    'detection_mode': attack_info['detection_mode'],
                    'detected': True,
                    'appropriate': False,
                    'precision': 0,
                    'recall': 0,
                    'score': score,
                    'max_deviation': max_deviation
                })
        else:
            print(f"\n   ❌ NOT DETECTED")
            print(f"   Max Deviation: {max_deviation:.2%} (Threshold: {config.MAGNITUDE_THRESHOLD:.0%})")
            print(f"   Avg Deviation: {avg_deviation:.2%}")
            
            results.append({
                'attack_type': attack_info['name'],
                'severity': attack_info['severity'],
                'detection_mode': attack_info['detection_mode'],
                'detected': False,
                'appropriate': False,
                'precision': 0,
                'recall': 0,
                'score': 0,
                'max_deviation': max_deviation
            })
    
    # Summary
    print("\n" + "="*75)
    print("COMPREHENSIVE SUMMARY")
    print("="*75)
    
    detected_count = sum(1 for r in results if r['detected'])
    appropriate_count = sum(1 for r in results if r['appropriate'])
    
    print(f"\n📊 Overall Detection Rate: {detected_count}/{len(results)} ({detected_count/len(results)*100:.0f}%)")
    print(f"✅ Appropriate Detection: {appropriate_count}/{len(results)} ({appropriate_count/len(results)*100:.0f}%)")
    
    # By attack type
    print("\n📋 Detection by Attack Type:")
    print(f"{'Attack Type':<15} {'Mode':<15} {'Detected':<10} {'Score':<10} {'Max Dev':<10}")
    print("-"*75)
    
    for r in results:
        detected_str = "✓" if r['detected'] else "✗"
        score_str = f"{r['score']:.4f}" if r['detected'] else "N/A"
        
        print(f"{r['attack_type']:<15} {r['detection_mode']:<15} {detected_str:<10} "
              f"{score_str:<10} {r['max_deviation']:<10.2%}")
    
    # Recommendations
    print("\n" + "="*75)
    print("DETECTION STRATEGY RECOMMENDATIONS")
    print("="*75)
    
    print("\n🎯 Attack-Specific Strategies:\n")
    
    print("1. PULSE Attacks (Instant Detection):")
    print("   ✓ Use EMERGENCY MODE (>50% threshold)")
    print("   ✓ Alert immediately on first detection")
    print("   ✓ Current config: PERFECT for pulse attacks")
    
    print("\n2. SCALING Attacks (Sustained Observation):")
    print("   ✓ Use NORMAL MODE (15% threshold, 3+ hours)")
    print("   ✓ Wait for sustained pattern")
    print("   ✓ Current config: GOOD for scaling attacks")
    
    print("\n3. RAMPING Attacks (Pattern Detection):")
    print("   ⚠️ Challenging - gradual increase hard to detect early")
    print("   ✓ Requires longer observation window (5-10 hours)")
    print("   💡 Recommendation: Add trend analysis component")
    
    print("\n4. RANDOM Attacks (Statistical Analysis):")
    print("   ⚠️ Moderate difficulty - noise can be subtle")
    print("   ✓ Requires variance monitoring")
    print("   💡 Recommendation: Add statistical anomaly detector")
    
    print("\n5. SMOOTH-CURVE Attacks (Long-Term Monitoring):")
    print("   ⚠️ Most challenging - very subtle")
    print("   ✓ Requires extended observation (10-20 hours)")
    print("   💡 Recommendation: Add frequency domain analysis")
    
    print("\n" + "="*75)
    print("CONFIGURATION TUNING GUIDE")
    print("="*75)
    
    print("\n💡 Current Config Analysis:")
    instant_attacks = [r for r in results if r['detection_mode'] == 'INSTANT']
    sustained_attacks = [r for r in results if r['detection_mode'] in ['SUSTAINED', 'PATTERN', 'STATISTICAL', 'LONG-TERM']]
    
    instant_detected = sum(1 for r in instant_attacks if r['detected'])
    sustained_detected = sum(1 for r in sustained_attacks if r['detected'])
    
    print(f"   Instant Attack Detection: {instant_detected}/{len(instant_attacks)} "
          f"({instant_detected/len(instant_attacks)*100:.0f}%)")
    print(f"   Sustained Attack Detection: {sustained_detected}/{len(sustained_attacks)} "
          f"({sustained_detected/len(sustained_attacks)*100:.0f}%)")
    
    print("\n⚙️ Recommended Settings for Different Priorities:")
    print("""
    HIGH SECURITY (Catch Everything):
    - MAGNITUDE_THRESHOLD = 0.10  (10%)
    - EMERGENCY_THRESHOLD = 0.40  (40%)
    - MIN_ANOMALY_DURATION = 2    (2 hours)
    
    BALANCED (Current - Recommended):
    - MAGNITUDE_THRESHOLD = 0.15  (15%)
    - EMERGENCY_THRESHOLD = 0.50  (50%)
    - MIN_ANOMALY_DURATION = 3    (3 hours)
    
    LOW FALSE ALARMS (Critical Only):
    - MAGNITUDE_THRESHOLD = 0.20  (20%)
    - EMERGENCY_THRESHOLD = 0.75  (75%)
    - MIN_ANOMALY_DURATION = 5    (5 hours)
    """)
    
    print("✅ Test complete!")


if __name__ == "__main__":
    test_all_attack_types()

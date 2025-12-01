"""
COMPREHENSIVE ATTACK TEMPLATE EVALUATION

Tests the MLAD system against all five attack templates from academic literature:
1. Pulse Attacks - Specific point modifications
2. Scaling Attacks - Duration-based multiplication
3. Ramping Attacks - Type I (up-ramping) and Type II (up + down ramping)
4. Random Attacks - Uniform random noise injection
5. Smooth-Curve Attacks - Polynomial curve replacement

Also categorizes by time series attack types:
- Point Attacks (single or few points)
- Contextual Attacks (depends on context)
- Collective Attacks (sequence of points)

Uses Phase 4 hybrid detection (DP + Statistical) for evaluation.
"""

import os
import sys
import numpy as np
import pandas as pd
import pickle
from tensorflow.keras.models import load_model

# Fix Windows console encoding
if sys.platform == 'win32':
    try:
        sys.stdout.reconfigure(encoding='utf-8')
    except AttributeError:
        import io
        sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8')

import config
from mlad_anomaly_detection import (
    load_and_preprocess_data,
    create_features,
    get_benchmark_for_period,
    detect_anomaly_with_segmentation,
    hybrid_detection
)
from comprehensive_model_evaluation import calculate_hour_level_metrics


class AdvancedAttackSimulator:
    """
    Comprehensive attack simulator implementing all five attack templates
    from academic literature on power grid cyberattacks.
    """
    
    @staticmethod
    def pulse_attack(data, start, duration=1, magnitude=3.0):
        """
        PULSE Attack: Sharp, instantaneous spike at specific point.
        
        Category: Point Attack
        Characteristics: 
        - Modifies load forecasts to higher/lower values at specific point
        - Ultra-short duration (typically 1 point)
        - Parameter: λ_P (pulse magnitude)
        
        Args:
            data: Original forecast data
            start: Attack start hour
            duration: Attack duration (typically 1 for pure pulse)
            magnitude: Pulse magnitude (λ_P)
        """
        attacked = data.copy()
        attacked[start:start+duration] *= magnitude
        return attacked, 'PULSE', 'POINT'
    
    @staticmethod
    def scaling_attack(data, start, duration=10, scale_factor=1.3):
        """
        SCALING Attack: Multiply values over specified duration.
        
        Category: Collective Attack
        Characteristics:
        - One significant upward stroke (SUS)
        - One significant downward stroke (SDS)
        - Parameter: λ_S (scaling factor)
        
        Args:
            data: Original forecast data
            start: Attack start hour
            duration: Attack duration
            scale_factor: Scaling parameter (λ_S)
        """
        attacked = data.copy()
        attacked[start:start+duration] *= scale_factor
        return attacked, 'SCALING', 'COLLECTIVE'
    
    @staticmethod
    def ramping_attack_type1(data, start, duration=15, max_increase=1.4):
        """
        RAMPING Attack Type I: Up-ramping only.
        
        Category: Collective Attack
        Characteristics:
        - Only upward ramping anomaly
        - Multiplied by ramping function λ_R * t
        - Simpler to detect than Type II
        
        Args:
            data: Original forecast data
            start: Attack start hour
            duration: Attack duration
            max_increase: Maximum ramping value (λ_R at end)
        """
        attacked = data.copy()
        ramp = np.linspace(1.0, max_increase, duration)
        attacked[start:start+duration] *= ramp
        return attacked, 'RAMPING-TYPE1', 'COLLECTIVE'
    
    @staticmethod
    def ramping_attack_type2(data, start, duration=20, max_increase=1.5):
        """
        RAMPING Attack Type II: Up-ramping then down-ramping.
        
        Category: Collective Attack
        Characteristics:
        - Both up-ramping and down-ramping anomalies
        - More challenging for operators to detect
        - One SUS and one SDS
        - MLAD can detect both phases
        
        Args:
            data: Original forecast data
            start: Attack start hour
            duration: Attack duration (split between up and down)
            max_increase: Peak ramping value (λ_R at middle)
        """
        attacked = data.copy()
        half_duration = duration // 2
        
        # Up-ramping phase
        up_ramp = np.linspace(1.0, max_increase, half_duration)
        attacked[start:start+half_duration] *= up_ramp
        
        # Down-ramping phase
        down_ramp = np.linspace(max_increase, 1.0, duration - half_duration)
        attacked[start+half_duration:start+duration] *= down_ramp
        
        return attacked, 'RAMPING-TYPE2', 'COLLECTIVE'
    
    @staticmethod
    def random_attack(data, start, duration=12, noise_level=0.3):
        """
        RANDOM Attack: Addition of uniform random values.
        
        Category: Collective Attack
        Characteristics:
        - Positive values from uniform random function
        - Scale factor: λ_RA = max(p_Ft)/2
        - Random start and end times
        - Multiple upward and downward strokes
        
        Args:
            data: Original forecast data
            start: Attack start hour (randomly chosen by attacker)
            duration: Attack duration
            noise_level: Noise intensity (fraction of λ_RA)
        """
        attacked = data.copy()
        np.random.seed(42 + start)  # Reproducible but varies by start
        
        # Scale factor based on maximum forecast value
        max_forecast = np.max(data)
        lambda_ra = max_forecast / 2
        
        # Generate uniform random noise (positive values)
        noise = np.random.uniform(0, noise_level * lambda_ra, duration)
        attacked[start:start+duration] += noise
        
        return attacked, 'RANDOM', 'COLLECTIVE'
    
    @staticmethod
    def smooth_curve_attack(data, start, duration=20, amplitude=1.25, polynomial_degree=3):
        """
        SMOOTH-CURVE Attack: Replace data with smooth polynomial curve.
        
        Category: Contextual Attack
        Characteristics:
        - Most challenging to detect (very secretive)
        - Polynomial fitting replaces original data
        - Smooth curve connects with neighboring points
        - Multiple upward and downward strokes possible
        
        Args:
            data: Original forecast data
            start: Attack start hour
            duration: Attack duration
            amplitude: Peak amplitude of curve
            polynomial_degree: Degree of polynomial for smoothness
        """
        attacked = data.copy()
        
        # Create smooth curve using sine wave (smoother than polynomial for attacks)
        curve = 1.0 + (amplitude - 1.0) * np.sin(np.linspace(0, np.pi, duration))
        attacked[start:start+duration] *= curve
        
        # Optional: Add polynomial smoothing at boundaries for seamless transition
        # This makes it blend better with neighboring points
        if start > 0 and start + duration < len(data):
            # Smooth transition at start
            transition_length = min(3, start)
            for i in range(transition_length):
                blend = (i + 1) / (transition_length + 1)
                idx = start - transition_length + i
                attacked[idx] = (1 - blend) * data[idx] + blend * attacked[start]
            
            # Smooth transition at end
            transition_length = min(3, len(data) - start - duration)
            for i in range(transition_length):
                blend = 1 - ((i + 1) / (transition_length + 1))
                idx = start + duration + i
                attacked[idx] = blend * attacked[start+duration-1] + (1 - blend) * data[idx]
        
        return attacked, 'SMOOTH-CURVE', 'CONTEXTUAL'
    
    @staticmethod
    def point_attack_burst(data, start, burst_count=3, spacing=2, magnitude=2.0):
        """
        POINT Attack: Multiple isolated point anomalies.
        
        Category: Point Attack
        Characteristics:
        - Multiple isolated points
        - Tests system's ability to detect scattered anomalies
        
        Args:
            data: Original forecast data
            start: First attack point
            burst_count: Number of isolated points
            spacing: Hours between burst points
            magnitude: Spike magnitude at each point
        """
        attacked = data.copy()
        for i in range(burst_count):
            point = start + (i * spacing)
            if point < len(data):
                attacked[point] *= magnitude
        return attacked, 'POINT-BURST', 'POINT'
    
    @staticmethod
    def contextual_seasonal_attack(data, start, duration=24, seasonal_shift=0.15):
        """
        CONTEXTUAL Attack: Shift that appears normal in isolation but anomalous in context.
        
        Category: Contextual Attack
        Characteristics:
        - Values appear reasonable individually
        - Anomalous in seasonal/daily context
        - Tests context-aware detection
        
        Args:
            data: Original forecast data
            start: Attack start hour
            duration: Attack duration (24h = one day)
            seasonal_shift: Percentage shift maintaining pattern
        """
        attacked = data.copy()
        attacked[start:start+duration] *= (1 + seasonal_shift)
        return attacked, 'CONTEXTUAL-SEASONAL', 'CONTEXTUAL'


def create_comprehensive_attack_scenarios(data_length):
    """
    Create comprehensive test scenarios covering all attack templates
    and time series attack types.
    
    Returns:
        list: Comprehensive list of attack scenarios
    """
    simulator = AdvancedAttackSimulator()
    scenarios = []
    
    # Define magnitude categories
    magnitudes = {
        'weak': [1.10, 1.15, 1.20],        # 10%, 15%, 20%
        'medium': [1.30, 1.50, 2.0],       # 30%, 50%, 100%
        'strong': [3.0, 5.0, 10.0]         # 200%, 400%, 900%
    }
    
    # Define duration categories
    durations = {
        'short': [1, 3, 5],
        'medium': [6, 12, 18],
        'long': [24, 36, 48]
    }
    
    current_position = 100
    spacing = 60  # More spacing for longer attacks
    
    print("\n🎯 Creating comprehensive attack template scenarios...")
    print("="*80)
    
    # ============================================================================
    # 1. PULSE ATTACKS - All magnitudes, ultra-short durations
    # ============================================================================
    print("📍 PULSE Attacks (Point Attacks)")
    pulse_count = 0
    for mag_category, mag_values in magnitudes.items():
        for mag in mag_values:
            for dur in [1, 2]:  # Pure pulse is 1 hour
                if current_position + dur + spacing < data_length:
                    scenarios.append({
                        'func': simulator.pulse_attack,
                        'params': {'start': current_position, 'duration': dur, 'magnitude': mag},
                        'description': f'{mag_category.upper()} PULSE: {int((mag-1)*100)}% for {dur}h',
                        'attack_type': 'PULSE',
                        'attack_template': 'PULSE',
                        'time_series_type': 'POINT',
                        'magnitude_category': mag_category,
                        'duration_category': 'short',
                        'magnitude_value': int((mag-1)*100),
                        'duration_value': dur
                    })
                    current_position += spacing
                    pulse_count += 1
    print(f"   Created {pulse_count} PULSE attack scenarios")
    
    # ============================================================================
    # 2. SCALING ATTACKS - All magnitudes, varied durations
    # ============================================================================
    print("📊 SCALING Attacks (Collective Attacks)")
    scaling_count = 0
    for mag_category in ['weak', 'medium', 'strong']:
        for mag in magnitudes[mag_category]:
            for dur_category, dur_values in durations.items():
                for dur in dur_values[:2]:  # Use first 2 durations per category
                    if current_position + dur + spacing < data_length:
                        scenarios.append({
                            'func': simulator.scaling_attack,
                            'params': {'start': current_position, 'duration': dur, 'scale_factor': mag},
                            'description': f'{mag_category.upper()} SCALING: {int((mag-1)*100)}% for {dur}h',
                            'attack_type': 'SCALING',
                            'attack_template': 'SCALING',
                            'time_series_type': 'COLLECTIVE',
                            'magnitude_category': mag_category,
                            'duration_category': dur_category,
                            'magnitude_value': int((mag-1)*100),
                            'duration_value': dur
                        })
                        current_position += spacing
                        scaling_count += 1
    print(f"   Created {scaling_count} SCALING attack scenarios")
    
    # ============================================================================
    # 3. RAMPING TYPE I ATTACKS - Up-ramping only
    # ============================================================================
    print("📈 RAMPING Type I Attacks (Up-ramping, Collective)")
    ramping1_count = 0
    for mag in magnitudes['medium']:
        for dur in [12, 18, 24]:
            if current_position + dur + spacing < data_length:
                scenarios.append({
                    'func': simulator.ramping_attack_type1,
                    'params': {'start': current_position, 'duration': dur, 'max_increase': mag},
                    'description': f'RAMPING-I: 0→{int((mag-1)*100)}% over {dur}h',
                    'attack_type': 'RAMPING-TYPE1',
                    'attack_template': 'RAMPING',
                    'time_series_type': 'COLLECTIVE',
                    'magnitude_category': 'medium',
                    'duration_category': 'medium' if dur <= 18 else 'long',
                    'magnitude_value': int((mag-1)*100),
                    'duration_value': dur
                })
                current_position += spacing
                ramping1_count += 1
    print(f"   Created {ramping1_count} RAMPING Type I scenarios")
    
    # ============================================================================
    # 4. RAMPING TYPE II ATTACKS - Up + Down ramping (more challenging)
    # ============================================================================
    print("📈📉 RAMPING Type II Attacks (Up+Down ramping, Collective)")
    ramping2_count = 0
    for mag in magnitudes['medium'] + [magnitudes['weak'][2]]:  # Medium + 20% weak
        for dur in [16, 24, 36]:
            if current_position + dur + spacing < data_length:
                scenarios.append({
                    'func': simulator.ramping_attack_type2,
                    'params': {'start': current_position, 'duration': dur, 'max_increase': mag},
                    'description': f'RAMPING-II: 0→{int((mag-1)*100)}%→0 over {dur}h',
                    'attack_type': 'RAMPING-TYPE2',
                    'attack_template': 'RAMPING',
                    'time_series_type': 'COLLECTIVE',
                    'magnitude_category': 'medium' if mag >= 1.3 else 'weak',
                    'duration_category': 'medium' if dur <= 24 else 'long',
                    'magnitude_value': int((mag-1)*100),
                    'duration_value': dur
                })
                current_position += spacing
                ramping2_count += 1
    print(f"   Created {ramping2_count} RAMPING Type II scenarios")
    
    # ============================================================================
    # 5. RANDOM ATTACKS - Multiple strokes, varied noise levels
    # ============================================================================
    print("🎲 RANDOM Attacks (Collective Attacks)")
    random_count = 0
    noise_levels = [0.10, 0.20, 0.30, 0.40]
    for noise in noise_levels:
        for dur in [6, 12, 18, 24]:
            if current_position + dur + spacing < data_length:
                mag_category = 'weak' if noise <= 0.15 else 'medium'
                scenarios.append({
                    'func': simulator.random_attack,
                    'params': {'start': current_position, 'duration': dur, 'noise_level': noise},
                    'description': f'RANDOM: ±{int(noise*100)}% noise for {dur}h',
                    'attack_type': 'RANDOM',
                    'attack_template': 'RANDOM',
                    'time_series_type': 'COLLECTIVE',
                    'magnitude_category': mag_category,
                    'duration_category': 'short' if dur <= 6 else ('medium' if dur <= 18 else 'long'),
                    'magnitude_value': int(noise*100),
                    'duration_value': dur
                })
                current_position += spacing
                random_count += 1
    print(f"   Created {random_count} RANDOM attack scenarios")
    
    # ============================================================================
    # 6. SMOOTH-CURVE ATTACKS - Most challenging (contextual)
    # ============================================================================
    print("🌊 SMOOTH-CURVE Attacks (Contextual, Most Challenging)")
    smooth_count = 0
    for amp in [1.15, 1.25, 1.35, 1.50]:
        for dur in [18, 24, 36, 48]:
            if current_position + dur + spacing < data_length:
                mag_category = 'weak' if amp < 1.25 else 'medium'
                scenarios.append({
                    'func': simulator.smooth_curve_attack,
                    'params': {'start': current_position, 'duration': dur, 'amplitude': amp},
                    'description': f'SMOOTH: Peak {int((amp-1)*100)}% over {dur}h',
                    'attack_type': 'SMOOTH-CURVE',
                    'attack_template': 'SMOOTH',
                    'time_series_type': 'CONTEXTUAL',
                    'magnitude_category': mag_category,
                    'duration_category': 'medium' if dur <= 24 else 'long',
                    'magnitude_value': int((amp-1)*100),
                    'duration_value': dur
                })
                current_position += spacing
                smooth_count += 1
    print(f"   Created {smooth_count} SMOOTH-CURVE attack scenarios")
    
    # ============================================================================
    # 7. POINT-BURST ATTACKS - Multiple isolated points
    # ============================================================================
    print("💥 POINT-BURST Attacks (Point Attacks)")
    burst_count = 0
    for burst_size in [2, 3, 5]:
        for mag in [1.5, 2.0, 3.0]:
            if current_position + (burst_size * 5) + spacing < data_length:
                scenarios.append({
                    'func': simulator.point_attack_burst,
                    'params': {'start': current_position, 'burst_count': burst_size, 
                              'spacing': 3, 'magnitude': mag},
                    'description': f'POINT-BURST: {burst_size} spikes of {int((mag-1)*100)}%',
                    'attack_type': 'POINT-BURST',
                    'attack_template': 'PULSE',
                    'time_series_type': 'POINT',
                    'magnitude_category': 'medium' if mag < 2.5 else 'strong',
                    'duration_category': 'short',
                    'magnitude_value': int((mag-1)*100),
                    'duration_value': burst_size
                })
                current_position += spacing
                burst_count += 1
    print(f"   Created {burst_count} POINT-BURST scenarios")
    
    # ============================================================================
    # 8. CONTEXTUAL-SEASONAL ATTACKS
    # ============================================================================
    print("🌐 CONTEXTUAL-SEASONAL Attacks (Contextual)")
    contextual_count = 0
    for shift in [0.10, 0.15, 0.20, 0.25]:
        for dur in [24, 48]:  # Daily or 2-day patterns
            if current_position + dur + spacing < data_length:
                scenarios.append({
                    'func': simulator.contextual_seasonal_attack,
                    'params': {'start': current_position, 'duration': dur, 'seasonal_shift': shift},
                    'description': f'CONTEXTUAL: {int(shift*100)}% seasonal shift for {dur}h',
                    'attack_type': 'CONTEXTUAL-SEASONAL',
                    'attack_template': 'SMOOTH',
                    'time_series_type': 'CONTEXTUAL',
                    'magnitude_category': 'weak' if shift <= 0.15 else 'medium',
                    'duration_category': 'long',
                    'magnitude_value': int(shift*100),
                    'duration_value': dur
                })
                current_position += spacing
                contextual_count += 1
    print(f"   Created {contextual_count} CONTEXTUAL-SEASONAL scenarios")
    
    # Summary
    total = len(scenarios)
    print("\n" + "="*80)
    print(f"✅ Total scenarios created: {total}")
    print(f"   - PULSE: {pulse_count}")
    print(f"   - SCALING: {scaling_count}")
    print(f"   - RAMPING Type I: {ramping1_count}")
    print(f"   - RAMPING Type II: {ramping2_count}")
    print(f"   - RANDOM: {random_count}")
    print(f"   - SMOOTH-CURVE: {smooth_count}")
    print(f"   - POINT-BURST: {burst_count}")
    print(f"   - CONTEXTUAL-SEASONAL: {contextual_count}")
    print(f"\nData required: {current_position + 100} hours")
    print("="*80)
    
    return scenarios


def evaluate_comprehensive_attacks():
    """
    Comprehensive evaluation of all attack templates using Phase 4 hybrid detection.
    """
    print("="*80)
    print("COMPREHENSIVE ATTACK TEMPLATE EVALUATION")
    print("Testing all 5 attack templates + variations")
    print("Detection Method: Phase 4 Hybrid (DP + Statistical)")
    print("="*80)
    
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
    test_data = df_features.iloc[split_idx:].copy()
    
    # Use maximum available test data
    test_window_size = min(6000, len(test_data))  # More data for comprehensive tests
    test_data = test_data.iloc[:test_window_size].copy()
    
    print(f"✅ Test data loaded: {len(test_data)} hours")
    
    feature_cols = [col for col in test_data.columns 
                   if col not in ['load', 'date', 'hour']]
    
    # Generate base forecast
    print("\n🔮 Generating base forecast...")
    X_test = test_data[feature_cols].values
    X_test_scaled = scaler.transform(X_test)
    X_test_reshaped = X_test_scaled.reshape((X_test_scaled.shape[0], 1, X_test_scaled.shape[1]))
    base_forecast = forecaster.predict(X_test_reshaped, verbose=0).flatten()
    print("✅ Base forecast generated")
    
    # Create comprehensive scenarios
    scenarios = create_comprehensive_attack_scenarios(len(base_forecast))
    
    # Initialize results storage
    all_results = []
    
    print(f"\n{'='*80}")
    print(f"RUNNING {len(scenarios)} COMPREHENSIVE ATTACK SCENARIOS")
    print(f"{'='*80}\n")
    
    MAX_DETECTIONS = 10
    
    # Run each scenario
    for idx, scenario in enumerate(scenarios, 1):
        # Inject attack
        attacked_forecast, attack_type, ts_type = scenario['func'](
            base_forecast, 
            **scenario['params']
        )
        
        # Get benchmark
        benchmark = get_benchmark_for_period(kmeans, attacked_forecast)
        scaling_data = attacked_forecast / (benchmark + 1e-6)
        
        # Ground truth
        start = scenario['params']['start']
        duration = scenario['params'].get('duration', scenario['params'].get('burst_count', 1))
        
        # Phase 4: Hybrid detection
        detections_raw = hybrid_detection(scaling_data, max_detections=MAX_DETECTIONS)
        
        # Convert to tuple format
        detections = []
        for det in detections_raw:
            if isinstance(det, dict):
                detections.append((det['start'], det['end'], 
                                  det.get('stat_score', det.get('dp_score', 0))))
            else:
                detections.append(det)
        
        # Match to ground truth
        best_match = None
        best_overlap_ratio = 0
        
        if detections and len(detections) > 0:
            for det_start, det_end, det_score in detections:
                overlap_start = max(start, det_start)
                overlap_end = min(start + duration - 1, det_end)
                
                if overlap_start <= overlap_end:
                    overlap_hours = overlap_end - overlap_start + 1
                    overlap_ratio = overlap_hours / duration
                    
                    if overlap_ratio > best_overlap_ratio:
                        best_overlap_ratio = overlap_ratio
                        best_match = (det_start, det_end, det_score)
        
        # Evaluation
        if best_match is not None:
            detected_start, detected_end, score = best_match
            attack_detected = True
            correct_detection = (best_overlap_ratio >= 0.3)
        else:
            detected_start, detected_end, score = None, None, 0
            attack_detected = False
            correct_detection = False
        
        # Hour-level metrics
        y_true = np.zeros(len(attacked_forecast), dtype=int)
        y_true[start:start+duration] = 1
        
        y_pred = np.zeros(len(attacked_forecast), dtype=int)
        if detected_start is not None:
            y_pred[detected_start:detected_end+1] = 1
        
        hour_metrics = calculate_hour_level_metrics(y_true, y_pred)
        
        # Deviation statistics
        attack_scaling = scaling_data[start:start+duration]
        max_deviation = np.max(np.abs(attack_scaling - 1.0))
        avg_deviation = np.mean(np.abs(attack_scaling - 1.0))
        
        # Store results
        result = {
            'scenario_id': idx,
            'description': scenario['description'],
            'attack_type': scenario['attack_type'],
            'attack_template': scenario['attack_template'],
            'time_series_type': scenario['time_series_type'],
            'magnitude_category': scenario['magnitude_category'],
            'duration_category': scenario['duration_category'],
            'magnitude_value': scenario['magnitude_value'],
            'duration_value': scenario['duration_value'],
            'attack_start': start,
            'attack_end': start + duration - 1,
            'attack_duration': duration,
            'detected': attack_detected,
            'correct_detection': correct_detection,
            'detected_start': detected_start if detected_start is not None else -1,
            'detected_end': detected_end if detected_end is not None else -1,
            'detection_score': score if score is not None else 0,
            'max_deviation': max_deviation,
            'avg_deviation': avg_deviation,
            **hour_metrics
        }
        
        all_results.append(result)
        
        # Progress
        if idx % 10 == 0 or idx == len(scenarios):
            print(f"✓ Completed {idx}/{len(scenarios)} scenarios")
    
    # Convert to DataFrame
    results_df = pd.DataFrame(all_results)
    
    # Print comprehensive results
    print_comprehensive_results(results_df)
    
    # Export
    output_file = os.path.join(config.ROOT_DIR, 'comprehensive_attack_evaluation_results.csv')
    results_df.to_csv(output_file, index=False)
    print(f"\n💾 Detailed results exported to: {output_file}")
    
    return results_df


def print_comprehensive_results(results_df):
    """Print comprehensive analysis of results."""
    
    print(f"\n{'='*80}")
    print("COMPREHENSIVE ATTACK EVALUATION RESULTS")
    print(f"{'='*80}\n")
    
    # Overall metrics
    total_tp = results_df['tp'].sum()
    total_tn = results_df['tn'].sum()
    total_fp = results_df['fp'].sum()
    total_fn = results_df['fn'].sum()
    
    overall_accuracy = (total_tp + total_tn) / (total_tp + total_tn + total_fp + total_fn)
    overall_precision = total_tp / (total_tp + total_fp) if (total_tp + total_fp) > 0 else 0
    overall_recall = total_tp / (total_tp + total_fn) if (total_tp + total_fn) > 0 else 0
    overall_f1 = 2 * (overall_precision * overall_recall) / (overall_precision + overall_recall) if (overall_precision + overall_recall) > 0 else 0
    overall_fpr = total_fp / (total_fp + total_tn) if (total_fp + total_tn) > 0 else 0
    
    print("📊 OVERALL HOUR-LEVEL CLASSIFICATION METRICS")
    print("-" * 80)
    print(f"  Accuracy:     {overall_accuracy:.4f} ({overall_accuracy*100:.2f}%)")
    print(f"  Precision:    {overall_precision:.4f} ({overall_precision*100:.2f}%)")
    print(f"  Recall:       {overall_recall:.4f} ({overall_recall*100:.2f}%)")
    print(f"  F1-Score:     {overall_f1:.4f}")
    print(f"  FPR:          {overall_fpr:.4f} ({overall_fpr*100:.2f}%)")
    
    # Attack-level detection
    print(f"\n🎯 ATTACK-LEVEL DETECTION RATES")
    print("-" * 80)
    total_scenarios = len(results_df)
    detected = results_df['detected'].sum()
    correct = results_df['correct_detection'].sum()
    print(f"  Total Scenarios:        {total_scenarios}")
    print(f"  Attacks Detected:       {detected} ({detected/total_scenarios*100:.1f}%)")
    print(f"  Correct Detections:     {correct} ({correct/total_scenarios*100:.1f}%)")
    
    # By attack template
    print(f"\n📋 DETECTION RATE BY ATTACK TEMPLATE")
    print("-" * 80)
    print(f"{'Template':<20} {'Total':<8} {'Detected':<10} {'Correct':<10} {'Rate':<10}")
    print("-" * 80)
    
    for template in ['PULSE', 'SCALING', 'RAMPING', 'RANDOM', 'SMOOTH']:
        template_data = results_df[results_df['attack_template'] == template]
        if len(template_data) > 0:
            total = len(template_data)
            det = template_data['detected'].sum()
            corr = template_data['correct_detection'].sum()
            rate = corr / total if total > 0 else 0
            print(f"{template:<20} {total:<8} {det:<10} {corr:<10} {rate*100:>6.1f}%")
    
    # By time series type
    print(f"\n🔍 DETECTION RATE BY TIME SERIES ATTACK TYPE")
    print("-" * 80)
    print(f"{'Type':<20} {'Total':<8} {'Detected':<10} {'Correct':<10} {'Rate':<10}")
    print("-" * 80)
    
    for ts_type in ['POINT', 'CONTEXTUAL', 'COLLECTIVE']:
        type_data = results_df[results_df['time_series_type'] == ts_type]
        if len(type_data) > 0:
            total = len(type_data)
            det = type_data['detected'].sum()
            corr = type_data['correct_detection'].sum()
            rate = corr / total if total > 0 else 0
            print(f"{ts_type:<20} {total:<8} {det:<10} {corr:<10} {rate*100:>6.1f}%")
    
    # By magnitude
    print(f"\n📊 DETECTION RATE BY MAGNITUDE CATEGORY")
    print("-" * 80)
    print(f"{'Category':<15} {'Total':<8} {'Detected':<10} {'Correct':<10} {'Rate':<10}")
    print("-" * 80)
    
    for mag_cat in ['weak', 'medium', 'strong']:
        cat_data = results_df[results_df['magnitude_category'] == mag_cat]
        if len(cat_data) > 0:
            total = len(cat_data)
            det = cat_data['detected'].sum()
            corr = cat_data['correct_detection'].sum()
            rate = corr / total if total > 0 else 0
            print(f"{mag_cat.upper():<15} {total:<8} {det:<10} {corr:<10} {rate*100:>6.1f}%")
    
    # By duration
    print(f"\n⏱️  DETECTION RATE BY DURATION CATEGORY")
    print("-" * 80)
    print(f"{'Category':<15} {'Total':<8} {'Detected':<10} {'Correct':<10} {'Rate':<10}")
    print("-" * 80)
    
    for dur_cat in ['short', 'medium', 'long']:
        cat_data = results_df[results_df['duration_category'] == dur_cat]
        if len(cat_data) > 0:
            total = len(cat_data)
            det = cat_data['detected'].sum()
            corr = cat_data['correct_detection'].sum()
            rate = corr / total if total > 0 else 0
            print(f"{dur_cat.upper():<15} {total:<8} {det:<10} {corr:<10} {rate*100:>6.1f}%")
    
    print(f"\n{'='*80}")
    print("✅ COMPREHENSIVE ATTACK EVALUATION COMPLETE!")
    print(f"{'='*80}")


if __name__ == "__main__":
    evaluate_comprehensive_attacks()


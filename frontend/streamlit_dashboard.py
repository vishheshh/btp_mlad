"""
Streamlit Dashboard - Real-time visualization for power grid protection system

Architecture:
- LIVE MODE: Fast HTTP polling with minimal UI (during simulation)
- ANALYSIS MODE: Full charts and detailed analysis (after simulation stops)

This solves the "laggy updates" problem by:
1. Using fast polling (1 second) with minimal UI during simulation
2. Rendering only essential metrics during live mode
3. Deferring heavy chart rendering until simulation stops
"""
import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
import requests
import time
from datetime import datetime, timedelta
import sys
import os

# Add parent directory to path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# API base URL
API_BASE_URL = "http://localhost:5000/api"

# Page configuration
st.set_page_config(
    page_title="Power Grid Protection Dashboard",
    page_icon="⚡",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Initialize session state for live data
if 'live_data' not in st.session_state:
    st.session_state.live_data = {
        'hours_processed': 0,
        'recent_actuals': [],
        'recent_forecasts': [],
        'recent_timestamps': [],
        'alerts_count': 0,
        'last_timestamp': None
    }

# Custom CSS with live mode styling
st.markdown("""
<style>
    .main-header {
        font-size: 2.5rem;
        font-weight: bold;
        color: #1f77b4;
        margin-bottom: 1rem;
    }
    .live-indicator {
        display: inline-block;
        width: 12px;
        height: 12px;
        background-color: #00ff00;
        border-radius: 50%;
        margin-right: 8px;
        animation: pulse 1s infinite;
    }
    @keyframes pulse {
        0% { opacity: 1; box-shadow: 0 0 0 0 rgba(0, 255, 0, 0.7); }
        70% { opacity: 0.7; box-shadow: 0 0 0 10px rgba(0, 255, 0, 0); }
        100% { opacity: 1; box-shadow: 0 0 0 0 rgba(0, 255, 0, 0); }
    }
    .live-metric-card {
        background: linear-gradient(135deg, #1a1a2e 0%, #16213e 100%);
        border-radius: 10px;
        padding: 15px;
        color: white;
        text-align: center;
        box-shadow: 0 4px 6px rgba(0, 0, 0, 0.3);
        margin-bottom: 10px;
    }
    .live-metric-value {
        font-size: 2rem;
        font-weight: bold;
        color: #00d4ff;
    }
    .live-metric-label {
        font-size: 0.85rem;
        color: #888;
        margin-top: 5px;
    }
    .alert-high { background-color: #ff6b6b; color: white; padding: 0.5rem; border-radius: 0.25rem; margin: 0.25rem 0; }
    .alert-medium { background-color: #ffa94d; color: white; padding: 0.5rem; border-radius: 0.25rem; margin: 0.25rem 0; }
    .alert-low { background-color: #51cf66; color: white; padding: 0.5rem; border-radius: 0.25rem; margin: 0.25rem 0; }
    .alert-emergency { background-color: #c92a2a; color: white; padding: 0.5rem; border-radius: 0.25rem; margin: 0.25rem 0; font-weight: bold; }
</style>
""", unsafe_allow_html=True)


def fetch_api_data(endpoint, params=None, timeout=2):
    """Fetch data from API endpoint with short timeout for live mode."""
    try:
        url = f"{API_BASE_URL}/{endpoint}"
        response = requests.get(url, params=params, timeout=timeout)
        if response.status_code == 200:
            return response.json()
        else:
            return None
    except Exception as e:
        return None


def post_api_data(endpoint, data=None):
    """Post data to API endpoint."""
    try:
        url = f"{API_BASE_URL}/{endpoint}"
        response = requests.post(url, json=data, timeout=5)
        if response.status_code == 200:
            return response.json()
        else:
            return None
    except Exception as e:
        st.error(f"Error posting to {endpoint}: {e}")
        return None


def create_mini_live_chart(recent_actuals, recent_forecasts, recent_scaling_ratios=None):
    """Create a lightweight mini-chart for live mode showing ML model response."""
    if len(recent_actuals) < 2:
        return None
    
    fig = go.Figure()
    
    # Use simple index for x-axis (faster than parsing timestamps)
    x_vals = list(range(len(recent_actuals)))
    
    # Forecast - dashed line (draw first so actual is on top)
    fig.add_trace(go.Scatter(
        x=x_vals,
        y=recent_forecasts,
        name='ML Forecast',
        line=dict(color='#00ff88', width=2, dash='dash'),
        mode='lines',
        hovertemplate='Forecast: %{y:.1f} MWh<extra></extra>'
    ))
    
    # Actual load - solid line
    fig.add_trace(go.Scatter(
        x=x_vals,
        y=recent_actuals,
        name='Actual Load',
        line=dict(color='#00d4ff', width=2),
        mode='lines',
        hovertemplate='Actual: %{y:.1f} MWh<extra></extra>'
    ))
    
    # Highlight deviation areas (where actual differs significantly from forecast)
    if recent_scaling_ratios:
        for i, ratio in enumerate(recent_scaling_ratios):
            deviation = abs(ratio - 1.0)
            if deviation > 0.09:  # Anomaly threshold
                # Color based on severity
                if deviation > 0.40:
                    color = 'rgba(220, 20, 60, 0.4)'  # Emergency - crimson
                elif deviation > 0.25:
                    color = 'rgba(255, 99, 71, 0.3)'  # High - tomato
                elif deviation > 0.15:
                    color = 'rgba(255, 165, 0, 0.3)'  # Medium - orange
                else:
                    color = 'rgba(144, 238, 144, 0.3)'  # Low - light green
                
                fig.add_vrect(
                    x0=i-0.5, x1=i+0.5,
                    fillcolor=color, layer="below",
                    line_width=0
                )
    
    fig.update_layout(
        height=250,
        margin=dict(l=40, r=10, t=10, b=30),
        paper_bgcolor='rgba(0,0,0,0)',
        plot_bgcolor='rgba(26,26,46,0.5)',
        xaxis=dict(
            showgrid=False,
            showticklabels=True,
            tickfont=dict(color='#888', size=9),
            title=dict(text="Hours", font=dict(color='#888', size=10)),
            zeroline=False
        ),
        yaxis=dict(
            showgrid=True,
            gridcolor='rgba(255,255,255,0.1)',
            showticklabels=True,
            tickfont=dict(color='#888', size=10),
            title=dict(text="Load (MWh)", font=dict(color='#888', size=10)),
            zeroline=False
        ),
        legend=dict(
            orientation="h",
            yanchor="bottom",
            y=1.02,
            xanchor="right",
            x=1,
            font=dict(color='#888', size=10)
        ),
        hovermode='x unified'
    )
    
    return fig


def render_live_mode(status_data):
    """
    Render minimal live mode UI during simulation.
    Uses fast HTTP polling instead of WebSocket for reliability.
    """
    # Header with live indicator
    st.markdown("""
        <div style="display: flex; align-items: center; margin-bottom: 15px;">
            <span class="live-indicator"></span>
            <span style="font-size: 1.5rem; font-weight: bold; color: #00ff00;">LIVE SIMULATION</span>
        </div>
    """, unsafe_allow_html=True)
    
    # Fetch latest data from API
    forecast_data = fetch_api_data("forecast", {"hours": 50}, timeout=2)
    alerts_data = fetch_api_data("alerts", {"limit": 10}, timeout=2)
    
    # Get ACTUAL total hours from status (not just the 50 we fetched)
    total_hours = 0
    if status_data and status_data.get('data_store'):
        total_hours = status_data['data_store'].get('total_hours', 0)
    
    # Extract latest metrics
    recent_scaling_ratios = []
    if forecast_data and forecast_data.get('data'):
        data_list = forecast_data['data']
        
        if len(data_list) > 0:
            latest = data_list[-1]
            actual = float(latest.get('actual', 0))
            forecast = float(latest.get('forecast', 0))
            scaling_ratio = float(latest.get('scaling_ratio', 1.0))
            timestamp = latest.get('timestamp', '')
            
            # Update session state
            st.session_state.live_data['hours_processed'] = total_hours
            st.session_state.live_data['last_timestamp'] = timestamp
            
            # Build rolling window from data (including scaling ratios for anomaly highlighting)
            recent_actuals = [float(d.get('actual', 0)) for d in data_list[-50:]]
            recent_forecasts = [float(d.get('forecast', 0)) for d in data_list[-50:]]
            recent_scaling_ratios = [float(d.get('scaling_ratio', 1.0)) for d in data_list[-50:]]
        else:
            actual = 0
            forecast = 0
            scaling_ratio = 1.0
            timestamp = ''
            recent_actuals = []
            recent_forecasts = []
            recent_scaling_ratios = []
    else:
        actual = 0
        forecast = 0
        scaling_ratio = 1.0
        timestamp = st.session_state.live_data.get('last_timestamp', '')
        recent_actuals = []
        recent_forecasts = []
        recent_scaling_ratios = []
    
    # Use total_hours from status, fallback to session state
    hours_processed = total_hours if total_hours > 0 else st.session_state.live_data.get('hours_processed', 0)
    
    # Get alerts count
    alerts_count = alerts_data.get('count', 0) if alerts_data else 0
    
    # Live metrics row - 5 columns
    col1, col2, col3, col4, col5 = st.columns(5)
    
    with col1:
        st.markdown(f"""
            <div class="live-metric-card">
                <div class="live-metric-value">{hours_processed}</div>
                <div class="live-metric-label">Hours Processed</div>
            </div>
        """, unsafe_allow_html=True)
    
    with col2:
        st.markdown(f"""
            <div class="live-metric-card">
                <div class="live-metric-value">{actual:.1f}</div>
                <div class="live-metric-label">Current Load (MWh)</div>
            </div>
        """, unsafe_allow_html=True)
    
    with col3:
        st.markdown(f"""
            <div class="live-metric-card">
                <div class="live-metric-value">{forecast:.1f}</div>
                <div class="live-metric-label">Forecast (MWh)</div>
            </div>
        """, unsafe_allow_html=True)
    
    with col4:
        deviation = abs(scaling_ratio - 1.0) * 100
        deviation_color = "#00ff00" if deviation < 5 else ("#ffa500" if deviation < 10 else "#ff0000")
        st.markdown(f"""
            <div class="live-metric-card">
                <div class="live-metric-value" style="color: {deviation_color}">{deviation:.1f}%</div>
                <div class="live-metric-label">Deviation</div>
            </div>
        """, unsafe_allow_html=True)
    
    with col5:
        alert_color = "#00ff00" if alerts_count == 0 else "#ff0000"
        st.markdown(f"""
            <div class="live-metric-card">
                <div class="live-metric-value" style="color: {alert_color}">{alerts_count}</div>
                <div class="live-metric-label">Alerts</div>
            </div>
        """, unsafe_allow_html=True)
    
    st.markdown("---")
    
    # Mini live chart with ML model response
    st.subheader("📈 ML Model Response - Live (Last 50 Hours)")
    st.caption("🟢 Green line = ML Forecast | 🔵 Blue line = Actual Load | Colored regions = Detected anomalies")
    
    if len(recent_actuals) > 1:
        mini_chart = create_mini_live_chart(recent_actuals, recent_forecasts, recent_scaling_ratios)
        if mini_chart:
            st.plotly_chart(mini_chart, use_container_width=True)
    else:
        st.info("⏳ Collecting data... Chart will appear after a few hours of simulation.")
    
    # Current timestamp and progress
    col_left, col_right = st.columns(2)
    
    with col_left:
        if timestamp:
            st.caption(f"📅 Current simulation time: **{timestamp}**")
    
    with col_right:
        # Simulation progress
        if status_data and status_data.get('simulation', {}).get('progress'):
            progress = status_data['simulation']['progress']
            progress_pct = progress.get('progress', 0)
            current_idx = progress.get('current_index', 0)
            total = progress.get('total', 1)
            st.progress(progress_pct / 100)
            st.caption(f"Progress: {current_idx:,} / {total:,} hours ({progress_pct:.1f}%)")
    
    # === MODEL REACTION PANEL ===
    st.markdown("---")
    st.subheader("🧠 Model Reaction Monitor")
    
    # Get active attacks info
    active_attacks_data = fetch_api_data("simulation/active_attacks", timeout=2)
    active_attacks = active_attacks_data.get('active_attacks', []) if active_attacks_data else []
    
    # Calculate max deviation from recent data (last 10 hours) for better detection
    max_recent_deviation = deviation  # Start with current
    if forecast_data and forecast_data.get('data'):
        recent_data = forecast_data['data'][-10:]  # Last 10 hours
        for d in recent_data:
            d_ratio = float(d.get('scaling_ratio', 1.0))
            d_dev = abs(d_ratio - 1.0) * 100
            if d_dev > max_recent_deviation:
                max_recent_deviation = d_dev
    
    # Use max recent deviation for status display
    display_deviation = max_recent_deviation
    
    # Create 3 columns for model reaction display
    react_col1, react_col2, react_col3 = st.columns(3)
    
    with react_col1:
        # Current Detection Status - matches config.py thresholds
        # EMERGENCY_THRESHOLD = 0.50 (50%)
        if display_deviation > 50:
            status_icon = "🔴"
            status_text = "EMERGENCY"
            status_color = "#DC143C"
            status_desc = "INSTANT ALERT - Grid threat!"
        elif display_deviation > 40:
            status_icon = "🟠"
            status_text = "HIGH"
            status_color = "#FF6347"
            status_desc = "Strong attack detected"
        elif display_deviation > 25:
            status_icon = "🟡"
            status_text = "MEDIUM"
            status_color = "#FFA500"
            status_desc = "Medium attack detected"
        elif display_deviation > 9:
            status_icon = "🟢"
            status_text = "LOW"
            status_color = "#90EE90"
            status_desc = "Weak attack detected"
        else:
            status_icon = "✅"
            status_text = "NORMAL"
            status_color = "#00ff00"
            status_desc = "Below 9% threshold"
        
        st.markdown(f"""
            <div style="background: linear-gradient(135deg, #1a1a2e 0%, #16213e 100%); 
                        border-radius: 10px; padding: 15px; text-align: center;
                        border-left: 4px solid {status_color};">
                <div style="font-size: 2rem;">{status_icon}</div>
                <div style="font-size: 1.2rem; font-weight: bold; color: {status_color};">{status_text}</div>
                <div style="font-size: 0.8rem; color: #888;">{status_desc}</div>
                <div style="font-size: 0.75rem; color: #666; margin-top: 5px;">Max: {display_deviation:.1f}%</div>
            </div>
        """, unsafe_allow_html=True)
    
    with react_col2:
        # Attack Status
        if active_attacks:
            attack = active_attacks[0]  # Show first active attack
            attack_type = attack.get('type', 'UNKNOWN')
            attack_mag = attack.get('magnitude', 0)
            st.markdown(f"""
                <div style="background: linear-gradient(135deg, #2d1b4e 0%, #1a1a2e 100%); 
                            border-radius: 10px; padding: 15px; text-align: center;
                            border-left: 4px solid #9932CC;">
                    <div style="font-size: 2rem;">💉</div>
                    <div style="font-size: 1.2rem; font-weight: bold; color: #9932CC;">ATTACK ACTIVE</div>
                    <div style="font-size: 0.9rem; color: #DDA0DD;">{attack_type}</div>
                    <div style="font-size: 0.8rem; color: #888;">Magnitude: {attack_mag:.2f}</div>
                </div>
            """, unsafe_allow_html=True)
        else:
            # Check if there are any alerts (attack might have just ended)
            has_recent_alerts = alerts_count > 0
            if has_recent_alerts:
                st.markdown(f"""
                    <div style="background: linear-gradient(135deg, #2d1b2e 0%, #1a1a2e 100%); 
                                border-radius: 10px; padding: 15px; text-align: center;
                                border-left: 4px solid #FF69B4;">
                        <div style="font-size: 2rem;">⚠️</div>
                        <div style="font-size: 1.2rem; font-weight: bold; color: #FF69B4;">ANOMALY</div>
                        <div style="font-size: 0.8rem; color: #888;">{alerts_count} alert(s) raised</div>
                    </div>
                """, unsafe_allow_html=True)
            else:
                st.markdown("""
                    <div style="background: linear-gradient(135deg, #1a1a2e 0%, #16213e 100%); 
                                border-radius: 10px; padding: 15px; text-align: center;
                                border-left: 4px solid #444;">
                        <div style="font-size: 2rem;">🛡️</div>
                        <div style="font-size: 1.2rem; font-weight: bold; color: #888;">NO ATTACK</div>
                        <div style="font-size: 0.8rem; color: #666;">System monitoring...</div>
                    </div>
                """, unsafe_allow_html=True)
    
    with react_col3:
        # Detection Method Active - matches config.py thresholds
        if display_deviation > 9:
            # Show which detection method caught it
            if display_deviation > 50:
                method = "Emergency Mode"
            elif display_deviation > 25:
                method = "DP + Statistical"
            else:
                method = "DP Algorithm"
            st.markdown(f"""
                <div style="background: linear-gradient(135deg, #1a2e1a 0%, #16213e 100%); 
                            border-radius: 10px; padding: 15px; text-align: center;
                            border-left: 4px solid #32CD32;">
                    <div style="font-size: 2rem;">🎯</div>
                    <div style="font-size: 1.2rem; font-weight: bold; color: #32CD32;">DETECTED</div>
                    <div style="font-size: 0.9rem; color: #90EE90;">{method}</div>
                    <div style="font-size: 0.8rem; color: #888;">Deviation: {display_deviation:.1f}%</div>
                </div>
            """, unsafe_allow_html=True)
        else:
            st.markdown("""
                <div style="background: linear-gradient(135deg, #1a1a2e 0%, #16213e 100%); 
                            border-radius: 10px; padding: 15px; text-align: center;
                            border-left: 4px solid #444;">
                    <div style="font-size: 2rem;">👁️</div>
                    <div style="font-size: 1.2rem; font-weight: bold; color: #888;">MONITORING</div>
                    <div style="font-size: 0.8rem; color: #666;">Awaiting anomaly...</div>
                </div>
            """, unsafe_allow_html=True)
    
    # Recent alerts (compact view)
    if alerts_data and alerts_data.get('alerts'):
        alerts = alerts_data['alerts'][-5:]  # Last 5 alerts
        if alerts:
            st.markdown("---")
            st.subheader("🚨 Recent Alerts")
            for alert in reversed(alerts):
                severity = alert.get('severity', 'LOW')
                method = alert.get('method', 'Unknown')
                severity_icons = {'LOW': '🟢', 'MEDIUM': '🟡', 'HIGH': '🟠', 'EMERGENCY': '🔴'}
                icon = severity_icons.get(severity, '🟢')
                st.caption(f"{icon} **{severity}** via {method} at {alert.get('start_time', 'N/A')}")
    
    # Speed context info
    current_speed = status_data.get('simulation', {}).get('speed', 100) if status_data else 100
    st.info(f"💡 **Live Mode** @ {current_speed}x speed (~{int(current_speed)} hrs/sec) | Refreshing every 1 second | Full analysis available when simulation stops.")


def create_attack_focused_chart(data_df, active_attacks):
    """Create a chart focused on injected attack periods."""
    if data_df is None or len(data_df) == 0 or not active_attacks:
        return None
    
    fig = go.Figure()
    
    # Convert timestamps
    timestamps = pd.to_datetime(data_df['timestamp'])
    
    # Find the time range that covers all attacks
    attack_ranges = []
    for attack in active_attacks:
        try:
            attack_start = pd.to_datetime(attack.get('start_time'))
            attack_end = pd.to_datetime(attack.get('end_time'))
            attack_ranges.append((attack_start, attack_end))
        except:
            continue
    
    if not attack_ranges:
        return None
    
    # Find min and max times across all attacks
    min_time = min([r[0] for r in attack_ranges])
    max_time = max([r[1] for r in attack_ranges])
    
    # Add buffer (2 hours before and after)
    buffer = pd.Timedelta(hours=2)
    chart_start = min_time - buffer
    chart_end = max_time + buffer
    
    # Filter data to attack period
    mask = (timestamps >= chart_start) & (timestamps <= chart_end)
    attack_data = data_df[mask].copy()
    attack_timestamps = pd.to_datetime(attack_data['timestamp'])
    
    if len(attack_data) == 0:
        return None
    
    # Add actual load line
    fig.add_trace(go.Scatter(
        x=attack_timestamps,
        y=attack_data['actual'],
        name='Actual Load',
        line=dict(color='#1f77b4', width=3),
        mode='lines+markers',
        marker=dict(size=6)
    ))
    
    # Add forecast line
    fig.add_trace(go.Scatter(
        x=attack_timestamps,
        y=attack_data['forecast'],
        name='Forecast',
        line=dict(color='#2ca02c', width=3, dash='dash'),
        mode='lines+markers',
        marker=dict(size=6)
    ))
    
    # Add benchmark line
    fig.add_trace(go.Scatter(
        x=attack_timestamps,
        y=attack_data['benchmark'],
        name='Benchmark',
        line=dict(color='#ff7f0e', width=3, dash='dot'),
        mode='lines+markers',
        marker=dict(size=6)
    ))
    
    # Identify attack regions
    attack_regions_list = []
    for attack in active_attacks:
        try:
            attack_start = pd.to_datetime(attack.get('start_time'))
            attack_end = pd.to_datetime(attack.get('end_time'))
            attack_regions_list.append((attack_start, attack_end))
        except:
            pass
    
    # Highlight detected anomalies (red)
    deviation = np.abs(attack_data['scaling_ratio'] - 1.0)
    anomaly_mask = deviation > 0.09
    
    if anomaly_mask.any():
        anomaly_regions = []
        in_anomaly = False
        start_idx = None
        
        for i, is_anomaly in enumerate(anomaly_mask):
            if is_anomaly and not in_anomaly:
                start_idx = i
                in_anomaly = True
            elif not is_anomaly and in_anomaly:
                anomaly_regions.append((start_idx, i-1))
                in_anomaly = False
        
        if in_anomaly:
            anomaly_regions.append((start_idx, len(anomaly_mask)-1))
        
        for start_idx, end_idx in anomaly_regions:
            region_start = attack_timestamps.iloc[start_idx]
            region_end = attack_timestamps.iloc[end_idx]
            
            is_in_attack_region = False
            for attack_start, attack_end in attack_regions_list:
                if not (region_end < attack_start or region_start > attack_end):
                    is_in_attack_region = True
                    break
            
            if not is_in_attack_region:
                fig.add_vrect(
                    x0=region_start,
                    x1=region_end,
                    fillcolor="rgba(255, 0, 0, 0.2)",
                    layer="below",
                    line_width=2,
                    line_color="red",
                    annotation_text="🚨 Detected",
                    annotation_position="top right",
                    annotation_font_size=10,
                    annotation_font_color="red"
                )
    
    # Highlight attack periods (purple)
    for idx, attack in enumerate(active_attacks):
        try:
            attack_start = pd.to_datetime(attack.get('start_time'))
            attack_end = pd.to_datetime(attack.get('end_time'))
            magnitude = attack.get('magnitude', 0.1)
            attack_type = attack.get('type', 'ATTACK')
            
            if magnitude <= 0.5:
                opacity = 0.15 + (magnitude / 0.5) * 0.15
            elif magnitude <= 1.0:
                opacity = 0.3 + ((magnitude - 0.5) / 0.5) * 0.2
            else:
                opacity = min(0.5 + ((magnitude - 1.0) / 2.0) * 0.2, 0.7)
            
            color = f"rgba(138, 43, 226, {opacity})"
            line_width = max(3, int(3 + magnitude * 2))
            
            fig.add_vrect(
                x0=attack_start,
                x1=attack_end,
                fillcolor=color,
                layer="below",
                line_width=line_width,
                line_color="purple",
                line_dash="dash",
                annotation_text=f"💉 {attack_type}<br>Mag: {magnitude:.2f}",
                annotation_position="top left",
                annotation_font_size=12,
                annotation_font_color="purple"
            )
            
            mid_time = attack_start + (attack_end - attack_start) / 2
            mid_idx = attack_timestamps.searchsorted(mid_time)
            if mid_idx < len(attack_data):
                mid_actual = attack_data.iloc[mid_idx]['actual']
                fig.add_trace(go.Scatter(
                    x=[mid_time],
                    y=[mid_actual],
                    mode='markers+text',
                    marker=dict(symbol='diamond', size=18, color='purple', line=dict(width=3, color='white')),
                    text=[f"{attack_type}"],
                    textposition="top center",
                    textfont=dict(color='purple', size=11, family='Arial Black'),
                    name=f"Attack: {attack_type}",
                    showlegend=False,
                    hovertemplate=f"<b>{attack_type} Attack</b><br>Magnitude: {magnitude:.2f}<extra></extra>"
                ))
        except:
            continue
    
    fig.update_layout(
        title=f"💉 Attack Injection View ({len(active_attacks)} attack(s))",
        xaxis_title="Time",
        yaxis_title="Load (MWh)",
        hovermode='x unified',
        height=400,
        legend=dict(yanchor="top", y=0.99, xanchor="left", x=0.01),
        xaxis=dict(range=[chart_start, chart_end])
    )
    
    return fig


def get_severity_color(deviation_pct):
    """
    Get color based on deviation severity.
    
    Matches config.py thresholds:
    - EMERGENCY: >50% deviation → Deep Red (INSTANT ALERT)
    - HIGH: 40-50% deviation → Red-Orange (Strong attack)
    - MEDIUM: 25-40% deviation → Orange (Medium attack)
    - LOW: 9-25% deviation → Light Green (Weak attack)
    """
    if deviation_pct < 0.25:  # LOW (9-25%)
        return "rgba(144, 238, 144, 0.3)", "#90EE90", "LOW"  # Light green
    elif deviation_pct < 0.40:  # MEDIUM (25-40%)
        return "rgba(255, 165, 0, 0.3)", "#FFA500", "MEDIUM"  # Orange
    elif deviation_pct < 0.50:  # HIGH (40-50%)
        return "rgba(255, 99, 71, 0.4)", "#FF6347", "HIGH"  # Tomato red
    else:  # EMERGENCY (>50%)
        return "rgba(220, 20, 60, 0.5)", "#DC143C", "EMERGENCY"  # Crimson


def create_load_chart(data_df, hours=24, active_attacks=None):
    """Create real-time load chart with multiple series and severity-based anomaly colors."""
    if data_df is None or len(data_df) == 0:
        return None
    
    fig = go.Figure()
    
    timestamps = pd.to_datetime(data_df['timestamp'])
    
    fig.add_trace(go.Scatter(
        x=timestamps, y=data_df['actual'],
        name='Actual Load', line=dict(color='#1f77b4', width=2), mode='lines'
    ))
    
    fig.add_trace(go.Scatter(
        x=timestamps, y=data_df['forecast'],
        name='Forecast', line=dict(color='#2ca02c', width=2, dash='dash'), mode='lines'
    ))
    
    fig.add_trace(go.Scatter(
        x=timestamps, y=data_df['benchmark'],
        name='Benchmark', line=dict(color='#ff7f0e', width=2, dash='dot'), mode='lines'
    ))
    
    # Attack regions (for exclusion from anomaly highlighting)
    attack_regions = []
    if active_attacks:
        for attack in active_attacks:
            try:
                attack_start = pd.to_datetime(attack.get('start_time'))
                attack_end = pd.to_datetime(attack.get('end_time'))
                if attack_start <= timestamps.iloc[-1] and attack_end >= timestamps.iloc[0]:
                    chart_start = max(attack_start, timestamps.iloc[0])
                    chart_end = min(attack_end, timestamps.iloc[-1])
                    attack_regions.append((chart_start, chart_end))
            except:
                pass
    
    # Anomaly regions with SEVERITY-BASED COLORS
    deviation = np.abs(data_df['scaling_ratio'] - 1.0)
    anomaly_mask = deviation > 0.09  # 9% threshold
    
    if anomaly_mask.any():
        # Find continuous anomaly regions with their max deviation
        anomaly_regions = []
        in_anomaly = False
        start_idx = None
        
        for i, is_anomaly in enumerate(anomaly_mask):
            if is_anomaly and not in_anomaly:
                start_idx = i
                in_anomaly = True
            elif not is_anomaly and in_anomaly:
                # Calculate max deviation in this region
                max_dev = deviation.iloc[start_idx:i].max()
                anomaly_regions.append((start_idx, i-1, max_dev))
                in_anomaly = False
        
        if in_anomaly:
            max_dev = deviation.iloc[start_idx:].max()
            anomaly_regions.append((start_idx, len(anomaly_mask)-1, max_dev))
        
        # Draw anomaly regions with severity-based colors
        for start_idx, end_idx, max_deviation in anomaly_regions:
            region_start = timestamps.iloc[start_idx]
            region_end = timestamps.iloc[end_idx]
            
            # Check if in attack region (skip if so - attacks shown separately)
            is_in_attack_region = False
            for attack_start, attack_end in attack_regions:
                if not (region_end < attack_start or region_start > attack_end):
                    is_in_attack_region = True
                    break
            
            if not is_in_attack_region:
                fill_color, line_color, severity = get_severity_color(max_deviation)
                
                # Severity icons
                severity_icons = {'LOW': '🟢', 'MEDIUM': '🟡', 'HIGH': '🟠', 'EMERGENCY': '🔴'}
                icon = severity_icons.get(severity, '⚠️')
                
                fig.add_vrect(
                    x0=region_start, x1=region_end,
                    fillcolor=fill_color, layer="below",
                    line_width=2, line_color=line_color,
                    annotation_text=f"{icon} {severity}<br>{max_deviation*100:.1f}%",
                    annotation_position="top right",
                    annotation_font_size=9,
                    annotation_font_color=line_color
                )
    
    # Attack regions (purple with DOTTED BORDERS)
    if active_attacks:
        for attack in active_attacks:
            try:
                attack_start = pd.to_datetime(attack.get('start_time'))
                attack_end = pd.to_datetime(attack.get('end_time'))
                magnitude = attack.get('magnitude', 0.1)
                attack_type = attack.get('type', 'ATTACK')
                
                if attack_start <= timestamps.iloc[-1] and attack_end >= timestamps.iloc[0]:
                    chart_start = max(attack_start, timestamps.iloc[0])
                    chart_end = min(attack_end, timestamps.iloc[-1])
                    
                    if magnitude <= 0.5:
                        opacity = 0.15 + (magnitude / 0.5) * 0.15
                    elif magnitude <= 1.0:
                        opacity = 0.3 + ((magnitude - 0.5) / 0.5) * 0.2
                    else:
                        opacity = min(0.5 + ((magnitude - 1.0) / 2.0) * 0.2, 0.7)
                    
                    # Add shaded region with dotted border
                    fig.add_vrect(
                        x0=chart_start, x1=chart_end,
                        fillcolor=f"rgba(138, 43, 226, {opacity})", layer="below",
                        line_width=3, line_color="purple", line_dash="dot",
                        annotation_text=f"💉 {attack_type}<br>Mag: {magnitude:.2f}",
                        annotation_position="top left", annotation_font_size=11, annotation_font_color="purple"
                    )
                    
                    # Add vertical dotted lines at attack boundaries for visibility
                    y_min = data_df['actual'].min() * 0.95
                    y_max = data_df['actual'].max() * 1.05
                    
                    # Start boundary line
                    fig.add_trace(go.Scatter(
                        x=[chart_start, chart_start],
                        y=[y_min, y_max],
                        mode='lines',
                        line=dict(color='purple', width=2, dash='dot'),
                        showlegend=False,
                        hoverinfo='skip'
                    ))
                    
                    # End boundary line
                    fig.add_trace(go.Scatter(
                        x=[chart_end, chart_end],
                        y=[y_min, y_max],
                        mode='lines',
                        line=dict(color='purple', width=2, dash='dot'),
                        showlegend=False,
                        hoverinfo='skip'
                    ))
            except:
                pass
    
    fig.update_layout(
        title="Real-Time Load Monitoring",
        xaxis_title="Time", yaxis_title="Load (MWh)",
        hovermode='x unified', height=500,
        legend=dict(yanchor="top", y=0.99, xanchor="left", x=0.01)
    )
    
    return fig


def render_analysis_mode(status_data):
    """Render full analysis mode UI (when simulation is stopped)."""
    st.markdown('<div class="main-header">⚡ Power Grid Protection Dashboard</div>', unsafe_allow_html=True)
    
    # Status row
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        if status_data:
            sim_status = status_data.get('simulation', {})
            status_text = "🟢 Running" if sim_status.get('is_running') else "⚪ Stopped"
            st.metric("Status", status_text)
    
    with col2:
        if status_data:
            store_stats = status_data.get('data_store', {})
            st.metric("Data Hours", store_stats.get('total_hours', 0))
    
    with col3:
        alerts_data = fetch_api_data("alerts", {"limit": 100})
        if alerts_data:
            st.metric("Active Alerts", alerts_data.get('count', 0))
    
    with col4:
        metrics_data = fetch_api_data("metrics")
        if metrics_data and 'metrics' in metrics_data:
            throughput = metrics_data['metrics'].get('throughput_hours_per_sec', 0)
            st.metric("Throughput", f"{throughput:.1f} hrs/sec")
    
    st.divider()
    
    # Get attacks
    injected_attacks_data = fetch_api_data("simulation/injected_attacks")
    active_attacks = injected_attacks_data.get('injected_attacks', []) if injected_attacks_data else []
    
    # Attack-focused chart with ML Model Response Analysis
    if active_attacks:
        st.header("🧠 ML Model Response to Attack Injection")
        st.caption("📊 Analysis of how the ML model detected and responded to injected attacks")
        
        # Fetch ALL available data to ensure we capture all attacks
        # Use a large number to get all stored data
        forecast_data = fetch_api_data("forecast", {"hours": 10000})
        alerts_for_analysis = fetch_api_data("alerts", {"limit": 100})
        
        if forecast_data and forecast_data.get('data'):
            data_df = pd.DataFrame(forecast_data['data'])
            
            # === ML MODEL RESPONSE SUMMARY ===
            st.subheader("📋 Detection Summary")
            
            # Analyze each attack
            for idx, attack in enumerate(active_attacks):
                attack_type = attack.get('type', 'UNKNOWN')
                attack_mag = attack.get('magnitude', 0)
                attack_start = attack.get('start_time', '')
                attack_end = attack.get('end_time', '')
                attack_duration = attack.get('duration', 0)
                
                # Find data during attack period
                try:
                    attack_start_dt = pd.to_datetime(attack_start)
                    attack_end_dt = pd.to_datetime(attack_end)
                    timestamps = pd.to_datetime(data_df['timestamp'])
                    
                    # Check if attack period is within our data range
                    data_start = timestamps.min()
                    data_end = timestamps.max()
                    
                    # Adjust attack times to be within data range if needed
                    effective_start = max(attack_start_dt, data_start)
                    effective_end = min(attack_end_dt, data_end)
                    
                    attack_mask = (timestamps >= effective_start) & (timestamps <= effective_end)
                    attack_data = data_df[attack_mask]
                    
                    if len(attack_data) > 0:
                        # Calculate detection metrics
                        deviations = np.abs(attack_data['scaling_ratio'] - 1.0) * 100
                        max_deviation = deviations.max()
                        avg_deviation = deviations.mean()
                        detected_hours = (deviations > 9).sum()  # Hours where anomaly was detected
                        
                        # Determine severity (matching config.py thresholds)
                        # EMERGENCY_THRESHOLD = 0.50 (50%)
                        # Strong: >40%, Medium: 25-40%, Weak: 9-25%
                        if max_deviation > 50:
                            severity = "EMERGENCY"
                            severity_color = "#DC143C"
                            severity_icon = "🔴"
                            severity_desc = "INSTANT ALERT - Critical grid threat!"
                        elif max_deviation > 40:
                            severity = "HIGH"
                            severity_color = "#FF6347"
                            severity_icon = "🟠"
                            severity_desc = "Strong attack - Fast detection (1hr)"
                        elif max_deviation > 25:
                            severity = "MEDIUM"
                            severity_color = "#FFA500"
                            severity_icon = "🟡"
                            severity_desc = "Medium attack - Standard detection (2hr)"
                        elif max_deviation > 9:
                            severity = "LOW"
                            severity_color = "#90EE90"
                            severity_icon = "🟢"
                            severity_desc = "Weak attack - Extended detection (3hr)"
                        else:
                            severity = "UNDETECTED"
                            severity_color = "#888"
                            severity_icon = "⚪"
                            severity_desc = "Below 9% threshold"
                        
                        # Detection method based on config thresholds
                        if max_deviation > 50:
                            method = "Emergency Mode (Instant)"
                        elif max_deviation > 40:
                            method = "DP + Statistical Hybrid"
                        elif max_deviation > 25:
                            method = "DP + Statistical Hybrid"
                        elif max_deviation > 9:
                            method = "DP Algorithm"
                        else:
                            method = "Below Threshold"
                        
                        # Display attack analysis card
                        st.markdown(f"""
                        <div style="background: linear-gradient(135deg, #1a1a2e 0%, #16213e 100%); 
                                    border-radius: 10px; padding: 20px; margin: 10px 0;
                                    border-left: 5px solid {severity_color};">
                            <div style="display: flex; justify-content: space-between; align-items: center;">
                                <div>
                                    <h3 style="color: white; margin: 0;">💉 Attack #{idx+1}: {attack_type}</h3>
                                    <p style="color: #888; margin: 5px 0;">Magnitude: {attack_mag:.2f} | Duration: {attack_duration}h</p>
                                    <p style="color: #666; font-size: 0.9rem;">{attack_start} → {attack_end}</p>
                                </div>
                                <div style="text-align: right;">
                                    <div style="font-size: 2rem;">{severity_icon}</div>
                                    <div style="color: {severity_color}; font-weight: bold;">{severity}</div>
                                </div>
                            </div>
                            <hr style="border-color: #333; margin: 15px 0;">
                            <div style="display: flex; justify-content: space-around; text-align: center;">
                                <div>
                                    <div style="color: #00d4ff; font-size: 1.5rem; font-weight: bold;">{max_deviation:.1f}%</div>
                                    <div style="color: #888; font-size: 0.8rem;">Max Deviation</div>
                                </div>
                                <div>
                                    <div style="color: #00ff88; font-size: 1.5rem; font-weight: bold;">{avg_deviation:.1f}%</div>
                                    <div style="color: #888; font-size: 0.8rem;">Avg Deviation</div>
                                </div>
                                <div>
                                    <div style="color: #ffa500; font-size: 1.5rem; font-weight: bold;">{detected_hours}/{attack_duration}</div>
                                    <div style="color: #888; font-size: 0.8rem;">Hours Detected</div>
                                </div>
                                <div>
                                    <div style="color: #DDA0DD; font-size: 1rem; font-weight: bold;">{method}</div>
                                    <div style="color: #888; font-size: 0.8rem;">Detection Method</div>
                                </div>
                            </div>
                        </div>
                        """, unsafe_allow_html=True)
                    else:
                        # Show why no data - helpful for debugging
                        st.warning(f"Attack #{idx+1} ({attack_type}): Attack period ({attack_start} to {attack_end}) has no matching data. Data range: {data_start} to {data_end}")
                except Exception as e:
                    st.warning(f"Attack #{idx+1}: Could not analyze - {str(e)}")
            
            st.divider()
            
            # === VISUAL CHART ===
            st.subheader("📈 Attack Period Visualization")
            attack_chart = create_attack_focused_chart(data_df, active_attacks)
            
            if attack_chart:
                st.plotly_chart(attack_chart, use_container_width=True)
        
        st.divider()
    
    # Main chart
    st.header("📊 Load Analysis Chart")
    
    forecast_data_all = fetch_api_data("forecast", {"hours": 168})
    
    if forecast_data_all and forecast_data_all.get('data') and len(forecast_data_all['data']) > 0:
        all_data_df = pd.DataFrame(forecast_data_all['data'])
        
        if len(all_data_df) > 1:
            max_hours = max(2, min(168, len(all_data_df)))  # Ensure at least 2
            default_hours = min(24, max_hours)
            
            hours_slider = st.slider(
                "Time Range (hours)", 
                min_value=1, 
                max_value=max_hours, 
                value=default_hours,
                key="analysis_hours_slider"
            )
            
            forecast_data = fetch_api_data("forecast", {"hours": hours_slider})
            
            if forecast_data and forecast_data.get('data') and len(forecast_data['data']) > 0:
                data_df = pd.DataFrame(forecast_data['data'])
                chart = create_load_chart(data_df, hours=hours_slider, active_attacks=active_attacks)
                if chart:
                    st.plotly_chart(chart, use_container_width=True)
                else:
                    st.warning("Could not render chart. Try adjusting the time range.")
            else:
                st.warning("No data available for the selected time range.")
        else:
            st.info("Not enough data points. Run simulation longer to see chart.")
    else:
        st.info("No data available. Start simulation to see chart.")
    
    # Legend
    with st.expander("📖 Chart Legend"):
        st.markdown("""
        **Lines:**
        - 🔵 **Actual Load** (Blue solid): Real measured power grid load
        - 🟢 **Forecast** (Green dashed): LSTM model prediction  
        - 🟠 **Benchmark** (Orange dotted): Normal pattern from K-means clustering
        
        **Attack Regions (Purple):**
        - 💉 Injected attacks shown with purple shading (darker = stronger magnitude)
        
        **Anomaly Detection Severity (Color-coded):**
        - 🟢 **LOW** (9-15% deviation): Light green - Minor anomaly
        - 🟡 **MEDIUM** (15-25% deviation): Orange - Moderate anomaly
        - 🟠 **HIGH** (25-40% deviation): Red-Orange - Significant anomaly
        - 🔴 **EMERGENCY** (>40% deviation): Deep Red - Critical anomaly
        """)
    
    # Alerts
    st.header("🚨 Alerts & Detections")
    
    alerts_data = fetch_api_data("alerts", {"limit": 50})
    if alerts_data and alerts_data.get('alerts'):
        alerts = alerts_data['alerts']
        
        severity_filter = st.multiselect("🔍 Filter by Severity", 
            ["LOW", "MEDIUM", "HIGH", "EMERGENCY"],
            default=["LOW", "MEDIUM", "HIGH", "EMERGENCY"])
        
        filtered_alerts = [a for a in alerts if a.get('severity') in severity_filter]
        
        if filtered_alerts:
            st.caption(f"Showing {len(filtered_alerts)} alert(s)")
            for alert in filtered_alerts[:20]:
                severity = alert.get('severity', 'LOW')
                severity_colors = {'LOW': '#51cf66', 'MEDIUM': '#ffa94d', 'HIGH': '#ff6b6b', 'EMERGENCY': '#c92a2a'}
                severity_icons = {'LOW': '🟢', 'MEDIUM': '🟡', 'HIGH': '🟠', 'EMERGENCY': '🔴'}
                
                st.markdown(f"""
                <div style="background-color: {severity_colors.get(severity, '#51cf66')}15; 
                    border-left: 4px solid {severity_colors.get(severity, '#51cf66')}; 
                    padding: 0.5rem; margin: 0.25rem 0; border-radius: 0.25rem;">
                    <strong>{severity_icons.get(severity, '🟢')} {severity}</strong> | 
                    📅 {alert.get('start_time', 'N/A')} → {alert.get('end_time', 'N/A')} | 
                    Method: {alert.get('method', 'N/A')}
                </div>
                """, unsafe_allow_html=True)
        else:
            st.info("ℹ️ No alerts match filters.")
    else:
        st.success("✅ No alerts - System operating normally!")
    
    # Performance metrics
    st.header("📈 Performance Metrics")
    
    metrics_data = fetch_api_data("metrics")
    if metrics_data and 'metrics' in metrics_data:
        metrics = metrics_data['metrics']
        
        col1, col2, col3, col4 = st.columns(4)
        
        with col1:
            latency = metrics.get('latency', {})
            latency_mean = latency.get('mean')
            st.metric("⏱️ Mean Latency", f"{latency_mean:.3f}s" if latency_mean else "N/A")
        
        with col2:
            st.metric("⚡ Throughput", f"{metrics.get('throughput_hours_per_sec', 0):.2f} hrs/sec")
        
        with col3:
            memory_mb = metrics.get('memory', {}).get('memory_mb', 0)
            st.metric("💾 Memory", f"{memory_mb:.0f} MB" if memory_mb > 0 else "N/A")
        
        with col4:
            st.metric("📊 Hours Processed", metrics.get('hours_processed', 0))


def main():
    """Main dashboard function with live/analysis mode switching."""
    
    # Sidebar
    with st.sidebar:
        st.header("⚙️ Control Panel")
        
        # Simulation control
        st.subheader("Simulation Control")
        status_data = fetch_api_data("status")
        
        is_running = False
        if status_data and status_data.get('simulation'):
            sim_status = status_data['simulation']
            is_running = sim_status.get('is_running', False)
            
            if is_running:
                st.success("🟢 Simulation Running")
                if st.button("⏹️ Stop Simulation"):
                    result = post_api_data("simulation/stop")
                    if result and result.get('status') == 'success':
                        st.success("Simulation stopped")
                        st.rerun()
            else:
                st.info("⚪ Simulation Stopped")
                speed = st.number_input("Speed (1x = real-time)", min_value=1.0, value=100.0, step=10.0)
                if st.button("▶️ Start Simulation"):
                    result = post_api_data("simulation/start", {"speed": speed})
                    if result and result.get('status') == 'success':
                        st.success("Simulation started")
                        st.rerun()
            
            # Speed control
            if is_running:
                current_speed = sim_status.get('speed', 100.0)
                new_speed = st.slider("Simulation Speed", min_value=1.0, max_value=1000.0, 
                                     value=float(current_speed), step=10.0)
                if new_speed != current_speed:
                    post_api_data("simulation/speed", {"speed": new_speed})
            
            # Reset
            st.divider()
            if st.button("🗑️ Reset Simulation"):
                if st.session_state.get('reset_confirmed', False):
                    result = post_api_data("simulation/reset")
                    if result and result.get('status') == 'success':
                        st.success("✅ Reset!")
                        st.session_state.reset_confirmed = False
                        st.session_state.live_data = {
                            'hours_processed': 0, 'recent_actuals': [], 'recent_forecasts': [],
                            'recent_timestamps': [], 'alerts_count': 0, 'last_timestamp': None
                        }
                        st.rerun()
                else:
                    st.session_state.reset_confirmed = True
                    st.warning("⚠️ Click again to confirm")
        
        st.divider()
        
        # Attack injection
        st.subheader("💉 Attack Injection")
        
        attack_type = st.selectbox("Attack Type",
            ["PULSE", "SCALING", "RAMPING", "RANDOM", "SMOOTH-CURVE", 
             "POINT-BURST", "CONTEXTUAL-SEASONAL", "RAMPING-TYPE2"])
        
        attack_start = st.number_input("Start Hour (offset)", min_value=0, value=5)
        attack_duration = st.number_input("Duration (hours)", min_value=1, value=2)
        attack_magnitude = st.slider("Magnitude", min_value=0.1, max_value=10.0, value=0.5, step=0.1)
        
        if st.button("💉 Inject Attack"):
            result = post_api_data("simulation/inject_attack", {
                "type": attack_type, "start_hour": attack_start,
                "duration": attack_duration, "magnitude": attack_magnitude
            })
            if result and result.get('status') == 'success':
                st.success("Attack injected!")
            else:
                st.error("Failed to inject attack")
        
        st.divider()
        
        # Display mode
        st.subheader("🖥️ Display Mode")
        if is_running:
            use_live_mode = st.checkbox("🔴 Live Mode", value=True, 
                                       help="Fast polling with minimal UI")
        else:
            use_live_mode = False
            st.caption("Live mode available when simulation is running")
    
    # Main content
    if is_running and use_live_mode:
        render_live_mode(status_data)
        
        # Fast refresh for live mode (1 second)
        time.sleep(1)
        st.rerun()
    else:
        render_analysis_mode(status_data)


if __name__ == "__main__":
    main()

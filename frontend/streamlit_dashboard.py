"""
Streamlit Dashboard - Real-time visualization for power grid protection system
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

# Custom CSS
st.markdown("""
<style>
    .main-header {
        font-size: 2.5rem;
        font-weight: bold;
        color: #1f77b4;
        margin-bottom: 1rem;
    }
    .alert-high {
        background-color: #ff6b6b;
        color: white;
        padding: 0.5rem;
        border-radius: 0.25rem;
        margin: 0.25rem 0;
    }
    .alert-medium {
        background-color: #ffa94d;
        color: white;
        padding: 0.5rem;
        border-radius: 0.25rem;
        margin: 0.25rem 0;
    }
    .alert-low {
        background-color: #51cf66;
        color: white;
        padding: 0.5rem;
        border-radius: 0.25rem;
        margin: 0.25rem 0;
    }
    .alert-emergency {
        background-color: #c92a2a;
        color: white;
        padding: 0.5rem;
        border-radius: 0.25rem;
        margin: 0.25rem 0;
        font-weight: bold;
    }
</style>
""", unsafe_allow_html=True)


def fetch_api_data(endpoint, params=None):
    """Fetch data from API endpoint."""
    try:
        url = f"{API_BASE_URL}/{endpoint}"
        response = requests.get(url, params=params, timeout=5)
        if response.status_code == 200:
            return response.json()
        else:
            return None
    except Exception as e:
        st.error(f"Error fetching {endpoint}: {e}")
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
    
    # First, identify attack regions
    attack_regions_list = []
    for attack in active_attacks:
        try:
            attack_start = pd.to_datetime(attack.get('start_time'))
            attack_end = pd.to_datetime(attack.get('end_time'))
            attack_regions_list.append((attack_start, attack_end))
        except:
            pass
    
    # Highlight detected anomalies FIRST (red) - but exclude attack regions
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
        
        # Add red regions for detected anomalies - only if not in attack region
        for start_idx, end_idx in anomaly_regions:
            region_start = attack_timestamps.iloc[start_idx]
            region_end = attack_timestamps.iloc[end_idx]
            
            # Check if this region overlaps with any attack region
            is_in_attack_region = False
            for attack_start, attack_end in attack_regions_list:
                if not (region_end < attack_start or region_start > attack_end):
                    is_in_attack_region = True
                    break
            
            # Only show red if not in attack region
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
    
    # Highlight each attack period with magnitude-based color intensity (lighter for visibility) - DRAW LAST so purple is visible
    for idx, attack in enumerate(active_attacks):
        try:
            attack_start = pd.to_datetime(attack.get('start_time'))
            attack_end = pd.to_datetime(attack.get('end_time'))
            magnitude = attack.get('magnitude', 0.1)
            attack_type = attack.get('type', 'ATTACK')
            
            # Calculate color intensity based on magnitude (more visible)
            if magnitude <= 0.5:
                opacity = 0.15 + (magnitude / 0.5) * 0.15  # 0.15 to 0.3
            elif magnitude <= 1.0:
                opacity = 0.3 + ((magnitude - 0.5) / 0.5) * 0.2  # 0.3 to 0.5
            else:
                opacity = min(0.5 + ((magnitude - 1.0) / 2.0) * 0.2, 0.7)  # 0.5 to 0.7 (capped)
            
            # Base purple color with variable opacity
            color = f"rgba(138, 43, 226, {opacity})"
            
            # Line width also varies with magnitude
            line_width = max(3, int(3 + magnitude * 2))
            
            fig.add_vrect(
                x0=attack_start,
                x1=attack_end,
                fillcolor=color,
                layer="below",  # Below lines but above red regions
                line_width=line_width,
                line_color="purple",
                line_dash="dash",
                annotation_text=f"💉 {attack_type}<br>Mag: {magnitude:.2f}",
                annotation_position="top left",
                annotation_font_size=12,
                annotation_font_color="purple"
            )
            
            # Add attack type marker at midpoint for visibility
            mid_time = attack_start + (attack_end - attack_start) / 2
            mid_idx = attack_timestamps.searchsorted(mid_time)
            if mid_idx < len(attack_data):
                mid_actual = attack_data.iloc[mid_idx]['actual']
                # Add marker for attack type
                fig.add_trace(go.Scatter(
                    x=[mid_time],
                    y=[mid_actual],
                    mode='markers+text',
                    marker=dict(
                        symbol='diamond',
                        size=18,
                        color='purple',
                        line=dict(width=3, color='white')
                    ),
                    text=[f"{attack_type}"],
                    textposition="top center",
                    textfont=dict(color='purple', size=11, family='Arial Black'),
                    name=f"Attack: {attack_type}",
                    showlegend=False,
                    hovertemplate=f"<b>{attack_type} Attack</b><br>Magnitude: {magnitude:.2f}<br>Duration: {attack.get('duration', 0)}h<extra></extra>"
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


def create_load_chart(data_df, hours=24, active_attacks=None):
    """
    Create real-time load chart with multiple series.
    
    Args:
        data_df: DataFrame with timestamp, actual, forecast, benchmark, scaling_ratio
        hours: Number of hours (ignored if data_df already filtered)
        active_attacks: List of active attack dictionaries
    """
    if data_df is None or len(data_df) == 0:
        return None
    
    fig = go.Figure()
    
    # Convert timestamps
    timestamps = pd.to_datetime(data_df['timestamp'])
    
    # Add actual load line
    fig.add_trace(go.Scatter(
        x=timestamps,
        y=data_df['actual'],
        name='Actual Load',
        line=dict(color='#1f77b4', width=2),
        mode='lines'
    ))
    
    # Add forecast line
    fig.add_trace(go.Scatter(
        x=timestamps,
        y=data_df['forecast'],
        name='Forecast',
        line=dict(color='#2ca02c', width=2, dash='dash'),
        mode='lines'
    ))
    
    # Add benchmark line
    fig.add_trace(go.Scatter(
        x=timestamps,
        y=data_df['benchmark'],
        name='Benchmark',
        line=dict(color='#ff7f0e', width=2, dash='dot'),
        mode='lines'
    ))
    
    # First, identify attack regions to exclude from red anomaly detection
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
    
    # Highlight anomalous regions (where scaling ratio deviates significantly) - RED
    # But exclude attack regions (they'll be purple)
    deviation = np.abs(data_df['scaling_ratio'] - 1.0)
    anomaly_mask = deviation > 0.09
    
    if anomaly_mask.any():
        # Find continuous anomaly regions
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
        
        # Add shaded regions for detected anomalies (RED) - only if not in attack region
        for start_idx, end_idx in anomaly_regions:
            region_start = timestamps.iloc[start_idx]
            region_end = timestamps.iloc[end_idx]
            
            # Check if this region overlaps with any attack region
            is_in_attack_region = False
            for attack_start, attack_end in attack_regions:
                if not (region_end < attack_start or region_start > attack_end):
                    is_in_attack_region = True
                    break
            
            # Only show red if not in attack region
            if not is_in_attack_region:
                fig.add_vrect(
                    x0=region_start,
                    x1=region_end,
                    fillcolor="rgba(255, 0, 0, 0.2)",  # Red for detected anomalies
                    layer="below",
                    line_width=1,
                    line_color="red",
                    annotation_text="Detected Anomaly",
                    annotation_position="top right"
                )
    
    # Highlight injected attack regions (purple) with magnitude-based intensity - DRAW LAST so it's on top
    if active_attacks:
        for attack in active_attacks:
            try:
                attack_start = pd.to_datetime(attack.get('start_time'))
                attack_end = pd.to_datetime(attack.get('end_time'))
                magnitude = attack.get('magnitude', 0.1)
                attack_type = attack.get('type', 'ATTACK')
                
                # Check if attack time is within chart range
                if attack_start <= timestamps.iloc[-1] and attack_end >= timestamps.iloc[0]:
                    # Clip to chart range
                    chart_start = max(attack_start, timestamps.iloc[0])
                    chart_end = min(attack_end, timestamps.iloc[-1])
                    
                    # Calculate color intensity based on magnitude (lighter for better visibility)
                    if magnitude <= 0.5:
                        opacity = 0.15 + (magnitude / 0.5) * 0.15  # 0.15 to 0.3 (more visible)
                    elif magnitude <= 1.0:
                        opacity = 0.3 + ((magnitude - 0.5) / 0.5) * 0.2  # 0.3 to 0.5
                    else:
                        opacity = min(0.5 + ((magnitude - 1.0) / 2.0) * 0.2, 0.7)  # 0.5 to 0.7 (capped)
                    
                    color = f"rgba(138, 43, 226, {opacity})"
                    line_width = max(3, int(3 + magnitude * 2))
                    
                    # Add purple shaded region
                    fig.add_vrect(
                        x0=chart_start,
                        x1=chart_end,
                        fillcolor=color,
                        layer="below",  # Below lines but above red regions
                        line_width=line_width,
                        line_color="purple",
                        line_dash="dash",
                        annotation_text=f"💉 {attack_type}<br>Mag: {magnitude:.2f}",
                        annotation_position="top left",
                        annotation_font_size=11,
                        annotation_font_color="purple"
                    )
                    
                    # Add attack type markers at start and midpoint for visibility
                    mid_time = chart_start + (chart_end - chart_start) / 2
                    mid_idx = timestamps.searchsorted(mid_time)
                    if mid_idx < len(data_df):
                        mid_actual = data_df.iloc[mid_idx]['actual']
                        # Add marker for attack type
                        fig.add_trace(go.Scatter(
                            x=[mid_time],
                            y=[mid_actual],
                            mode='markers+text',
                            marker=dict(
                                symbol='diamond',
                                size=15,
                                color='purple',
                                line=dict(width=2, color='white')
                            ),
                            text=[f"{attack_type}"],
                            textposition="top center",
                            textfont=dict(color='purple', size=10, family='Arial Black'),
                            name=f"Attack: {attack_type}",
                            showlegend=False,
                            hovertemplate=f"<b>{attack_type} Attack</b><br>Magnitude: {magnitude:.2f}<extra></extra>"
                        ))
            except Exception as e:
                pass  # Skip if timestamp conversion fails
    
    fig.update_layout(
        title="Real-Time Load Monitoring",
        xaxis_title="Time",
        yaxis_title="Load (MWh)",
        hovermode='x unified',
        height=500,
        legend=dict(yanchor="top", y=0.99, xanchor="left", x=0.01)
    )
    
    return fig


def main():
    """Main dashboard function."""
    # Header
    st.markdown('<div class="main-header">⚡ Power Grid Protection Dashboard</div>', unsafe_allow_html=True)
    
    # Sidebar
    with st.sidebar:
        st.header("⚙️ Control Panel")
        
        # Simulation control
        st.subheader("Simulation Control")
        status_data = fetch_api_data("status")
        
        if status_data and status_data.get('simulation'):
            sim_status = status_data['simulation']
            
            if sim_status.get('is_running'):
                st.success("🟢 Simulation Running")
                if st.button("⏹️ Stop Simulation"):
                    result = post_api_data("simulation/stop")
                    if result and result.get('status') == 'success':
                        st.success("Simulation stopped")
                        st.rerun()
            else:
                st.info("⚪ Simulation Stopped")
                if st.button("▶️ Start Simulation"):
                    speed = st.number_input("Speed (1x = real-time)", min_value=1.0, value=100.0, step=10.0)
                    result = post_api_data("simulation/start", {"speed": speed})
                    if result and result.get('status') == 'success':
                        st.success("Simulation started")
                        st.rerun()
            
            # Speed control
            if sim_status.get('is_running'):
                current_speed = sim_status.get('speed', 100.0)
                new_speed = st.slider("Simulation Speed", min_value=1.0, max_value=1000.0, 
                                     value=float(current_speed), step=10.0)
                if new_speed != current_speed:
                    post_api_data("simulation/speed", {"speed": new_speed})
            
            # Reset button
            st.divider()
            st.subheader("🔄 Reset")
            if st.button("🗑️ Reset Simulation", help="Clear all data, attacks, and reset simulation state"):
                if st.session_state.get('reset_confirmed', False):
                    result = post_api_data("simulation/reset")
                    if result and result.get('status') == 'success':
                        st.success("✅ Simulation reset successfully!")
                        st.session_state.reset_confirmed = False
                        st.rerun()
                    else:
                        st.error("Failed to reset simulation")
                else:
                    st.session_state.reset_confirmed = True
                    st.warning("⚠️ Click again to confirm reset (this will clear all data)")
        
        st.divider()
        
        # Attack injection
        st.subheader("Attack Injection")
        
        attack_type = st.selectbox(
            "Attack Type",
            ["PULSE", "SCALING", "RAMPING", "RANDOM", "SMOOTH-CURVE", 
             "POINT-BURST", "CONTEXTUAL-SEASONAL", "RAMPING-TYPE2"]
        )
        
        attack_start = st.number_input("Start Hour (offset)", min_value=0, value=5)
        attack_duration = st.number_input("Duration (hours)", min_value=1, value=2)
        attack_magnitude = st.slider("Magnitude", min_value=0.1, max_value=2.0, value=0.2, step=0.1)
        
        if st.button("💉 Inject Attack"):
            result = post_api_data("simulation/inject_attack", {
                "type": attack_type,
                "start_hour": attack_start,
                "duration": attack_duration,
                "magnitude": attack_magnitude
            })
            if result and result.get('status') == 'success':
                st.success("Attack injected!")
            else:
                st.error("Failed to inject attack")
    
    # Main content
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
    
    # Get attacks for main chart visualization (use injected_attacks if simulation stopped)
    if status_data and status_data.get('simulation', {}).get('is_running'):
        active_attacks_data = fetch_api_data("simulation/active_attacks")
        active_attacks = active_attacks_data.get('active_attacks', []) if active_attacks_data else []
    else:
        # When stopped, use all injected attacks
        injected_attacks_data = fetch_api_data("simulation/injected_attacks")
        active_attacks = injected_attacks_data.get('injected_attacks', []) if injected_attacks_data else []
    
    # Attack-focused chart (separate, zoomed view) - ONLY SHOW WHEN SIMULATION IS STOPPED
    if status_data and status_data.get('simulation'):
        sim_status = status_data['simulation']
        is_running = sim_status.get('is_running', False)
        
        # Only show attack view when simulation is stopped
        if not is_running:
            # Get all injected attacks (including completed ones)
            injected_attacks_data = fetch_api_data("simulation/injected_attacks")
            injected_attacks = injected_attacks_data.get('injected_attacks', []) if injected_attacks_data else []
            
            if injected_attacks:
                st.header("💉 Attack Injection Analysis View")
                st.caption("📊 Zoomed view showing only the time periods where attacks were injected (shown only when simulation is stopped)")
                
                # Get enough data to cover all attacks
                forecast_data = fetch_api_data("forecast", {"hours": 168})  # Get more data to ensure we cover attacks
                
                if forecast_data and forecast_data.get('data'):
                    data_df = pd.DataFrame(forecast_data['data'])
                    attack_chart = create_attack_focused_chart(data_df, injected_attacks)
                    
                    if attack_chart:
                        st.plotly_chart(attack_chart, width='stretch')
                        
                        # Show attack details
                        st.info(f"**{len(injected_attacks)} Attack(s) Injected During Simulation:**")
                        for idx, attack in enumerate(injected_attacks, 1):
                            st.caption(f"  {idx}. **{attack.get('type', 'UNKNOWN')}** attack: "
                                     f"{attack.get('start_time', '')} to {attack.get('end_time', '')} "
                                     f"(Magnitude: {attack.get('magnitude', 0):.2f}, Duration: {attack.get('duration', 0)} hours)")
                    else:
                        st.warning("⚠️ Attack data not available in the displayed time range. The attack may have occurred outside the current data window.")
                else:
                    st.info("ℹ️ No data available. Run simulation and inject attacks to see this view.")
                
                st.divider()
    
    # Main real-time chart
    st.header("📊 Real-Time Load Chart")
    
    # Get available data to determine time range
    forecast_data_all = fetch_api_data("forecast", {"hours": 168})  # Get max data to determine range
    
    if forecast_data_all and forecast_data_all.get('data'):
        all_data_df = pd.DataFrame(forecast_data_all['data'])
        
        if len(all_data_df) > 0:
            # Convert timestamps
            all_timestamps = pd.to_datetime(all_data_df['timestamp'])
            min_time = all_timestamps.min()
            max_time = all_timestamps.max()
            
            # Time range selector
            use_custom_range = st.checkbox("📅 Use custom time range", value=False, help="Select specific start and end times instead of 'last N hours'")
            
            if use_custom_range:
                # Custom time range selector
                # Convert to datetime objects for slider
                min_timestamp = min_time.to_pydatetime()
                max_timestamp = max_time.to_pydatetime()
                
                # Default to last 24 hours
                default_start = max(min_timestamp, max_timestamp - timedelta(hours=24))
                default_end = max_timestamp
                
                # Create time range slider with proper initialization
                time_range = st.slider(
                    "Select Time Range",
                    min_value=min_timestamp,
                    max_value=max_timestamp,
                    value=(default_start, default_end),
                    format="YYYY-MM-DD HH:mm",
                    key="time_range_slider"
                )
                
                start_time, end_time = time_range
                
                # Filter data to selected range
                mask = (all_timestamps >= pd.to_datetime(start_time)) & (all_timestamps <= pd.to_datetime(end_time))
                filtered_data = all_data_df[mask].copy()
                
                if len(filtered_data) > 0:
                    chart = create_load_chart(filtered_data, hours=None, active_attacks=active_attacks)
                    if chart:
                        st.plotly_chart(chart, width='stretch')
                    hours_span = (end_time - start_time).total_seconds() / 3600
                    st.caption(f"📊 Showing {len(filtered_data)} hours from {start_time.strftime('%Y-%m-%d %H:%M')} to {end_time.strftime('%Y-%m-%d %H:%M')}")
                else:
                    st.warning("⚠️ No data in selected time range. Try adjusting the range.")
            else:
                # Default: Last N hours
                max_hours = min(168, len(all_data_df))
                hours_slider = st.slider("Time Range (hours)", min_value=1, max_value=max_hours, value=24, key="hours_slider")
                
                # Get last N hours
                forecast_data = fetch_api_data("forecast", {"hours": hours_slider})
                
                if forecast_data and forecast_data.get('data'):
                    data_df = pd.DataFrame(forecast_data['data'])
                    chart = create_load_chart(data_df, hours=hours_slider, active_attacks=active_attacks)
                    if chart:
                        st.plotly_chart(chart, width='stretch')
                else:
                    st.info("No data available.")
        else:
            st.info("No data available. Start simulation to see chart.")
    else:
        st.info("No data available. Start simulation to see chart.")
    
    # Legend for chart colors
    with st.expander("📖 Chart Legend"):
        st.markdown("""
        **Lines:**
        - **Actual Load** (Blue solid): Real measured power grid load
        - **Forecast** (Green dashed): LSTM model prediction
        - **Benchmark** (Orange dotted): Normal pattern from K-means clustering
        
        **Shaded Regions:**
        - **Purple regions**: Injected attacks (darker = stronger attack, lighter = weaker attack)
        - **Red regions**: Detected anomalies (system detected potential attacks)
        
        **Attack Color Intensity:**
        - **Light Purple** (0.1-0.5 magnitude): Weak attacks
        - **Medium Purple** (0.5-1.0 magnitude): Medium attacks
        - **Dark Purple** (1.0+ magnitude): Strong attacks
        """)
    
    # Alerts panel - Beautiful UI
    st.header("🚨 Alerts & Detections")
    
    alerts_data = fetch_api_data("alerts", {"limit": 50})
    if alerts_data and alerts_data.get('alerts'):
        alerts = alerts_data['alerts']
        
        # Filter options in a nice layout
        filter_col1, filter_col2, filter_col3 = st.columns([2, 2, 1])
        with filter_col1:
            severity_filter = st.multiselect(
                "🔍 Filter by Severity", 
                ["LOW", "MEDIUM", "HIGH", "EMERGENCY"],
                default=["LOW", "MEDIUM", "HIGH", "EMERGENCY"],
                help="Select which alert severities to display"
            )
        with filter_col2:
            show_acknowledged = st.checkbox("✅ Show Acknowledged Alerts", value=False, help="Toggle to show/hide acknowledged alerts")
        with filter_col3:
            st.metric("Total Alerts", len(alerts))
        
        # Display alerts
        filtered_alerts = alerts
        if severity_filter:
            filtered_alerts = [a for a in filtered_alerts if a.get('severity') in severity_filter]
        if not show_acknowledged:
            filtered_alerts = [a for a in filtered_alerts if not a.get('acknowledged', False)]
        
        if filtered_alerts:
            st.caption(f"Showing {len(filtered_alerts)} alert(s)")
            
            # Display alerts in cards
            for idx, alert in enumerate(filtered_alerts[:20]):  # Show first 20
                severity = alert.get('severity', 'LOW')
                
                # Severity colors
                severity_colors = {
                    'LOW': '#51cf66',
                    'MEDIUM': '#ffa94d',
                    'HIGH': '#ff6b6b',
                    'EMERGENCY': '#c92a2a'
                }
                severity_icons = {
                    'LOW': '🟢',
                    'MEDIUM': '🟡',
                    'HIGH': '🟠',
                    'EMERGENCY': '🔴'
                }
                
                color = severity_colors.get(severity, '#51cf66')
                icon = severity_icons.get(severity, '🟢')
                
                # Create alert card
                with st.container():
                    st.markdown(f"""
                    <div style="
                        background-color: {color}15;
                        border-left: 4px solid {color};
                        padding: 1rem;
                        margin: 0.5rem 0;
                        border-radius: 0.25rem;
                        box-shadow: 0 2px 4px rgba(0,0,0,0.1);
                    ">
                        <div style="display: flex; justify-content: space-between; align-items: center;">
                            <div>
                                <strong>{icon} {severity}</strong>
                                <br>
                                <small style="color: #666;">
                                    📅 {alert.get('start_time', 'N/A')} → {alert.get('end_time', 'N/A')}
                                </small>
                            </div>
                            <div style="text-align: right;">
                                <small><strong>Method:</strong> {alert.get('method', 'N/A')}</small>
                            </div>
                        </div>
                    </div>
                    """, unsafe_allow_html=True)
                    
                    # Acknowledge button
                    if not alert.get('acknowledged', False):
                        if st.button(f"✅ Acknowledge", key=f"ack_{idx}_{alert.get('start_time', '')}"):
                            st.success("Alert acknowledged (Note: API endpoint needed)")
                    
                    st.markdown("---")
        else:
            st.info("ℹ️ No alerts match the selected filters.")
    else:
        st.success("✅ No alerts detected - System operating normally!")
    
    # Performance metrics with tooltips
    st.header("📈 Performance Metrics")
    st.caption("Hover over each metric for detailed explanation")
    
    metrics_data = fetch_api_data("metrics")
    if metrics_data and 'metrics' in metrics_data:
        metrics = metrics_data['metrics']
        
        # Check if metrics are actually populated
        has_data = (
            metrics.get('hours_processed', 0) > 0 or
            metrics.get('throughput_hours_per_sec', 0) > 0 or
            metrics.get('latency', {}).get('count', 0) > 0
        )
        
        if not has_data and status_data and status_data.get('simulation', {}).get('is_running'):
            st.warning("⚠️ Performance metrics are being collected. They will appear after processing a few hours of data.")
        elif not has_data:
            st.info("ℹ️ Performance metrics will be available when simulation is running.")
        
        # Metrics with tooltips using expandable sections
        col1, col2, col3, col4 = st.columns(4)
        
        with col1:
            latency = metrics.get('latency', {})
            latency_mean = latency.get('mean')
            if latency_mean is not None and latency.get('count', 0) > 0:
                st.metric("⏱️ Mean Latency", f"{latency_mean:.3f}s")
            else:
                st.metric("⏱️ Mean Latency", "N/A")
            with st.expander("ℹ️ What is Mean Latency?"):
                st.caption("Average time taken to process one hour of data. Lower is better. Includes forecast generation, benchmark lookup, and anomaly detection.")
        
        with col2:
            throughput = metrics.get('throughput_hours_per_sec', 0)
            if throughput > 0:
                st.metric("⚡ Throughput", f"{throughput:.2f} hrs/sec")
            else:
                st.metric("⚡ Throughput", "0.00 hrs/sec")
            with st.expander("ℹ️ What is Throughput?"):
                st.caption("Number of hours of data processed per second. Higher is better. Shows how fast the system can analyze historical data.")
        
        with col3:
            memory = metrics.get('memory', {})
            memory_mb = memory.get('memory_mb', 0)
            if memory_mb > 0:
                st.metric("💾 Memory Usage", f"{memory_mb:.0f} MB")
            else:
                st.metric("💾 Memory Usage", "N/A")
            with st.expander("ℹ️ What is Memory Usage?"):
                st.caption("Current RAM consumption by the system. Includes data storage (30-day retention), models (LSTM, K-means), and processing buffers.")
        
        with col4:
            hours_processed = metrics.get('hours_processed', 0)
            st.metric("📊 Hours Processed", hours_processed)
            with st.expander("ℹ️ What is Hours Processed?"):
                st.caption("Total number of hours of data analyzed since simulation started. Used to track system progress and data coverage.")
        
        # Latency distribution chart with explanation
        if latency.get('count', 0) > 0:
            st.subheader("📈 Latency Distribution")
            with st.expander("ℹ️ Understanding Latency Percentiles"):
                st.markdown("""
                - **Mean**: Average processing time
                - **P50 (Median)**: 50% of requests complete faster than this
                - **P95**: 95% of requests complete faster than this (handles outliers)
                - **P99**: 99% of requests complete faster than this (worst-case scenarios)
                
                Lower values indicate better performance.
                """)
            latency_data = {
                'Mean': latency.get('mean', 0),
                'P50': latency.get('p50', 0),
                'P95': latency.get('p95', 0),
                'P99': latency.get('p99', 0)
            }
            st.bar_chart(latency_data)
    
    # Auto-refresh - only refresh if simulation is running
    auto_refresh_enabled = st.checkbox("🔄 Auto-refresh", value=False)
    
    if auto_refresh_enabled:
        # Check if simulation is running before refreshing
        if status_data and status_data.get('simulation', {}).get('is_running'):
            refresh_interval = st.selectbox("Refresh Interval", [5, 10, 15, 30], index=0, key="refresh_interval")
            st.caption(f"Auto-refreshing every {refresh_interval} seconds (simulation running)")
            time.sleep(refresh_interval)
            st.rerun()
        else:
            st.caption("⏸️ Auto-refresh paused (simulation stopped)")


if __name__ == "__main__":
    main()

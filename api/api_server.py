"""
API Server - REST API and WebSocket endpoints
"""
from flask import Flask, jsonify, request
from flask_cors import CORS
from flask_socketio import SocketIO
import sys
import os

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from api.data_store import DataStore
from api.simulation_controller import SimulationController
from api.websocket_handler import WebSocketHandler


# Create Flask app
app = Flask(__name__)
app.config['SECRET_KEY'] = 'power-grid-protection-secret-key'
CORS(app)  # Enable CORS for frontend

# Initialize SocketIO
socketio = SocketIO(app, cors_allowed_origins="*", async_mode='threading')

# Initialize components
data_store = DataStore()
simulation_controller = SimulationController(data_store)
websocket_handler = WebSocketHandler(socketio)

# Register WebSocket callback
def websocket_callback(update_data):
    """Callback to send updates via WebSocket."""
    websocket_handler.broadcast_update(update_data)

simulation_controller.register_callback(websocket_callback)


# REST API Endpoints

@app.route('/api/status', methods=['GET'])
def get_status():
    """Get simulation status."""
    status = simulation_controller.get_status()
    store_stats = data_store.get_statistics()
    
    return jsonify({
        'status': 'success',
        'simulation': status,
        'data_store': store_stats
    })


@app.route('/api/forecast', methods=['GET'])
def get_forecast():
    """Get forecast data."""
    hours = request.args.get('hours', default=24, type=int)
    data = data_store.get_recent_data(hours=hours)
    
    if len(data) == 0:
        return jsonify({
            'status': 'success',
            'data': [],
            'message': 'No data available'
        })
    
    # Convert DataFrame to dict and ensure timestamps are strings
    records = data.to_dict('records')
    
    # Convert timestamps and numpy types to native Python types
    import numpy as np
    import pandas as pd
    
    for record in records:
        if 'timestamp' in record:
            if isinstance(record['timestamp'], (pd.Timestamp, pd.DatetimeIndex)):
                record['timestamp'] = str(record['timestamp'])
            elif pd.notna(record['timestamp']):
                record['timestamp'] = str(record['timestamp'])
        # Convert numpy types
        for key, value in record.items():
            if isinstance(value, (np.integer, np.int64, np.int32)):
                record[key] = int(value)
            elif isinstance(value, (np.floating, np.float64, np.float32)):
                record[key] = float(value)
    
    return jsonify({
        'status': 'success',
        'data': records
    })


@app.route('/api/detections', methods=['GET'])
def get_detections():
    """Get detection results."""
    limit = request.args.get('limit', default=100, type=int)
    detections = data_store.get_detections(limit=limit)
    
    return jsonify({
        'status': 'success',
        'detections': detections,
        'count': len(detections)
    })


@app.route('/api/alerts', methods=['GET'])
def get_alerts():
    """Get alerts."""
    limit = request.args.get('limit', default=100, type=int)
    severity = request.args.get('severity', default=None, type=str)
    acknowledged = request.args.get('acknowledged', default=None, type=str)
    
    alerts = data_store.get_alerts(limit=limit)
    
    # Convert numpy/pandas types to native Python types for JSON serialization
    def convert_to_native(obj):
        """Convert numpy/pandas types to native Python types."""
        import numpy as np
        import pandas as pd
        
        if isinstance(obj, (np.integer, np.int64, np.int32)):
            return int(obj)
        elif isinstance(obj, (np.floating, np.float64, np.float32)):
            return float(obj)
        elif isinstance(obj, (pd.Timestamp, pd.DatetimeIndex)):
            return str(obj)
        elif isinstance(obj, dict):
            return {k: convert_to_native(v) for k, v in obj.items()}
        elif isinstance(obj, (list, tuple)):
            return [convert_to_native(item) for item in obj]
        else:
            return obj
    
    # Convert alerts to JSON-serializable format
    alerts = [convert_to_native(a) for a in alerts]
    
    # Filter by severity
    if severity:
        alerts = [a for a in alerts if a.get('severity') == severity]
    
    # Filter by acknowledgment status
    if acknowledged is not None:
        acknowledged_bool = acknowledged.lower() == 'true'
        alerts = [a for a in alerts if a.get('acknowledged', False) == acknowledged_bool]
    
    return jsonify({
        'status': 'success',
        'alerts': alerts,
        'count': len(alerts)
    })


@app.route('/api/metrics', methods=['GET'])
def get_metrics():
    """Get performance metrics."""
    if simulation_controller.monitor:
        metrics = simulation_controller.monitor.get_all_metrics()
    else:
        metrics = {}
    
    # Add detection latency
    if simulation_controller.detector:
        latency_stats = simulation_controller.detector.get_detection_latency_stats()
        metrics['detection_latency'] = latency_stats
    
    # Convert numpy/pandas types to native Python types
    def convert_to_native(obj):
        """Convert numpy/pandas types to native Python types."""
        import numpy as np
        import pandas as pd
        
        if isinstance(obj, (np.integer, np.int64, np.int32)):
            return int(obj)
        elif isinstance(obj, (np.floating, np.float64, np.float32)):
            return float(obj)
        elif isinstance(obj, (pd.Timestamp, pd.DatetimeIndex)):
            return str(obj)
        elif isinstance(obj, dict):
            return {k: convert_to_native(v) for k, v in obj.items()}
        elif isinstance(obj, (list, tuple)):
            return [convert_to_native(item) for item in obj]
        else:
            return obj
    
    metrics = convert_to_native(metrics)
    
    return jsonify({
        'status': 'success',
        'metrics': metrics
    })


@app.route('/api/simulation/start', methods=['POST'])
def start_simulation():
    """Start simulation."""
    data = request.get_json() or {}
    speed = data.get('speed', None)
    
    result = simulation_controller.start(speed=speed)
    return jsonify(result)


@app.route('/api/simulation/stop', methods=['POST'])
def stop_simulation():
    """Stop simulation."""
    result = simulation_controller.stop()
    return jsonify(result)


@app.route('/api/simulation/speed', methods=['POST'])
def set_speed():
    """Set simulation speed."""
    data = request.get_json() or {}
    speed = data.get('speed', 1.0)
    
    result = simulation_controller.set_speed(speed)
    return jsonify(result)


@app.route('/api/simulation/inject_attack', methods=['POST'])
def inject_attack():
    """Inject attack into simulation."""
    data = request.get_json() or {}
    
    attack_type = data.get('type', 'PULSE')
    start_hour = data.get('start_hour', 0)
    duration = data.get('duration', 1)
    magnitude = data.get('magnitude', 0.1)
    
    result = simulation_controller.inject_attack(
        attack_type=attack_type,
        start_hour=start_hour,
        duration=duration,
        magnitude=magnitude,
        **data.get('parameters', {})
    )
    
    return jsonify(result)


@app.route('/api/simulation/active_attacks', methods=['GET'])
def get_active_attacks():
    """Get list of active attacks."""
    if simulation_controller.simulator and simulation_controller.is_running:
        current_hour = simulation_controller.simulator.current_index - 1
        active_attacks = []
        
        for attack in simulation_controller.active_attacks:
            attack_start = attack['start_hour']
            attack_end = attack_start + attack['duration']
            
            # Calculate actual timestamps if we have data
            if simulation_controller.simulator.data is not None:
                try:
                    start_idx = current_hour + attack_start
                    end_idx = current_hour + attack_end
                    
                    if start_idx < len(simulation_controller.simulator.data.index):
                        start_time = simulation_controller.simulator.data.index[start_idx]
                        end_time = simulation_controller.simulator.data.index[min(end_idx, len(simulation_controller.simulator.data.index) - 1)]
                        
                        active_attacks.append({
                            'type': attack['type'],
                            'start_hour': attack_start,
                            'duration': attack['duration'],
                            'magnitude': attack['magnitude'],
                            'start_time': str(start_time),
                            'end_time': str(end_time),
                            'injected_at': str(attack.get('injected_at', ''))
                        })
                except:
                    pass
        
        return jsonify({
            'status': 'success',
            'active_attacks': active_attacks,
            'count': len(active_attacks)
        })
    else:
        return jsonify({
            'status': 'success',
            'active_attacks': [],
            'count': 0
        })


@app.route('/api/simulation/reset', methods=['POST'])
def reset_simulation():
    """Reset simulation - clear all data, attacks, and state."""
    try:
        simulation_controller.reset()
        return jsonify({
            'status': 'success',
            'message': 'Simulation reset successfully'
        })
    except Exception as e:
        return jsonify({
            'status': 'error',
            'message': str(e)
        }), 500


@app.route('/api/simulation/injected_attacks', methods=['GET'])
def get_injected_attacks():
    """Get list of all injected attacks (including completed ones) with timestamps."""
    all_attacks = []
    
    # Try to get timestamps from simulator data first
    simulator_data = None
    if simulation_controller.simulator and simulation_controller.simulator.data is not None:
        simulator_data = simulation_controller.simulator.data
    
    # Fallback: try to get timestamps from data_store
    data_store_timestamps = list(data_store.timestamps) if len(data_store.timestamps) > 0 else None
    
    for attack in simulation_controller.all_injected_attacks:
        try:
            # Calculate when attack actually occurred
            injection_hour = attack.get('injection_hour', 0)
            attack_start_hour = injection_hour + attack['start_hour']
            attack_end_hour = attack_start_hour + attack['duration']
            
            start_time = None
            end_time = None
            
            # Try simulator data first
            if simulator_data is not None:
                if 0 <= attack_start_hour < len(simulator_data.index):
                    start_time = simulator_data.index[attack_start_hour]
                    end_idx = min(attack_end_hour, len(simulator_data.index) - 1)
                    if end_idx >= attack_start_hour:
                        end_time = simulator_data.index[end_idx]
            
            # Fallback to data_store timestamps
            if start_time is None and data_store_timestamps is not None:
                if 0 <= attack_start_hour < len(data_store_timestamps):
                    start_time = data_store_timestamps[attack_start_hour]
                    end_idx = min(attack_end_hour, len(data_store_timestamps) - 1)
                    if end_idx >= attack_start_hour:
                        end_time = data_store_timestamps[end_idx]
            
            # If we have valid timestamps, add the attack
            if start_time is not None and end_time is not None:
                all_attacks.append({
                    'type': attack['type'],
                    'start_hour': attack['start_hour'],
                    'duration': attack['duration'],
                    'magnitude': attack['magnitude'],
                    'start_time': str(start_time),
                    'end_time': str(end_time),
                    'injected_at': str(attack.get('injected_at', ''))
                })
        except Exception as e:
            print(f"Error processing attack: {e}")
            continue
    
    return jsonify({
        'status': 'success',
        'injected_attacks': all_attacks,
        'count': len(all_attacks)
    })


# WebSocket Events are handled by WebSocketHandler
# The handlers are set up automatically when WebSocketHandler is initialized


if __name__ == '__main__':
    print("Starting API server...")
    print("API endpoints available at http://localhost:5000/api/")
    print("WebSocket available at ws://localhost:5000")
    
    socketio.run(app, host='0.0.0.0', port=5000, debug=True)


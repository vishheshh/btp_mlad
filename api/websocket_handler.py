"""
WebSocket Handler - Real-time updates via WebSocket
"""
from flask_socketio import SocketIO, emit
from flask import request
from typing import List, Callable
import json


class WebSocketHandler:
    """
    Handles WebSocket connections and real-time updates.
    """

    def __init__(self, socketio: SocketIO):
        """
        Initialize WebSocket handler.

        Args:
            socketio: Flask-SocketIO instance
        """
        self.socketio = socketio
        self.connected_clients = set()
        self.setup_handlers()

    def setup_handlers(self):
        """Setup WebSocket event handlers."""
        
        @self.socketio.on('connect')
        def handle_connect():
            """Handle client connection."""
            self.connected_clients.add(request.sid)
            print(f"Client connected: {request.sid}")
            emit('connected', {'status': 'success', 'message': 'Connected to simulation stream'})

        @self.socketio.on('disconnect')
        def handle_disconnect():
            """Handle client disconnection."""
            if request.sid in self.connected_clients:
                self.connected_clients.remove(request.sid)
            print(f"Client disconnected: {request.sid}")

        @self.socketio.on('subscribe')
        def handle_subscribe(data):
            """Handle subscription request."""
            # Client can subscribe to specific channels
            emit('subscribed', {'status': 'success', 'channels': data.get('channels', [])})

    def broadcast_update(self, update_data: dict):
        """
        Broadcast update to all connected clients.

        Args:
            update_data: Dictionary with update data
        """
        # Convert numpy/pandas types to native Python types
        import numpy as np
        import pandas as pd
        
        def convert_to_native(obj):
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
        
        update_data = convert_to_native(update_data)
        
        # Emit to all connected clients (broadcast by default when no room specified)
        self.socketio.emit('update', update_data)

    def send_alert(self, alert_data: dict):
        """
        Send alert to all connected clients.

        Args:
            alert_data: Alert dictionary
        """
        # Convert numpy/pandas types to native Python types
        import numpy as np
        import pandas as pd
        
        def convert_to_native(obj):
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
        
        alert_data = convert_to_native(alert_data)
        
        # Emit to all connected clients (broadcast by default when no room specified)
        self.socketio.emit('alert', alert_data)

    def get_connected_count(self) -> int:
        """Get number of connected clients."""
        return len(self.connected_clients)


"""
WebSocket Client - Real-time data streaming for Streamlit dashboard
Uses python-socketio client to receive live updates from the API server.
"""
import socketio
import threading
import queue
import time
from typing import Optional, Callable, Dict, Any
from dataclasses import dataclass, field
from collections import deque


@dataclass
class LiveMetrics:
    """Container for live simulation metrics."""
    timestamp: str = ""
    forecast: float = 0.0
    actual: float = 0.0
    benchmark: float = 0.0
    scaling_ratio: float = 1.0
    hours_processed: int = 0
    alerts_count: int = 0
    is_connected: bool = False
    last_update_time: float = 0.0
    
    # Rolling window for mini-chart (last 50 points)
    recent_actuals: deque = field(default_factory=lambda: deque(maxlen=50))
    recent_forecasts: deque = field(default_factory=lambda: deque(maxlen=50))
    recent_timestamps: deque = field(default_factory=lambda: deque(maxlen=50))


class StreamlitWebSocketClient:
    """
    WebSocket client optimized for Streamlit's execution model.
    
    Streamlit reruns the entire script on each interaction, so we need:
    1. A singleton pattern to maintain connection across reruns
    2. Thread-safe data access
    3. A message queue for buffering updates
    """
    
    _instance: Optional['StreamlitWebSocketClient'] = None
    _lock = threading.Lock()
    
    def __new__(cls, server_url: str = "http://localhost:5000"):
        """Singleton pattern - only one client instance."""
        with cls._lock:
            if cls._instance is None:
                cls._instance = super().__new__(cls)
                cls._instance._initialized = False
            return cls._instance
    
    def __init__(self, server_url: str = "http://localhost:5000"):
        """Initialize the WebSocket client."""
        if self._initialized:
            return
            
        self.server_url = server_url
        self.sio = socketio.Client(
            reconnection=True,
            reconnection_attempts=5,
            reconnection_delay=1,
            reconnection_delay_max=5,
            logger=False,
            engineio_logger=False
        )
        
        # Thread-safe metrics storage
        self.metrics = LiveMetrics()
        self._metrics_lock = threading.Lock()
        
        # Message queue for alerts
        self.alert_queue: queue.Queue = queue.Queue(maxsize=100)
        
        # Connection state
        self._connected = False
        self._connection_thread: Optional[threading.Thread] = None
        
        # Setup event handlers
        self._setup_handlers()
        
        self._initialized = True
    
    def _setup_handlers(self):
        """Setup SocketIO event handlers."""
        
        @self.sio.event
        def connect():
            """Handle connection."""
            print("[WebSocket] Connected to server")
            self._connected = True
            with self._metrics_lock:
                self.metrics.is_connected = True
        
        @self.sio.event
        def disconnect():
            """Handle disconnection."""
            print("[WebSocket] Disconnected from server")
            self._connected = False
            with self._metrics_lock:
                self.metrics.is_connected = False
        
        @self.sio.event
        def connected(data):
            """Handle server confirmation."""
            print(f"[WebSocket] Server says: {data.get('message', 'Connected')}")
        
        @self.sio.on('update')
        def on_update(data: Dict[str, Any]):
            """Handle real-time data updates."""
            with self._metrics_lock:
                self.metrics.timestamp = data.get('timestamp', '')
                self.metrics.forecast = float(data.get('forecast', 0))
                self.metrics.actual = float(data.get('actual', 0))
                self.metrics.benchmark = float(data.get('benchmark', 0))
                self.metrics.scaling_ratio = float(data.get('scaling_ratio', 1.0))
                self.metrics.hours_processed += 1
                self.metrics.last_update_time = time.time()
                
                # Add to rolling window for mini-chart
                self.metrics.recent_actuals.append(self.metrics.actual)
                self.metrics.recent_forecasts.append(self.metrics.forecast)
                self.metrics.recent_timestamps.append(self.metrics.timestamp)
                
                # Count alerts
                alerts = data.get('alerts', [])
                if alerts:
                    self.metrics.alerts_count += len(alerts)
                    for alert in alerts:
                        try:
                            self.alert_queue.put_nowait(alert)
                        except queue.Full:
                            # Remove oldest alert if queue is full
                            try:
                                self.alert_queue.get_nowait()
                                self.alert_queue.put_nowait(alert)
                            except:
                                pass
        
        @self.sio.on('alert')
        def on_alert(data: Dict[str, Any]):
            """Handle alert notifications."""
            try:
                self.alert_queue.put_nowait(data)
                with self._metrics_lock:
                    self.metrics.alerts_count += 1
            except queue.Full:
                pass
    
    def connect(self) -> bool:
        """
        Connect to the WebSocket server.
        
        Returns:
            True if connected successfully, False otherwise
        """
        if self._connected:
            return True
        
        try:
            self.sio.connect(self.server_url, wait_timeout=5)
            return True
        except Exception as e:
            print(f"[WebSocket] Connection failed: {e}")
            return False
    
    def connect_async(self):
        """Connect in a background thread (non-blocking)."""
        if self._connected:
            return
        
        if self._connection_thread is not None and self._connection_thread.is_alive():
            return
        
        def _connect():
            try:
                self.sio.connect(self.server_url, wait_timeout=10)
            except Exception as e:
                print(f"[WebSocket] Async connection failed: {e}")
        
        self._connection_thread = threading.Thread(target=_connect, daemon=True)
        self._connection_thread.start()
    
    def disconnect(self):
        """Disconnect from the server."""
        if self._connected:
            try:
                self.sio.disconnect()
            except:
                pass
            self._connected = False
    
    def get_metrics(self) -> LiveMetrics:
        """
        Get current metrics (thread-safe copy).
        
        Returns:
            Copy of current LiveMetrics
        """
        with self._metrics_lock:
            # Return a copy to avoid race conditions
            return LiveMetrics(
                timestamp=self.metrics.timestamp,
                forecast=self.metrics.forecast,
                actual=self.metrics.actual,
                benchmark=self.metrics.benchmark,
                scaling_ratio=self.metrics.scaling_ratio,
                hours_processed=self.metrics.hours_processed,
                alerts_count=self.metrics.alerts_count,
                is_connected=self.metrics.is_connected,
                last_update_time=self.metrics.last_update_time,
                recent_actuals=deque(self.metrics.recent_actuals, maxlen=50),
                recent_forecasts=deque(self.metrics.recent_forecasts, maxlen=50),
                recent_timestamps=deque(self.metrics.recent_timestamps, maxlen=50)
            )
    
    def get_recent_alerts(self, max_count: int = 10) -> list:
        """
        Get recent alerts from the queue.
        
        Args:
            max_count: Maximum number of alerts to return
            
        Returns:
            List of alert dictionaries
        """
        alerts = []
        while len(alerts) < max_count:
            try:
                alert = self.alert_queue.get_nowait()
                alerts.append(alert)
            except queue.Empty:
                break
        return alerts
    
    def reset_metrics(self):
        """Reset metrics (call when simulation resets)."""
        with self._metrics_lock:
            self.metrics = LiveMetrics()
            self.metrics.is_connected = self._connected
        
        # Clear alert queue
        while not self.alert_queue.empty():
            try:
                self.alert_queue.get_nowait()
            except:
                break
    
    @property
    def is_connected(self) -> bool:
        """Check if connected to server."""
        return self._connected
    
    @classmethod
    def get_instance(cls, server_url: str = "http://localhost:5000") -> 'StreamlitWebSocketClient':
        """Get or create the singleton instance."""
        return cls(server_url)
    
    @classmethod
    def reset_instance(cls):
        """Reset the singleton instance (for testing)."""
        with cls._lock:
            if cls._instance is not None:
                cls._instance.disconnect()
                cls._instance = None


# Convenience function for Streamlit
def get_websocket_client(server_url: str = "http://localhost:5000") -> StreamlitWebSocketClient:
    """
    Get the WebSocket client singleton.
    
    Usage in Streamlit:
        from websocket_client import get_websocket_client
        
        ws_client = get_websocket_client()
        ws_client.connect_async()
        metrics = ws_client.get_metrics()
    """
    return StreamlitWebSocketClient.get_instance(server_url)


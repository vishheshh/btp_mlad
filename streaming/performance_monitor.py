"""
Performance Monitor - Track system performance metrics
"""
import time
import psutil
import os
from datetime import datetime
from typing import Dict, List
from collections import deque
import sys

# Add parent directory to path for imports
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from streaming.streaming_config import *


class PerformanceMonitor:
    """
    Monitors system performance metrics.
    """

    def __init__(self):
        """Initialize the performance monitor."""
        self.process = psutil.Process(os.getpid())
        self.start_time = None
        self.hours_processed = 0
        self.latency_history = deque(maxlen=1000)  # Last 1000 measurements
        self.memory_history = deque(maxlen=100)  # Last 100 measurements

    def start(self):
        """Start monitoring."""
        self.start_time = time.time()
        self.hours_processed = 0
        print("Performance monitoring started")

    def record_processing(self, processing_time: float):
        """
        Record processing time for one hour.

        Args:
            processing_time: Time taken to process one hour (seconds)
        """
        self.latency_history.append(processing_time)
        self.hours_processed += 1

        # Record memory usage periodically
        if self.hours_processed % 10 == 0:
            memory_mb = self.process.memory_info().rss / 1024 / 1024
            self.memory_history.append({
                'timestamp': datetime.now(),
                'memory_mb': memory_mb,
                'hours_processed': self.hours_processed
            })

    def get_latency_stats(self) -> Dict:
        """Get latency statistics."""
        if len(self.latency_history) == 0:
            return {
                'mean': None,
                'p50': None,
                'p95': None,
                'p99': None,
                'max': None,
                'count': 0
            }

        latencies = list(self.latency_history)
        latencies_sorted = sorted(latencies)

        return {
            'mean': sum(latencies) / len(latencies),
            'p50': latencies_sorted[len(latencies_sorted) // 2],
            'p95': latencies_sorted[int(len(latencies_sorted) * 0.95)],
            'p99': latencies_sorted[int(len(latencies_sorted) * 0.99)],
            'max': max(latencies),
            'count': len(latencies)
        }

    def get_throughput(self) -> float:
        """Get throughput (hours per second)."""
        if self.start_time is None:
            return 0.0

        elapsed = time.time() - self.start_time
        if elapsed == 0:
            return 0.0

        return self.hours_processed / elapsed

    def reset(self):
        """Reset all performance metrics."""
        self.start_time = None
        self.hours_processed = 0
        self.latency_history.clear()
        self.memory_history.clear()
    
    def get_memory_usage(self) -> Dict:
        """Get current memory usage."""
        memory_info = self.process.memory_info()
        memory_mb = memory_info.rss / 1024 / 1024
        memory_gb = memory_mb / 1024

        return {
            'memory_mb': memory_mb,
            'memory_gb': memory_gb,
            'memory_percent': self.process.memory_percent(),
            'within_limit': memory_gb < MAX_MEMORY_GB
        }

    def get_all_metrics(self) -> Dict:
        """Get all performance metrics."""
        return {
            'latency': self.get_latency_stats(),
            'throughput_hours_per_sec': self.get_throughput(),
            'memory': self.get_memory_usage(),
            'hours_processed': self.hours_processed,
            'uptime_seconds': time.time() - self.start_time if self.start_time else 0,
            'targets': {
                'target_latency_seconds': TARGET_LATENCY_SECONDS,
                'target_throughput': TARGET_THROUGHPUT_HOURS_PER_SEC,
                'max_memory_gb': MAX_MEMORY_GB
            }
        }


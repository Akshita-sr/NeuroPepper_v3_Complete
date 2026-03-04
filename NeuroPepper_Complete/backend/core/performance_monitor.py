# backend/core/performance_monitor.py
# ============================================
# System Performance Monitor
# ============================================

import threading
import time
import psutil
from typing import Dict, Any, Optional


class PerformanceMonitor:
    """
    Background performance monitoring for the NeuroPepper system.
    """
    
    def __init__(self):
        self._running = False
        self._thread: Optional[threading.Thread] = None
        self._latest_stats: Dict[str, Any] = {}
    
    def start(self):
        """Start the monitoring thread."""
        if self._running:
            return
        
        self._running = True
        self._thread = threading.Thread(target=self._monitor_loop, daemon=True)
        self._thread.start()
        print("📊 Performance monitor started")
    
    def stop(self):
        """Stop the monitoring thread."""
        self._running = False
        if self._thread:
            self._thread.join(timeout=1)
        print("📊 Performance monitor stopped")
    
    def _monitor_loop(self):
        """Background monitoring loop."""
        while self._running:
            try:
                self._latest_stats = {
                    "cpu_percent": psutil.cpu_percent(),
                    "memory_percent": psutil.virtual_memory().percent,
                    "timestamp": time.time()
                }
            except Exception:
                pass
            time.sleep(2)
    
    def get_stats(self) -> Dict[str, Any]:
        """Get latest performance statistics."""
        return self._latest_stats.copy()


# Singleton instance
performance_monitor = PerformanceMonitor()

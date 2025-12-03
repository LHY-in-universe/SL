"""
Log Manager for centralized log collection and filtering

Provides:
- Thread-safe log collection
- Level-based filtering (DEBUG, INFO, WARNING, ERROR)
- Log export functionality
- Configurable buffer size
"""

import threading
from collections import deque
from typing import List, Optional, Dict
from datetime import datetime
from enum import Enum


class LogLevel(Enum):
    """Log level enumeration"""
    DEBUG = "DEBUG"
    INFO = "INFO"
    WARNING = "WARNING"
    ERROR = "ERROR"


class LogManager:
    """
    Thread-safe log manager for collecting and managing logs

    Features:
    - Circular buffer with configurable size
    - Level-based filtering
    - Export to file
    - Thread-safe operations

    Example:
        >>> manager = LogManager(max_logs=1000)
        >>> manager.add_log(LogLevel.INFO, "Server started")
        >>> logs = manager.get_logs(level_filter=LogLevel.ERROR)
        >>> manager.export_logs("/tmp/logs.txt")
    """

    def __init__(self, max_logs: int = 1000):
        """
        Initialize the log manager

        Args:
            max_logs: Maximum number of logs to retain in memory
        """
        self.max_logs = max_logs
        self._logs = deque(maxlen=max_logs)
        self._lock = threading.Lock()

    def add_log(
        self,
        level: LogLevel,
        message: str,
        source: str = "server",
        **kwargs
    ):
        """
        Add a log entry

        Args:
            level: Log level (DEBUG, INFO, WARNING, ERROR)
            message: Log message
            source: Source of the log (default: "server")
            **kwargs: Additional context to include in the log
        """
        with self._lock:
            log_entry = {
                'timestamp': datetime.now(),
                'level': level.value if isinstance(level, LogLevel) else level,
                'message': message,
                'source': source,
                **kwargs
            }
            self._logs.append(log_entry)

    def get_logs(
        self,
        level_filter: Optional[str] = None,
        limit: Optional[int] = None,
        source_filter: Optional[str] = None
    ) -> List[Dict]:
        """
        Get logs with optional filtering

        Args:
            level_filter: Filter by log level (e.g., "ERROR", "INFO")
                         Use "ALL" or None to get all logs
            limit: Maximum number of logs to return (most recent first)
            source_filter: Filter by source

        Returns:
            List of log dictionaries sorted by timestamp (most recent first)
        """
        with self._lock:
            logs = list(self._logs)

            # Filter by level
            if level_filter and level_filter != "ALL":
                logs = [log for log in logs if log['level'] == level_filter]

            # Filter by source
            if source_filter:
                logs = [log for log in logs if log['source'] == source_filter]

            # Sort by timestamp (most recent first)
            logs.sort(key=lambda x: x['timestamp'], reverse=True)

            # Apply limit
            if limit:
                logs = logs[:limit]

            return logs

    def get_log_count(self, level_filter: Optional[str] = None) -> int:
        """
        Get count of logs

        Args:
            level_filter: Optional level filter

        Returns:
            Number of logs matching the filter
        """
        with self._lock:
            if not level_filter or level_filter == "ALL":
                return len(self._logs)
            return sum(1 for log in self._logs if log['level'] == level_filter)

    def format_logs(self, logs: List[Dict]) -> str:
        """
        Format logs as a human-readable string

        Args:
            logs: List of log dictionaries

        Returns:
            Formatted log string
        """
        if not logs:
            return "No logs available"

        formatted_lines = []
        for log in logs:
            timestamp_str = log['timestamp'].strftime("%Y-%m-%d %H:%M:%S.%f")[:-3]
            level = log['level']
            source = log['source']
            message = log['message']

            # Add level emoji/icon
            level_icon = {
                'DEBUG': '🔍',
                'INFO': 'ℹ️',
                'WARNING': '⚠️',
                'ERROR': '❌'
            }.get(level, '📝')

            line = f"[{timestamp_str}] {level_icon} {level:8} [{source:10}] {message}"

            # Add any additional context
            for key, value in log.items():
                if key not in ['timestamp', 'level', 'source', 'message']:
                    line += f"\n    {key}: {value}"

            formatted_lines.append(line)

        return "\n".join(formatted_lines)

    def export_logs(
        self,
        filepath: str,
        level_filter: Optional[str] = None,
        source_filter: Optional[str] = None
    ) -> int:
        """
        Export logs to a file

        Args:
            filepath: Path to the output file
            level_filter: Optional level filter
            source_filter: Optional source filter

        Returns:
            Number of logs exported
        """
        logs = self.get_logs(
            level_filter=level_filter,
            source_filter=source_filter
        )

        formatted_logs = self.format_logs(logs)

        with open(filepath, 'w', encoding='utf-8') as f:
            f.write(formatted_logs)

        return len(logs)

    def clear(self):
        """Clear all logs"""
        with self._lock:
            self._logs.clear()

    def get_summary(self) -> Dict[str, int]:
        """
        Get a summary of log counts by level

        Returns:
            Dictionary with counts for each log level
        """
        with self._lock:
            summary = {
                'DEBUG': 0,
                'INFO': 0,
                'WARNING': 0,
                'ERROR': 0,
                'TOTAL': len(self._logs)
            }

            for log in self._logs:
                level = log['level']
                if level in summary:
                    summary[level] += 1

            return summary

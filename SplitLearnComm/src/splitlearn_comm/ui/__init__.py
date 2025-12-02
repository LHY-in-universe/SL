"""
Gradio UI components for splitlearn-comm

This module provides built-in UI components for both client-side interactive
text generation and server-side monitoring dashboards.
"""

from .client_ui import ClientUI
from .server_ui import ServerMonitoringUI

__all__ = [
    'ClientUI',
    'ServerMonitoringUI',
]

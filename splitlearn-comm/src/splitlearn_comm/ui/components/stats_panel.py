"""
Statistics panel component for displaying real-time metrics
"""

from typing import Dict, Any, Optional
from datetime import datetime


class StatsPanel:
    """
    Helper class for formatting and managing statistics displays
    """

    @staticmethod
    def format_generation_stats(
        tokens_generated: int,
        max_tokens: int,
        elapsed_time: float,
        status: str = "generating"
    ) -> str:
        """
        Format generation statistics for display

        Args:
            tokens_generated: Number of tokens generated so far
            max_tokens: Maximum tokens to generate
            elapsed_time: Time elapsed in seconds
            status: Current status ("generating", "completed", "stopped", "error")

        Returns:
            Formatted statistics string
        """
        avg_speed = tokens_generated / elapsed_time if elapsed_time > 0 else 0

        status_emoji = {
            "generating": "🔄",
            "completed": "✅",
            "stopped": "🛑",
            "error": "❌"
        }

        emoji = status_emoji.get(status, "⏱️")

        if status == "generating":
            return (
                f"{emoji} 生成统计\n\n"
                f"Tokens: {tokens_generated}/{max_tokens}\n"
                f"速度: {avg_speed:.2f} tokens/s\n"
                f"耗时: {elapsed_time:.2f}s"
            )
        elif status == "completed":
            return (
                f"{emoji} 生成完成\n\n"
                f"生成了 {tokens_generated} 个 tokens\n"
                f"平均速度: {avg_speed:.2f} tokens/s\n"
                f"总耗时: {elapsed_time:.2f}s"
            )
        elif status == "stopped":
            return (
                f"{emoji} 生成已停止\n\n"
                f"生成了 {tokens_generated} 个 tokens\n"
                f"平均速度: {avg_speed:.2f} tokens/s\n"
                f"耗时: {elapsed_time:.2f}s"
            )
        else:  # error
            return (
                f"{emoji} 生成出错\n\n"
                f"已生成 {tokens_generated} tokens\n"
                f"耗时: {elapsed_time:.2f}s"
            )

    @staticmethod
    def format_server_stats(
        total_requests: int,
        success_rate: float,
        avg_compute_time: float,
        uptime_seconds: float,
        active_connections: int = 0
    ) -> str:
        """
        Format server statistics for display

        Args:
            total_requests: Total number of requests processed
            success_rate: Success rate (0-1)
            avg_compute_time: Average compute time in milliseconds
            uptime_seconds: Server uptime in seconds
            active_connections: Number of active connections

        Returns:
            Formatted statistics string
        """
        uptime_hours = uptime_seconds / 3600
        success_percent = success_rate * 100

        return (
            f"📊 服务器统计\n\n"
            f"总请求数: {total_requests}\n"
            f"成功率: {success_percent:.1f}%\n"
            f"平均计算时间: {avg_compute_time:.2f}ms\n"
            f"运行时间: {uptime_hours:.1f}h\n"
            f"活跃连接: {active_connections}"
        )

    @staticmethod
    def format_client_stats(
        request_count: int,
        avg_network_time: float,
        avg_compute_time: float,
        avg_total_time: float
    ) -> str:
        """
        Format client statistics for display

        Args:
            request_count: Total number of requests sent
            avg_network_time: Average network time in milliseconds
            avg_compute_time: Average server compute time in milliseconds
            avg_total_time: Average total time in milliseconds

        Returns:
            Formatted statistics string
        """
        return (
            f"📈 客户端统计\n\n"
            f"总请求数: {request_count}\n"
            f"平均网络时间: {avg_network_time:.2f}ms\n"
            f"平均计算时间: {avg_compute_time:.2f}ms\n"
            f"平均总时间: {avg_total_time:.2f}ms"
        )

    @staticmethod
    def format_connection_status(
        connected: bool,
        server_address: str,
        server_version: Optional[str] = None,
        server_device: Optional[str] = None
    ) -> str:
        """
        Format connection status for display

        Args:
            connected: Whether connected to server
            server_address: Server address
            server_version: Server version (if available)
            server_device: Server device (if available)

        Returns:
            Formatted status string
        """
        if connected:
            status = f"✅ 已连接\n\n服务器: {server_address}"
            if server_version:
                status += f"\n版本: {server_version}"
            if server_device:
                status += f"\n设备: {server_device}"
            return status
        else:
            return f"❌ 未连接\n\n服务器: {server_address}\n\n请点击'初始化'按钮连接"

"""
Split Learning 系统监控面板
实时显示服务器状态和统计信息
"""
import sys
import os
import time
import datetime
from rich.console import Console
from rich.table import Table
from rich.live import Live
from rich.layout import Layout
from rich.panel import Panel

# 添加路径
current_dir = os.path.dirname(os.path.abspath(__file__))
project_root = os.path.dirname(current_dir)
sys.path.append(os.path.join(project_root, 'splitlearn-comm', 'src'))

from splitlearn_comm import GRPCComputeClient

def get_server_info(client):
    """获取服务器信息"""
    try:
        info = client.get_service_info()
        if info:
            return info
        return {"status": "Unknown", "error": "No info returned"}
    except Exception as e:
        return {"status": "Error", "error": str(e)}

def create_dashboard(client):
    """创建监控面板布局"""
    info = get_server_info(client)
    
    # 1. 服务器概览表
    server_table = Table(show_header=False, box=None)
    if "error" in info:
        server_table.add_row("[red]连接失败[/red]", info["error"])
    else:
        uptime = str(datetime.timedelta(seconds=int(info.get("uptime_seconds", 0))))
        server_table.add_row("服务名称", f"[bold cyan]{info.get('service_name', 'Unknown')}[/bold cyan]")
        server_table.add_row("版本", info.get("version", "Unknown"))
        server_table.add_row("运行设备", f"[yellow]{info.get('device', 'Unknown')}[/yellow]")
        server_table.add_row("运行时间", uptime)
        server_table.add_row("总处理请求", f"[green]{info.get('total_requests', 0)}[/green]")
        
        # 资源信息
        custom = info.get("custom_info", {})
        if custom:
            cpu = custom.get("cpu_percent", "N/A")
            mem_mb = custom.get("memory_mb", "N/A")
            mem_pct = custom.get("memory_percent", "N/A")
            server_table.add_row("CPU 使用率", f"{cpu}%")
            server_table.add_row("内存使用", f"{mem_mb} MB ({mem_pct}%)")

    # 2. 客户端连接测试表
    client_stats = client.get_statistics()
    perf_table = Table(title="客户端视角性能", show_header=True)
    perf_table.add_column("指标", style="cyan")
    perf_table.add_column("数值", style="magenta")
    
    perf_table.add_row("本地已发请求", str(client_stats.get("total_requests", 0)))
    perf_table.add_row("平均网络延迟", f"{client_stats.get('avg_network_time_ms', 0):.2f} ms")
    perf_table.add_row("平均计算耗时", f"{client_stats.get('avg_compute_time_ms', 0):.2f} ms")
    perf_table.add_row("平均总耗时", f"{client_stats.get('avg_total_time_ms', 0):.2f} ms")

    # 布局
    layout = Layout()
    layout.split_column(
        Layout(Panel(server_table, title="🌍 服务器状态 (Trunk)", border_style="blue")),
        Layout(Panel(perf_table, title="🚀 性能监控", border_style="green"))
    )
    return layout

def main():
    console = Console()
    console.print("[bold]正在连接监控系统...[/bold]")
    
    # 连接服务器
    client = GRPCComputeClient("127.0.0.1:50053", timeout=5.0)
    
    # 尝试连接
    if not client.connect():
        console.print("[bold red]❌ 无法连接到服务器！请确保 start_server.py 正在运行。[/bold red]")
        return

    console.print("[bold green]✅ 已连接！启动实时监控... (按 Ctrl+C 退出)[/bold green]")
    time.sleep(1)

    try:
        with Live(create_dashboard(client), refresh_per_second=1) as live:
            while True:
                # 模拟一次心跳请求来更新客户端统计 (可选，这里只获取 info)
                # client.health_check() 
                
                live.update(create_dashboard(client))
                time.sleep(1)
    except KeyboardInterrupt:
        console.print("\n[bold]监控已停止[/bold]")

if __name__ == "__main__":
    # 安装 rich 库 (如果还没有)
    try:
        import rich
    except ImportError:
        os.system(f"{sys.executable} -m pip install rich")
        import rich
        
    main()

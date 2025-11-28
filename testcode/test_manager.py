"""
测试 splitlearn-manager 的完整流程
包含：模型保存 -> 服务器启动 -> 客户端调用
使用 multiprocessing 避免信号处理问题
"""
import sys
import os
import time
import multiprocessing
import torch
import torch.nn as nn
import logging

# 添加路径
current_dir = os.path.dirname(os.path.abspath(__file__))
project_root = os.path.dirname(current_dir)
sys.path.append(os.path.join(project_root, 'splitlearn-manager', 'src'))
sys.path.append(os.path.join(project_root, 'splitlearn-comm', 'src'))

# 配置日志
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger("TestManager")

# 1. 定义一个简单的模型 (模拟 Trunk)
class SimpleTrunk(nn.Module):
    def __init__(self):
        super().__init__()
        self.linear = nn.Linear(10, 10)
        
    def forward(self, x):
        return self.linear(x)

def run_server(stop_event):
    """在独立进程中运行服务器"""
    try:
        # 重新配置日志（因为是在新进程中）
        logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
        logger = logging.getLogger("ServerProcess")
        
        from splitlearn_manager import ManagedServer, ServerConfig, ModelConfig
        
        # 配置服务器
        server_config = ServerConfig(
            host="localhost",
            port=50052,
            metrics_port=8001,
            log_level="INFO"
        )
        
        server = ManagedServer(config=server_config)
        
        # 保存临时模型
        model_path = os.path.join(current_dir, "temp_trunk.pt")
        model = SimpleTrunk()
        torch.save(model, model_path)
        logger.info(f"临时模型已保存到: {model_path}")
        
        # 加载模型
        model_config = ModelConfig(
            model_id="simple-trunk",
            model_path=model_path,
            model_type="pytorch",
            device="cpu"
        )
        server.load_model(model_config)
        logger.info("模型加载成功")
        
        # 启动服务器
        logger.info("正在启动服务器...")
        server.start()
        
        # 等待停止信号
        while not stop_event.is_set():
            time.sleep(0.5)
            
        logger.info("正在停止服务器...")
        server.stop()
        
        # 清理临时文件
        if os.path.exists(model_path):
            os.remove(model_path)
            
    except Exception as e:
        logger.error(f"服务器错误: {e}")
        import traceback
        traceback.print_exc()

def run_client():
    """运行客户端测试"""
    try:
        # 等待服务器启动
        time.sleep(5) # 给服务器更多启动时间
        
        from splitlearn_comm import GRPCComputeClient
        
        logger.info("正在连接服务器...")
        client = GRPCComputeClient("localhost:50052")
        
        if not client.connect():
            logger.error("无法连接到服务器")
            return False
            
        logger.info("已连接!")
        
        # 准备数据
        input_tensor = torch.randn(1, 10)
        logger.info(f"发送输入: {input_tensor.shape}")
        
        # 发送计算请求
        output = client.compute(input_tensor)
        
        logger.info(f"收到输出: {output.shape}")
        logger.info("测试成功!")
        
        client.close()
        return True
        
    except Exception as e:
        logger.error(f"客户端错误: {e}")
        import traceback
        traceback.print_exc()
        return False

if __name__ == "__main__":
    print("=== 开始 splitlearn-manager 测试 ===")
    
    # 设置启动方法为 spawn (macOS 默认，但显式设置更安全)
    multiprocessing.set_start_method('spawn', force=True)
    
    # 创建停止事件
    stop_event = multiprocessing.Event()
    
    # 启动服务器进程
    server_process = multiprocessing.Process(target=run_server, args=(stop_event,))
    server_process.start()
    
    # 运行客户端
    success = run_client()
    
    # 停止服务器
    stop_event.set()
    server_process.join()
    
    if success:
        print("\n✅ 测试通过")
    else:
        print("\n❌ 测试失败")

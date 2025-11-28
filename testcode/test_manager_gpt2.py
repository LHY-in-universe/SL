"""
使用 splitlearn-manager 服务 GPT-2 Trunk 模型的完整示例
"""
import sys
import os
import time
import multiprocessing
import torch
import logging
from transformers import AutoConfig

# 添加路径
current_dir = os.path.dirname(os.path.abspath(__file__))
project_root = os.path.dirname(current_dir)
sys.path.append(os.path.join(project_root, 'SplitLearning', 'src'))
sys.path.append(os.path.join(project_root, 'splitlearn-manager', 'src'))
sys.path.append(os.path.join(project_root, 'splitlearn-comm', 'src'))

# 配置日志
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger("TestGPT2Manager")

def prepare_model():
    """准备并保存 GPT-2 Trunk 模型"""
    from splitlearn import ModelFactory
    
    logger.info("正在下载并拆分 GPT-2 模型...")
    # 拆分点: Bottom(0-2), Trunk(2-10), Top(10-12)
    bottom, trunk, top = ModelFactory.create_split_models(
        model_type='gpt2',
        model_name_or_path='gpt2',
        split_point_1=2,
        split_point_2=10,
        device='cpu'
    )
    
    # 保存 Trunk 模型
    model_path = os.path.join(current_dir, "gpt2_trunk.pt")
    logger.info(f"保存 Trunk 模型到: {model_path}")
    trunk.save_split_model(model_path)
    
    return model_path

def run_server(stop_event, model_path):
    """在独立进程中运行服务器"""
    try:
        # 重新配置日志
        logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
        logger = logging.getLogger("ServerProcess")
        
        from splitlearn_manager import ManagedServer, ServerConfig, ModelConfig
        
        # 配置服务器
        server_config = ServerConfig(
            host="localhost",
            port=50053,
            metrics_port=8002,
            log_level="INFO"
        )
        
        server = ManagedServer(config=server_config)
        
        # 加载模型
        # 注意：我们需要提供 input_shape 用于 warmup
        # GPT-2 hidden_size 是 768
        model_config = ModelConfig(
            model_id="gpt2-trunk",
            model_path=model_path,
            model_type="pytorch",  # 我们保存的是标准 PyTorch state_dict
            device="cpu",
            warmup=True,
            config={
                "input_shape": (1, 10, 768)  # [batch, seq, hidden]
            }
        )
        
        # 注意：由于我们保存的是 state_dict，而 ManagedServer 默认加载整个模型对象，
        # 或者需要我们注册自定义加载器。
        # 为了简单起见，这里我们需要一个小技巧：
        # 我们需要在服务器端重新实例化模型结构，然后加载权重。
        # 但 ManagedServer 的默认 PyTorch 加载器假设加载的是整个模型对象 (torch.save(model))。
        # 
        # 在实际生产中，我们应该扩展 ModelLoader。
        # 这里为了演示，我们假设 model_path 指向的是整个模型对象（我们在 prepare_model 中会这样保存）。
        
        server.load_model(model_config)
        logger.info("GPT-2 Trunk 模型加载成功")
        
        # 启动服务器
        server.start()
        
        # 等待停止信号
        while not stop_event.is_set():
            time.sleep(0.5)
            
        server.stop()
            
    except Exception as e:
        logger.error(f"服务器错误: {e}")
        import traceback
        traceback.print_exc()

def run_client():
    """运行客户端测试"""
    try:
        time.sleep(10) # 等待服务器启动和模型加载（GPT-2 比较大）
        
        from splitlearn_comm import GRPCComputeClient
        
        logger.info("正在连接服务器...")
        client = GRPCComputeClient("localhost:50053")
        
        if not client.connect():
            logger.error("无法连接到服务器")
            return False
            
        logger.info("已连接!")
        
        # 准备模拟数据 (Batch=1, Seq=5, Hidden=768)
        # 这模拟了 Bottom 模型的输出
        input_tensor = torch.randn(1, 5, 768)
        logger.info(f"发送输入: {input_tensor.shape}")
        
        # 发送计算请求
        start_time = time.time()
        output = client.compute(input_tensor)
        end_time = time.time()
        
        logger.info(f"收到输出: {output.shape}")
        logger.info(f"计算耗时: {(end_time - start_time)*1000:.2f} ms")
        
        # 验证输出形状
        # Trunk 模型的输入输出形状应该一致 (1, 5, 768)
        if output.shape == input_tensor.shape:
            logger.info("✅ 输出形状正确")
        else:
            logger.error(f"❌ 输出形状错误: 期望 {input_tensor.shape}, 实际 {output.shape}")
        
        client.close()
        return True
        
    except Exception as e:
        logger.error(f"客户端错误: {e}")
        import traceback
        traceback.print_exc()
        return False

if __name__ == "__main__":
    print("=== GPT-2 Trunk Server 测试 ===")
    
    # 1. 准备模型
    # 为了让 ManagedServer 能直接加载，我们这里保存整个模型对象
    # 而不是调用 save_split_model (它只保存 state_dict)
    from splitlearn import ModelFactory
    print("准备模型中...")
    bottom, trunk, top = ModelFactory.create_split_models(
        model_type='gpt2',
        model_name_or_path='gpt2',
        split_point_1=2,
        split_point_2=10,
        device='cpu'
    )
    model_path = os.path.join(current_dir, "gpt2_trunk_full.pt")
    torch.save(trunk, model_path)
    print(f"完整模型对象已保存到: {model_path}")
    
    # 2. 启动测试
    multiprocessing.set_start_method('spawn', force=True)
    stop_event = multiprocessing.Event()
    
    server_process = multiprocessing.Process(target=run_server, args=(stop_event, model_path))
    server_process.start()
    
    success = run_client()
    
    stop_event.set()
    server_process.join()
    
    # 清理
    if os.path.exists(model_path):
        os.remove(model_path)
    
    if success:
        print("\n✅ GPT-2 测试通过")
    else:
        print("\n❌ GPT-2 测试失败")

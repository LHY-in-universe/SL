"""
启动 GPT-2 Trunk 服务器 (持久运行)
"""
import sys
import os
import torch
import logging
import signal

# 添加路径
current_dir = os.path.dirname(os.path.abspath(__file__))
project_root = os.path.dirname(current_dir)
sys.path.append(os.path.join(project_root, 'SplitLearning', 'src'))
sys.path.append(os.path.join(project_root, 'splitlearn-manager', 'src'))
sys.path.append(os.path.join(project_root, 'splitlearn-comm', 'src'))

# 配置日志
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger("GPT2Server")

def main():
    from splitlearn import ModelFactory
    from splitlearn_manager import ManagedServer, ServerConfig, ModelConfig
    
    # 1. 准备模型文件
    model_path = os.path.join(current_dir, "gpt2_trunk_full.pt")
    if not os.path.exists(model_path):
        logger.info("正在下载并拆分 GPT-2 模型以获取 Trunk...")
        bottom, trunk, top = ModelFactory.create_split_models(
            model_type='gpt2',
            model_name_or_path='gpt2',
            split_point_1=2,
            split_point_2=10,
            device='cpu'
        )
        torch.save(trunk, model_path)
        logger.info(f"Trunk 模型已保存到: {model_path}")
    else:
        logger.info(f"使用现有的 Trunk 模型: {model_path}")

    # 2. 配置服务器
    server_config = ServerConfig(
        host="127.0.0.1",
        port=50053,
        metrics_port=8002,
        log_level="INFO"
    )
    
    server = ManagedServer(config=server_config)
    
    # 3. 加载模型
    model_config = ModelConfig(
        model_id="gpt2-trunk",
        model_path=model_path,
        model_type="pytorch",
        device="cpu",
        warmup=True,
        config={"input_shape": (1, 10, 768)}
    )
    
    server.load_model(model_config)
    logger.info("GPT-2 Trunk 模型加载成功")
    
    # 4. 启动服务
    logger.info("启动服务器在 port 50053...")
    server.start()
    
    # 5. 等待终止
    try:
        server.wait_for_termination()
    except KeyboardInterrupt:
        logger.info("正在停止服务器...")
        server.stop()

if __name__ == "__main__":
    main()

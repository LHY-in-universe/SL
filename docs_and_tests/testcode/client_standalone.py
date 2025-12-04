"""
Standalone GPT-2 Split Learning Client
完全独立的客户端脚本，不依赖 splitlearn 包，解决 import 死锁问题。
"""
import os
import sys
import time
import torch
import torch.nn as nn
from typing import Optional
from transformers import GPT2Config, AutoTokenizer
from transformers.models.gpt2.modeling_gpt2 import GPT2Block

# ================= 配置 =================
SERVER_IP = '192.168.0.144'
SERVER_PORT = 50053
MODEL_ID = 'gpt2-trunk'
# ========================================

# ================= 模型定义 (从 splitlearn 复制) =================

class BaseSplitModel(nn.Module):
    def __init__(self, config, start_layer: int, end_layer: int):
        super().__init__()
        self.config = config
        self.start_layer = start_layer
        self.end_layer = end_layer

class GPT2BottomModel(BaseSplitModel):
    def __init__(self, config: GPT2Config, end_layer: int = 2):
        super().__init__(config, start_layer=0, end_layer=end_layer)
        self.wte = nn.Embedding(config.vocab_size, config.hidden_size)
        self.wpe = nn.Embedding(config.n_positions, config.hidden_size)
        self.drop = nn.Dropout(config.embd_pdrop)
        self.h = nn.ModuleList([GPT2Block(config, layer_idx=i) for i in range(end_layer)])

    def forward(self, input_ids):
        batch_size, seq_len = input_ids.shape
        device = input_ids.device
        
        # Embeddings
        inputs_embeds = self.wte(input_ids)
        position_ids = torch.arange(0, seq_len, dtype=torch.long, device=device)
        position_ids = position_ids.unsqueeze(0).expand(batch_size, -1)
        position_embeds = self.wpe(position_ids)
        hidden_states = self.drop(inputs_embeds + position_embeds)
        
        # Transformer blocks
        # 注意：这里简化了 attention_mask 处理，假设是基本的自回归生成
        attention_mask = None 
        
        for block in self.h:
            outputs = block(hidden_states, attention_mask=attention_mask)
            hidden_states = outputs[0]
            
        return hidden_states

class GPT2TopModel(BaseSplitModel):
    def __init__(self, config: GPT2Config, start_layer: int = 10):
        super().__init__(config, start_layer=start_layer, end_layer=config.n_layer)
        self.h = nn.ModuleList([GPT2Block(config, layer_idx=i) for i in range(start_layer, config.n_layer)])
        self.ln_f = nn.LayerNorm(config.hidden_size, eps=config.layer_norm_epsilon)

    def forward(self, hidden_states):
        # Transformer blocks
        for block in self.h:
            outputs = block(hidden_states)
            hidden_states = outputs[0]
            
        hidden_states = self.ln_f(hidden_states)
        
        # LM Head (GPT2LMHeadModel 的 lm_head 权重绑定到 wte)
        # 这里我们需要动态绑定或者加载权重
        return hidden_states

# ================= 主逻辑 =================

def main():
    print("=" * 60)
    print("GPT-2 独立客户端 (Standalone)")
    print("=" * 60)
    
    # 1. 加载 Tokenizer
    print("\n[1] 加载 Tokenizer...")
    tokenizer = AutoTokenizer.from_pretrained('gpt2')
    config = GPT2Config.from_pretrained('gpt2')
    print("    ✅ 完成")
    
    # 2. 加载模型权重
    # 注意：因为我们重定义了类，不能直接 torch.load(obj)，需要 load state_dict
    # 或者我们用 hack 的方式：先把类定义注入到 splitlearn 命名空间（如果 pickle 需要）
    # 但为了稳健，我们还是直接加载完整模型并提取权重，这样最安全
    
    print("\n[2] 从 HuggingFace 加载完整模型 (用于提取权重)...")
    from transformers import GPT2LMHeadModel
    full_model = GPT2LMHeadModel.from_pretrained('gpt2')
    full_model.eval()
    print("    ✅ 完成")
    
    # 3. 构建 Bottom 和 Top
    print("\n[3] 构建拆分模型...")
    
    # Bottom (0-2层)
    bottom = GPT2BottomModel(config, end_layer=3) # 注意：这里用3层
    # 复制权重
    bottom.wte.weight = full_model.transformer.wte.weight
    bottom.wpe.weight = full_model.transformer.wpe.weight
    bottom.h = full_model.transformer.h[:3]
    bottom.eval()
    
    # Top (10-12层)
    top_blocks = full_model.transformer.h[10:]
    ln_f = full_model.transformer.ln_f
    lm_head = full_model.lm_head
    
    print("    ✅ 完成 (Bottom: 0-2, Trunk: 3-10, Top: 10-12)")
    
    # 4. 连接 gRPC
    # 添加路径以导入 splitlearn_comm
    current_dir = os.path.dirname(os.path.abspath(__file__))
    project_root = os.path.dirname(current_dir)
    sys.path.insert(0, os.path.join(project_root, 'splitlearn-comm', 'src'))
    
    print(f"\n[4] 连接服务器 {SERVER_IP}:{SERVER_PORT}...")
    from splitlearn_comm import GRPCComputeClient
    client = GRPCComputeClient(f"{SERVER_IP}:{SERVER_PORT}", timeout=20.0)
    
    if not client.connect():
        print("    ❌ 连接失败！请检查服务器是否启动")
        return
    print("    ✅ 连接成功！")
    
    # 5. 交互循环
    print("\n" + "=" * 60)
    print("准备就绪！输入 prompt 生成文本")
    print("=" * 60)
    
    while True:
        try:
            prompt = input("\n>>> ").strip()
            if prompt.lower() == 'quit':
                break
            if not prompt:
                continue
            
            print("生成中: ", end='')
            input_ids = tokenizer.encode(prompt, return_tensors="pt")
            
            for _ in range(20):
                with torch.no_grad():
                    # Bottom
                    h = bottom(input_ids)
                    
                    # Trunk (Remote)
                    # 注意：Trunk 期望的输入形状和输出形状需要匹配
                    h = client.compute(h, model_id=MODEL_ID)
                    
                    # Top
                    # Top 部分包含剩下的 layers + ln_f + lm_head
                    for block in top_blocks:
                        h = block(h)[0]
                    h = ln_f(h)
                    logits = lm_head(h)
                    
                    next_logits = logits[:, -1, :]
                    next_id = next_logits.argmax(dim=-1).unsqueeze(-1)
                    
                    input_ids = torch.cat([input_ids, next_id], dim=-1)
                    print(tokenizer.decode(next_id[0]), end='', flush=True)
                    
                    if next_id.item() == tokenizer.eos_token_id:
                        break
            print()
            
        except KeyboardInterrupt:
            print("\n")
            break
        except Exception as e:
            print(f"\n❌ 错误: {e}")
            import traceback
            traceback.print_exc()

    client.close()

if __name__ == "__main__":
    main()




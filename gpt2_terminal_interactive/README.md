# GPT-2 终端交互式生成工具

直接在终端使用，不使用 Gradio，使用 monitor 库进行详细的性能统计。

## 安装依赖

```bash
pip install -r requirements.txt
```

## 使用方法

```bash
cd /Users/lhy/Desktop/Git/SL/gpt2_terminal_interactive
PYTHONPATH=../SplitLearnCore/src:../SplitLearnComm/src python gpt2_interactive.py
```

## 功能

- 终端交互式生成
- 实时性能监控（CPU、内存）
- 详细的 token 级别统计
- 可配置参数（max_tokens, temperature, top_k）

## 命令

- `help` - 显示帮助
- `stats` - 显示统计信息
- `max_tokens N` - 设置最大生成 token 数
- `temperature N` - 设置温度参数
- `top_k N` - 设置 Top-k 采样
- `quit/exit` - 退出程序

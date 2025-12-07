# GPT-2 交互式对话

简单的 GPT-2 交互式对话脚本，使用 **SplitLearnCore** 的 `load_full_model` 函数加载模型。

## 功能

- 使用 SplitLearnCore 统一接口加载 Hugging Face 模型
- 交互式对话功能
- 记录每次交互的时间（编码、推理、解码）
- 显示会话统计信息
- 手动生成方式，避免 Bus error

## 安装依赖

首次运行前需要创建虚拟环境并安装依赖：

```bash
# 创建虚拟环境（使用 Python.framework 3.11）
/Library/Frameworks/Python.framework/Versions/3.11/bin/python3 -m venv venv

# 激活虚拟环境并安装依赖
source venv/bin/activate
pip install -r requirements.txt
```

**注意**: 脚本会自动检测并使用项目中的 SplitLearnCore 库（位于 `../SplitLearnCore/src`）。

## 使用方法

### 方法 1: 使用运行脚本（推荐，自动使用虚拟环境）

```bash
# 使用运行脚本（会自动使用虚拟环境）
./run.sh
```

脚本会自动检测并使用虚拟环境中的 Python，避免 mutex 问题和 Bus error。

### 方法 2: 使用当前环境的 Python

```bash
python3 interactive_gpt2.py
```

**注意**: 如果遇到 `[mutex.cc : 452] RAW: Lock blocking` 错误，请使用方法 1 使用系统自带的 Python。

## 使用说明

1. 运行脚本后，程序会使用 SplitLearnCore 的 `load_full_model` 函数自动下载并加载模型（首次运行需要下载）
2. 输入你的问题或对话内容
3. 输入 'quit'、'exit' 或 '退出' 来结束程序
4. 程序会显示每次交互的详细时间统计（编码、推理、解码）和会话统计信息

## 技术细节

- **模型加载**: 使用 `splitlearn_core.quickstart.load_full_model()` 统一接口
- **生成方式**: 手动循环生成，避免使用 `model.generate()` 可能导致的 Bus error
- **线程安全**: 自动配置单线程模式，避免多线程竞争
- **设备管理**: 强制使用 CPU，避免 MPS 相关问题

## 注意事项

- 首次运行需要下载模型，可能需要一些时间
- 模型会加载到内存中，确保有足够的 RAM
- 生成回复的质量取决于输入和模型参数
- 如果遇到 Bus error，脚本会自动回退到更安全的生成方式
- 可以在 `interactive_gpt2_splitlearn.py` 中修改 `model_name` 变量来使用不同的模型（如 "sshleifer/tiny-gpt2" 用于快速测试）

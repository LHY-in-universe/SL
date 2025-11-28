# Model Storage Configuration Guide

本文档说明如何在 split learning 系统中配置和使用模型存储功能。

## 概述

新的模型存储功能允许您：
- 自动保存分割后的模型组件（Bottom、Trunk、Top）
- 组织模型文件到结构化的目录中
- 生成包含模型信息的元数据文件
- 通过配置文件管理存储设置

## 目录结构

启用自动保存后，模型将被保存到以下目录结构：

```
./models/
├── bottom/
│   ├── gpt2_2-10_bottom.pt
│   └── gpt2_2-10_bottom_metadata.json
├── trunk/
│   ├── gpt2_2-10_trunk.pt
│   └── gpt2_2-10_trunk_metadata.json
└── top/
    ├── gpt2_2-10_top.pt
    └── gpt2_2-10_top_metadata.json
```

文件命名规则：`{模型名}_{分割配置}_{组件类型}.pt`

## 使用方法

### 1. 基本用法 - 自动保存

```python
from splitlearn import ModelFactory

# 创建分割模型并自动保存
bottom, trunk, top = ModelFactory.create_split_models(
    model_type='gpt2',
    model_name_or_path='gpt2',
    split_point_1=2,
    split_point_2=10,
    storage_path='./models',  # 存储路径
    auto_save=True             # 启用自动保存
)
```

输出示例：
```
============================================================
Saving split models to storage...
============================================================
Storage directory: ./models
  Bottom model saved: models/bottom/gpt2_2-10_bottom.pt
  Trunk model saved: models/trunk/gpt2_2-10_trunk.pt
  Top model saved: models/top/gpt2_2-10_top.pt
============================================================
```

### 2. 向后兼容 - 不保存

默认情况下不会保存模型，保持原有行为：

```python
# 不保存模型（默认行为）
bottom, trunk, top = ModelFactory.create_split_models(
    model_type='gpt2',
    model_name_or_path='gpt2',
    split_point_1=2,
    split_point_2=10
)
# 不会创建任何文件
```

### 3. 手动保存

您也可以在创建模型后手动保存：

```python
from pathlib import Path

# 创建模型（不自动保存）
bottom, trunk, top = ModelFactory.create_split_models(...)

# 稍后手动保存
bottom.save_split_model(Path('./custom_path/bottom_model.pt'))
trunk.save_split_model(Path('./custom_path/trunk_model.pt'))
top.save_split_model(Path('./custom_path/top_model.pt'))
```

### 4. 使用 StorageManager 工具

```python
from splitlearn import StorageManager

# 生成标准化路径
path = StorageManager.get_split_model_path(
    base_path="./models",
    model_name="gpt2",
    component="bottom",
    split_config="2-10"
)
# 返回: models/bottom/gpt2_2-10_bottom.pt

# 创建存储目录
dirs = StorageManager.create_storage_directories("./models")
# 创建: ./models/bottom/, ./models/trunk/, ./models/top/

# 获取元数据文件路径
model_path = Path("./models/bottom/gpt2_2-10_bottom.pt")
metadata_path = StorageManager.get_model_metadata_path(model_path)
# 返回: models/bottom/gpt2_2-10_bottom_metadata.json
```

## 配置管理

### Server 配置

在 `ServerConfig` 中配置模型存储设置：

```python
from splitlearn_manager.config import ServerConfig

# 创建配置
config = ServerConfig(
    host="0.0.0.0",
    port=50051,
    model_storage_dir="./models",      # 模型存储目录
    auto_save_split_models=True        # 是否自动保存
)

# 保存到 YAML
config.to_yaml("server_config.yaml")
```

YAML 配置示例：
```yaml
host: 0.0.0.0
port: 50051
max_workers: 10
max_models: 5
metrics_port: 8000
health_check_interval: 30.0
enable_monitoring: true
log_level: INFO
model_storage_dir: ./models        # 新增
auto_save_split_models: true       # 新增
config: {}
```

### Model 配置

在 `ModelConfig` 中配置每个模型的存储设置：

```python
from splitlearn_manager.config import ModelConfig

# 创建配置
config = ModelConfig(
    model_id="my_gpt2",
    model_path="./models/gpt2.pt",
    model_type="gpt2",
    split_storage_config={          # 分割模型存储配置
        "storage_path": "./models",
        "auto_save": True,
        "split_config": "2-10"
    }
)

# 保存到 YAML
config.to_yaml("model_config.yaml")
```

## 元数据文件

每个保存的模型都会生成一个 JSON 元数据文件，包含：

```json
{
  "model_class": "GPT2BottomModel",
  "component": "bottom",
  "start_layer": 0,
  "end_layer": 2,
  "num_layers": 2,
  "num_parameters": 12345678,
  "memory_mb": 47.23,
  "config": { ... },
  "saved_at": "2025-11-28T10:30:45.123456"
}
```

读取元数据：

```python
import json
from pathlib import Path

metadata_path = Path("./models/bottom/gpt2_2-10_bottom_metadata.json")
with open(metadata_path, 'r') as f:
    metadata = json.load(f)

print(f"Component: {metadata['component']}")
print(f"Parameters: {metadata['num_parameters']:,}")
print(f"Memory: {metadata['memory_mb']} MB")
```

## 加载保存的模型

使用 PyTorch 标准方法加载模型：

```python
import torch
from splitlearn import ModelFactory

# 首先创建模型结构
bottom, trunk, top = ModelFactory.create_split_models(
    model_type='gpt2',
    model_name_or_path='gpt2',
    split_point_1=2,
    split_point_2=10
)

# 加载保存的权重
bottom.load_state_dict(torch.load('./models/bottom/gpt2_2-10_bottom.pt'))
trunk.load_state_dict(torch.load('./models/trunk/gpt2_2-10_trunk.pt'))
top.load_state_dict(torch.load('./models/top/gpt2_2-10_top.pt'))
```

## 最佳实践

1. **开发环境**：使用项目相对路径（如 `./models`）
2. **生产环境**：使用绝对路径或配置文件
3. **版本管理**：文件名中包含分割配置（如 `2-10`）以区分不同的分割方案
4. **元数据检查**：在加载模型前检查元数据确保兼容性
5. **磁盘空间**：大型模型会占用大量空间，定期清理不需要的模型

## 示例脚本

查看以下示例脚本了解更多用法：

- `examples/test_model_storage.py` - 测试所有存储功能
- `examples/split_gpt2_with_storage.py` - 完整的 GPT-2 分割和保存示例

## 故障排除

### 问题：模型未保存

**原因**：`auto_save=False`（默认值）

**解决**：明确设置 `auto_save=True` 和 `storage_path`

### 问题：目录权限错误

**原因**：没有写入权限

**解决**：检查目录权限或使用有权限的路径

### 问题：磁盘空间不足

**原因**：大型模型占用空间大

**解决**：清理旧模型或使用更大的存储设备

## API 参考

### ModelFactory.create_split_models()

```python
def create_split_models(
    model_type: str,
    model_name_or_path: str,
    split_point_1: int,
    split_point_2: int,
    device: str = 'cpu',
    storage_path: Optional[str] = None,    # 新增
    auto_save: bool = False,               # 新增
) -> Tuple[BottomModel, TrunkModel, TopModel]
```

### BaseSplitModel.save_split_model()

```python
def save_split_model(
    self,
    save_path: Union[str, Path],
    save_metadata: bool = True
) -> None
```

### StorageManager

```python
class StorageManager:
    @staticmethod
    def get_split_model_path(base_path, model_name, component, split_config) -> Path

    @staticmethod
    def create_storage_directories(base_path: str) -> Dict[str, Path]

    @staticmethod
    def get_model_metadata_path(model_path: Path) -> Path
```

## 更新历史

- **2025-11-28**: 初始实现
  - 添加自动保存功能
  - 添加存储管理工具
  - 添加配置选项
  - 添加元数据生成

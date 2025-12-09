"""
Qwen3-VL split model components.

结构与 Qwen2-VL 拆分一致：视觉塔在 bottom，文本 Transformer 按层拆分到 trunk/top。
"""

from .bottom import Qwen3VLBottomModel  # noqa: F401
from .trunk import Qwen3VLTrunkModel    # noqa: F401
from .top import Qwen3VLTopModel        # noqa: F401

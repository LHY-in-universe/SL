"""
Qwen2-VL split model components.

Provides bottom/trunk/top registrations for Qwen2-VL (vision-language) models,
splitting at the visual encoder -> text decoder boundary and along decoder layers.
"""

from .bottom import Qwen2VLBottomModel  # noqa: F401
from .trunk import Qwen2VLTrunkModel    # noqa: F401
from .top import Qwen2VLTopModel        # noqa: F401

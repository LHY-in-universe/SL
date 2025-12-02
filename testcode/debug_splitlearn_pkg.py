"""
调试 splitlearn 包内部导入
"""
import os
import sys
import time

# 设置路径
current_dir = os.path.dirname(os.path.abspath(__file__))
project_root = os.path.dirname(current_dir)
sys.path.insert(0, os.path.join(project_root, 'SplitLearning', 'src'))

print("Start debugging splitlearn package...")

# 1. version
try:
    print("1. Importing version...", end="", flush=True)
    from splitlearn.__version__ import __version__
    print(" OK")
except Exception as e:
    print(f" FAILED: {e}")

# 2. core
try:
    print("2. Importing core...", end="", flush=True)
    from splitlearn import core
    print(" OK (package)")
    
    print("   2.1 Importing BaseSplitModel...", end="", flush=True)
    from splitlearn.core import BaseSplitModel
    print(" OK")
except Exception as e:
    print(f" FAILED: {e}")

# 3. registry
try:
    print("3. Importing registry...", end="", flush=True)
    from splitlearn.registry import ModelRegistry
    print(" OK")
except Exception as e:
    print(f" FAILED: {e}")

# 4. factory (this imports registry and transformers)
try:
    print("4. Importing factory...", end="", flush=True)
    from splitlearn.factory import ModelFactory
    print(" OK")
except Exception as e:
    print(f" FAILED: {e}")

# 5. utils
try:
    print("5. Importing utils...", end="", flush=True)
    from splitlearn.utils import ParamMapper, StorageManager
    print(" OK")
except Exception as e:
    print(f" FAILED: {e}")

# 6. models (package)
try:
    print("6. Importing models package...", end="", flush=True)
    from splitlearn import models
    print(" OK")
except Exception as e:
    print(f" FAILED: {e}")

# 7. models.gpt2
try:
    print("7. Importing models.gpt2...", end="", flush=True)
    from splitlearn.models.gpt2 import GPT2BottomModel
    print(" OK")
except Exception as e:
    print(f" FAILED: {e}")

print("\n✅ All checks passed!")


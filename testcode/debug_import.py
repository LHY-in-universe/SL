import sys
import os

print("Step 1: Importing sys and os... Done")
sys.stdout.flush()

print("Step 2: Importing torch...")
sys.stdout.flush()
import torch
print(f"Torch version: {torch.__version__}")
sys.stdout.flush()

print("Step 3: Importing transformers...")
sys.stdout.flush()
import transformers
print(f"Transformers version: {transformers.__version__}")
sys.stdout.flush()

print("Step 4: Importing AutoTokenizer...")
sys.stdout.flush()
from transformers import AutoTokenizer
print("AutoTokenizer imported.")
sys.stdout.flush()

print("Step 5: Importing AutoModelForCausalLM...")
sys.stdout.flush()
from transformers import AutoModelForCausalLM
print("AutoModelForCausalLM imported.")
sys.stdout.flush()

print("Step 6: Adding splitlearn to path...")
sys.stdout.flush()
current_dir = os.path.dirname(os.path.abspath(__file__))
project_root = os.path.dirname(current_dir)
splitlearn_path = os.path.join(project_root, 'SplitLearning', 'src')
sys.path.append(splitlearn_path)
print(f"Path added: {splitlearn_path}")
sys.stdout.flush()

print("Step 7: Importing splitlearn...")
sys.stdout.flush()
import splitlearn
print(f"Splitlearn version: {splitlearn.__version__}")
sys.stdout.flush()

print("Step 8: Importing ModelFactory...")
sys.stdout.flush()
from splitlearn import ModelFactory
print("ModelFactory imported.")
sys.stdout.flush()

print("All imports successful.")

#!/usr/bin/env python3
"""
Simple Test - Basic functionality verification
"""

import sys
sys.path.insert(0, '/Users/lhy/Desktop/Git/splitlearn-manager/src')

from splitlearn_manager import ModelConfig, ServerConfig, ResourceManager

print("=" * 70)
print("splitlearn-manager - Simple Test")
print("=" * 70)

# Test 1: Configuration
print("\n[Test 1] Configuration...")
model_config = ModelConfig(
    model_id="test_model",
    model_path="/tmp/test.pt",
    device="cpu"
)
assert model_config.validate()
print("  ✓ ModelConfig works")

server_config = ServerConfig(port=50051)
assert server_config.validate()
print("  ✓ ServerConfig works")

# Test 2: Resource Manager
print("\n[Test 2] ResourceManager...")
rm = ResourceManager()
usage = rm.get_current_usage()
print(f"  ✓ CPU: {usage.cpu_percent:.1f}%")
print(f"  ✓ Memory: {usage.memory_mb:.0f}MB")

device = rm.find_best_device(prefer_gpu=False)
print(f"  ✓ Best device: {device}")

print("\n" + "=" * 70)
print("✓ All basic tests passed!")
print("=" * 70)

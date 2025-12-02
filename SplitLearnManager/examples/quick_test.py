#!/usr/bin/env python3
"""
Quick Test - Verify splitlearn-manager basic functionality

This script tests:
1. Model configuration
2. Model manager
3. Resource manager
4. Basic model loading/unloading
"""

import sys
import torch
import torch.nn as nn

# Add parent directory to path
sys.path.insert(0, '/Users/lhy/Desktop/Git/splitlearn-manager/src')

from splitlearn_manager import (
    ModelConfig,
    ServerConfig,
    ModelManager,
    ResourceManager,
)


def create_test_model():
    """Create a simple test model."""
    return nn.Sequential(
        nn.Linear(10, 20),
        nn.ReLU(),
        nn.Linear(20, 10)
    )


def test_configuration():
    """Test configuration classes."""
    print("\n[Test 1] Configuration...")

    # Model config
    model_config = ModelConfig(
        model_id="test_model",
        model_path="/tmp/test_model.pt",
        device="cpu",
        batch_size=32
    )
    assert model_config.validate()
    print("  ✓ ModelConfig validated")

    # Server config
    server_config = ServerConfig(
        port=50051,
        max_workers=10
    )
    assert server_config.validate()
    print("  ✓ ServerConfig validated")


def test_resource_manager():
    """Test resource manager."""
    print("\n[Test 2] ResourceManager...")

    rm = ResourceManager()

    # Get usage
    usage = rm.get_current_usage()
    assert usage.cpu_percent >= 0
    assert usage.memory_mb > 0
    print(f"  ✓ CPU: {usage.cpu_percent:.1f}%")
    print(f"  ✓ Memory: {usage.memory_mb:.0f}MB")

    # Find best device
    device = rm.find_best_device(prefer_gpu=False)
    assert device == "cpu"
    print(f"  ✓ Best device: {device}")


def test_model_manager():
    """Test model manager."""
    print("\n[Test 3] ModelManager...")

    # Create model manager
    mm = ModelManager(max_models=3)

    # Create and save test model
    model = create_test_model()
    model_path = "/tmp/test_model.pt"
    torch.save(model, model_path)

    # Test model loading
    config = ModelConfig(
        model_id="test_model",
        model_path=model_path,
        model_type="pytorch",
        device="cpu",
        warmup=False  # Skip warmup for speed
    )

    success = mm.load_model(config)
    assert success
    print("  ✓ Model loaded")

    # Test model retrieval
    managed_model = mm.get_model("test_model")
    assert managed_model is not None
    print("  ✓ Model retrieved")

    # Test model info
    info = mm.get_model_info("test_model")
    assert info["model_id"] == "test_model"
    print(f"  ✓ Model info: {info['num_parameters']} parameters")

    # Test list models
    models = mm.list_models()
    assert len(models) == 1
    print(f"  ✓ Listed {len(models)} model(s)")

    # Test statistics
    stats = mm.get_statistics()
    assert stats["total_models"] == 1
    print(f"  ✓ Statistics: {stats['total_requests']} requests")

    # Test model unloading
    success = mm.unload_model("test_model")
    assert success
    print("  ✓ Model unloaded")

    # Verify unloaded
    assert len(mm.list_models()) == 0
    print("  ✓ Verified unload")


def test_multi_model():
    """Test loading multiple models."""
    print("\n[Test 4] Multi-Model Management...")

    mm = ModelManager(max_models=3)

    # Load multiple models
    for i in range(3):
        model = create_test_model()
        model_path = f"/tmp/test_model_{i}.pt"
        torch.save(model, model_path)

        config = ModelConfig(
            model_id=f"model_{i}",
            model_path=model_path,
            device="cpu",
            warmup=False
        )

        mm.load_model(config)

    assert len(mm.list_models()) == 3
    print("  ✓ Loaded 3 models")

    # Test LRU eviction by loading a 4th model
    model = create_test_model()
    model_path = "/tmp/test_model_3.pt"
    torch.save(model, model_path)

    config = ModelConfig(
        model_id="model_3",
        model_path=model_path,
        device="cpu",
        warmup=False
    )

    mm.load_model(config)
    assert len(mm.list_models()) == 3  # Should still be 3 (LRU evicted)
    print("  ✓ LRU eviction working")

    # Cleanup
    mm.shutdown()
    print("  ✓ Shutdown complete")


def main():
    print("=" * 70)
    print("splitlearn-manager - Quick Test")
    print("=" * 70)

    try:
        test_configuration()
        test_resource_manager()
        test_model_manager()
        test_multi_model()

        print("\n" + "=" * 70)
        print("✓ All tests passed!")
        print("=" * 70)

    except Exception as e:
        print(f"\n✗ Test failed: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)


if __name__ == "__main__":
    main()

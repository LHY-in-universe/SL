"""
Tests for ComputeFunction and ModelComputeFunction.
"""

import pytest
import torch
import torch.nn as nn

from splitlearn_comm.core import ComputeFunction, ModelComputeFunction


class CustomComputeFunction(ComputeFunction):
    """Custom implementation for testing."""

    def __init__(self, value: float = 1.0):
        self.value = value
        self.setup_called = False
        self.teardown_called = False

    def compute(self, input_tensor: torch.Tensor) -> torch.Tensor:
        return input_tensor + self.value

    def get_info(self):
        return {"name": "CustomCompute", "value": str(self.value)}

    def setup(self):
        self.setup_called = True

    def teardown(self):
        self.teardown_called = True


class TestComputeFunction:
    """Tests for ComputeFunction abstract base class."""

    def test_custom_implementation(self, sample_tensor):
        """Test custom ComputeFunction implementation."""
        compute_fn = CustomComputeFunction(value=5.0)

        # Test compute
        output = compute_fn.compute(sample_tensor)
        expected = sample_tensor + 5.0
        assert torch.allclose(output, expected)

    def test_get_info(self):
        """Test get_info method."""
        compute_fn = CustomComputeFunction(value=3.14)
        info = compute_fn.get_info()

        assert info["name"] == "CustomCompute"
        assert info["value"] == "3.14"

    def test_setup_teardown(self):
        """Test setup and teardown hooks."""
        compute_fn = CustomComputeFunction()

        assert not compute_fn.setup_called
        assert not compute_fn.teardown_called

        compute_fn.setup()
        assert compute_fn.setup_called

        compute_fn.teardown()
        assert compute_fn.teardown_called

    def test_cannot_instantiate_abstract_class(self):
        """Test that ComputeFunction cannot be instantiated directly."""
        with pytest.raises(TypeError):
            ComputeFunction()


class TestModelComputeFunction:
    """Tests for ModelComputeFunction."""

    @pytest.fixture
    def simple_model(self):
        """Simple PyTorch model for testing."""
        return nn.Sequential(
            nn.Linear(10, 20),
            nn.ReLU(),
            nn.Linear(20, 10)
        )

    def test_initialization_cpu(self, simple_model):
        """Test initialization with CPU device."""
        compute_fn = ModelComputeFunction(
            model=simple_model,
            device="cpu",
            model_name="TestModel"
        )

        assert compute_fn.model is simple_model
        assert compute_fn.device == "cpu"
        assert compute_fn.model_name == "TestModel"

    @pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA not available")
    def test_initialization_cuda(self, simple_model):
        """Test initialization with CUDA device."""
        compute_fn = ModelComputeFunction(
            model=simple_model,
            device="cuda",
            model_name="TestModel"
        )

        assert compute_fn.device == "cuda"
        # Check model is on CUDA
        for param in compute_fn.model.parameters():
            assert param.is_cuda

    def test_compute_cpu(self, simple_model):
        """Test computation on CPU."""
        compute_fn = ModelComputeFunction(
            model=simple_model,
            device="cpu"
        )

        input_tensor = torch.randn(1, 10)
        output = compute_fn.compute(input_tensor)

        assert output.shape == (1, 10)
        assert not output.requires_grad  # Should be in eval mode with no_grad

    def test_compute_batch(self, simple_model):
        """Test computation with batch input."""
        compute_fn = ModelComputeFunction(model=simple_model, device="cpu")

        input_tensor = torch.randn(32, 10)  # Batch of 32
        output = compute_fn.compute(input_tensor)

        assert output.shape == (32, 10)

    def test_model_name_inference(self, simple_model):
        """Test automatic model name inference."""
        compute_fn = ModelComputeFunction(model=simple_model, device="cpu")

        info = compute_fn.get_info()
        assert info["name"] == "Sequential"

    def test_get_info(self, simple_model):
        """Test get_info returns correct information."""
        compute_fn = ModelComputeFunction(
            model=simple_model,
            device="cpu",
            model_name="MyModel"
        )

        info = compute_fn.get_info()

        assert info["name"] == "MyModel"
        assert info["device"] == "cpu"
        assert "parameters" in info

    def test_parameter_count(self, simple_model):
        """Test parameter counting."""
        compute_fn = ModelComputeFunction(model=simple_model, device="cpu")

        # Count parameters manually
        expected_params = sum(p.numel() for p in simple_model.parameters())

        info = compute_fn.get_info()
        assert info["parameters"] == expected_params

    def test_eval_mode(self, simple_model):
        """Test that model is in eval mode during compute."""
        simple_model.train()  # Set to training mode
        assert simple_model.training

        compute_fn = ModelComputeFunction(model=simple_model, device="cpu")
        input_tensor = torch.randn(1, 10)

        # After compute, should be in eval mode
        compute_fn.compute(input_tensor)
        assert not simple_model.training

    def test_no_gradient(self, simple_model):
        """Test that gradients are not computed."""
        compute_fn = ModelComputeFunction(model=simple_model, device="cpu")

        input_tensor = torch.randn(1, 10, requires_grad=True)
        output = compute_fn.compute(input_tensor)

        assert not output.requires_grad

    def test_different_input_shapes(self, simple_model):
        """Test with different input shapes."""
        compute_fn = ModelComputeFunction(model=simple_model, device="cpu")

        shapes = [(1, 10), (5, 10), (100, 10)]

        for shape in shapes:
            input_tensor = torch.randn(*shape)
            output = compute_fn.compute(input_tensor)
            assert output.shape == shape

    def test_deterministic_output(self, simple_model):
        """Test that same input produces same output."""
        compute_fn = ModelComputeFunction(model=simple_model, device="cpu")

        input_tensor = torch.randn(1, 10)

        output1 = compute_fn.compute(input_tensor)
        output2 = compute_fn.compute(input_tensor)

        assert torch.allclose(output1, output2)


class TestComputeFunctionEdgeCases:
    """Tests for edge cases and error conditions."""

    def test_zero_dimension_tensor(self):
        """Test with scalar tensor."""
        class ScalarFunction(ComputeFunction):
            def compute(self, input_tensor):
                return input_tensor * 2

        compute_fn = ScalarFunction()
        scalar = torch.tensor(5.0)
        output = compute_fn.compute(scalar)

        assert output.item() == 10.0

    def test_large_tensor(self):
        """Test with large tensor."""
        class IdentityFunction(ComputeFunction):
            def compute(self, input_tensor):
                return input_tensor

        compute_fn = IdentityFunction()
        large_tensor = torch.randn(100, 100, 100)  # 1M elements
        output = compute_fn.compute(large_tensor)

        assert output.shape == large_tensor.shape

    def test_empty_tensor(self):
        """Test with empty tensor."""
        class IdentityFunction(ComputeFunction):
            def compute(self, input_tensor):
                return input_tensor

        compute_fn = IdentityFunction()
        empty_tensor = torch.tensor([])
        output = compute_fn.compute(empty_tensor)

        assert output.shape == torch.Size([0])

    def test_compute_function_error_handling(self):
        """Test error propagation in compute function."""
        class ErrorFunction(ComputeFunction):
            def compute(self, input_tensor):
                raise ValueError("Intentional error")

        compute_fn = ErrorFunction()
        input_tensor = torch.randn(2, 2)

        with pytest.raises(ValueError, match="Intentional error"):
            compute_fn.compute(input_tensor)

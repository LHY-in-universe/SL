"""
Pytest configuration and fixtures for splitlearn-comm tests.
"""

import pytest
import torch
import threading
import time
from typing import Generator

from splitlearn_comm import GRPCComputeServer, GRPCComputeClient
from splitlearn_comm.core import ComputeFunction


class DummyComputeFunction(ComputeFunction):
    """Simple compute function for testing."""

    def __init__(self, multiplier: float = 2.0, delay: float = 0.0):
        self.multiplier = multiplier
        self.delay = delay
        self.call_count = 0

    def compute(self, input_tensor: torch.Tensor) -> torch.Tensor:
        """Multiply input by multiplier."""
        self.call_count += 1
        if self.delay > 0:
            time.sleep(self.delay)
        return input_tensor * self.multiplier

    def get_info(self):
        return {
            "name": "DummyComputeFunction",
            "multiplier": str(self.multiplier),
            "call_count": str(self.call_count)
        }

    def setup(self):
        """Setup called when server starts."""
        self.call_count = 0

    def teardown(self):
        """Teardown called when server stops."""
        pass


class ErrorComputeFunction(ComputeFunction):
    """Compute function that raises errors for testing."""

    def __init__(self, error_message: str = "Test error"):
        self.error_message = error_message

    def compute(self, input_tensor: torch.Tensor) -> torch.Tensor:
        raise RuntimeError(self.error_message)

    def get_info(self):
        return {"name": "ErrorComputeFunction"}


@pytest.fixture
def dummy_compute_fn():
    """Fixture providing a dummy compute function."""
    return DummyComputeFunction(multiplier=2.0)


@pytest.fixture
def error_compute_fn():
    """Fixture providing an error-raising compute function."""
    return ErrorComputeFunction()


@pytest.fixture
def test_port():
    """Fixture providing a test port number."""
    return 50099


@pytest.fixture
def test_server(dummy_compute_fn, test_port) -> Generator[GRPCComputeServer, None, None]:
    """
    Fixture providing a running test server.

    The server is started in a background thread and stopped after the test.
    """
    server = GRPCComputeServer(
        compute_fn=dummy_compute_fn,
        host="localhost",
        port=test_port,
        max_workers=5
    )

    # Start server in background thread
    server_thread = threading.Thread(target=server.start, daemon=True)
    server_thread.start()

    # Wait for server to start
    time.sleep(0.5)

    yield server

    # Cleanup
    server.stop(grace=1.0)


@pytest.fixture
def test_client(test_port) -> Generator[GRPCComputeClient, None, None]:
    """
    Fixture providing a test client.

    The client is connected and closed after the test.
    """
    client = GRPCComputeClient(
        server_address=f"localhost:{test_port}",
        timeout=5.0
    )

    # Connect to server
    if not client.connect():
        pytest.skip("Could not connect to test server")

    yield client

    # Cleanup
    client.close()


@pytest.fixture
def sample_tensor():
    """Fixture providing a sample tensor for testing."""
    return torch.randn(2, 3, 4)


@pytest.fixture
def sample_tensors():
    """Fixture providing multiple sample tensors."""
    return [
        torch.randn(1, 10),
        torch.randn(2, 5, 8),
        torch.randn(1, 1, 1, 1),
        torch.ones(3, 3),
        torch.zeros(2, 2, 2)
    ]

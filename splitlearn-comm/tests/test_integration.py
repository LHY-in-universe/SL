"""
Integration tests for server-client communication.
"""

import pytest
import torch
import time
import threading

from splitlearn_comm import (
    GRPCComputeServer,
    GRPCComputeClient,
    ExponentialBackoff,
    FixedDelay
)
from splitlearn_comm.core import ComputeFunction, ModelComputeFunction, CompressedTensorCodec

from .conftest import DummyComputeFunction, ErrorComputeFunction


class TestBasicIntegration:
    """Basic integration tests."""

    def test_simple_compute(self, test_server, test_client, sample_tensor):
        """Test basic compute operation."""
        output = test_client.compute(sample_tensor)

        # DummyComputeFunction multiplies by 2.0
        expected = sample_tensor * 2.0

        assert torch.allclose(output, expected)
        assert output.shape == sample_tensor.shape

    def test_multiple_requests(self, test_server, test_client):
        """Test multiple sequential requests."""
        num_requests = 10

        for i in range(num_requests):
            input_tensor = torch.randn(2, 5, 8)
            output = test_client.compute(input_tensor)

            expected = input_tensor * 2.0
            assert torch.allclose(output, expected)

        # Check statistics
        stats = test_client.get_statistics()
        assert stats['total_requests'] == num_requests

    def test_different_tensor_shapes(self, test_server, test_client, sample_tensors):
        """Test with various tensor shapes."""
        for tensor in sample_tensors:
            output = test_client.compute(tensor)

            expected = tensor * 2.0
            assert torch.allclose(output, expected)
            assert output.shape == tensor.shape

    def test_health_check(self, test_server, test_client):
        """Test health check functionality."""
        is_healthy = test_client.health_check()
        assert is_healthy is True

    def test_service_info(self, test_server, test_client):
        """Test getting service information."""
        info = test_client.get_service_info()

        assert 'service_name' in info
        assert 'version' in info
        assert 'device' in info
        assert 'uptime_seconds' in info
        assert info['service_name'] == 'DummyComputeFunction'

    def test_statistics_tracking(self, test_server, test_client):
        """Test that statistics are tracked correctly."""
        # Perform some requests
        for _ in range(5):
            input_tensor = torch.randn(1, 10)
            test_client.compute(input_tensor)

        stats = test_client.get_statistics()

        assert stats['total_requests'] == 5
        assert stats['avg_total_time_ms'] > 0


class TestModelIntegration:
    """Integration tests with actual PyTorch models."""

    @pytest.fixture
    def model_server(self, test_port):
        """Fixture providing server with a real PyTorch model."""
        import torch.nn as nn

        model = nn.Sequential(
            nn.Linear(10, 20),
            nn.ReLU(),
            nn.Linear(20, 10)
        )

        compute_fn = ModelComputeFunction(
            model=model,
            device="cpu",
            model_name="TestModel"
        )

        server = GRPCComputeServer(
            compute_fn=compute_fn,
            host="localhost",
            port=test_port,
        )

        server_thread = threading.Thread(target=server.start, daemon=True)
        server_thread.start()
        time.sleep(0.5)

        yield server

        server.stop(grace=1.0)

    def test_model_inference(self, model_server, test_client):
        """Test inference with a real model."""
        input_tensor = torch.randn(5, 10)
        output = test_client.compute(input_tensor)

        assert output.shape == (5, 10)
        assert not torch.isnan(output).any()  # No NaN values

    def test_batch_processing(self, model_server, test_client):
        """Test processing batches."""
        batch_sizes = [1, 16, 32, 64]

        for batch_size in batch_sizes:
            input_tensor = torch.randn(batch_size, 10)
            output = test_client.compute(input_tensor)

            assert output.shape == (batch_size, 10)


class TestErrorHandling:
    """Integration tests for error handling."""

    @pytest.fixture
    def error_server(self, test_port):
        """Server that raises errors."""
        compute_fn = ErrorComputeFunction()

        server = GRPCComputeServer(
            compute_fn=compute_fn,
            host="localhost",
            port=test_port + 1,  # Different port
        )

        server_thread = threading.Thread(target=server.start, daemon=True)
        server_thread.start()
        time.sleep(0.5)

        yield server

        server.stop(grace=1.0)

    @pytest.fixture
    def error_client(self, test_port):
        """Client connecting to error server."""
        client = GRPCComputeClient(
            server_address=f"localhost:{test_port + 1}",
            timeout=5.0
        )
        client.connect()

        yield client

        client.close()

    def test_server_error_handling(self, error_server, error_client):
        """Test that server errors are propagated to client."""
        input_tensor = torch.randn(2, 3)

        # Server will raise error, client should handle it
        with pytest.raises(Exception):
            error_client.compute(input_tensor)

        # Request should be counted
        stats = error_client.get_statistics()
        assert stats['total_requests'] >= 1

    def test_connection_failure(self):
        """Test handling of connection failure."""
        # Try to connect to non-existent server
        client = GRPCComputeClient("localhost:99999", timeout=1.0)

        # Connection should fail
        connected = client.connect()
        assert connected is False

    def test_timeout_handling(self, test_port):
        """Test request timeout handling."""
        # Create server with slow compute function
        slow_fn = DummyComputeFunction(delay=2.0)

        server = GRPCComputeServer(
            compute_fn=slow_fn,
            host="localhost",
            port=test_port + 2,
        )

        server_thread = threading.Thread(target=server.start, daemon=True)
        server_thread.start()
        time.sleep(0.5)

        # Client with short timeout
        client = GRPCComputeClient(
            server_address=f"localhost:{test_port + 2}",
            timeout=0.5  # Shorter than server delay
        )
        client.connect()

        input_tensor = torch.randn(2, 3)

        # Should timeout
        with pytest.raises(Exception):
            client.compute(input_tensor)

        client.close()
        server.stop(grace=1.0)


class TestRetryIntegration:
    """Integration tests for retry mechanisms."""

    def test_retry_on_temporary_failure(self, test_port):
        """Test that retry works on temporary failures."""

        class TemporaryFailureFunction(ComputeFunction):
            def __init__(self):
                self.attempt_count = 0

            def compute(self, input_tensor):
                self.attempt_count += 1
                if self.attempt_count < 3:
                    raise RuntimeError("Temporary failure")
                return input_tensor * 2.0

        compute_fn = TemporaryFailureFunction()

        server = GRPCComputeServer(
            compute_fn=compute_fn,
            host="localhost",
            port=test_port + 3,
        )

        server_thread = threading.Thread(target=server.start, daemon=True)
        server_thread.start()
        time.sleep(0.5)

        # Client with retry strategy
        retry = ExponentialBackoff(max_retries=5, initial_delay=0.1)
        client = GRPCComputeClient(
            server_address=f"localhost:{test_port + 3}",
            retry_strategy=retry
        )
        client.connect()

        input_tensor = torch.randn(2, 3)

        # Should succeed after retries
        output = client.compute(input_tensor)
        assert torch.allclose(output, input_tensor * 2.0)

        client.close()
        server.stop(grace=1.0)

    def test_fixed_delay_retry(self, test_port):
        """Test FixedDelay retry strategy."""

        class CountingFunction(ComputeFunction):
            def __init__(self):
                self.count = 0

            def compute(self, input_tensor):
                self.count += 1
                if self.count == 1:
                    raise RuntimeError("First attempt fails")
                return input_tensor

        compute_fn = CountingFunction()

        server = GRPCComputeServer(
            compute_fn=compute_fn,
            host="localhost",
            port=test_port + 4,
        )

        server_thread = threading.Thread(target=server.start, daemon=True)
        server_thread.start()
        time.sleep(0.5)

        retry = FixedDelay(max_retries=3, delay=0.1)
        client = GRPCComputeClient(
            server_address=f"localhost:{test_port + 4}",
            retry_strategy=retry
        )
        client.connect()

        input_tensor = torch.randn(2, 3)
        output = client.compute(input_tensor)

        assert torch.allclose(output, input_tensor)

        client.close()
        server.stop(grace=1.0)


class TestCompression:
    """Integration tests for compressed communication."""

    @pytest.fixture
    def compressed_server(self, test_port):
        """Server with compression enabled."""
        compute_fn = DummyComputeFunction()
        codec = CompressedTensorCodec(compression_level=6)

        server = GRPCComputeServer(
            compute_fn=compute_fn,
            codec=codec,
            host="localhost",
            port=test_port + 5,
        )

        server_thread = threading.Thread(target=server.start, daemon=True)
        server_thread.start()
        time.sleep(0.5)

        yield server

        server.stop(grace=1.0)

    @pytest.fixture
    def compressed_client(self, test_port):
        """Client with compression enabled."""
        codec = CompressedTensorCodec(compression_level=6)
        client = GRPCComputeClient(
            server_address=f"localhost:{test_port + 5}",
            codec=codec
        )
        client.connect()

        yield client

        client.close()

    def test_compressed_communication(self, compressed_server, compressed_client):
        """Test that compressed communication works."""
        input_tensor = torch.randn(10, 100, 100)  # Large tensor
        output = compressed_client.compute(input_tensor)

        expected = input_tensor * 2.0
        assert torch.allclose(output, expected)

    def test_compression_with_different_sizes(self, compressed_server, compressed_client):
        """Test compression with various tensor sizes."""
        sizes = [
            (10, 10),
            (100, 100),
            (1000, 100),
        ]

        for size in sizes:
            input_tensor = torch.randn(*size)
            output = compressed_client.compute(input_tensor)

            expected = input_tensor * 2.0
            assert torch.allclose(output, expected)


class TestConcurrency:
    """Integration tests for concurrent access."""

    def test_concurrent_clients(self, test_server, test_port):
        """Test multiple clients accessing server concurrently."""
        num_clients = 10
        requests_per_client = 5

        results = []
        errors = []

        def client_worker(client_id):
            try:
                client = GRPCComputeClient(f"localhost:{test_port}")
                if not client.connect():
                    errors.append((client_id, "Connection failed"))
                    return

                for i in range(requests_per_client):
                    input_tensor = torch.randn(2, 3)
                    output = client.compute(input_tensor)
                    results.append((client_id, i, output.shape))

                client.close()
            except Exception as e:
                errors.append((client_id, str(e)))

        threads = [
            threading.Thread(target=client_worker, args=(i,))
            for i in range(num_clients)
        ]

        for t in threads:
            t.start()

        for t in threads:
            t.join()

        # Check results
        assert len(errors) == 0, f"Errors occurred: {errors}"
        assert len(results) == num_clients * requests_per_client

    def test_concurrent_requests_same_client(self, test_server, test_client):
        """Test concurrent requests from same client."""
        num_requests = 20

        results = []
        errors = []

        def make_request(request_id):
            try:
                input_tensor = torch.randn(2, 3)
                output = test_client.compute(input_tensor)
                results.append((request_id, output.shape))
            except Exception as e:
                errors.append((request_id, str(e)))

        threads = [
            threading.Thread(target=make_request, args=(i,))
            for i in range(num_requests)
        ]

        for t in threads:
            t.start()

        for t in threads:
            t.join()

        # All requests should succeed
        assert len(errors) == 0
        assert len(results) == num_requests


class TestContextManagers:
    """Integration tests for context manager support."""

    def test_server_context_manager(self, test_port):
        """Test server as context manager."""
        compute_fn = DummyComputeFunction()

        # Server should start and stop automatically
        with GRPCComputeServer(compute_fn, port=test_port + 6) as server:
            time.sleep(0.5)

            # Connect client
            client = GRPCComputeClient(f"localhost:{test_port + 6}")
            assert client.connect()

            # Make request
            output = client.compute(torch.randn(2, 3))
            assert output is not None

            client.close()

        # Server should be stopped now

    def test_client_context_manager(self, test_server, test_port):
        """Test client as context manager."""
        with GRPCComputeClient(f"localhost:{test_port}") as client:
            # Should auto-connect
            output = client.compute(torch.randn(2, 3))
            assert output is not None

        # Client should be closed now


class TestEdgeCases:
    """Integration tests for edge cases."""

    def test_very_large_tensor(self, test_server, test_client):
        """Test with very large tensor."""
        # 10M elements
        large_tensor = torch.randn(100, 100, 1000)

        output = test_client.compute(large_tensor)

        expected = large_tensor * 2.0
        assert output.shape == expected.shape
        # Sample check (full check would be slow)
        assert torch.allclose(output[0, 0, :100], expected[0, 0, :100])

    def test_empty_tensor(self, test_server, test_client):
        """Test with empty tensor."""
        empty_tensor = torch.tensor([])

        output = test_client.compute(empty_tensor)

        assert output.shape == torch.Size([0])

    def test_scalar_tensor(self, test_server, test_client):
        """Test with scalar tensor."""
        scalar = torch.tensor(3.14)

        output = test_client.compute(scalar)

        expected = scalar * 2.0
        assert torch.allclose(output, expected)

    def test_reconnection(self, test_server, test_client, test_port):
        """Test client reconnection after disconnect."""
        # First request
        output1 = test_client.compute(torch.randn(2, 3))
        assert output1 is not None

        # Close and reconnect
        test_client.close()
        time.sleep(0.2)

        new_client = GRPCComputeClient(f"localhost:{test_port}")
        assert new_client.connect()

        # Second request
        output2 = new_client.compute(torch.randn(2, 3))
        assert output2 is not None

        new_client.close()

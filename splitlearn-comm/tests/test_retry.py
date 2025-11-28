"""
Tests for retry strategies.
"""

import pytest
import time
from unittest.mock import Mock, MagicMock
import grpc

from splitlearn_comm.client import RetryStrategy, ExponentialBackoff, FixedDelay


class MockRpcError(grpc.RpcError):
    """Mock RPC error for testing."""

    def __init__(self, code=grpc.StatusCode.UNAVAILABLE, details="Test error"):
        self._code = code
        self._details = details

    def code(self):
        return self._code

    def details(self):
        return self._details


def create_mock_rpc_error(code=grpc.StatusCode.UNAVAILABLE):
    """Create a mock gRPC error."""
    return MockRpcError(code=code)


class TestExponentialBackoff:
    """Tests for ExponentialBackoff retry strategy."""

    def test_initialization_default(self):
        """Test initialization with default parameters."""
        retry = ExponentialBackoff()

        assert retry.max_retries == 3
        assert retry.initial_delay == 1.0
        assert retry.max_delay == 30.0
        assert retry.jitter == 0.25

    def test_initialization_custom(self):
        """Test initialization with custom parameters."""
        retry = ExponentialBackoff(
            max_retries=5,
            initial_delay=2.0,
            max_delay=60.0,
            jitter=0.5
        )

        assert retry.max_retries == 5
        assert retry.initial_delay == 2.0
        assert retry.max_delay == 60.0
        assert retry.jitter == 0.5

    def test_immediate_success(self):
        """Test function that succeeds immediately."""
        retry = ExponentialBackoff(max_retries=3)

        call_count = [0]

        def successful_func():
            call_count[0] += 1
            return "success"

        result = retry.execute(successful_func)

        assert result == "success"
        assert call_count[0] == 1

    def test_retry_on_grpc_error(self):
        """Test retry on gRPC errors."""
        retry = ExponentialBackoff(max_retries=3, initial_delay=0.01)

        call_count = [0]

        def sometimes_failing():
            call_count[0] += 1
            if call_count[0] < 3:
                raise create_mock_rpc_error(grpc.StatusCode.UNAVAILABLE)
            return "success"

        result = retry.execute(sometimes_failing)

        assert result == "success"
        assert call_count[0] == 3

    def test_max_retries_exceeded(self):
        """Test that exception is raised when max retries exceeded."""
        retry = ExponentialBackoff(max_retries=3, initial_delay=0.01)

        def always_failing():
            raise create_mock_rpc_error(grpc.StatusCode.UNAVAILABLE)

        with pytest.raises(grpc.RpcError):
            retry.execute(always_failing)

    def test_non_retriable_error_not_retried(self):
        """Test that non-retriable errors are not retried."""
        retry = ExponentialBackoff(max_retries=3, initial_delay=0.01)

        call_count = [0]

        def invalid_arg_error():
            call_count[0] += 1
            raise create_mock_rpc_error(grpc.StatusCode.INVALID_ARGUMENT)

        with pytest.raises(grpc.RpcError):
            retry.execute(invalid_arg_error)

        # Should only be called once (no retries for INVALID_ARGUMENT)
        assert call_count[0] == 1

    def test_exponential_backoff_timing(self):
        """Test that delays follow exponential backoff pattern."""
        retry = ExponentialBackoff(
            max_retries=4,
            initial_delay=0.05,
            jitter=0.0  # No jitter for predictable timing
        )

        timestamps = []

        def failing_func():
            timestamps.append(time.time())
            if len(timestamps) < 4:
                raise create_mock_rpc_error(grpc.StatusCode.UNAVAILABLE)
            return "success"

        retry.execute(failing_func)

        # Calculate actual delays
        delays = [timestamps[i + 1] - timestamps[i] for i in range(len(timestamps) - 1)]

        # Expected delays: 0.05, 0.1, 0.2 (exponential)
        expected_delays = [0.05, 0.1, 0.2]

        for actual, expected in zip(delays, expected_delays):
            assert abs(actual - expected) < 0.02  # Small tolerance

    def test_returns_function_result(self):
        """Test that the function's return value is passed through."""
        retry = ExponentialBackoff()

        def returns_value():
            return {"key": "value"}

        result = retry.execute(returns_value)
        assert result == {"key": "value"}


class TestFixedDelay:
    """Tests for FixedDelay retry strategy."""

    def test_initialization_default(self):
        """Test initialization with default parameters."""
        retry = FixedDelay()

        assert retry.max_retries == 3
        assert retry.delay == 1.0

    def test_initialization_custom(self):
        """Test initialization with custom parameters."""
        retry = FixedDelay(max_retries=5, delay=2.0)

        assert retry.max_retries == 5
        assert retry.delay == 2.0

    def test_immediate_success(self):
        """Test function that succeeds immediately."""
        retry = FixedDelay(max_retries=3)

        call_count = [0]

        def successful_func():
            call_count[0] += 1
            return "success"

        result = retry.execute(successful_func)

        assert result == "success"
        assert call_count[0] == 1

    def test_retry_with_fixed_delay(self):
        """Test that delays are fixed."""
        retry = FixedDelay(max_retries=4, delay=0.05)

        timestamps = []

        def failing_func():
            timestamps.append(time.time())
            if len(timestamps) < 4:
                raise create_mock_rpc_error(grpc.StatusCode.UNAVAILABLE)
            return "success"

        retry.execute(failing_func)

        # Calculate delays
        delays = [timestamps[i + 1] - timestamps[i] for i in range(len(timestamps) - 1)]

        # All delays should be approximately equal to 0.05
        for delay in delays:
            assert abs(delay - 0.05) < 0.02

    def test_max_retries_exceeded(self):
        """Test exception when max retries exceeded."""
        retry = FixedDelay(max_retries=2, delay=0.01)

        call_count = [0]

        def always_failing():
            call_count[0] += 1
            raise create_mock_rpc_error(grpc.StatusCode.UNAVAILABLE)

        with pytest.raises(grpc.RpcError):
            retry.execute(always_failing)

        assert call_count[0] == 2

    def test_zero_delay(self):
        """Test with zero delay."""
        retry = FixedDelay(max_retries=3, delay=0.0)

        call_count = [0]

        def sometimes_failing():
            call_count[0] += 1
            if call_count[0] < 3:
                raise create_mock_rpc_error(grpc.StatusCode.UNAVAILABLE)
            return "success"

        start_time = time.time()
        result = retry.execute(sometimes_failing)
        elapsed = time.time() - start_time

        assert result == "success"
        # Should complete very quickly with no delays
        assert elapsed < 0.1


class TestRetryEdgeCases:
    """Tests for edge cases and special scenarios."""

    def test_max_retries_zero(self):
        """Test retry strategy with max_retries=0."""
        retry = ExponentialBackoff(max_retries=0)

        call_count = [0]

        def failing_func():
            call_count[0] += 1
            raise create_mock_rpc_error(grpc.StatusCode.UNAVAILABLE)

        # Should fail immediately, no retries
        with pytest.raises(RuntimeError, match="Retry failed without exception"):
            retry.execute(failing_func)

        assert call_count[0] == 0

    def test_function_returns_none(self):
        """Test retry with function that returns None."""
        retry = ExponentialBackoff(max_retries=3)

        def returns_none():
            return None

        result = retry.execute(returns_none)
        assert result is None

    def test_different_error_codes(self):
        """Test retry behavior with different gRPC error codes."""
        retry = ExponentialBackoff(max_retries=3, initial_delay=0.01)

        # UNAVAILABLE should be retried
        call_count_unavailable = [0]

        def unavailable_then_success():
            call_count_unavailable[0] += 1
            if call_count_unavailable[0] == 1:
                raise create_mock_rpc_error(grpc.StatusCode.UNAVAILABLE)
            return "success"

        result = retry.execute(unavailable_then_success)
        assert result == "success"
        assert call_count_unavailable[0] == 2

        # DEADLINE_EXCEEDED should be retried
        call_count_deadline = [0]

        def deadline_then_success():
            call_count_deadline[0] += 1
            if call_count_deadline[0] == 1:
                raise create_mock_rpc_error(grpc.StatusCode.DEADLINE_EXCEEDED)
            return "success"

        result = retry.execute(deadline_then_success)
        assert result == "success"
        assert call_count_deadline[0] == 2

    def test_permission_denied_not_retried(self):
        """Test that PERMISSION_DENIED is not retried."""
        retry = ExponentialBackoff(max_retries=5, initial_delay=0.01)

        call_count = [0]

        def permission_denied():
            call_count[0] += 1
            raise create_mock_rpc_error(grpc.StatusCode.PERMISSION_DENIED)

        with pytest.raises(grpc.RpcError):
            retry.execute(permission_denied)

        # Should only be called once
        assert call_count[0] == 1

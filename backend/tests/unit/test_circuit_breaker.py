"""Tests for ClaudeClient circuit breaker (per-instance state).

Verifies that the circuit breaker opens after consecutive failures,
resets on success, auto-resets after the timeout, and that separate
ClaudeClient instances have independent circuit breaker state.
"""

from unittest.mock import MagicMock, patch

import pytest


@pytest.fixture()
def _mock_settings():
    """Provide fake settings so ClaudeClient.__init__ doesn't need real env vars."""
    fake = MagicMock()
    fake.claude_api_key.get_secret_value.return_value = "sk-fake-key"
    fake.claude_model = "claude-3-5-sonnet-20241022"
    fake.claude_temperature = 0.0
    fake.claude_max_tokens = 4096
    with patch("src.llm.claude_client.get_settings", return_value=fake):
        yield


@pytest.fixture()
def client(_mock_settings):
    """Return a fresh ClaudeClient with mocked settings."""
    from src.llm.claude_client import ClaudeClient

    return ClaudeClient()


class TestCircuitBreakerState:
    """Per-instance circuit breaker state management."""

    def test_fresh_client_circuit_closed(self, client):
        assert client._consecutive_failures == 0
        assert client._circuit_open_until == 0.0
        # Should not raise
        client._check_circuit_breaker()

    @pytest.mark.asyncio
    async def test_failures_below_threshold_keep_circuit_closed(self, client):
        for _ in range(4):
            await client._record_failure()
        assert client._consecutive_failures == 4
        assert client._circuit_open_until == 0.0
        # Should not raise
        client._check_circuit_breaker()

    @pytest.mark.asyncio
    async def test_five_failures_open_circuit(self, client):
        for _ in range(5):
            await client._record_failure()
        assert client._consecutive_failures == 5
        assert client._circuit_open_until > 0.0
        with pytest.raises(RuntimeError, match="circuit breaker is open"):
            client._check_circuit_breaker()

    @pytest.mark.asyncio
    async def test_success_resets_circuit(self, client):
        for _ in range(5):
            await client._record_failure()
        # Circuit is open
        assert client._circuit_open_until > 0.0

        await client._record_success()
        assert client._consecutive_failures == 0
        assert client._circuit_open_until == 0.0
        # Should not raise
        client._check_circuit_breaker()

    @pytest.mark.asyncio
    async def test_circuit_auto_resets_after_timeout(self, client):
        with patch("src.llm.claude_client.time.time") as mock_time:
            # Open the circuit at t=1000
            mock_time.return_value = 1000.0
            for _ in range(5):
                await client._record_failure()

            # Still within the 60-second window at t=1050
            mock_time.return_value = 1050.0
            with pytest.raises(RuntimeError, match="circuit breaker is open"):
                client._check_circuit_breaker()

            # After timeout at t=1061 -- circuit allows calls again
            mock_time.return_value = 1061.0
            client._check_circuit_breaker()  # Should not raise

    @pytest.mark.asyncio
    async def test_partial_failures_then_success_resets_counter(self, client):
        await client._record_failure()
        await client._record_failure()
        await client._record_failure()
        assert client._consecutive_failures == 3

        await client._record_success()
        assert client._consecutive_failures == 0

        # Need 5 more consecutive failures to open
        for _ in range(4):
            await client._record_failure()
        client._check_circuit_breaker()  # Still closed


class TestCircuitBreakerIsolation:
    """Separate ClaudeClient instances have independent circuit breaker state."""

    @pytest.mark.asyncio
    async def test_independent_instances(self, _mock_settings):
        from src.llm.claude_client import ClaudeClient

        client_a = ClaudeClient()
        client_b = ClaudeClient()

        # Open circuit on client_a
        for _ in range(5):
            await client_a._record_failure()

        # client_a is open
        with pytest.raises(RuntimeError):
            client_a._check_circuit_breaker()

        # client_b should be unaffected
        assert client_b._consecutive_failures == 0
        client_b._check_circuit_breaker()  # Should not raise

    @pytest.mark.asyncio
    async def test_reset_client_gives_fresh_state(self, _mock_settings):
        from src.llm.claude_client import get_claude_client, reset_claude_client

        client1 = get_claude_client()
        for _ in range(5):
            await client1._record_failure()
        assert client1._consecutive_failures == 5

        reset_claude_client()
        client2 = get_claude_client()
        assert client2._consecutive_failures == 0
        assert client2 is not client1


class TestCircuitBreakerConstants:
    """Verify class-level constants are accessible and correct."""

    def test_default_thresholds(self, client):
        assert client.MAX_CONSECUTIVE_FAILURES == 5
        assert client.CIRCUIT_OPEN_DURATION == 60.0

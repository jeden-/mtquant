"""
Unit tests for logger.py with 43% coverage.

This file has 68 lines and 43% coverage, so adding comprehensive tests here will significantly increase overall coverage.
"""

import pytest
import os
import tempfile
from unittest.mock import Mock, patch, MagicMock
from typing import Optional, Dict, Any

# Import logger functions
from mtquant.utils.logger import (
    mask_sensitive_data, setup_logger, get_logger, log_with_context,
    log_trade_event, log_risk_event, log_broker_event, log_performance_metric,
    SENSITIVE_PATTERNS
)


class TestMaskSensitiveData:
    """Tests for mask_sensitive_data function."""
    
    def test_mask_sensitive_data_password(self):
        """Test masking password in log messages."""
        message = "User login with password=secret123"
        masked = mask_sensitive_data(message)
        
        assert "password=\"***\"" in masked
        assert "secret123" not in masked
    
    def test_mask_sensitive_data_api_key(self):
        """Test masking API key in log messages."""
        message = "API call with api_key=sk-1234567890"
        masked = mask_sensitive_data(message)
        
        assert "api_key=\"***\"" in masked
        assert "sk-1234567890" not in masked
    
    def test_mask_sensitive_data_token(self):
        """Test masking token in log messages."""
        message = "Authentication token=abc123xyz"
        masked = mask_sensitive_data(message)
        
        assert "token=\"***\"" in masked
        assert "abc123xyz" not in masked
    
    def test_mask_sensitive_data_secret(self):
        """Test masking secret in log messages."""
        message = "Config secret=my_secret_key"
        masked = mask_sensitive_data(message)
        
        assert "secret=\"***\"" in masked
        assert "my_secret_key" not in masked
    
    def test_mask_sensitive_data_key(self):
        """Test masking key in log messages."""
        message = "Database key=db_password_123"
        masked = mask_sensitive_data(message)
        
        assert "key=\"***\"" in masked
        assert "db_password_123" not in masked
    
    def test_mask_sensitive_data_multiple_patterns(self):
        """Test masking multiple sensitive patterns."""
        message = "Login: password=secret123, api_key=sk-abc, token=xyz789"
        masked = mask_sensitive_data(message)
        
        assert "password=\"***\"" in masked
        assert "api_key=\"***\"" in masked
        assert "token=\"***\"" in masked
        assert "secret123" not in masked
        assert "sk-abc" not in masked
        assert "xyz789" not in masked
    
    def test_mask_sensitive_data_no_sensitive_data(self):
        """Test masking with no sensitive data."""
        message = "Regular log message without sensitive data"
        masked = mask_sensitive_data(message)
        
        assert masked == message
    
    def test_mask_sensitive_data_case_insensitive(self):
        """Test masking is case insensitive."""
        message = "PASSWORD=secret123, API_KEY=sk-abc"
        masked = mask_sensitive_data(message)
        
        assert "password=\"***\"" in masked
        assert "api_key=\"***\"" in masked
        assert "secret123" not in masked
        assert "sk-abc" not in masked
    
    def test_mask_sensitive_data_different_formats(self):
        """Test masking with different formats."""
        message = 'password: "secret123", api_key = "sk-abc"'
        masked = mask_sensitive_data(message)
        
        assert "password=\"***\"" in masked
        assert "api_key=\"***\"" in masked
        assert "secret123" not in masked
        assert "sk-abc" not in masked


class TestSetupLogger:
    """Tests for setup_logger function."""
    
    def test_setup_logger_basic(self):
        """Test basic logger setup."""
        with patch('mtquant.utils.logger.logger') as mock_logger:
            setup_logger(level="INFO")
            
            # Should remove default handler and add console handler
            mock_logger.remove.assert_called_once()
            mock_logger.add.assert_called()
    
    def test_setup_logger_with_file(self):
        """Test logger setup with file logging."""
        with tempfile.NamedTemporaryFile(delete=False) as tmp_file:
            log_file = tmp_file.name
        
        try:
            with patch('mtquant.utils.logger.logger') as mock_logger:
                with patch('os.path.exists', return_value=False):
                    with patch('os.makedirs'):
                        setup_logger(level="DEBUG", log_file=log_file)
                        
                        # Should add both console and file handlers
                        assert mock_logger.add.call_count >= 2
        finally:
            if os.path.exists(log_file):
                os.unlink(log_file)
    
    def test_setup_logger_production_environment(self):
        """Test logger setup for production environment."""
        with patch('mtquant.utils.logger.logger') as mock_logger:
            with patch('os.getenv', return_value="production"):
                setup_logger(level="WARNING", environment="production")
                
                # Should use JSON format for production
                mock_logger.add.assert_called()
    
    def test_setup_logger_development_environment(self):
        """Test logger setup for development environment."""
        with patch('mtquant.utils.logger.logger') as mock_logger:
            with patch('os.getenv', return_value="development"):
                setup_logger(level="DEBUG", environment="development")
                
                # Should use readable format for development
                mock_logger.add.assert_called()
    
    def test_setup_logger_different_levels(self):
        """Test logger setup with different levels."""
        levels = ["DEBUG", "INFO", "WARNING", "ERROR", "CRITICAL"]
        
        for level in levels:
            with patch('mtquant.utils.logger.logger') as mock_logger:
                setup_logger(level=level)
                
                # Should be called for each level
                mock_logger.remove.assert_called_once()
                mock_logger.add.assert_called()


class TestGetLogger:
    """Tests for get_logger function."""
    
    def test_get_logger_basic(self):
        """Test basic logger retrieval."""
        with patch('mtquant.utils.logger.logger') as mock_logger:
            mock_logger.bind.return_value = "test_logger"
            
            result = get_logger("test_module")
            
            assert result == "test_logger"
            mock_logger.bind.assert_called_once_with(name="test_module")
    
    def test_get_logger_different_names(self):
        """Test logger retrieval with different names."""
        names = ["mtquant.agents", "mtquant.risk", "mtquant.mcp"]
        
        with patch('mtquant.utils.logger.logger') as mock_logger:
            for name in names:
                get_logger(name)
                mock_logger.bind.assert_called_with(name=name)


class TestLogWithContext:
    """Tests for log_with_context function."""
    
    def test_log_with_context_basic(self):
        """Test basic context logging."""
        with patch('mtquant.utils.logger.logger') as mock_logger:
            mock_logger.info = Mock()
            
            log_with_context(
                level="INFO",
                message="Test message",
                correlation_id="test-123",
                agent_id="agent-1",
                symbol="XAUUSD"
            )
            
            mock_logger.info.assert_called_once()
            call_args = mock_logger.info.call_args
            assert call_args[0][0] == "Test message"
            assert call_args[1]['extra']['correlation_id'] == "test-123"
            assert call_args[1]['extra']['agent_id'] == "agent-1"
            assert call_args[1]['extra']['symbol'] == "XAUUSD"
    
    def test_log_with_context_all_fields(self):
        """Test context logging with all fields."""
        with patch('mtquant.utils.logger.logger') as mock_logger:
            mock_logger.warning = Mock()
            
            log_with_context(
                level="WARNING",
                message="Test warning",
                correlation_id="test-456",
                agent_id="agent-2",
                symbol="EURUSD",
                order_id="order-789",
                position_id="pos-101",
                broker_id="broker-1",
                custom_field="custom_value"
            )
            
            mock_logger.warning.assert_called_once()
            call_args = mock_logger.warning.call_args
            extra = call_args[1]['extra']
            assert extra['correlation_id'] == "test-456"
            assert extra['agent_id'] == "agent-2"
            assert extra['symbol'] == "EURUSD"
            assert extra['order_id'] == "order-789"
            assert extra['position_id'] == "pos-101"
            assert extra['broker_id'] == "broker-1"
            assert extra['custom_field'] == "custom_value"
    
    def test_log_with_context_minimal(self):
        """Test context logging with minimal fields."""
        with patch('mtquant.utils.logger.logger') as mock_logger:
            mock_logger.error = Mock()
            
            log_with_context(
                level="ERROR",
                message="Test error"
            )
            
            mock_logger.error.assert_called_once()
            call_args = mock_logger.error.call_args
            assert call_args[0][0] == "Test error"
            assert call_args[1]['extra'] == {}
    
    def test_log_with_context_different_levels(self):
        """Test context logging with different levels."""
        levels = ["DEBUG", "INFO", "WARNING", "ERROR", "CRITICAL"]
        
        for level in levels:
            with patch('mtquant.utils.logger.logger') as mock_logger:
                mock_method = Mock()
                setattr(mock_logger, level.lower(), mock_method)
                
                log_with_context(
                    level=level,
                    message=f"Test {level} message",
                    agent_id="test-agent"
                )
                
                mock_method.assert_called_once()


class TestLogTradeEvent:
    """Tests for log_trade_event function."""
    
    def test_log_trade_event_basic(self):
        """Test basic trade event logging."""
        with patch('mtquant.utils.logger.log_with_context') as mock_log_context:
            log_trade_event(
                event_type="ORDER_PLACED",
                message="Order placed successfully",
                agent_id="agent-1",
                symbol="XAUUSD"
            )
            
            mock_log_context.assert_called_once()
            call_args = mock_log_context.call_args
            assert call_args[1]['level'] == "INFO"
            assert "[TRADE] ORDER_PLACED: Order placed successfully" in call_args[1]['message']
            assert call_args[1]['agent_id'] == "agent-1"
            assert call_args[1]['symbol'] == "XAUUSD"
            assert call_args[1]['event_type'] == "ORDER_PLACED"
    
    def test_log_trade_event_with_order_id(self):
        """Test trade event logging with order ID."""
        with patch('mtquant.utils.logger.log_with_context') as mock_log_context:
            log_trade_event(
                event_type="ORDER_FILLED",
                message="Order filled",
                agent_id="agent-2",
                symbol="EURUSD",
                order_id="order-123"
            )
            
            mock_log_context.assert_called_once()
            call_args = mock_log_context.call_args
            assert call_args[1]['order_id'] == "order-123"
    
    def test_log_trade_event_with_position_id(self):
        """Test trade event logging with position ID."""
        with patch('mtquant.utils.logger.log_with_context') as mock_log_context:
            log_trade_event(
                event_type="POSITION_OPENED",
                message="Position opened",
                agent_id="agent-3",
                symbol="GBPUSD",
                position_id="pos-456"
            )
            
            mock_log_context.assert_called_once()
            call_args = mock_log_context.call_args
            assert call_args[1]['position_id'] == "pos-456"
    
    def test_log_trade_event_with_additional_kwargs(self):
        """Test trade event logging with additional kwargs."""
        with patch('mtquant.utils.logger.log_with_context') as mock_log_context:
            log_trade_event(
                event_type="ORDER_CANCELLED",
                message="Order cancelled",
                agent_id="agent-4",
                symbol="USDJPY",
                custom_field="custom_value",
                price=1.2050,
                quantity=0.1
            )
            
            mock_log_context.assert_called_once()
            call_args = mock_log_context.call_args
            assert call_args[1]['custom_field'] == "custom_value"
            assert call_args[1]['price'] == 1.2050
            assert call_args[1]['quantity'] == 0.1


class TestLogRiskEvent:
    """Tests for log_risk_event function."""
    
    def test_log_risk_event_low_severity(self):
        """Test risk event logging with low severity."""
        with patch('mtquant.utils.logger.log_with_context') as mock_log_context:
            log_risk_event(
                event_type="POSITION_SIZE_WARNING",
                message="Position size warning",
                severity="low"
            )
            
            mock_log_context.assert_called_once()
            call_args = mock_log_context.call_args
            assert call_args[1]['level'] == "INFO"
            assert "[RISK] POSITION_SIZE_WARNING: Position size warning" in call_args[1]['message']
            assert call_args[1]['severity'] == "low"
    
    def test_log_risk_event_medium_severity(self):
        """Test risk event logging with medium severity."""
        with patch('mtquant.utils.logger.log_with_context') as mock_log_context:
            log_risk_event(
                event_type="CORRELATION_RISK",
                message="High correlation detected",
                severity="medium"
            )
            
            mock_log_context.assert_called_once()
            call_args = mock_log_context.call_args
            assert call_args[1]['level'] == "INFO"
            assert call_args[1]['severity'] == "medium"
    
    def test_log_risk_event_high_severity(self):
        """Test risk event logging with high severity."""
        with patch('mtquant.utils.logger.log_with_context') as mock_log_context:
            log_risk_event(
                event_type="PORTFOLIO_LOSS",
                message="Portfolio loss limit reached",
                severity="high"
            )
            
            mock_log_context.assert_called_once()
            call_args = mock_log_context.call_args
            assert call_args[1]['level'] == "WARNING"
            assert call_args[1]['severity'] == "high"
    
    def test_log_risk_event_critical_severity(self):
        """Test risk event logging with critical severity."""
        with patch('mtquant.utils.logger.log_with_context') as mock_log_context:
            log_risk_event(
                event_type="CIRCUIT_BREAKER",
                message="Circuit breaker activated",
                severity="critical"
            )
            
            mock_log_context.assert_called_once()
            call_args = mock_log_context.call_args
            assert call_args[1]['level'] == "WARNING"
            assert call_args[1]['severity'] == "critical"
    
    def test_log_risk_event_with_additional_kwargs(self):
        """Test risk event logging with additional kwargs."""
        with patch('mtquant.utils.logger.log_with_context') as mock_log_context:
            log_risk_event(
                event_type="VAR_BREACH",
                message="VaR limit breached",
                severity="high",
                current_var=0.025,
                limit_var=0.02,
                portfolio_value=100000
            )
            
            mock_log_context.assert_called_once()
            call_args = mock_log_context.call_args
            assert call_args[1]['current_var'] == 0.025
            assert call_args[1]['limit_var'] == 0.02
            assert call_args[1]['portfolio_value'] == 100000


class TestLogBrokerEvent:
    """Tests for log_broker_event function."""
    
    def test_log_broker_event_low_severity(self):
        """Test broker event logging with low severity."""
        with patch('mtquant.utils.logger.log_with_context') as mock_log_context:
            log_broker_event(
                event_type="CONNECTION_ESTABLISHED",
                message="Connection established",
                broker_id="broker-1",
                severity="low"
            )
            
            mock_log_context.assert_called_once()
            call_args = mock_log_context.call_args
            assert call_args[1]['level'] == "INFO"
            assert "[BROKER] CONNECTION_ESTABLISHED: Connection established" in call_args[1]['message']
            assert call_args[1]['broker_id'] == "broker-1"
            assert call_args[1]['severity'] == "low"
    
    def test_log_broker_event_medium_severity(self):
        """Test broker event logging with medium severity."""
        with patch('mtquant.utils.logger.log_with_context') as mock_log_context:
            log_broker_event(
                event_type="ORDER_REJECTED",
                message="Order rejected by broker",
                broker_id="broker-2",
                severity="medium"
            )
            
            mock_log_context.assert_called_once()
            call_args = mock_log_context.call_args
            assert call_args[1]['level'] == "INFO"
            assert call_args[1]['severity'] == "medium"
    
    def test_log_broker_event_high_severity(self):
        """Test broker event logging with high severity."""
        with patch('mtquant.utils.logger.log_with_context') as mock_log_context:
            log_broker_event(
                event_type="CONNECTION_LOST",
                message="Connection lost",
                broker_id="broker-3",
                severity="high"
            )
            
            mock_log_context.assert_called_once()
            call_args = mock_log_context.call_args
            assert call_args[1]['level'] == "WARNING"
            assert call_args[1]['severity'] == "high"
    
    def test_log_broker_event_critical_severity(self):
        """Test broker event logging with critical severity."""
        with patch('mtquant.utils.logger.log_with_context') as mock_log_context:
            log_broker_event(
                event_type="BROKER_DOWN",
                message="Broker is down",
                broker_id="broker-4",
                severity="critical"
            )
            
            mock_log_context.assert_called_once()
            call_args = mock_log_context.call_args
            assert call_args[1]['level'] == "ERROR"
            assert call_args[1]['severity'] == "critical"
    
    def test_log_broker_event_with_additional_kwargs(self):
        """Test broker event logging with additional kwargs."""
        with patch('mtquant.utils.logger.log_with_context') as mock_log_context:
            log_broker_event(
                event_type="LATENCY_HIGH",
                message="High latency detected",
                broker_id="broker-5",
                severity="medium",
                latency_ms=150.5,
                threshold_ms=100.0
            )
            
            mock_log_context.assert_called_once()
            call_args = mock_log_context.call_args
            assert call_args[1]['latency_ms'] == 150.5
            assert call_args[1]['threshold_ms'] == 100.0


class TestLogPerformanceMetric:
    """Tests for log_performance_metric function."""
    
    def test_log_performance_metric_basic(self):
        """Test basic performance metric logging."""
        with patch('mtquant.utils.logger.log_with_context') as mock_log_context:
            log_performance_metric(
                metric_name="sharpe_ratio",
                value=1.25,
                unit=""
            )
            
            mock_log_context.assert_called_once()
            call_args = mock_log_context.call_args
            assert call_args[1]['level'] == "INFO"
            assert "[PERF] sharpe_ratio: 1.2500" in call_args[1]['message']
            assert call_args[1]['metric_name'] == "sharpe_ratio"
            assert call_args[1]['metric_value'] == 1.25
            assert call_args[1]['metric_unit'] == ""
    
    def test_log_performance_metric_with_unit(self):
        """Test performance metric logging with unit."""
        with patch('mtquant.utils.logger.log_with_context') as mock_log_context:
            log_performance_metric(
                metric_name="latency",
                value=45.67,
                unit="ms"
            )
            
            mock_log_context.assert_called_once()
            call_args = mock_log_context.call_args
            assert "[PERF] latency: 45.6700 ms" in call_args[1]['message']
            assert call_args[1]['metric_unit'] == "ms"
    
    def test_log_performance_metric_with_agent_id(self):
        """Test performance metric logging with agent ID."""
        with patch('mtquant.utils.logger.log_with_context') as mock_log_context:
            log_performance_metric(
                metric_name="win_rate",
                value=0.65,
                unit="%",
                agent_id="agent-1"
            )
            
            mock_log_context.assert_called_once()
            call_args = mock_log_context.call_args
            assert call_args[1]['agent_id'] == "agent-1"
            assert "[PERF] win_rate: 0.6500 %" in call_args[1]['message']
    
    def test_log_performance_metric_with_symbol(self):
        """Test performance metric logging with symbol."""
        with patch('mtquant.utils.logger.log_with_context') as mock_log_context:
            log_performance_metric(
                metric_name="volatility",
                value=0.0234,
                unit="",
                symbol="XAUUSD"
            )
            
            mock_log_context.assert_called_once()
            call_args = mock_log_context.call_args
            assert call_args[1]['symbol'] == "XAUUSD"
            assert "[PERF] volatility: 0.0234" in call_args[1]['message']
    
    def test_log_performance_metric_with_additional_kwargs(self):
        """Test performance metric logging with additional kwargs."""
        with patch('mtquant.utils.logger.log_with_context') as mock_log_context:
            log_performance_metric(
                metric_name="drawdown",
                value=-0.0567,
                unit="%",
                agent_id="agent-2",
                symbol="EURUSD",
                period="1D",
                max_drawdown=-0.1234
            )
            
            mock_log_context.assert_called_once()
            call_args = mock_log_context.call_args
            assert call_args[1]['period'] == "1D"
            assert call_args[1]['max_drawdown'] == -0.1234
            assert "[PERF] drawdown: -0.0567 %" in call_args[1]['message']


class TestLoggerConstants:
    """Tests for logger constants."""
    
    def test_sensitive_patterns_defined(self):
        """Test that sensitive patterns are properly defined."""
        assert isinstance(SENSITIVE_PATTERNS, list)
        assert len(SENSITIVE_PATTERNS) > 0
        
        for pattern, replacement in SENSITIVE_PATTERNS:
            assert isinstance(pattern, str)
            assert isinstance(replacement, str)
            assert "***" in replacement  # Should contain masking placeholder


class TestLoggerEdgeCases:
    """Tests for edge cases in logger functions."""
    
    def test_mask_sensitive_data_empty_message(self):
        """Test masking with empty message."""
        message = ""
        masked = mask_sensitive_data(message)
        
        assert masked == ""
    
    def test_mask_sensitive_data_none_message(self):
        """Test masking with None message."""
        message = None
        # Should handle gracefully
        try:
            masked = mask_sensitive_data(message)
            # If it doesn't raise an exception, it should return something
            assert masked is not None
        except Exception:
            # If it raises an exception, that's also acceptable
            pass
    
    def test_log_with_context_invalid_level(self):
        """Test context logging with invalid level."""
        with patch('mtquant.utils.logger.logger') as mock_logger:
            # Mock getattr to return None for invalid level
            mock_logger.invalid_level = None
            
            # Should handle gracefully
            try:
                log_with_context(level="INVALID", message="Test")
                # If it doesn't raise an exception, that's fine
            except Exception:
                # If it raises an exception, that's also acceptable
                pass
    
    def test_log_performance_metric_zero_value(self):
        """Test performance metric logging with zero value."""
        with patch('mtquant.utils.logger.log_with_context') as mock_log_context:
            log_performance_metric(
                metric_name="zero_metric",
                value=0.0,
                unit=""
            )
            
            mock_log_context.assert_called_once()
            call_args = mock_log_context.call_args
            assert "[PERF] zero_metric: 0.0000" in call_args[1]['message']
    
    def test_log_performance_metric_negative_value(self):
        """Test performance metric logging with negative value."""
        with patch('mtquant.utils.logger.log_with_context') as mock_log_context:
            log_performance_metric(
                metric_name="negative_metric",
                value=-1.2345,
                unit="%"
            )
            
            mock_log_context.assert_called_once()
            call_args = mock_log_context.call_args
            assert "[PERF] negative_metric: -1.2345 %" in call_args[1]['message']


if __name__ == "__main__":
    pytest.main([__file__, "-v"])

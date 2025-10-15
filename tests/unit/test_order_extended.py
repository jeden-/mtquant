"""
Extended unit tests for order.py to increase coverage.
"""

import pytest
from datetime import datetime
from unittest.mock import patch

from mtquant.mcp_integration.models.order import (
    Order,
    OrderSide,
    OrderType,
    OrderStatus
)


class TestOrderEnums:
    """Test Order enums."""
    
    def test_order_side_enum(self):
        """Test OrderSide enum values."""
        assert OrderSide.BUY.value == "buy"
        assert OrderSide.SELL.value == "sell"
        assert len(OrderSide) == 2
    
    def test_order_type_enum(self):
        """Test OrderType enum values."""
        assert OrderType.MARKET.value == "market"
        assert OrderType.LIMIT.value == "limit"
        assert OrderType.STOP.value == "stop"
        assert len(OrderType) == 3
    
    def test_order_status_enum(self):
        """Test OrderStatus enum values."""
        assert OrderStatus.PENDING.value == "pending"
        assert OrderStatus.FILLED.value == "filled"
        assert OrderStatus.CANCELLED.value == "cancelled"
        assert OrderStatus.REJECTED.value == "rejected"
        assert len(OrderStatus) == 4


class TestOrderCreation:
    """Test Order creation and basic functionality."""
    
    def test_order_creation_basic(self):
        """Test basic order creation."""
        order = Order(
            agent_id="test_agent",
            symbol="XAUUSD",
            side="buy",
            order_type="market",
            quantity=0.1,
            signal=0.5
        )
        
        assert order.agent_id == "test_agent"
        assert order.symbol == "XAUUSD"
        assert order.side == "buy"
        assert order.order_type == "market"
        assert order.quantity == 0.1
        assert order.signal == 0.5
        assert order.status == "pending"
        assert order.order_id is None
        assert order.price is None
        assert order.stop_loss is None
        assert order.take_profit is None
        assert order.broker_id is None
        assert order.metadata == {}
        assert isinstance(order.created_at, datetime)
    
    def test_order_creation_with_all_fields(self):
        """Test order creation with all fields."""
        order = Order(
            order_id="ORDER123",
            agent_id="test_agent",
            symbol="XAUUSD",
            side="sell",
            order_type="limit",
            quantity=0.2,
            price=2050.0,
            stop_loss=2060.0,
            take_profit=2040.0,
            signal=-0.3,
            status="filled",
            broker_id="broker1",
            metadata={"test": "value"}
        )
        
        assert order.order_id == "ORDER123"
        assert order.agent_id == "test_agent"
        assert order.symbol == "XAUUSD"
        assert order.side == "sell"
        assert order.order_type == "limit"
        assert order.quantity == 0.2
        assert order.price == 2050.0
        assert order.stop_loss == 2060.0
        assert order.take_profit == 2040.0
        assert order.signal == -0.3
        assert order.status == "filled"
        assert order.broker_id == "broker1"
        assert order.metadata == {"test": "value"}
    
    def test_order_creation_with_custom_datetime(self):
        """Test order creation with custom datetime."""
        custom_time = datetime(2024, 1, 1, 12, 0, 0)
        
        with patch('mtquant.mcp_integration.models.order.datetime') as mock_datetime:
            mock_datetime.utcnow.return_value = custom_time
            
            order = Order(
                agent_id="test_agent",
                symbol="XAUUSD",
                side="buy",
                order_type="market",
                quantity=0.1,
                signal=0.5,
                created_at=custom_time
            )
            
            assert order.created_at == custom_time


class TestOrderValidation:
    """Test Order validation logic."""
    
    def test_signal_validation_valid_range(self):
        """Test signal validation with valid range."""
        # Test valid signals
        valid_signals = [-1.0, -0.5, 0.0, 0.5, 1.0]
        
        for signal in valid_signals:
            order = Order(
                agent_id="test_agent",
                symbol="XAUUSD",
                side="buy",
                order_type="market",
                quantity=0.1,
                signal=signal
            )
            assert order.signal == signal
    
    def test_signal_validation_invalid_range(self):
        """Test signal validation with invalid range."""
        invalid_signals = [-1.1, 1.1, -2.0, 2.0]
        
        for signal in invalid_signals:
            with pytest.raises(ValueError, match="Signal must be between -1 and 1"):
                Order(
                    agent_id="test_agent",
                    symbol="XAUUSD",
                    side="buy",
                    order_type="market",
                    quantity=0.1,
                    signal=signal
                )
    
    def test_quantity_validation_positive(self):
        """Test quantity validation with positive values."""
        valid_quantities = [0.01, 0.1, 1.0, 10.0]
        
        for quantity in valid_quantities:
            order = Order(
                agent_id="test_agent",
                symbol="XAUUSD",
                side="buy",
                order_type="market",
                quantity=quantity,
                signal=0.5
            )
            assert order.quantity == quantity
    
    def test_quantity_validation_negative_or_zero(self):
        """Test quantity validation with negative or zero values."""
        invalid_quantities = [0.0, -0.1, -1.0]
        
        for quantity in invalid_quantities:
            with pytest.raises(ValueError, match="Quantity must be positive"):
                Order(
                    agent_id="test_agent",
                    symbol="XAUUSD",
                    side="buy",
                    order_type="market",
                    quantity=quantity,
                    signal=0.5
                )
    
    def test_side_validation_valid(self):
        """Test side validation with valid values."""
        valid_sides = ['buy', 'sell']
        
        for side in valid_sides:
            order = Order(
                agent_id="test_agent",
                symbol="XAUUSD",
                side=side,
                order_type="market",
                quantity=0.1,
                signal=0.5
            )
            assert order.side == side
    
    def test_side_validation_invalid(self):
        """Test side validation with invalid values."""
        invalid_sides = ['BUY', 'SELL', 'hold', 'invalid']
        
        for side in invalid_sides:
            with pytest.raises(ValueError, match="Invalid side"):
                Order(
                    agent_id="test_agent",
                    symbol="XAUUSD",
                    side=side,
                    order_type="market",
                    quantity=0.1,
                    signal=0.5
                )
    
    def test_order_type_validation_valid(self):
        """Test order type validation with valid values."""
        valid_types = ['market', 'limit', 'stop']
        
        for order_type in valid_types:
            order = Order(
                agent_id="test_agent",
                symbol="XAUUSD",
                side="buy",
                order_type=order_type,
                quantity=0.1,
                signal=0.5,
                price=2050.0 if order_type != 'market' else None
            )
            assert order.order_type == order_type
    
    def test_order_type_validation_invalid(self):
        """Test order type validation with invalid values."""
        invalid_types = ['MARKET', 'LIMIT', 'STOP', 'invalid']
        
        for order_type in invalid_types:
            with pytest.raises(ValueError, match="Invalid order type"):
                Order(
                    agent_id="test_agent",
                    symbol="XAUUSD",
                    side="buy",
                    order_type=order_type,
                    quantity=0.1,
                    signal=0.5
                )
    
    def test_price_validation_limit_order(self):
        """Test price validation for limit orders."""
        # Limit order without price should fail
        with pytest.raises(ValueError, match="Price required for limit orders"):
            Order(
                agent_id="test_agent",
                symbol="XAUUSD",
                side="buy",
                order_type="limit",
                quantity=0.1,
                signal=0.5
            )
        
        # Limit order with price should succeed
        order = Order(
            agent_id="test_agent",
            symbol="XAUUSD",
            side="buy",
            order_type="limit",
            quantity=0.1,
            signal=0.5,
            price=2050.0
        )
        assert order.price == 2050.0
    
    def test_price_validation_stop_order(self):
        """Test price validation for stop orders."""
        # Stop order without price should fail
        with pytest.raises(ValueError, match="Price required for stop orders"):
            Order(
                agent_id="test_agent",
                symbol="XAUUSD",
                side="buy",
                order_type="stop",
                quantity=0.1,
                signal=0.5
            )
        
        # Stop order with price should succeed
        order = Order(
            agent_id="test_agent",
            symbol="XAUUSD",
            side="buy",
            order_type="stop",
            quantity=0.1,
            signal=0.5,
            price=2050.0
        )
        assert order.price == 2050.0
    
    def test_price_validation_market_order(self):
        """Test price validation for market orders."""
        # Market order without price should succeed
        order = Order(
            agent_id="test_agent",
            symbol="XAUUSD",
            side="buy",
            order_type="market",
            quantity=0.1,
            signal=0.5
        )
        assert order.price is None
        
        # Market order with price should also succeed
        order = Order(
            agent_id="test_agent",
            symbol="XAUUSD",
            side="buy",
            order_type="market",
            quantity=0.1,
            signal=0.5,
            price=2050.0
        )
        assert order.price == 2050.0
    
    def test_stop_loss_validation_positive(self):
        """Test stop loss validation with positive values."""
        valid_stop_losses = [0.01, 1.0, 100.0]
        
        for stop_loss in valid_stop_losses:
            order = Order(
                agent_id="test_agent",
                symbol="XAUUSD",
                side="buy",
                order_type="market",
                quantity=0.1,
                signal=0.5,
                stop_loss=stop_loss
            )
            assert order.stop_loss == stop_loss
    
    def test_stop_loss_validation_negative_or_zero(self):
        """Test stop loss validation with negative or zero values."""
        invalid_stop_losses = [0.0, -1.0, -100.0]
        
        for stop_loss in invalid_stop_losses:
            with pytest.raises(ValueError, match="Stop loss must be positive"):
                Order(
                    agent_id="test_agent",
                    symbol="XAUUSD",
                    side="buy",
                    order_type="market",
                    quantity=0.1,
                    signal=0.5,
                    stop_loss=stop_loss
                )
    
    def test_take_profit_validation_positive(self):
        """Test take profit validation with positive values."""
        valid_take_profits = [0.01, 1.0, 100.0]
        
        for take_profit in valid_take_profits:
            order = Order(
                agent_id="test_agent",
                symbol="XAUUSD",
                side="buy",
                order_type="market",
                quantity=0.1,
                signal=0.5,
                take_profit=take_profit
            )
            assert order.take_profit == take_profit
    
    def test_take_profit_validation_negative_or_zero(self):
        """Test take profit validation with negative or zero values."""
        invalid_take_profits = [0.0, -1.0, -100.0]
        
        for take_profit in invalid_take_profits:
            with pytest.raises(ValueError, match="Take profit must be positive"):
                Order(
                    agent_id="test_agent",
                    symbol="XAUUSD",
                    side="buy",
                    order_type="market",
                    quantity=0.1,
                    signal=0.5,
                    take_profit=take_profit
                )
    
    def test_stop_loss_take_profit_logic_buy_order(self):
        """Test stop loss/take profit logic for buy orders."""
        # Buy order: stop loss should be below take profit
        order = Order(
            agent_id="test_agent",
            symbol="XAUUSD",
            side="buy",
            order_type="market",
            quantity=0.1,
            signal=0.5,
            stop_loss=2040.0,  # Below take profit
            take_profit=2060.0
        )
        assert order.stop_loss == 2040.0
        assert order.take_profit == 2060.0
        
        # Buy order: stop loss above take profit should fail
        with pytest.raises(ValueError, match="For buy orders, stop loss must be below take profit"):
            Order(
                agent_id="test_agent",
                symbol="XAUUSD",
                side="buy",
                order_type="market",
                quantity=0.1,
                signal=0.5,
                stop_loss=2060.0,  # Above take profit
                take_profit=2040.0
            )
    
    def test_stop_loss_take_profit_logic_sell_order(self):
        """Test stop loss/take profit logic for sell orders."""
        # Sell order: stop loss should be above take profit
        order = Order(
            agent_id="test_agent",
            symbol="XAUUSD",
            side="sell",
            order_type="market",
            quantity=0.1,
            signal=-0.5,
            stop_loss=2060.0,  # Above take profit
            take_profit=2040.0
        )
        assert order.stop_loss == 2060.0
        assert order.take_profit == 2040.0
        
        # Sell order: stop loss below take profit should fail
        with pytest.raises(ValueError, match="For sell orders, stop loss must be above take profit"):
            Order(
                agent_id="test_agent",
                symbol="XAUUSD",
                side="sell",
                order_type="market",
                quantity=0.1,
                signal=-0.5,
                stop_loss=2040.0,  # Below take profit
                take_profit=2060.0
            )


class TestOrderSerialization:
    """Test Order serialization methods."""
    
    def test_to_dict_basic(self):
        """Test to_dict method with basic order."""
        order = Order(
            agent_id="test_agent",
            symbol="XAUUSD",
            side="buy",
            order_type="market",
            quantity=0.1,
            signal=0.5
        )
        
        result = order.to_dict()
        
        assert isinstance(result, dict)
        assert result['agent_id'] == "test_agent"
        assert result['symbol'] == "XAUUSD"
        assert result['side'] == "buy"
        assert result['order_type'] == "market"
        assert result['quantity'] == 0.1
        assert result['signal'] == 0.5
        assert result['status'] == "pending"
        assert result['order_id'] is None
        assert result['price'] is None
        assert result['stop_loss'] is None
        assert result['take_profit'] is None
        assert result['broker_id'] is None
        assert result['metadata'] == {}
        assert isinstance(result['created_at'], str)
    
    def test_to_dict_with_all_fields(self):
        """Test to_dict method with all fields."""
        custom_time = datetime(2024, 1, 1, 12, 0, 0)
        
        order = Order(
            order_id="ORDER123",
            agent_id="test_agent",
            symbol="XAUUSD",
            side="sell",
            order_type="limit",
            quantity=0.2,
            price=2050.0,
            stop_loss=2060.0,
            take_profit=2040.0,
            signal=-0.3,
            status="filled",
            broker_id="broker1",
            metadata={"test": "value"},
            created_at=custom_time
        )
        
        result = order.to_dict()
        
        assert result['order_id'] == "ORDER123"
        assert result['agent_id'] == "test_agent"
        assert result['symbol'] == "XAUUSD"
        assert result['side'] == "sell"
        assert result['order_type'] == "limit"
        assert result['quantity'] == 0.2
        assert result['price'] == 2050.0
        assert result['stop_loss'] == 2060.0
        assert result['take_profit'] == 2040.0
        assert result['signal'] == -0.3
        assert result['status'] == "filled"
        assert result['broker_id'] == "broker1"
        assert result['metadata'] == {"test": "value"}
        assert result['created_at'] == custom_time.isoformat()
    
    def test_from_dict_basic(self):
        """Test from_dict method with basic order."""
        data = {
            'agent_id': 'test_agent',
            'symbol': 'XAUUSD',
            'side': 'buy',
            'order_type': 'market',
            'quantity': 0.1,
            'signal': 0.5,
            'created_at': '2024-01-01T12:00:00',
            'status': 'pending',
            'order_id': None,
            'price': None,
            'stop_loss': None,
            'take_profit': None,
            'broker_id': None,
            'metadata': {}
        }
        
        order = Order.from_dict(data)
        
        assert order.agent_id == "test_agent"
        assert order.symbol == "XAUUSD"
        assert order.side == "buy"
        assert order.order_type == "market"
        assert order.quantity == 0.1
        assert order.signal == 0.5
        assert order.status == "pending"
        assert order.order_id is None
        assert order.price is None
        assert order.stop_loss is None
        assert order.take_profit is None
        assert order.broker_id is None
        assert order.metadata == {}
        assert isinstance(order.created_at, datetime)
    
    def test_from_dict_with_all_fields(self):
        """Test from_dict method with all fields."""
        data = {
            'order_id': 'ORDER123',
            'agent_id': 'test_agent',
            'symbol': 'XAUUSD',
            'side': 'sell',
            'order_type': 'limit',
            'quantity': 0.2,
            'price': 2050.0,
            'stop_loss': 2060.0,
            'take_profit': 2040.0,
            'signal': -0.3,
            'created_at': '2024-01-01T12:00:00',
            'status': 'filled',
            'broker_id': 'broker1',
            'metadata': {'test': 'value'}
        }
        
        order = Order.from_dict(data)
        
        assert order.order_id == "ORDER123"
        assert order.agent_id == "test_agent"
        assert order.symbol == "XAUUSD"
        assert order.side == "sell"
        assert order.order_type == "limit"
        assert order.quantity == 0.2
        assert order.price == 2050.0
        assert order.stop_loss == 2060.0
        assert order.take_profit == 2040.0
        assert order.signal == -0.3
        assert order.status == "filled"
        assert order.broker_id == "broker1"
        assert order.metadata == {'test': 'value'}
        assert isinstance(order.created_at, datetime)
    
    def test_round_trip_serialization(self):
        """Test round-trip serialization (to_dict -> from_dict)."""
        original_order = Order(
            order_id="ORDER123",
            agent_id="test_agent",
            symbol="XAUUSD",
            side="sell",
            order_type="limit",
            quantity=0.2,
            price=2050.0,
            stop_loss=2060.0,
            take_profit=2040.0,
            signal=-0.3,
            status="filled",
            broker_id="broker1",
            metadata={"test": "value"}
        )
        
        # Convert to dict and back
        data = original_order.to_dict()
        restored_order = Order.from_dict(data)
        
        # Compare all fields except created_at (due to datetime parsing)
        assert restored_order.order_id == original_order.order_id
        assert restored_order.agent_id == original_order.agent_id
        assert restored_order.symbol == original_order.symbol
        assert restored_order.side == original_order.side
        assert restored_order.order_type == original_order.order_type
        assert restored_order.quantity == original_order.quantity
        assert restored_order.price == original_order.price
        assert restored_order.stop_loss == original_order.stop_loss
        assert restored_order.take_profit == original_order.take_profit
        assert restored_order.signal == original_order.signal
        assert restored_order.status == original_order.status
        assert restored_order.broker_id == original_order.broker_id
        assert restored_order.metadata == original_order.metadata


class TestOrderEdgeCases:
    """Test Order edge cases and error conditions."""
    
    def test_order_with_none_values(self):
        """Test order with None values for optional fields."""
        order = Order(
            agent_id="test_agent",
            symbol="XAUUSD",
            side="buy",
            order_type="market",
            quantity=0.1,
            signal=0.5,
            order_id=None,
            price=None,
            stop_loss=None,
            take_profit=None,
            broker_id=None
        )
        
        assert order.order_id is None
        assert order.price is None
        assert order.stop_loss is None
        assert order.take_profit is None
        assert order.broker_id is None
    
    def test_order_with_empty_metadata(self):
        """Test order with empty metadata."""
        order = Order(
            agent_id="test_agent",
            symbol="XAUUSD",
            side="buy",
            order_type="market",
            quantity=0.1,
            signal=0.5,
            metadata={}
        )
        
        assert order.metadata == {}
    
    def test_order_with_complex_metadata(self):
        """Test order with complex metadata."""
        metadata = {
            "strategy": "momentum",
            "confidence": "0.85",
            "risk_level": "medium",
            "notes": "High volatility expected"
        }
        
        order = Order(
            agent_id="test_agent",
            symbol="XAUUSD",
            side="buy",
            order_type="market",
            quantity=0.1,
            signal=0.5,
            metadata=metadata
        )
        
        assert order.metadata == metadata
    
    def test_order_status_values(self):
        """Test all order status values."""
        statuses = ['pending', 'filled', 'cancelled', 'rejected']
        
        for status in statuses:
            order = Order(
                agent_id="test_agent",
                symbol="XAUUSD",
                side="buy",
                order_type="market",
                quantity=0.1,
                signal=0.5,
                status=status
            )
            assert order.status == status
    
    def test_order_with_extreme_signal_values(self):
        """Test order with extreme signal values."""
        # Test boundary values
        extreme_signals = [-1.0, 1.0]
        
        for signal in extreme_signals:
            order = Order(
                agent_id="test_agent",
                symbol="XAUUSD",
                side="buy",
                order_type="market",
                quantity=0.1,
                signal=signal
            )
            assert order.signal == signal
    
    def test_order_with_small_quantity(self):
        """Test order with very small quantity."""
        order = Order(
            agent_id="test_agent",
            symbol="XAUUSD",
            side="buy",
            order_type="market",
            quantity=0.001,  # Very small quantity
            signal=0.5
        )
        
        assert order.quantity == 0.001
    
    def test_order_with_large_quantity(self):
        """Test order with large quantity."""
        order = Order(
            agent_id="test_agent",
            symbol="XAUUSD",
            side="buy",
            order_type="market",
            quantity=1000.0,  # Large quantity
            signal=0.5
        )
        
        assert order.quantity == 1000.0


class TestOrderUtilityMethods:
    """Test Order utility methods."""
    
    def test_is_market_order(self):
        """Test is_market_order method."""
        market_order = Order(
            agent_id="test_agent",
            symbol="XAUUSD",
            side="buy",
            order_type="market",
            quantity=0.1,
            signal=0.5
        )
        
        limit_order = Order(
            agent_id="test_agent",
            symbol="XAUUSD",
            side="buy",
            order_type="limit",
            quantity=0.1,
            signal=0.5,
            price=2050.0
        )
        
        stop_order = Order(
            agent_id="test_agent",
            symbol="XAUUSD",
            side="buy",
            order_type="stop",
            quantity=0.1,
            signal=0.5,
            price=2050.0
        )
        
        assert market_order.is_market_order() is True
        assert limit_order.is_market_order() is False
        assert stop_order.is_market_order() is False
    
    def test_is_limit_order(self):
        """Test is_limit_order method."""
        market_order = Order(
            agent_id="test_agent",
            symbol="XAUUSD",
            side="buy",
            order_type="market",
            quantity=0.1,
            signal=0.5
        )
        
        limit_order = Order(
            agent_id="test_agent",
            symbol="XAUUSD",
            side="buy",
            order_type="limit",
            quantity=0.1,
            signal=0.5,
            price=2050.0
        )
        
        stop_order = Order(
            agent_id="test_agent",
            symbol="XAUUSD",
            side="buy",
            order_type="stop",
            quantity=0.1,
            signal=0.5,
            price=2050.0
        )
        
        assert market_order.is_limit_order() is False
        assert limit_order.is_limit_order() is True
        assert stop_order.is_limit_order() is False
    
    def test_is_stop_order(self):
        """Test is_stop_order method."""
        market_order = Order(
            agent_id="test_agent",
            symbol="XAUUSD",
            side="buy",
            order_type="market",
            quantity=0.1,
            signal=0.5
        )
        
        limit_order = Order(
            agent_id="test_agent",
            symbol="XAUUSD",
            side="buy",
            order_type="limit",
            quantity=0.1,
            signal=0.5,
            price=2050.0
        )
        
        stop_order = Order(
            agent_id="test_agent",
            symbol="XAUUSD",
            side="buy",
            order_type="stop",
            quantity=0.1,
            signal=0.5,
            price=2050.0
        )
        
        assert market_order.is_stop_order() is False
        assert limit_order.is_stop_order() is False
        assert stop_order.is_stop_order() is True
    
    def test_has_stop_loss(self):
        """Test has_stop_loss method."""
        order_without_sl = Order(
            agent_id="test_agent",
            symbol="XAUUSD",
            side="buy",
            order_type="market",
            quantity=0.1,
            signal=0.5
        )
        
        order_with_sl = Order(
            agent_id="test_agent",
            symbol="XAUUSD",
            side="buy",
            order_type="market",
            quantity=0.1,
            signal=0.5,
            stop_loss=2040.0
        )
        
        assert order_without_sl.has_stop_loss() is False
        assert order_with_sl.has_stop_loss() is True
    
    def test_has_take_profit(self):
        """Test has_take_profit method."""
        order_without_tp = Order(
            agent_id="test_agent",
            symbol="XAUUSD",
            side="buy",
            order_type="market",
            quantity=0.1,
            signal=0.5
        )
        
        order_with_tp = Order(
            agent_id="test_agent",
            symbol="XAUUSD",
            side="buy",
            order_type="market",
            quantity=0.1,
            signal=0.5,
            take_profit=2060.0
        )
        
        assert order_without_tp.has_take_profit() is False
        assert order_with_tp.has_take_profit() is True
    
    def test_get_risk_reward_ratio_buy_order(self):
        """Test get_risk_reward_ratio method for buy order."""
        order = Order(
            agent_id="test_agent",
            symbol="XAUUSD",
            side="buy",
            order_type="limit",
            quantity=0.1,
            signal=0.5,
            price=2050.0,
            stop_loss=2040.0,
            take_profit=2060.0
        )
        
        ratio = order.get_risk_reward_ratio()
        assert ratio is not None
        assert ratio == 1.0  # (2060-2050)/(2050-2040) = 10/10 = 1.0
    
    def test_get_risk_reward_ratio_sell_order(self):
        """Test get_risk_reward_ratio method for sell order."""
        order = Order(
            agent_id="test_agent",
            symbol="XAUUSD",
            side="sell",
            order_type="limit",
            quantity=0.1,
            signal=-0.5,
            price=2050.0,
            stop_loss=2060.0,
            take_profit=2040.0
        )
        
        ratio = order.get_risk_reward_ratio()
        assert ratio is not None
        assert ratio == 1.0  # (2050-2040)/(2060-2050) = 10/10 = 1.0
    
    def test_get_risk_reward_ratio_no_stop_loss(self):
        """Test get_risk_reward_ratio method without stop loss."""
        order = Order(
            agent_id="test_agent",
            symbol="XAUUSD",
            side="buy",
            order_type="limit",
            quantity=0.1,
            signal=0.5,
            price=2050.0,
            take_profit=2060.0
        )
        
        ratio = order.get_risk_reward_ratio()
        assert ratio is None
    
    def test_get_risk_reward_ratio_no_take_profit(self):
        """Test get_risk_reward_ratio method without take profit."""
        order = Order(
            agent_id="test_agent",
            symbol="XAUUSD",
            side="buy",
            order_type="limit",
            quantity=0.1,
            signal=0.5,
            price=2050.0,
            stop_loss=2040.0
        )
        
        ratio = order.get_risk_reward_ratio()
        assert ratio is None
    
    def test_get_risk_reward_ratio_no_price(self):
        """Test get_risk_reward_ratio method without price."""
        order = Order(
            agent_id="test_agent",
            symbol="XAUUSD",
            side="buy",
            order_type="market",
            quantity=0.1,
            signal=0.5,
            stop_loss=2040.0,
            take_profit=2060.0
        )
        
        ratio = order.get_risk_reward_ratio()
        assert ratio is None
    
    def test_get_risk_reward_ratio_zero_risk(self):
        """Test get_risk_reward_ratio method with zero risk."""
        order = Order(
            agent_id="test_agent",
            symbol="XAUUSD",
            side="buy",
            order_type="limit",
            quantity=0.1,
            signal=0.5,
            price=2050.0,
            stop_loss=2050.0,  # Same as price = zero risk
            take_profit=2060.0
        )
        
        ratio = order.get_risk_reward_ratio()
        assert ratio is None
    
    def test_repr(self):
        """Test __repr__ method."""
        order = Order(
            order_id="ORDER123",
            agent_id="test_agent",
            symbol="XAUUSD",
            side="sell",
            order_type="limit",
            quantity=0.2,
            signal=-0.3,
            price=2050.0,
            status="filled"
        )
        
        repr_str = repr(order)
        assert isinstance(repr_str, str)
        assert "ORDER123" in repr_str
        assert "test_agent" in repr_str
        assert "XAUUSD" in repr_str
        assert "sell" in repr_str
        assert "limit" in repr_str
        assert "0.2" in repr_str
        assert "-0.300" in repr_str
        assert "filled" in repr_str
    
    def test_str(self):
        """Test __str__ method."""
        order = Order(
            order_id="ORDER123",
            agent_id="test_agent",
            symbol="XAUUSD",
            side="sell",
            order_type="limit",
            quantity=0.2,
            signal=-0.3,
            price=2050.0,
            status="filled"
        )
        
        str_repr = str(order)
        assert isinstance(str_repr, str)
        assert "ORDER123" in str_repr
        assert "SELL" in str_repr
        assert "0.2" in str_repr
        assert "XAUUSD" in str_repr
        assert "2050.0" in str_repr
        assert "-0.300" in str_repr
        assert "filled" in str_repr
    
    def test_str_new_order(self):
        """Test __str__ method for new order without order_id."""
        order = Order(
            agent_id="test_agent",
            symbol="XAUUSD",
            side="buy",
            order_type="market",
            quantity=0.1,
            signal=0.5
        )
        
        str_repr = str(order)
        assert isinstance(str_repr, str)
        assert "NEW" in str_repr
        assert "BUY" in str_repr
        assert "0.1" in str_repr
        assert "XAUUSD" in str_repr
        assert "MARKET" in str_repr
        assert "0.500" in str_repr
        assert "pending" in str_repr

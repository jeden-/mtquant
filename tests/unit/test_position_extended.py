"""
Extended unit tests for Position model
"""

import pytest
from datetime import datetime, timedelta
from unittest.mock import patch

from mtquant.mcp_integration.models.position import Position


class TestPositionInitialization:
    """Test Position initialization and basic properties."""
    
    def test_position_initialization_basic(self):
        """Test basic position initialization."""
        position = Position(
            position_id="pos_1",
            agent_id="agent_1",
            symbol="EURUSD",
            side="long",
            quantity=0.1,
            entry_price=1.1000,
            current_price=1.1050
        )
        
        assert position.position_id == "pos_1"
        assert position.agent_id == "agent_1"
        assert position.symbol == "EURUSD"
        assert position.side == "long"
        assert position.quantity == 0.1
        assert position.entry_price == 1.1000
        assert position.current_price == 1.1050
        assert position.broker_id == "default"
        assert isinstance(position.opened_at, datetime)
        assert position.unrealized_pnl == pytest.approx(0.0005, abs=1e-6)  # (1.105 - 1.1) * 0.1
    
    def test_position_initialization_with_stop_loss_take_profit(self):
        """Test position initialization with stop loss and take profit."""
        position = Position(
            position_id="pos_2",
            agent_id="agent_2",
            symbol="XAUUSD",
            side="short",
            quantity=0.05,
            entry_price=2000.0,
            current_price=1995.0,
            stop_loss=2010.0,
            take_profit=1980.0,
            broker_id="broker_1"
        )
        
        assert position.stop_loss == 2010.0
        assert position.take_profit == 1980.0
        assert position.broker_id == "broker_1"
        assert position.unrealized_pnl == 0.25  # (2000.0 - 1995.0) * 0.05
    
    def test_position_initialization_with_metadata(self):
        """Test position initialization with metadata."""
        metadata = {"strategy": "momentum", "timeframe": "1H"}
        position = Position(
            position_id="pos_3",
            agent_id="agent_3",
            symbol="SPX500",
            side="long",
            quantity=1.0,
            entry_price=4000.0,
            current_price=4010.0,
            metadata=metadata
        )
        
        assert position.metadata == metadata


class TestPositionCalculations:
    """Test Position calculation methods and properties."""
    
    def test_calculate_unrealized_pnl_long(self):
        """Test unrealized P&L calculation for long position."""
        position = Position(
            position_id="pos_1",
            agent_id="agent_1",
            symbol="EURUSD",
            side="long",
            quantity=0.1,
            entry_price=1.1000,
            current_price=1.1050
        )
        
        # P&L should be (1.105 - 1.1) * 0.1 = 0.0005
        assert position.unrealized_pnl == pytest.approx(0.0005, abs=1e-6)
    
    def test_calculate_unrealized_pnl_short(self):
        """Test unrealized P&L calculation for short position."""
        position = Position(
            position_id="pos_2",
            agent_id="agent_2",
            symbol="EURUSD",
            side="short",
            quantity=0.1,
            entry_price=1.1000,
            current_price=1.0950
        )
        
        # P&L should be (1.1 - 1.095) * 0.1 = 0.0005
        assert position.unrealized_pnl == pytest.approx(0.0005, abs=1e-6)
    
    def test_calculate_unrealized_pnl_negative(self):
        """Test unrealized P&L calculation for losing position."""
        position = Position(
            position_id="pos_3",
            agent_id="agent_3",
            symbol="EURUSD",
            side="long",
            quantity=0.1,
            entry_price=1.1000,
            current_price=1.0950
        )
        
        # P&L should be (1.095 - 1.1) * 0.1 = -0.0005
        assert position.unrealized_pnl == pytest.approx(-0.0005, abs=1e-6)
    
    def test_unrealized_pnl_pct(self):
        """Test unrealized P&L percentage calculation."""
        position = Position(
            position_id="pos_1",
            agent_id="agent_1",
            symbol="EURUSD",
            side="long",
            quantity=0.1,
            entry_price=1.1000,
            current_price=1.1050
        )
        
        # P&L % should be (0.0005 / (1.1 * 0.1)) * 100 = 0.4545%
        expected_pct = (0.0005 / (1.1 * 0.1)) * 100
        assert position.unrealized_pnl_pct == pytest.approx(expected_pct, abs=0.001)
    
    def test_unrealized_pnl_pct_zero_entry_price(self):
        """Test unrealized P&L percentage with zero entry price."""
        position = Position(
            position_id="pos_1",
            agent_id="agent_1",
            symbol="EURUSD",
            side="long",
            quantity=0.1,
            entry_price=0.0,
            current_price=1.1050
        )
        
        assert position.unrealized_pnl_pct == 0.0
    
    def test_position_value(self):
        """Test position value calculation."""
        position = Position(
            position_id="pos_1",
            agent_id="agent_1",
            symbol="EURUSD",
            side="long",
            quantity=0.1,
            entry_price=1.1000,
            current_price=1.1050
        )
        
        # Position value should be 0.1 * 1.1050 = 0.1105
        assert position.position_value == 0.1105
    
    def test_entry_value(self):
        """Test entry value calculation."""
        position = Position(
            position_id="pos_1",
            agent_id="agent_1",
            symbol="EURUSD",
            side="long",
            quantity=0.1,
            entry_price=1.1000,
            current_price=1.1050
        )
        
        # Entry value should be 0.1 * 1.1000 = 0.11
        assert position.entry_value == pytest.approx(0.11, abs=1e-6)


class TestPositionDuration:
    """Test Position duration calculations."""
    
    def test_duration_hours(self):
        """Test duration calculation in hours."""
        # Create position with specific opened_at time
        opened_at = datetime.utcnow() - timedelta(hours=2)
        position = Position(
            position_id="pos_1",
            agent_id="agent_1",
            symbol="EURUSD",
            side="long",
            quantity=0.1,
            entry_price=1.1000,
            current_price=1.1050
        )
        position.opened_at = opened_at
        
        # Duration should be approximately 2 hours
        assert abs(position.duration_hours - 2.0) < 0.1
    
    def test_duration_days(self):
        """Test duration calculation in days."""
        # Create position with specific opened_at time
        opened_at = datetime.utcnow() - timedelta(days=1, hours=12)
        position = Position(
            position_id="pos_1",
            agent_id="agent_1",
            symbol="EURUSD",
            side="long",
            quantity=0.1,
            entry_price=1.1000,
            current_price=1.1050
        )
        position.opened_at = opened_at
        
        # Duration should be approximately 1.5 days
        assert abs(position.duration_days - 1.5) < 0.1


class TestPositionStatus:
    """Test Position status properties."""
    
    def test_is_winning(self):
        """Test is_winning property."""
        winning_position = Position(
            position_id="pos_1",
            agent_id="agent_1",
            symbol="EURUSD",
            side="long",
            quantity=0.1,
            entry_price=1.1000,
            current_price=1.1050
        )
        
        losing_position = Position(
            position_id="pos_2",
            agent_id="agent_2",
            symbol="EURUSD",
            side="long",
            quantity=0.1,
            entry_price=1.1000,
            current_price=1.0950
        )
        
        breakeven_position = Position(
            position_id="pos_3",
            agent_id="agent_3",
            symbol="EURUSD",
            side="long",
            quantity=0.1,
            entry_price=1.1000,
            current_price=1.1000
        )
        
        assert winning_position.is_winning == True
        assert losing_position.is_winning == False
        assert breakeven_position.is_winning == False
    
    def test_is_losing(self):
        """Test is_losing property."""
        winning_position = Position(
            position_id="pos_1",
            agent_id="agent_1",
            symbol="EURUSD",
            side="long",
            quantity=0.1,
            entry_price=1.1000,
            current_price=1.1050
        )
        
        losing_position = Position(
            position_id="pos_2",
            agent_id="agent_2",
            symbol="EURUSD",
            side="long",
            quantity=0.1,
            entry_price=1.1000,
            current_price=1.0950
        )
        
        breakeven_position = Position(
            position_id="pos_3",
            agent_id="agent_3",
            symbol="EURUSD",
            side="long",
            quantity=0.1,
            entry_price=1.1000,
            current_price=1.1000
        )
        
        assert winning_position.is_losing == False
        assert losing_position.is_losing == True
        assert breakeven_position.is_losing == False
    
    def test_is_at_breakeven(self):
        """Test is_at_breakeven property."""
        winning_position = Position(
            position_id="pos_1",
            agent_id="agent_1",
            symbol="EURUSD",
            side="long",
            quantity=0.1,
            entry_price=1.1000,
            current_price=1.1050
        )
        
        losing_position = Position(
            position_id="pos_2",
            agent_id="agent_2",
            symbol="EURUSD",
            side="long",
            quantity=0.1,
            entry_price=1.1000,
            current_price=1.0950
        )
        
        breakeven_position = Position(
            position_id="pos_3",
            agent_id="agent_3",
            symbol="EURUSD",
            side="long",
            quantity=0.1,
            entry_price=1.1000,
            current_price=1.1000
        )
        
        assert winning_position.is_at_breakeven == False
        assert losing_position.is_at_breakeven == False
        assert breakeven_position.is_at_breakeven == True


class TestPositionStopLossTakeProfit:
    """Test Position stop loss and take profit properties."""
    
    def test_has_stop_loss(self):
        """Test has_stop_loss property."""
        position_with_sl = Position(
            position_id="pos_1",
            agent_id="agent_1",
            symbol="EURUSD",
            side="long",
            quantity=0.1,
            entry_price=1.1000,
            current_price=1.1050,
            stop_loss=1.0950
        )
        
        position_without_sl = Position(
            position_id="pos_2",
            agent_id="agent_2",
            symbol="EURUSD",
            side="long",
            quantity=0.1,
            entry_price=1.1000,
            current_price=1.1050
        )
        
        assert position_with_sl.has_stop_loss == True
        assert position_without_sl.has_stop_loss == False
    
    def test_has_take_profit(self):
        """Test has_take_profit property."""
        position_with_tp = Position(
            position_id="pos_1",
            agent_id="agent_1",
            symbol="EURUSD",
            side="long",
            quantity=0.1,
            entry_price=1.1000,
            current_price=1.1050,
            take_profit=1.1100
        )
        
        position_without_tp = Position(
            position_id="pos_2",
            agent_id="agent_2",
            symbol="EURUSD",
            side="long",
            quantity=0.1,
            entry_price=1.1000,
            current_price=1.1050
        )
        
        assert position_with_tp.has_take_profit == True
        assert position_without_tp.has_take_profit == False
    
    def test_stop_loss_distance_long(self):
        """Test stop loss distance calculation for long position."""
        position = Position(
            position_id="pos_1",
            agent_id="agent_1",
            symbol="EURUSD",
            side="long",
            quantity=0.1,
            entry_price=1.1000,
            current_price=1.1050,
            stop_loss=1.0950
        )
        
        # For long position: entry_price - stop_loss = 1.1000 - 1.0950 = 0.005
        assert position.stop_loss_distance == pytest.approx(0.005, abs=1e-6)
    
    def test_stop_loss_distance_short(self):
        """Test stop loss distance calculation for short position."""
        position = Position(
            position_id="pos_2",
            agent_id="agent_2",
            symbol="EURUSD",
            side="short",
            quantity=0.1,
            entry_price=1.1000,
            current_price=1.0950,
            stop_loss=1.1050
        )
        
        # For short position: stop_loss - entry_price = 1.1050 - 1.1000 = 0.005
        assert position.stop_loss_distance == pytest.approx(0.005, abs=1e-6)
    
    def test_stop_loss_distance_none(self):
        """Test stop loss distance when no stop loss is set."""
        position = Position(
            position_id="pos_1",
            agent_id="agent_1",
            symbol="EURUSD",
            side="long",
            quantity=0.1,
            entry_price=1.1000,
            current_price=1.1050
        )
        
        assert position.stop_loss_distance is None
    
    def test_take_profit_distance_long(self):
        """Test take profit distance calculation for long position."""
        position = Position(
            position_id="pos_1",
            agent_id="agent_1",
            symbol="EURUSD",
            side="long",
            quantity=0.1,
            entry_price=1.1000,
            current_price=1.1050,
            take_profit=1.1100
        )
        
        # For long position: take_profit - entry_price = 1.1100 - 1.1000 = 0.01
        assert position.take_profit_distance == pytest.approx(0.01, abs=1e-6)
    
    def test_take_profit_distance_short(self):
        """Test take profit distance calculation for short position."""
        position = Position(
            position_id="pos_2",
            agent_id="agent_2",
            symbol="EURUSD",
            side="short",
            quantity=0.1,
            entry_price=1.1000,
            current_price=1.0950,
            take_profit=1.0900
        )
        
        # For short position: entry_price - take_profit = 1.1000 - 1.0900 = 0.01
        assert position.take_profit_distance == pytest.approx(0.01, abs=1e-6)
    
    def test_take_profit_distance_none(self):
        """Test take profit distance when no take profit is set."""
        position = Position(
            position_id="pos_1",
            agent_id="agent_1",
            symbol="EURUSD",
            side="long",
            quantity=0.1,
            entry_price=1.1000,
            current_price=1.1050
        )
        
        assert position.take_profit_distance is None
    
    def test_risk_reward_ratio(self):
        """Test risk-reward ratio calculation."""
        position = Position(
            position_id="pos_1",
            agent_id="agent_1",
            symbol="EURUSD",
            side="long",
            quantity=0.1,
            entry_price=1.1000,
            current_price=1.1050,
            stop_loss=1.0950,  # Risk: 0.005
            take_profit=1.1100  # Reward: 0.01
        )
        
        # Risk-reward ratio should be 0.01 / 0.005 = 2.0
        assert position.risk_reward_ratio == pytest.approx(2.0, abs=1e-6)
    
    def test_risk_reward_ratio_none(self):
        """Test risk-reward ratio when stop loss or take profit is missing."""
        position_no_sl = Position(
            position_id="pos_1",
            agent_id="agent_1",
            symbol="EURUSD",
            side="long",
            quantity=0.1,
            entry_price=1.1000,
            current_price=1.1050,
            take_profit=1.1100
        )
        
        position_no_tp = Position(
            position_id="pos_2",
            agent_id="agent_2",
            symbol="EURUSD",
            side="long",
            quantity=0.1,
            entry_price=1.1000,
            current_price=1.1050,
            stop_loss=1.0950
        )
        
        assert position_no_sl.risk_reward_ratio is None
        assert position_no_tp.risk_reward_ratio is None
    
    def test_risk_reward_ratio_zero_risk(self):
        """Test risk-reward ratio when risk distance is zero."""
        position = Position(
            position_id="pos_1",
            agent_id="agent_1",
            symbol="EURUSD",
            side="long",
            quantity=0.1,
            entry_price=1.1000,
            current_price=1.1050,
            stop_loss=1.1000,  # Risk: 0.0
            take_profit=1.1100  # Reward: 0.01
        )
        
        assert position.risk_reward_ratio is None


class TestPositionUpdates:
    """Test Position update methods."""
    
    def test_update_current_price(self):
        """Test updating current price."""
        position = Position(
            position_id="pos_1",
            agent_id="agent_1",
            symbol="EURUSD",
            side="long",
            quantity=0.1,
            entry_price=1.1000,
            current_price=1.1050
        )
        
        # Update price
        position.update_current_price(1.1100)
        
        assert position.current_price == 1.1100
        assert position.unrealized_pnl == pytest.approx(0.001, abs=1e-6)  # (1.11 - 1.1) * 0.1
    
    def test_update_current_price_negative(self):
        """Test updating current price with negative value."""
        position = Position(
            position_id="pos_1",
            agent_id="agent_1",
            symbol="EURUSD",
            side="long",
            quantity=0.1,
            entry_price=1.1000,
            current_price=1.1050
        )
        
        with pytest.raises(ValueError, match="Price must be positive"):
            position.update_current_price(-1.0)
    
    def test_is_stop_loss_hit_long(self):
        """Test stop loss hit detection for long position."""
        position = Position(
            position_id="pos_1",
            agent_id="agent_1",
            symbol="EURUSD",
            side="long",
            quantity=0.1,
            entry_price=1.1000,
            current_price=1.1050,
            stop_loss=1.0950
        )
        
        # Price above stop loss - not hit
        assert position.is_stop_loss_hit() == False
        
        # Price at stop loss - hit
        position.update_current_price(1.0950)
        assert position.is_stop_loss_hit() == True
        
        # Price below stop loss - hit
        position.update_current_price(1.0900)
        assert position.is_stop_loss_hit() == True
    
    def test_is_stop_loss_hit_short(self):
        """Test stop loss hit detection for short position."""
        position = Position(
            position_id="pos_2",
            agent_id="agent_2",
            symbol="EURUSD",
            side="short",
            quantity=0.1,
            entry_price=1.1000,
            current_price=1.0950,
            stop_loss=1.1050
        )
        
        # Price below stop loss - not hit
        assert position.is_stop_loss_hit() == False
        
        # Price at stop loss - hit
        position.update_current_price(1.1050)
        assert position.is_stop_loss_hit() == True
        
        # Price above stop loss - hit
        position.update_current_price(1.1100)
        assert position.is_stop_loss_hit() == True
    
    def test_is_stop_loss_hit_no_stop_loss(self):
        """Test stop loss hit detection when no stop loss is set."""
        position = Position(
            position_id="pos_1",
            agent_id="agent_1",
            symbol="EURUSD",
            side="long",
            quantity=0.1,
            entry_price=1.1000,
            current_price=1.1050
        )
        
        assert position.is_stop_loss_hit() == False
    
    def test_is_take_profit_hit_long(self):
        """Test take profit hit detection for long position."""
        position = Position(
            position_id="pos_1",
            agent_id="agent_1",
            symbol="EURUSD",
            side="long",
            quantity=0.1,
            entry_price=1.1000,
            current_price=1.1050,
            take_profit=1.1100
        )
        
        # Price below take profit - not hit
        assert position.is_take_profit_hit() == False
        
        # Price at take profit - hit
        position.update_current_price(1.1100)
        assert position.is_take_profit_hit() == True
        
        # Price above take profit - hit
        position.update_current_price(1.1150)
        assert position.is_take_profit_hit() == True
    
    def test_is_take_profit_hit_short(self):
        """Test take profit hit detection for short position."""
        position = Position(
            position_id="pos_2",
            agent_id="agent_2",
            symbol="EURUSD",
            side="short",
            quantity=0.1,
            entry_price=1.1000,
            current_price=1.0950,
            take_profit=1.0900
        )
        
        # Price above take profit - not hit
        assert position.is_take_profit_hit() == False
        
        # Price at take profit - hit
        position.update_current_price(1.0900)
        assert position.is_take_profit_hit() == True
        
        # Price below take profit - hit
        position.update_current_price(1.0850)
        assert position.is_take_profit_hit() == True
    
    def test_is_take_profit_hit_no_take_profit(self):
        """Test take profit hit detection when no take profit is set."""
        position = Position(
            position_id="pos_1",
            agent_id="agent_1",
            symbol="EURUSD",
            side="long",
            quantity=0.1,
            entry_price=1.1000,
            current_price=1.1050
        )
        
        assert position.is_take_profit_hit() == False
    
    def test_should_close(self):
        """Test should_close method."""
        position = Position(
            position_id="pos_1",
            agent_id="agent_1",
            symbol="EURUSD",
            side="long",
            quantity=0.1,
            entry_price=1.1000,
            current_price=1.1050,
            stop_loss=1.0950,
            take_profit=1.1100
        )
        
        # Price between stop loss and take profit - should not close
        assert position.should_close() == False
        
        # Price hits stop loss - should close
        position.update_current_price(1.0950)
        assert position.should_close() == True
        
        # Reset and hit take profit - should close
        position.update_current_price(1.1050)
        position.update_current_price(1.1100)
        assert position.should_close() == True


class TestPositionSerialization:
    """Test Position serialization methods."""
    
    def test_to_dict(self):
        """Test converting position to dictionary."""
        position = Position(
            position_id="pos_1",
            agent_id="agent_1",
            symbol="EURUSD",
            side="long",
            quantity=0.1,
            entry_price=1.1000,
            current_price=1.1050,
            stop_loss=1.0950,
            take_profit=1.1100,
            broker_id="broker_1",
            metadata={"strategy": "momentum"}
        )
        
        result = position.to_dict()
        
        assert result['position_id'] == "pos_1"
        assert result['agent_id'] == "agent_1"
        assert result['symbol'] == "EURUSD"
        assert result['side'] == "long"
        assert result['quantity'] == 0.1
        assert result['entry_price'] == 1.1000
        assert result['current_price'] == 1.1050
        assert result['stop_loss'] == 1.0950
        assert result['take_profit'] == 1.1100
        assert result['unrealized_pnl'] == pytest.approx(0.0005, abs=1e-6)
        assert result['broker_id'] == "broker_1"
        assert result['metadata'] == {"strategy": "momentum"}
        assert 'opened_at' in result
    
    def test_from_dict(self):
        """Test creating position from dictionary."""
        data = {
            'position_id': 'pos_1',
            'agent_id': 'agent_1',
            'symbol': 'EURUSD',
            'side': 'long',
            'quantity': 0.1,
            'entry_price': 1.1000,
            'current_price': 1.1050,
            'stop_loss': 1.0950,
            'take_profit': 1.1100,
            'broker_id': 'broker_1',
            'metadata': {'strategy': 'momentum'},
            'opened_at': '2024-01-01T12:00:00'
        }
        
        position = Position.from_dict(data)
        
        assert position.position_id == "pos_1"
        assert position.agent_id == "agent_1"
        assert position.symbol == "EURUSD"
        assert position.side == "long"
        assert position.quantity == 0.1
        assert position.entry_price == 1.1000
        assert position.current_price == 1.1050
        assert position.stop_loss == 1.0950
        assert position.take_profit == 1.1100
        assert position.broker_id == "broker_1"
        assert position.metadata == {"strategy": "momentum"}
        assert isinstance(position.opened_at, datetime)
    
    def test_from_dict_with_none_values(self):
        """Test creating position from dictionary with None values."""
        data = {
            'position_id': 'pos_1',
            'agent_id': 'agent_1',
            'symbol': 'EURUSD',
            'side': 'long',
            'quantity': 0.1,
            'entry_price': 1.1000,
            'current_price': 1.1050,
            'stop_loss': None,
            'take_profit': None,
            'broker_id': 'broker_1',
            'metadata': None,
            'opened_at': '2024-01-01T12:00:00'
        }
        
        position = Position.from_dict(data)
        
        assert position.position_id == "pos_1"
        assert position.stop_loss is None
        assert position.take_profit is None
        assert position.metadata == {}


class TestPositionStringRepresentation:
    """Test Position string representation methods."""
    
    def test_repr(self):
        """Test __repr__ method."""
        position = Position(
            position_id="pos_1",
            agent_id="agent_1",
            symbol="EURUSD",
            side="long",
            quantity=0.1,
            entry_price=1.1000,
            current_price=1.1050
        )
        
        repr_str = repr(position)
        
        assert "Position(" in repr_str
        assert "position_id=pos_1" in repr_str
        assert "agent_id=agent_1" in repr_str
        assert "symbol=EURUSD" in repr_str
        assert "side=long" in repr_str
        assert "quantity=0.1" in repr_str
        assert "entry_price=1.1" in repr_str
        assert "current_price=1.105" in repr_str
        assert "unrealized_pnl=0.00" in repr_str  # Rounded to 2 decimal places
    
    def test_str(self):
        """Test __str__ method."""
        position = Position(
            position_id="pos_1",
            agent_id="agent_1",
            symbol="EURUSD",
            side="long",
            quantity=0.1,
            entry_price=1.1000,
            current_price=1.1050
        )
        
        str_repr = str(position)
        
        assert "Position pos_1:" in str_repr
        assert "LONG 0.1 EURUSD" in str_repr
        assert "@ 1.1 -> 1.105" in str_repr
        assert "P&L: +0.00" in str_repr  # Rounded to 2 decimal places
        assert "%" in str_repr
    
    def test_str_negative_pnl(self):
        """Test __str__ method with negative P&L."""
        position = Position(
            position_id="pos_2",
            agent_id="agent_2",
            symbol="EURUSD",
            side="long",
            quantity=0.1,
            entry_price=1.1000,
            current_price=1.0950
        )
        
        str_repr = str(position)
        
        assert "P&L: -0.00" in str_repr  # Rounded to 2 decimal places

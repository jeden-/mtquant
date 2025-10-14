# Risk Management System

## Overview

The MTQuant Risk Management System provides a comprehensive 3-tier defense mechanism to protect trading capital and ensure regulatory compliance. The system operates at multiple levels to prevent catastrophic losses while allowing profitable trading strategies to operate effectively.

## Architecture

```
┌─────────────────────────────────────────────────────────────┐
│                    Risk Management System                   │
├─────────────────────────────────────────────────────────────┤
│  Tier 1: Pre-Trade Validation (<50ms)                      │
│  ├── Price Band Validation                                  │
│  ├── Position Size Limits                                   │
│  ├── Capital Verification                                   │
│  ├── Portfolio Exposure Check                               │
│  ├── Regulatory Compliance                                  │
│  └── Correlation Risk Assessment                            │
├─────────────────────────────────────────────────────────────┤
│  Tier 2: Position Sizing & Risk Calculation                │
│  ├── Kelly Criterion Method                                 │
│  ├── Volatility-Based Sizing                                │
│  ├── Fixed Fractional Method                                │
│  └── Dynamic Risk Adjustment                                │
├─────────────────────────────────────────────────────────────┤
│  Tier 3: Circuit Breaker System                            │
│  ├── Level 1: Warning (5% loss)                            │
│  ├── Level 2: Reduce Positions (10% loss)                  │
│  ├── Level 3: Full Halt (15% loss)                         │
│  └── Automatic Recovery & Reset                             │
└─────────────────────────────────────────────────────────────┘
```

## Components

### 1. PreTradeChecker

**Purpose**: Validates orders before execution to prevent invalid or dangerous trades.

**Key Features**:
- **Sub-50ms execution time** for real-time trading
- **Comprehensive validation** of all order parameters
- **Market condition awareness** with price band checks
- **Regulatory compliance** enforcement

**Validation Checks**:

```python
# Example usage
from mtquant.risk_management import PreTradeChecker

checker = PreTradeChecker()
result = await checker.validate(order, portfolio_state)

if result.is_valid:
    # Proceed with order execution
    pass
else:
    # Handle validation failure
    logger.warning(f"Order rejected: {result.reason}")
```

**Validation Rules**:

1. **Price Band Validation**
   - Current price within ±5% of last known price
   - Prevents execution on stale or erroneous data
   - Configurable tolerance per instrument

2. **Position Size Limits**
   - Maximum position size: 10% of portfolio equity
   - Minimum position size: 0.01 lots
   - Position size within broker limits

3. **Capital Verification**
   - Sufficient margin available
   - Free margin > required margin × 1.5
   - Account equity > minimum threshold

4. **Portfolio Exposure Check**
   - Total gross exposure < 150% of equity
   - Net exposure < 100% of equity
   - Per-instrument exposure < 15% of equity

5. **Regulatory Compliance**
   - Maximum leverage limits
   - Instrument-specific restrictions
   - Trading hours compliance
   - Position limits per instrument

6. **Correlation Risk Assessment**
   - Monitor correlation between positions
   - Reduce exposure if correlation > 0.7
   - Prevent over-concentration in correlated assets

### 2. PositionSizer

**Purpose**: Calculates optimal position sizes based on risk parameters and market conditions.

**Sizing Methods**:

#### Kelly Criterion Method
```python
# Optimal position sizing based on historical performance
kelly_pct = (win_rate * avg_win - (1 - win_rate) * avg_loss) / avg_loss
position_size = kelly_pct * 0.25 * portfolio_equity  # Fractional Kelly
```

**Advantages**:
- Mathematically optimal for known win rates
- Maximizes long-term growth
- Accounts for both wins and losses

**Disadvantages**:
- Requires accurate historical data
- Can be volatile with changing market conditions
- May suggest very large positions

#### Volatility-Based Sizing
```python
# Position size based on instrument volatility
risk_per_trade = 0.02  # 2% of portfolio
position_size = (risk_per_trade * portfolio_equity) / (atr * multiplier)
```

**Advantages**:
- Adapts to current market volatility
- Consistent risk per trade
- Works well with ATR (Average True Range)

**Disadvantages**:
- May under-size in low volatility
- Requires accurate volatility estimates
- Less responsive to changing conditions

#### Fixed Fractional Method
```python
# Fixed percentage of portfolio per trade
fraction = 0.02  # 2% of portfolio
position_size = fraction * portfolio_equity
```

**Advantages**:
- Simple and predictable
- Easy to understand and implement
- Consistent risk management

**Disadvantages**:
- Doesn't adapt to market conditions
- May over-size in volatile markets
- Doesn't consider instrument-specific risks

**Method Selection**:
The system automatically selects the best method based on:
- Portfolio history and performance
- Current market volatility
- Instrument characteristics
- Risk tolerance settings

### 3. CircuitBreaker

**Purpose**: Provides automatic trading halt mechanisms to prevent catastrophic losses.

**Circuit Breaker Levels**:

#### Level 1: Warning (5% Daily Loss)
```python
if daily_pnl_pct <= -5.0:
    circuit_breaker.activate_level_1()
    # Actions:
    # - Send warning alerts
    # - Reduce position sizes by 50%
    # - Increase monitoring frequency
    # - Log risk event
```

**Triggers**:
- Daily P&L ≤ -5% of starting equity
- Rapid loss acceleration (>2% in 1 hour)
- Multiple consecutive losing trades

**Actions**:
- Position size multiplier: 0.5
- Alert notifications to risk team
- Enhanced monitoring and logging
- Trading continues with reduced risk

#### Level 2: Reduce Positions (10% Daily Loss)
```python
if daily_pnl_pct <= -10.0:
    circuit_breaker.activate_level_2()
    # Actions:
    # - Halt new position opening
    # - Close 50% of existing positions
    # - Reduce position sizes by 75%
    # - Escalate to risk team
```

**Triggers**:
- Daily P&L ≤ -10% of starting equity
- Level 1 activation for >30 minutes
- Multiple risk violations

**Actions**:
- Position size multiplier: 0.25
- Halt new position opening
- Close 50% of existing positions
- Escalate to risk management team
- Trading continues with minimal risk

#### Level 3: Full Halt (15% Daily Loss)
```python
if daily_pnl_pct <= -15.0:
    circuit_breaker.activate_level_3()
    # Actions:
    # - Halt ALL trading
    # - Close ALL positions
    # - Lock system
    # - Require manual intervention
```

**Triggers**:
- Daily P&L ≤ -15% of starting equity
- Level 2 activation for >15 minutes
- System malfunction or data corruption

**Actions**:
- Position size multiplier: 0.0
- Halt ALL trading operations
- Close ALL existing positions
- Lock system for manual review
- Require risk team approval to resume

**Recovery Process**:
1. **Manual Review**: Risk team analyzes loss causes
2. **System Check**: Verify all components are functioning
3. **Configuration Review**: Check risk parameters and limits
4. **Gradual Restart**: Resume with reduced limits
5. **Monitoring**: Enhanced monitoring for 24 hours

## Configuration

### Risk Limits Configuration

Create `config/risk-limits.yaml`:

```yaml
# Risk Management Configuration
risk_limits:
  # Pre-trade validation limits
  pre_trade:
    max_position_size_pct: 0.10      # 10% of portfolio
    min_position_size: 0.01          # Minimum 0.01 lots
    max_gross_exposure_pct: 1.50     # 150% of equity
    max_net_exposure_pct: 1.00       # 100% of equity
    max_instrument_exposure_pct: 0.15 # 15% per instrument
    price_band_tolerance_pct: 0.05   # 5% price band
    
  # Position sizing parameters
  position_sizing:
    default_method: "volatility"     # kelly, volatility, fixed
    kelly_fraction: 0.25            # Fractional Kelly (25%)
    volatility_risk_pct: 0.02       # 2% risk per trade
    fixed_fraction_pct: 0.02        # 2% fixed fraction
    max_position_size_pct: 0.10     # Maximum 10% per trade
    
  # Circuit breaker thresholds
  circuit_breaker:
    level_1_threshold_pct: -5.0     # 5% daily loss
    level_2_threshold_pct: -10.0    # 10% daily loss
    level_3_threshold_pct: -15.0    # 15% daily loss
    cooldown_period_minutes: 60     # 1 hour cooldown
    recovery_threshold_pct: -2.0    # Recover at -2% loss
    
  # Correlation limits
  correlation:
    max_correlation: 0.7            # Maximum correlation
    correlation_lookback_days: 30   # 30-day correlation
    max_correlated_positions: 3     # Max correlated positions
    
  # Regulatory limits
  regulatory:
    max_leverage: 100               # Maximum leverage
    trading_hours_only: true        # Trade only during market hours
    max_daily_trades: 1000          # Maximum trades per day
    max_positions: 100              # Maximum open positions
```

### Environment Variables

```bash
# Risk Management Settings
RISK_MANAGEMENT_ENABLED=true
CIRCUIT_BREAKER_ENABLED=true
PRE_TRADE_VALIDATION_ENABLED=true
POSITION_SIZING_ENABLED=true

# Alert Settings
RISK_ALERT_EMAIL=risk@company.com
RISK_ALERT_WEBHOOK=https://hooks.slack.com/...
RISK_ALERT_THRESHOLD_PCT=5.0

# Logging
RISK_LOG_LEVEL=INFO
RISK_LOG_FILE=logs/risk_management.log
AUDIT_LOG_ENABLED=true
```

## Usage Examples

### Basic Risk Management Setup

```python
from mtquant.risk_management import PreTradeChecker, PositionSizer, CircuitBreaker

# Initialize components
pre_trade_checker = PreTradeChecker()
position_sizer = PositionSizer()
circuit_breaker = CircuitBreaker()

# Portfolio state
portfolio_state = {
    'equity': 100000.0,
    'daily_pnl': 0.0,
    'positions': [],
    'daily_trades': 0
}

# Order validation
order = Order(symbol='XAUUSD', side='buy', quantity=0.1, ...)
validation_result = await pre_trade_checker.validate(order, portfolio_state)

if validation_result.is_valid:
    # Calculate position size
    sizing_result = position_sizer.calculate(
        signal=0.5,
        portfolio_equity=portfolio_state['equity'],
        instrument_volatility=20.0,
        method='volatility'
    )
    
    # Check circuit breaker
    if circuit_breaker.is_trading_allowed():
        # Execute trade
        pass
    else:
        logger.warning("Trading halted by circuit breaker")
else:
    logger.warning(f"Order rejected: {validation_result.reason}")
```

### Advanced Risk Management

```python
# Custom risk limits
custom_limits = {
    'max_position_size_pct': 0.05,  # 5% instead of 10%
    'level_1_threshold_pct': -3.0,  # 3% instead of 5%
    'correlation_threshold': 0.5    # 0.5 instead of 0.7
}

# Initialize with custom limits
pre_trade_checker = PreTradeChecker(custom_limits)
circuit_breaker = CircuitBreaker(custom_limits)

# Monitor risk metrics
risk_metrics = {
    'daily_pnl_pct': -2.5,
    'portfolio_volatility': 0.15,
    'max_drawdown_pct': -8.0,
    'sharpe_ratio': 1.2,
    'win_rate': 0.55
}

# Update circuit breaker
circuit_breaker.update_daily_pnl(
    daily_pnl=risk_metrics['daily_pnl_pct'] * portfolio_state['equity'],
    current_equity=portfolio_state['equity']
)

# Get current status
status = circuit_breaker.get_status()
multiplier = circuit_breaker.get_position_size_multiplier()
```

## Monitoring and Alerts

### Risk Metrics Dashboard

The system provides real-time risk metrics:

- **Daily P&L**: Current daily profit/loss
- **Portfolio Exposure**: Gross and net exposure percentages
- **Position Count**: Number of open positions
- **Correlation Matrix**: Correlation between positions
- **Circuit Breaker Status**: Current circuit breaker level
- **Risk Violations**: Count of risk limit violations

### Alert System

```python
# Risk alert configuration
alerts = {
    'daily_loss_threshold': -5.0,    # Alert at 5% loss
    'position_limit_threshold': 0.9,  # Alert at 90% of limit
    'correlation_threshold': 0.8,     # Alert at 80% correlation
    'circuit_breaker_activation': True # Alert on circuit breaker
}

# Alert channels
channels = {
    'email': 'risk@company.com',
    'slack': 'https://hooks.slack.com/...',
    'sms': '+1234567890',
    'webhook': 'https://api.company.com/alerts'
}
```

### Audit Logging

All risk management decisions are logged for compliance:

```python
# Audit log entry
audit_entry = {
    'timestamp': '2024-01-15T10:30:00Z',
    'event_type': 'ORDER_REJECTED',
    'order_id': 'ORD-12345',
    'reason': 'Position size exceeds limit',
    'risk_metrics': {
        'portfolio_equity': 100000.0,
        'requested_size': 0.15,
        'max_allowed_size': 0.10
    },
    'user_id': 'system',
    'agent_id': 'XAUUSD_agent'
}
```

## Best Practices

### 1. Conservative Risk Limits
- Start with conservative limits (5% max position, 3% daily loss)
- Gradually increase limits as system proves stable
- Never exceed regulatory requirements

### 2. Regular Monitoring
- Monitor risk metrics every 15 minutes
- Review circuit breaker status daily
- Analyze risk violations weekly

### 3. Stress Testing
- Test circuit breaker activation regularly
- Simulate extreme market conditions
- Verify failover mechanisms work

### 4. Documentation
- Document all risk limit changes
- Maintain audit trail of decisions
- Regular risk management reviews

### 5. Team Training
- Train team on risk management procedures
- Establish escalation procedures
- Regular risk management drills

## Troubleshooting

### Common Issues

1. **Orders Rejected by Pre-Trade Checker**
   - Check position size limits
   - Verify sufficient margin
   - Review price band settings

2. **Circuit Breaker Activation**
   - Analyze loss causes
   - Check market conditions
   - Review risk parameters

3. **Position Sizing Issues**
   - Verify volatility calculations
   - Check Kelly Criterion inputs
   - Review method selection logic

### Debug Mode

Enable debug logging for detailed risk management information:

```python
import logging
logging.getLogger('mtquant.risk_management').setLevel(logging.DEBUG)
```

This will provide detailed information about:
- Validation decisions
- Position size calculations
- Circuit breaker status changes
- Risk metric updates

## Conclusion

The MTQuant Risk Management System provides comprehensive protection for trading operations while allowing profitable strategies to operate effectively. The 3-tier defense mechanism ensures that losses are contained while maintaining system flexibility and responsiveness to market conditions.

Regular monitoring, conservative limits, and proper configuration are essential for effective risk management. The system is designed to be both protective and flexible, allowing for adaptation to changing market conditions while maintaining strict risk controls.

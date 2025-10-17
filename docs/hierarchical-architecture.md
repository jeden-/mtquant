# Hierarchical Multi-Agent Trading Architecture

## Overview

The MTQuant hierarchical multi-agent system implements a sophisticated 3-level architecture for portfolio-level trading decisions. This document provides a comprehensive overview of the system architecture, components, and decision flow.

## Architecture Diagram

```
┌─────────────────────────────────────────────────────────────────┐
│                    HIERARCHICAL TRADING SYSTEM                  │
│                                                                 │
│  Level 1: META-CONTROLLER (Portfolio Manager)                  │
│  ┌─────────────────────────────────────────────────────────┐    │
│  │  • Portfolio-level decisions                            │    │
│  │  • Capital allocation to specialists                    │    │
│  │  • Risk appetite management                            │    │
│  │  • Market regime detection                             │    │
│  │  • 74-dimensional state space                          │    │
│  └──────────┬──────────────┬──────────────────────────────┘    │
│             │              │                                  │
│  Level 2: SPECIALISTS (Domain Experts)                        │
│  ┌──────────▼───┐  ┌──────▼────┐  ┌────────▼──┐              │
│  │   FOREX      │  │ COMMODITIES│  │  EQUITY   │              │
│  │  Specialist  │  │ Specialist │  │ Specialist│              │
│  │ (EUR,GBP,JPY)│  │ (XAU,WTIUSD)│  │(SPX500,NAS100,US30)│    │
│  └──────┬───────┘  └─────┬──────┘  └─────┬─────┘              │
│         │                │               │                    │
│  Level 3: INSTRUMENT AGENTS (Execution)                       │
│  ┌──────▼──────┐  ┌─────▼──────┐  ┌─────▼──────┐             │
│  │EURUSD│GBPUSD│  │XAUUSD│WTIUSD│  │SPX500│NAS100│            │
│  │      │USDJPY│  │      │      │  │      │US30  │            │
│  └─────────────┘  └────────────┘  └──────────────┘            │
│                                                                 │
│  PORTFOLIO RISK MANAGER (Cross-cutting)                       │
│  ┌─────────────────────────────────────────────────────────┐    │
│  │ • VaR monitoring • Correlation tracking                │    │
│  │ • Sector exposure • Margin requirements                │    │
│  │ • Circuit breakers • Position limits                   │    │
│  └─────────────────────────────────────────────────────────┘    │
│                                                                 │
│  COMMUNICATION HUB (Inter-agent messaging)                    │
│  ┌─────────────────────────────────────────────────────────┐    │
│  │ • Allocation messages • Performance reports            │    │
│  │ • Coordination signals • Alert broadcasting            │    │
│  └─────────────────────────────────────────────────────────┘    │
└─────────────────────────────────────────────────────────────────┘
```

## Core Components

### 1. Meta-Controller

**Purpose**: Portfolio-level decision maker that allocates capital and manages risk appetite.

**Key Features**:
- **State Space**: 74-dimensional portfolio state representation
- **Outputs**: 
  - Specialist allocations (3-dim softmax)
  - Risk appetite (0-1 continuous)
  - State value estimate
- **Architecture**: 3-layer neural network (256→128→outputs)

**State Components**:
```python
portfolio_state = [
    # Portfolio metrics (32 dims)
    portfolio_returns_30d,      # 30 values
    portfolio_volatility,       # 1 value
    current_drawdown,           # 1 value
    
    # Correlation matrix (28 dims)
    correlation_matrix_upper,   # 8x8 upper triangle
    
    # Specialist performance (9 dims)
    specialist_sharpes,         # 3 values
    specialist_win_rates,       # 3 values
    specialist_utilizations,    # 3 values
    
    # Macro indicators (5 dims)
    vix_level,                  # 1 value
    dxy_level,                  # 1 value
    interest_rates,             # 3 values
]
```

### 2. Specialists

**Purpose**: Domain experts that make instrument-specific trading decisions.

#### Forex Specialist
- **Instruments**: EURUSD, GBPUSD, USDJPY
- **Market Features**: DXY, interest rate spreads, carry trade indicators, FX volatility
- **Specialization**: Currency correlation, central bank policy, economic indicators

#### Commodities Specialist  
- **Instruments**: XAUUSD, WTIUSD
- **Market Features**: Inflation expectations, geopolitical risk, supply/demand indicators
- **Specialization**: Safe-haven flows, OPEC decisions, inventory reports

#### Equity Specialist
- **Instruments**: SPX500, NAS100, US30
- **Market Features**: P/E ratios, earnings season, Fed policy, market breadth
- **Specialization**: Sector rotation, earnings calendar, volatility regimes

### 3. Portfolio Risk Manager

**Purpose**: Multi-layer risk validation and portfolio-level limits.

**Risk Layers**:
1. **Portfolio VaR**: 2% daily VaR limit at 95% confidence
2. **Correlation Exposure**: Max 70% correlation concentration
3. **Sector Allocation**: Max 40% per asset class
4. **Margin Requirements**: Sufficient margin for all positions

**VaR Calculation Methods**:
- **Variance-Covariance**: Parametric method (default, ~5ms)
- **Historical Simulation**: Non-parametric method (~10ms)
- **Monte Carlo**: Stress testing method (~50ms)

### 4. Communication Hub

**Purpose**: Inter-agent messaging and coordination.

**Message Types**:
- **AllocationMessage**: Meta → Specialists (capital allocation)
- **PerformanceReport**: Specialists → Meta (performance metrics)
- **CoordinationSignal**: Specialist ↔ Specialist (hedge opportunities)

## Decision Flow

### 1. Market Observation
```python
# Meta-controller observes portfolio state
portfolio_state = get_portfolio_state(portfolio, specialists)

# Specialists observe market conditions
for specialist in specialists:
    market_state = get_domain_features(market_data, specialist.domain)
    instrument_states = get_instrument_observations(market_data, specialist.instruments)
```

### 2. Meta-Controller Decision
```python
# Meta-controller makes portfolio-level decisions
allocations, risk_appetite, value = meta_controller(portfolio_state)

# allocations: [forex_allocation, commodities_allocation, equity_allocation]
# risk_appetite: 0.0 (defensive) to 1.0 (aggressive)
```

### 3. Specialist Decisions
```python
# Each specialist makes instrument-specific decisions
for specialist_name, specialist in specialists.items():
    allocation = allocations[specialist_name]
    actions, confidence = specialist(market_state, instrument_states, allocation)
    
    # actions: {instrument: action_probabilities}
    # confidence: 0.0 to 1.0
```

### 4. Risk Validation
```python
# Portfolio risk manager validates all decisions
proposed_positions = generate_positions_from_actions(actions)
is_valid, reason = portfolio_risk_manager.check_portfolio_risk(
    proposed_positions, current_portfolio
)

if not is_valid:
    # Scale down positions or reject orders
    proposed_positions = scale_down_to_risk_limit(proposed_positions)
```

### 5. Order Execution
```python
# Generate and execute orders
orders = create_orders_from_positions(proposed_positions)
executed_orders = execute_orders(orders)

# Update portfolio and positions
update_portfolio(executed_orders)
```

## Training Pipeline

### Phase 1: Individual Specialist Training (500K timesteps)
- **Objective**: Train each specialist independently
- **Environment**: Single-instrument trading environments
- **Reward**: Instrument-specific Sharpe ratio
- **Duration**: ~16 hours

### Phase 2: Meta-Controller Pre-training (300K timesteps)
- **Objective**: Train meta-controller with fixed specialists
- **Environment**: Portfolio-level environment
- **Reward**: Portfolio Sharpe ratio
- **Duration**: ~12 hours

### Phase 3: Joint Fine-tuning (1M timesteps)
- **Objective**: Joint optimization of all components
- **Environment**: Full hierarchical environment
- **Reward**: Risk-adjusted portfolio performance
- **Duration**: ~20 hours

## Performance Requirements

| **Metric** | **Target** | **Measurement** |
|------------|------------|-----------------|
| **Decision Latency** | <100ms | End-to-end decision time |
| **Meta-Controller Forward** | <10ms | Single forward pass |
| **Specialist Forward** | <5ms | Per specialist |
| **VaR Calculation** | <10ms | Portfolio VaR |
| **Memory Usage** | <500MB | System memory |
| **Throughput** | >1000 decisions/s | Batch processing |

## Risk Management

### Pre-Trade Validation
- Price band validation (±5-10% from last known)
- Position size limits (<5% Average Daily Volume)
- Capital verification (sufficient margin)
- Regulatory compliance (max leverage, instrument restrictions)

### Intra-Trade Monitoring
- Dynamic stop-loss adjustment
- P&L tracking and alerts
- Correlation monitoring
- Position concentration limits

### Circuit Breakers
- **Level 1** (5% daily loss): Warning alerts, reduce position sizes
- **Level 2** (10% daily loss): Halt new positions, close risky positions  
- **Level 3** (15-20% daily loss): Full trading halt, flatten all positions

## Configuration

### Agent Configuration (`config/agents.yaml`)
```yaml
meta_controller:
  state_dim: 74
  hidden_dim: 256
  hidden_dim_2: 128
  dropout: 0.2
  learning_rate: 0.0003

specialists:
  forex:
    type: ForexSpecialist
    instruments: [EURUSD, GBPUSD, USDJPY]
    market_features_dim: 8
    observation_dim: 50
    hidden_dim: 64
  
  commodities:
    type: CommoditiesSpecialist
    instruments: [XAUUSD, WTIUSD]
    market_features_dim: 6
    observation_dim: 50
    hidden_dim: 64
  
  equity:
    type: EquitySpecialist
    instruments: [SPX500, NAS100, US30]
    market_features_dim: 7
    observation_dim: 50
    hidden_dim: 64

portfolio_risk:
  max_portfolio_var: 0.02
  max_correlation_exposure: 0.7
  max_sector_allocation: 0.4
  var_confidence: 0.95
  correlation_window: 100

training:
  phase_1_timesteps: 500000
  phase_2_timesteps: 300000
  phase_3_timesteps: 1000000
  batch_size: 256
  n_epochs: 10
  gamma: 0.99
  gae_lambda: 0.95
  clip_range: 0.2
```

## Usage Examples

### Basic System Usage
```python
from mtquant.agents.hierarchical.hierarchical_system import HierarchicalTradingSystem
from mtquant.agents.hierarchical.specialist_factory import SpecialistRegistry

# Create system
registry = SpecialistRegistry()
specialists = {
    'forex': registry.create_specialist('forex', forex_config),
    'commodities': registry.create_specialist('commodities', commodities_config),
    'equity': registry.create_specialist('equity', equity_config)
}

meta_controller = MetaController(state_dim=74, hidden_dim=256, hidden_dim_2=128)
portfolio_risk_manager = PortfolioRiskManager()
communication_hub = CommunicationHub()

system = HierarchicalTradingSystem(
    meta_controller=meta_controller,
    specialists=specialists,
    portfolio_risk_manager=portfolio_risk_manager,
    communication_hub=communication_hub
)

# Execute decision cycle
orders = system.step(market_data, portfolio, current_positions)
```

### Training Pipeline Usage
```python
from mtquant.agents.training.phase3_joint_training import create_phase3_trainer

# Create trainer
trainer = create_phase3_trainer(
    config_path="config/agents.yaml",
    market_data=market_data
)

# Train the system
results = trainer.train()

# Evaluate the trained system
eval_results = trainer.evaluate(n_episodes=100)
```

### Command Line Usage
```bash
# Train complete pipeline
python scripts/training_pipeline.py --mode train

# Train specific phase
python scripts/training_pipeline.py --mode train --phase phase1

# Resume training
python scripts/training_pipeline.py --mode resume

# Evaluate trained model
python scripts/training_pipeline.py --mode eval --eval-episodes 200
```

## Monitoring and Debugging

### Training Monitoring
- **TensorBoard**: Real-time training metrics
- **Logging**: Comprehensive training logs
- **Checkpointing**: Model versioning and recovery
- **Evaluation**: Periodic performance evaluation

### Runtime Monitoring
- **Decision Latency**: End-to-end timing
- **Memory Usage**: System resource monitoring
- **Risk Metrics**: VaR, correlation, drawdown tracking
- **Performance Metrics**: Sharpe ratio, win rate, P&L

### Debugging Tools
- **Message Tracing**: Inter-agent communication logs
- **State Inspection**: Portfolio and market state visualization
- **Decision Analysis**: Meta-controller and specialist decision breakdown
- **Risk Analysis**: Risk validation and limit tracking

## Extensibility

### Adding New Specialists
1. Create new specialist class inheriting from `BaseSpecialist`
2. Implement required methods: `forward()`, `get_instruments()`, `get_domain_features()`
3. Register in `SpecialistRegistry`
4. Update configuration files

### Adding New Instruments
1. Add instrument to `config/symbols.yaml`
2. Update specialist configurations
3. Add market data processing
4. Update risk management parameters

### Custom Risk Rules
1. Extend `PortfolioRiskManager`
2. Implement custom risk validation methods
3. Update configuration parameters
4. Add monitoring and alerting

## Best Practices

### Development
- Always test individual components before integration
- Use comprehensive unit tests for each specialist
- Validate risk management rules thoroughly
- Monitor training progress and adjust hyperparameters

### Production
- Start with paper trading validation
- Gradual capital allocation (10% → 50% → 100%)
- Continuous monitoring and alerting
- Regular model retraining and updates

### Risk Management
- Never exceed VaR limits
- Monitor correlation exposure continuously
- Implement circuit breakers at multiple levels
- Maintain comprehensive audit trails

## Troubleshooting

### Common Issues
1. **High Latency**: Check batch sizes, optimize forward passes
2. **Memory Issues**: Reduce batch sizes, use gradient checkpointing
3. **Training Instability**: Adjust learning rates, add gradient clipping
4. **Risk Violations**: Review risk limits, check position sizing

### Performance Optimization
1. **GPU Utilization**: Ensure >80% GPU usage during training
2. **Parallel Processing**: Use multiple environments for training
3. **Memory Management**: Monitor memory usage, optimize data loading
4. **Network Architecture**: Balance model complexity vs. performance

## Future Enhancements

### Planned Features
- **Online Learning**: Continuous model updates during trading
- **Multi-Broker Support**: Distribute positions across brokers
- **Advanced Risk Models**: GARCH, regime-switching models
- **Alternative Data**: News sentiment, satellite data, social media

### Research Directions
- **Hierarchical Attention**: Attention mechanisms for specialist coordination
- **Meta-Learning**: Fast adaptation to new market regimes
- **Multi-Objective Optimization**: Balance returns, risk, and transaction costs
- **Explainable AI**: Interpretable decision making for regulatory compliance


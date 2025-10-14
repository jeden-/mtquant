# Reinforcement Learning Agents

## Overview

The MTQuant RL Agent system uses Proximal Policy Optimization (PPO) to train intelligent trading agents that can adapt to changing market conditions. Each agent is specialized for a specific instrument (XAUUSD, BTCUSD, etc.) and learns optimal trading strategies through interaction with a realistic trading environment.

## Architecture

```
┌─────────────────────────────────────────────────────────────┐
│                    RL Agent Architecture                    │
├─────────────────────────────────────────────────────────────┤
│  Agent Layer                                                │
│  ├── PPO Model (Stable-Baselines3)                         │
│  ├── Policy Network (Actor-Critic)                         │
│  ├── Value Network (State Value Estimation)                │
│  └── Experience Replay Buffer                              │
├─────────────────────────────────────────────────────────────┤
│  Environment Layer                                          │
│  ├── MTQuantTradingEnv (Gymnasium Compatible)              │
│  ├── State Space (Market + Position + Risk Features)       │
│  ├── Action Space (Continuous: -1 to +1)                   │
│  ├── Reward Function (Risk-Adjusted Returns)               │
│  └── Episode Management (Reset, Step, Done)                │
├─────────────────────────────────────────────────────────────┤
│  Data Layer                                                 │
│  ├── Feature Engineering (Technical Indicators)            │
│  ├── Market Data (OHLCV + Ticks)                           │
│  ├── Position Data (Current Holdings)                      │
│  └── Risk Data (Portfolio Metrics)                         │
└─────────────────────────────────────────────────────────────┘
```

## Agent Design

### State Space Design

The agent's state space is designed to be **stationary** and **informative**, providing all necessary information for trading decisions without look-ahead bias.

#### Market Features (Normalized)
```python
# Technical Indicators (0-1 normalized)
rsi = normalize(market_data['rsi'], 0, 100)           # RSI: 0-100 → 0-1
macd = normalize(market_data['macd'], -5, 5)          # MACD: -5 to 5 → 0-1
bb_position = normalize(market_data['bb_position'], 0, 1)  # Bollinger position
sma_ratio = market_data['close'] / market_data['sma_20']   # Price vs SMA ratio

# Log Returns (Stationary)
log_returns_1m = np.log(close / close.shift(1))       # 1-minute returns
log_returns_5m = np.log(close / close.shift(5))       # 5-minute returns
log_returns_15m = np.log(close / close.shift(15))     # 15-minute returns

# Volatility Measures
atr_normalized = normalize(market_data['atr'], 0, 50) # ATR normalized
volatility = market_data['returns'].rolling(20).std() # Rolling volatility
```

#### Position Features
```python
# Current Position State
holdings_ratio = current_position / max_position_size  # Position size ratio
unrealized_pnl_pct = unrealized_pnl / portfolio_equity # P&L percentage
position_age = min(position_duration_hours / 24, 1.0)  # Position age (0-1)

# Position History
recent_trades = get_recent_trades_count(24)            # Trades in last 24h
avg_trade_duration = get_avg_trade_duration()          # Average trade duration
```

#### Risk Features
```python
# Portfolio Risk Metrics
portfolio_volatility = calculate_portfolio_volatility() # Portfolio volatility
current_drawdown = portfolio.drawdown_pct              # Current drawdown
daily_pnl_pct = daily_pnl / portfolio_equity          # Daily P&L percentage

# Market Risk
correlation_risk = calculate_correlation_risk()        # Correlation with other positions
liquidity_risk = calculate_liquidity_risk()           # Market liquidity risk
```

### Action Space

The agent uses a **continuous action space** from -1 to +1:

```python
# Action Interpretation
action_value = model.predict(state)  # Returns value in [-1, 1]

if abs(action_value) > action_threshold:  # Default: 0.02
    if action_value > 0:
        # Buy signal - open long position
        target_position = calculate_position_size(action_value)
    else:
        # Sell signal - open short position  
        target_position = -calculate_position_size(abs(action_value))
else:
    # Hold signal - maintain current position
    target_position = current_position
```

**Action Thresholds**:
- **Minimum Action Threshold**: 0.02 (2% signal strength required)
- **Minimum Position Change**: 0.001 (0.1% position change required)
- **Maximum Position Size**: 10% of portfolio equity

### Reward Function

The reward function is designed to encourage **risk-adjusted returns** while penalizing excessive trading and losses.

#### Base Reward Structure
```python
def calculate_reward(self, trade_executed, trade_pnl, transaction_cost):
    reward = 0.0
    
    if trade_executed:
        # Base reward for executing a trade
        reward += 1.0
        
        # P&L-based reward (scaled appropriately)
        if trade_pnl != 0.0:
            net_pnl = trade_pnl - transaction_cost
            if self.current_capital > 0:
                # Scale P&L reward (more conservative)
                pnl_reward = (net_pnl / self.current_capital) * 100
                reward += pnl_reward
                
                # Bonus for profitable trades
                if net_pnl > 0:
                    reward += 2.0
                # Penalty for losing trades
                else:
                    reward -= 1.0
    
    # Penalty for inaction (encourage trading)
    if not trade_executed:
        reward -= 0.1
    
    return reward
```

#### Risk-Adjusted Rewards
```python
# Sharpe Ratio Component
if len(returns_history) > 20:
    sharpe_ratio = np.mean(returns_history) / np.std(returns_history)
    reward += sharpe_ratio * 10  # Scale Sharpe ratio

# Drawdown Penalty
if current_drawdown < -0.05:  # 5% drawdown
    reward -= abs(current_drawdown) * 100

# Transaction Cost Penalty
transaction_penalty = transaction_cost * 1000  # Scale up penalty
reward -= transaction_penalty
```

## Training Process

### Environment Setup

```python
from mtquant.agents.training.train_ppo import create_env, prepare_data

# Prepare training data
training_data = prepare_data(
    symbol='XAUUSD',
    start_date=None,  # Use default date range
    seed=42
)

# Create training environment
env = create_env(
    data=training_data,
    config={'ppo_agent': {
        'initial_capital': 10000,
        'transaction_cost': 0.001
    }},
    symbol='XAUUSD'
)
```

### PPO Configuration

```python
# PPO Hyperparameters (config/agents.yaml)
ppo_config = {
    'learning_rate': 0.0001,      # Lower for stability
    'n_steps': 8192,              # Larger batch size
    'batch_size': 256,            # Larger batch size
    'n_epochs': 10,               # Multiple epochs per update
    'gamma': 0.999,               # High discount factor
    'gae_lambda': 0.95,           # GAE parameter
    'clip_range': 0.2,            # PPO clip range
    'ent_coef': 0.01,             # Entropy coefficient
    'vf_coef': 0.5,               # Value function coefficient
    'max_grad_norm': 0.5,         # Gradient clipping
    'policy_kwargs': {
        'net_arch': [512, 256, 128, 64],  # Network architecture
        'activation_fn': 'tanh'            # Activation function
    }
}
```

### Training Loop

```python
from stable_baselines3 import PPO
from stable_baselines3.common.callbacks import EvalCallback

# Create PPO model
model = PPO(
    'MlpPolicy',
    env,
    learning_rate=0.0001,
    n_steps=8192,
    batch_size=256,
    n_epochs=10,
    gamma=0.999,
    verbose=1
)

# Setup evaluation callback
eval_callback = EvalCallback(
    eval_env,
    best_model_save_path='models/checkpoints/',
    log_path='logs/',
    eval_freq=25000,
    deterministic=True,
    render=False
)

# Train model
model.learn(
    total_timesteps=500000,
    callback=eval_callback,
    log_interval=5000
)

# Save final model
model.save('models/checkpoints/XAUUSD_ppo_final.zip')
```

### Training Metrics

The training process tracks several key metrics:

#### Episode Metrics
```python
@dataclass
class EpisodeMetrics:
    total_trades: int = 0
    winning_trades: int = 0
    losing_trades: int = 0
    total_pnl: float = 0.0
    win_rate: float = 0.0
    avg_win: float = 0.0
    avg_loss: float = 0.0
    profit_factor: float = 0.0
    sharpe_ratio: float = 0.0
    max_drawdown: float = 0.0
    avg_trade_duration: float = 0.0
```

#### Training Metrics
- **Episode Reward**: Cumulative reward per episode
- **Episode Length**: Number of steps per episode
- **Learning Rate**: Current learning rate
- **Policy Loss**: PPO policy loss
- **Value Loss**: Value function loss
- **Entropy**: Policy entropy (exploration measure)

### Evaluation Process

```python
from mtquant.agents.training.train_ppo import evaluate_agent

# Evaluate trained model
results = evaluate_agent(
    model=model,
    test_env=test_env,
    n_episodes=10
)

# Results structure
{
    'n_episodes': 10,
    'mean_reward': 1847.23,
    'std_reward': 234.56,
    'mean_sharpe_ratio': 0.054,
    'mean_win_rate': 0.5205,
    'episode_rewards': [1847.23, 1923.45, ...],
    'episode_metrics': [
        {
            'total_trades': 1689,
            'winning_trades': 879,
            'losing_trades': 810,
            'total_pnl': 2468.46,
            'win_rate': 0.5205,
            'sharpe_ratio': 0.054
        },
        ...
    ]
}
```

## Model Architecture

### Policy Network (Actor)

```python
# Policy Network Architecture
PolicyNetwork(
    input_size=state_dim,           # State space dimension
    hidden_layers=[512, 256, 128, 64],  # Hidden layer sizes
    output_size=1,                  # Single continuous action
    activation='tanh',              # Activation function
    output_activation='tanh'        # Output activation (-1 to 1)
)
```

### Value Network (Critic)

```python
# Value Network Architecture  
ValueNetwork(
    input_size=state_dim,           # Same as policy input
    hidden_layers=[512, 256, 128, 64],  # Same as policy
    output_size=1,                  # Single value estimate
    activation='tanh'               # Activation function
)
```

### Network Initialization

```python
# Xavier/Glorot initialization for stable training
def init_weights(m):
    if isinstance(m, nn.Linear):
        torch.nn.init.xavier_uniform_(m.weight)
        m.bias.data.fill_(0.01)

policy_net.apply(init_weights)
value_net.apply(init_weights)
```

## Feature Engineering

### Technical Indicators

```python
from mtquant.data.processors.feature_engineering import add_technical_indicators

# Add technical indicators to market data
data_with_indicators = add_technical_indicators(data)

# Available indicators:
indicators = {
    'rsi': calculate_rsi(data, period=14),
    'macd': calculate_macd(data, fast=12, slow=26, signal=9),
    'bb_upper': calculate_bollinger_upper(data, period=20, std=2),
    'bb_lower': calculate_bollinger_lower(data, period=20, std=2),
    'bb_position': calculate_bollinger_position(data, period=20, std=2),
    'sma_20': calculate_sma(data, period=20),
    'sma_50': calculate_sma(data, period=50),
    'atr': calculate_atr(data, period=14),
    'stoch_k': calculate_stochastic_k(data, period=14),
    'stoch_d': calculate_stochastic_d(data, period=14)
}
```

### Data Preprocessing

```python
# Log returns (stationary)
data['log_returns'] = np.log(data['close'] / data['close'].shift(1))

# Normalization (0-1 range)
def normalize_series(series, min_val, max_val):
    return (series - min_val) / (max_val - min_val)

data['rsi_norm'] = normalize_series(data['rsi'], 0, 100)
data['macd_norm'] = normalize_series(data['macd'], -5, 5)

# Handle NaN values
data = data.fillna(method='ffill').fillna(0)
```

### Feature Selection

```python
# Feature importance analysis
feature_importance = {
    'log_returns_1m': 0.15,
    'rsi': 0.12,
    'macd': 0.10,
    'bb_position': 0.08,
    'atr': 0.07,
    'position_ratio': 0.06,
    'unrealized_pnl_pct': 0.05,
    'portfolio_volatility': 0.04,
    # ... other features
}

# Select top features
selected_features = [
    'log_returns_1m', 'rsi', 'macd', 'bb_position', 'atr',
    'position_ratio', 'unrealized_pnl_pct', 'portfolio_volatility'
]
```

## Training Best Practices

### 1. Data Quality
- Use high-quality, clean market data
- Remove outliers and handle missing values
- Ensure sufficient data volume (minimum 10,000 bars)
- Use diverse market conditions (trending, ranging, volatile)

### 2. Hyperparameter Tuning
- Start with conservative learning rates (0.0001)
- Use larger batch sizes for stability (256-512)
- Adjust entropy coefficient for exploration (0.01-0.1)
- Monitor training metrics for overfitting

### 3. Environment Design
- Design realistic transaction costs
- Use appropriate position sizing
- Implement proper episode termination
- Balance reward function components

### 4. Training Monitoring
- Monitor episode rewards and lengths
- Track policy and value losses
- Watch for overfitting (training vs validation)
- Use early stopping if performance plateaus

### 5. Evaluation
- Use separate test data for evaluation
- Evaluate on multiple episodes (10-20)
- Check for consistent performance
- Validate on different market conditions

## Model Deployment

### Model Loading

```python
from stable_baselines3 import PPO

# Load trained model
model = PPO.load('models/checkpoints/XAUUSD_ppo_final.zip')

# Create inference environment
inference_env = create_env(test_data, config, 'XAUUSD')

# Make predictions
obs, info = inference_env.reset()
action, _ = model.predict(obs, deterministic=True)
```

### Real-Time Inference

```python
class TradingAgent:
    def __init__(self, model_path, symbol):
        self.model = PPO.load(model_path)
        self.symbol = symbol
        self.env = create_inference_env(symbol)
    
    def get_signal(self, market_data):
        # Update environment with new data
        self.env.update_data(market_data)
        
        # Get current state
        obs, info = self.env.get_current_state()
        
        # Predict action
        action, _ = self.model.predict(obs, deterministic=False)
        
        return float(action[0])
    
    def should_trade(self, signal, threshold=0.02):
        return abs(signal) > threshold
```

### Integration with Risk Management

```python
# Agent signal processing with risk management
def process_agent_signal(agent, market_data, portfolio_state):
    # Get agent signal
    signal = agent.get_signal(market_data)
    
    # Risk management validation
    if abs(signal) > 0.02:  # Significant signal
        # Create order
        order = create_order_from_signal(signal, market_data)
        
        # Pre-trade validation
        validation_result = await pre_trade_checker.validate(order, portfolio_state)
        
        if validation_result.is_valid:
            # Position sizing
            sizing_result = position_sizer.calculate(
                signal=signal,
                portfolio_equity=portfolio_state['equity'],
                instrument_volatility=market_data['atr']
            )
            
            # Execute trade
            if circuit_breaker.is_trading_allowed():
                order_id = await broker_manager.place_order(order)
                return order_id
    
    return None
```

## Performance Optimization

### Training Optimization

```python
# Use GPU if available
import torch
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Parallel environments for faster training
from stable_baselines3.common.vec_env import DummyVecEnv, SubprocVecEnv

def make_env():
    return create_env(training_data, config, 'XAUUSD')

# Create vectorized environment
env = SubprocVecEnv([make_env for _ in range(4)])  # 4 parallel environments

# Train with vectorized environment
model = PPO('MlpPolicy', env, ...)
model.learn(total_timesteps=500000)
```

### Inference Optimization

```python
# Batch inference for multiple symbols
def batch_inference(models, market_data_batch):
    signals = {}
    
    for symbol, data in market_data_batch.items():
        if symbol in models:
            model = models[symbol]
            signal = model.predict(data, deterministic=True)
            signals[symbol] = signal
    
    return signals

# Model quantization for faster inference
import torch.quantization as quantization

quantized_model = quantization.quantize_dynamic(
    model, {torch.nn.Linear}, dtype=torch.qint8
)
```

## Troubleshooting

### Common Training Issues

1. **Poor Performance**
   - Check data quality and preprocessing
   - Adjust reward function weights
   - Increase training time
   - Try different hyperparameters

2. **Overfitting**
   - Reduce model complexity
   - Increase regularization
   - Use more diverse training data
   - Implement early stopping

3. **Unstable Training**
   - Reduce learning rate
   - Increase batch size
   - Adjust clip range
   - Check gradient clipping

4. **No Trading Activity**
   - Lower action thresholds
   - Adjust reward function
   - Check environment setup
   - Verify data preprocessing

### Debug Mode

```python
# Enable debug logging
import logging
logging.getLogger('mtquant.agents').setLevel(logging.DEBUG)

# Debug environment
env = create_env(data, config, 'XAUUSD', debug=True)

# Debug model predictions
action, state = model.predict(obs, deterministic=True)
print(f"Action: {action}, State: {state}")
```

## Conclusion

The MTQuant RL Agent system provides a robust framework for training intelligent trading agents. The combination of PPO algorithm, realistic trading environment, and comprehensive risk management creates agents capable of adapting to changing market conditions while maintaining strict risk controls.

Key success factors include:
- High-quality, diverse training data
- Well-designed reward function
- Proper hyperparameter tuning
- Comprehensive evaluation
- Integration with risk management

Regular monitoring and retraining ensure agents remain effective as market conditions change.

# Training Pipeline Documentation

## Overview

The MTQuant hierarchical multi-agent training pipeline implements a sophisticated 3-phase training strategy that progressively builds from individual specialist training to joint optimization. This document provides comprehensive guidance on training setup, execution, and monitoring.

## Training Strategy

### Phase 1: Individual Specialist Training
**Objective**: Train each specialist independently on their domain-specific instruments.

**Key Features**:
- **Duration**: 500,000 timesteps (~16 hours)
- **Environment**: Single-instrument trading environments
- **Reward**: Instrument-specific Sharpe ratio
- **Parallel Training**: All specialists trained simultaneously

**Training Process**:
1. Create individual environments for each instrument
2. Train each specialist using PPO algorithm
3. Optimize for instrument-specific performance
4. Save specialist models for Phase 2

### Phase 2: Meta-Controller Pre-training
**Objective**: Train meta-controller to make portfolio-level decisions using fixed specialists.

**Key Features**:
- **Duration**: 300,000 timesteps (~12 hours)
- **Environment**: Portfolio-level environment
- **Reward**: Portfolio Sharpe ratio
- **Fixed Specialists**: Use pre-trained specialists from Phase 1

**Training Process**:
1. Load pre-trained specialist models
2. Create portfolio-level environment
3. Train meta-controller with fixed specialists
4. Optimize for portfolio-level performance

### Phase 3: Joint Fine-tuning
**Objective**: Joint optimization of meta-controller and specialists for optimal portfolio performance.

**Key Features**:
- **Duration**: 1,000,000 timesteps (~20 hours)
- **Environment**: Full hierarchical environment
- **Reward**: Risk-adjusted portfolio performance
- **Joint Optimization**: All components trained together

**Training Process**:
1. Load pre-trained meta-controller and specialists
2. Create joint training environment
3. Coordinate gradient updates between components
4. Fine-tune for optimal coordination

## Training Components

### 1. Phase1Trainer
**Location**: `mtquant/agents/training/phase1_trainer.py`

**Responsibilities**:
- Individual specialist training
- Environment creation and management
- PPO algorithm implementation
- Model checkpointing and evaluation

**Key Methods**:
```python
class Phase1Trainer:
    def train(self) -> Dict[str, Any]:
        """Train all specialists individually."""
    
    def train_specialist(self, specialist_name: str) -> Dict[str, Any]:
        """Train a single specialist."""
    
    def evaluate_specialist(self, specialist_name: str, n_episodes: int = 100) -> Dict[str, Any]:
        """Evaluate a trained specialist."""
```

### 2. Phase2Trainer
**Location**: `mtquant/agents/training/phase2_trainer.py`

**Responsibilities**:
- Meta-controller pre-training
- Portfolio-level environment management
- Specialist integration
- Portfolio reward optimization

**Key Methods**:
```python
class Phase2Trainer:
    def train(self) -> Dict[str, Any]:
        """Train meta-controller with fixed specialists."""
    
    def load_pretrained_specialists(self) -> None:
        """Load Phase 1 specialist models."""
    
    def create_portfolio_environment(self) -> MetaControllerTrainingEnv:
        """Create portfolio-level training environment."""
```

### 3. Phase3JointTrainer
**Location**: `mtquant/agents/training/phase3_joint_training.py`

**Responsibilities**:
- Joint fine-tuning coordination
- Gradient coordination between components
- Curriculum learning implementation
- Comprehensive monitoring and evaluation

**Key Methods**:
```python
class Phase3JointTrainer:
    def train(self) -> Dict[str, Any]:
        """Execute joint training of all components."""
    
    def load_pretrained_models(self) -> None:
        """Load Phase 1 and Phase 2 models."""
    
    def create_joint_training_env(self) -> JointTrainingEnv:
        """Create joint training environment."""
    
    def evaluate(self, n_episodes: int = 100) -> Dict[str, Any]:
        """Evaluate the trained hierarchical system."""
```

## Training Environments

### 1. Specialist Environments
**Purpose**: Individual instrument trading environments for Phase 1.

**Features**:
- Single instrument focus
- Instrument-specific reward functions
- Technical indicator integration
- Risk management rules

**Configuration**:
```python
env_config = {
    'instrument': 'EURUSD',
    'episode_length': 1000,
    'initial_capital': 100000.0,
    'transaction_cost': 0.003,
    'max_position_size': 0.1,
    'stop_loss_pct': 0.02
}
```

### 2. Meta-Controller Environment
**Purpose**: Portfolio-level environment for Phase 2.

**Features**:
- Multi-instrument portfolio management
- Portfolio-level reward functions
- Specialist coordination
- Risk management integration

**Configuration**:
```python
env_config = {
    'instruments': ['EURUSD', 'GBPUSD', 'USDJPY', 'XAUUSD', 'WTIUSD', 'SPX500', 'NAS100', 'US30'],
    'episode_length': 1000,
    'initial_capital': 100000.0,
    'max_portfolio_var': 0.02,
    'max_correlation_exposure': 0.7
}
```

### 3. Joint Training Environment
**Purpose**: Full hierarchical environment for Phase 3.

**Features**:
- Complete hierarchical system integration
- Joint reward optimization
- Risk-aware decision making
- Communication protocol testing

**Configuration**:
```python
env_config = {
    'instruments': ['EURUSD', 'GBPUSD', 'USDJPY', 'XAUUSD', 'WTIUSD', 'SPX500', 'NAS100', 'US30'],
    'episode_length': 1000,
    'initial_capital': 100000.0,
    'max_portfolio_var': 0.02,
    'max_correlation_exposure': 0.7,
    'gradient_coordination': True,
    'curriculum_learning': True
}
```

## Training Scripts

### 1. Individual Phase Scripts

#### Phase 1 Training
```bash
python scripts/run_phase1_training.py \
    --config config/agents.yaml \
    --timesteps 500000 \
    --output-dir models/checkpoints/phase1 \
    --log-dir logs/phase1 \
    --device cuda
```

#### Phase 2 Training
```bash
python scripts/run_phase2_training.py \
    --config config/agents.yaml \
    --timesteps 300000 \
    --output-dir models/checkpoints/phase2 \
    --log-dir logs/phase2 \
    --device cuda
```

#### Phase 3 Training
```bash
python scripts/run_phase3_training.py \
    --config config/agents.yaml \
    --timesteps 1000000 \
    --output-dir models/checkpoints/phase3 \
    --log-dir logs/phase3 \
    --device cuda \
    --eval-episodes 100
```

### 2. Complete Pipeline Script
```bash
# Train complete pipeline
python scripts/training_pipeline.py --mode train

# Train with custom configuration
python scripts/training_pipeline.py \
    --mode train \
    --config config/agents.yaml \
    --output-dir models/checkpoints \
    --log-dir logs \
    --device cuda

# Resume training from last completed phase
python scripts/training_pipeline.py --mode resume

# Evaluate trained model
python scripts/training_pipeline.py \
    --mode eval \
    --eval-episodes 200
```

### 3. End-to-End Training Script
```bash
python scripts/run_end_to_end_training.py \
    --config config/agents.yaml \
    --output-dir models/checkpoints \
    --log-dir logs \
    --device cuda \
    --phase1-timesteps 500000 \
    --phase2-timesteps 300000 \
    --phase3-timesteps 1000000
```

## Training Configuration

### Hyperparameters

#### PPO Algorithm
```yaml
training:
  # PPO hyperparameters
  learning_rate: 0.0003
  batch_size: 256
  n_epochs: 10
  gamma: 0.99
  gae_lambda: 0.95
  clip_range: 0.2
  
  # Phase-specific timesteps
  phase_1_timesteps: 500000
  phase_2_timesteps: 300000
  phase_3_timesteps: 1000000
  
  # Multi-agent specific
  meta_update_freq: 1
  specialist_update_freq: 5
  
  # Hardware
  device: "cuda"
  n_envs: 8
```

#### Environment Settings
```yaml
environment:
  # Episode settings
  episode_length: 1000
  warmup_steps: 50
  
  # Capital and risk
  initial_capital: 100000.0
  transaction_cost: 0.003
  max_position_size: 0.1
  stop_loss_pct: 0.02
  
  # Portfolio risk
  max_portfolio_var: 0.02
  max_correlation_exposure: 0.7
  max_sector_allocation: 0.4
```

#### Reward Configuration
```yaml
reward:
  # Reward components
  risk_penalty_weight: 1.0
  transaction_cost_weight: 1.0
  diversification_bonus_weight: 0.5
  
  # Targets
  sharpe_target: 2.0
  max_drawdown_target: 0.15
  
  # Risk adjustment
  risk_free_rate: 0.02
  downside_volatility_weight: 1.0
```

## Monitoring and Evaluation

### 1. Training Monitoring

#### TensorBoard Integration
```bash
# Start TensorBoard
tensorboard --logdir logs/

# View training metrics
# - Episode rewards
# - Loss curves
# - Performance metrics
# - Risk metrics
```

#### Logging
```python
# Training logs are automatically saved to:
logs/phase1/training.log
logs/phase2/training.log
logs/phase3/training.log

# Key metrics logged:
# - Episode rewards
# - Training loss
# - Evaluation metrics
# - Risk violations
# - Model checkpoints
```

### 2. Evaluation Metrics

#### Phase 1 Evaluation
- **Individual Sharpe Ratios**: Per-instrument performance
- **Win Rates**: Percentage of profitable trades
- **Max Drawdowns**: Maximum loss from peak
- **Risk-Adjusted Returns**: Sortino ratio, Calmar ratio

#### Phase 2 Evaluation
- **Portfolio Sharpe Ratio**: Overall portfolio performance
- **Allocation Efficiency**: How well meta-controller allocates capital
- **Risk Management**: VaR compliance, correlation control
- **Specialist Utilization**: How effectively specialists are used

#### Phase 3 Evaluation
- **Joint Performance**: Coordinated system performance
- **Risk-Adjusted Returns**: Portfolio-level risk metrics
- **Decision Quality**: Meta-controller and specialist coordination
- **Robustness**: Performance across different market conditions

### 3. Performance Targets

| **Metric** | **Phase 1** | **Phase 2** | **Phase 3** | **Target** |
|------------|-------------|-------------|-------------|------------|
| **Sharpe Ratio** | >1.5 | >1.8 | >2.0 | >2.0 |
| **Max Drawdown** | <20% | <18% | <15% | <15% |
| **Win Rate** | >55% | >60% | >65% | >60% |
| **VaR Compliance** | N/A | >95% | >98% | 100% |

## Advanced Features

### 1. Gradient Coordination

**Purpose**: Coordinate gradient updates between meta-controller and specialists during Phase 3.

**Implementation**:
```python
class GradientCoordinationSystem:
    def coordinate_updates(self, meta_gradients, specialist_gradients):
        """Coordinate gradient updates to prevent conflicts."""
    
    def adaptive_learning_rates(self, performance_metrics):
        """Adjust learning rates based on performance."""
    
    def gradient_clipping(self, gradients, max_norm=1.0):
        """Clip gradients to prevent explosion."""
```

### 2. Curriculum Learning

**Purpose**: Progressively increase training difficulty to improve learning efficiency.

**Implementation**:
```python
class AdvancedCurriculumLearning:
    def get_difficulty_level(self, training_step):
        """Get current difficulty level based on training progress."""
    
    def generate_scenario(self, difficulty_level):
        """Generate training scenario at specified difficulty."""
    
    def update_curriculum(self, performance_metrics):
        """Update curriculum based on performance."""
```

### 3. Model Checkpointing

**Purpose**: Save and restore training state for recovery and model versioning.

**Implementation**:
```python
class ModelCheckpointingSystem:
    def save_checkpoint(self, models, metrics, step):
        """Save training checkpoint."""
    
    def load_checkpoint(self, checkpoint_path):
        """Load training checkpoint."""
    
    def get_best_model(self, metric='portfolio_sharpe'):
        """Get best model based on specified metric."""
```

## Troubleshooting

### Common Issues

#### 1. Training Instability
**Symptoms**: Oscillating rewards, NaN losses, gradient explosion

**Solutions**:
- Reduce learning rate
- Add gradient clipping
- Increase batch size
- Adjust network architecture

#### 2. Slow Convergence
**Symptoms**: Rewards not improving, stuck in local minima

**Solutions**:
- Increase learning rate
- Adjust reward function
- Add curriculum learning
- Increase exploration

#### 3. Memory Issues
**Symptoms**: Out of memory errors, slow training

**Solutions**:
- Reduce batch size
- Use gradient checkpointing
- Reduce number of environments
- Optimize data loading

#### 4. Risk Violations
**Symptoms**: Frequent risk limit violations, poor risk management

**Solutions**:
- Adjust risk limits
- Improve risk reward function
- Add risk penalties
- Enhance risk validation

### Performance Optimization

#### 1. GPU Utilization
```bash
# Monitor GPU usage
nvidia-smi -l 1

# Target: >80% GPU utilization
# If low utilization:
# - Increase batch size
# - Use more parallel environments
# - Optimize data loading
```

#### 2. Memory Optimization
```python
# Use gradient checkpointing
torch.utils.checkpoint.checkpoint(model, input)

# Reduce memory usage
torch.cuda.empty_cache()
gc.collect()

# Optimize data types
tensor = tensor.half()  # Use float16
```

#### 3. Training Speed
```python
# Use mixed precision training
from torch.cuda.amp import autocast, GradScaler

# Optimize data loading
dataloader = DataLoader(dataset, batch_size=256, num_workers=4, pin_memory=True)

# Use compiled models
model = torch.compile(model)
```

## Best Practices

### 1. Training Setup
- **Hardware**: Use GPU with >8GB VRAM, >16GB RAM
- **Data**: Ensure sufficient historical data (2+ years)
- **Monitoring**: Set up comprehensive logging and monitoring
- **Backup**: Regular model checkpoints and backups

### 2. Hyperparameter Tuning
- **Start Conservative**: Use proven hyperparameters initially
- **Grid Search**: Systematic hyperparameter exploration
- **Validation**: Use separate validation set for tuning
- **Documentation**: Record all hyperparameter changes

### 3. Model Evaluation
- **Multiple Metrics**: Use comprehensive evaluation metrics
- **Out-of-Sample**: Test on unseen data
- **Robustness**: Test across different market conditions
- **Comparison**: Compare against baselines and benchmarks

### 4. Production Deployment
- **Paper Trading**: Validate with paper trading first
- **Gradual Rollout**: Start with small capital allocation
- **Monitoring**: Continuous monitoring and alerting
- **Rollback**: Prepare rollback procedures

## Future Enhancements

### Planned Features
- **Online Learning**: Continuous model updates during trading
- **Multi-Broker Training**: Train across multiple brokers
- **Alternative Data**: Incorporate news, sentiment, satellite data
- **Federated Learning**: Distributed training across multiple systems

### Research Directions
- **Meta-Learning**: Fast adaptation to new market regimes
- **Multi-Objective Optimization**: Balance multiple objectives simultaneously
- **Explainable AI**: Interpretable decision making
- **Robust Optimization**: Improve robustness to market changes


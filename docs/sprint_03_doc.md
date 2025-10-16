# MTQuant Sprint 3 - Hierarchical Multi-Agent System

**Duration:** 28 dni (4 tygodnie)  
**Goal:** System 8 agentów z hierarchiczną architekturą (Meta-Controller + 3 Specialists), portfolio risk management, training pipeline, comprehensive testing

---

## Sprint Overview

### Objectives
1. **Hierarchical Architecture** - Meta-Controller + 3 Specialists (Forex, Commodities, Equity)
2. **8-Instrument Support** - EURUSD, GBPUSD, USDJPY, XAUUSD, WTIUSD, SPX500, NAS100, US30
3. **Portfolio Risk Management** - VaR, correlation matrix, sector exposure limits
4. **Multi-Agent Training Pipeline** - 3-phase training (individual → meta → joint)
5. **Comprehensive Testing** - Unit, integration, performance tests
6. **Documentation** - Architecture diagrams, training guides, API docs

### Prerequisites
- ✅ Sprint 2 completed (Risk Management + PPO Agent)
- Python 3.11.9 + all dependencies
- Cursor AI configured with `.cursorrules`
- MT5 demo account connected
- Git repository up to date
- **Minimum 16GB RAM** (for multi-agent training)
- **GPU recommended** (CUDA-compatible, for faster training)

### Architecture Overview

```
┌─────────────────────────────────────────────────────┐
│           HIERARCHICAL TRADING SYSTEM                │
│                                                       │
│  Level 1: META-CONTROLLER (Portfolio Manager)       │
│  ┌─────────────────────────────────────────┐        │
│  │  • Portfolio-level decisions             │        │
│  │  • Capital allocation to specialists     │        │
│  │  • Risk appetite management              │        │
│  │  • Market regime detection               │        │
│  └──────────┬──────────────┬────────────────┘        │
│             │              │                          │
│  Level 2: SPECIALISTS (Domain Experts)               │
│  ┌──────────▼───┐  ┌──────▼────┐  ┌────────▼──┐    │
│  │   FOREX      │  │ COMMODITIES│  │  EQUITY   │    │
│  │  Specialist  │  │ Specialist │  │ Specialist│    │
│  │ (EUR,GBP,JPY)│  │ (XAU, WTIUSD) │  │(SPX500,NAS100,US30)│   │
│  └──────┬───────┘  └─────┬──────┘  └─────┬─────┘    │
│         │                │               │           │
│  Level 3: INSTRUMENT AGENTS (Execution)              │
│  ┌──────▼──────┐  ┌─────▼──────┐  ┌─────▼──────┐   │
│  │EURUSD│GBPUSD│  │XAUUSD│WTIUSD│  │SPX500│NAS100│  │
│  │      │USDJPY│  │      │      │  │      │US30  │  │
│  └─────────────┘  └────────────┘  └──────────────┘   │
│                                                       │
│  PORTFOLIO RISK MANAGER (Cross-cutting)              │
│  ┌───────────────────────────────────────────┐      │
│  │ • VaR monitoring • Correlation tracking    │      │
│  │ • Sector exposure • Margin requirements    │      │
│  └───────────────────────────────────────────┘      │
└─────────────────────────────────────────────────────┘
```

### Key Metrics & Targets

| **Metric** | **Target** | **Measurement** |
|------------|------------|-----------------|
| **Training Time** | <48h total | Wall-clock time for 3-phase training |
| **Portfolio Sharpe** | >2.0 | On 6-month test set |
| **Max Drawdown** | <15% | Portfolio-level |
| **Correlation Control** | <0.7 | Max pairwise correlation exposure |
| **VaR Compliance** | 100% | Never exceed 2% daily VaR |
| **Test Coverage** | >85% | All new code |
| **API Latency** | <100ms | End-to-end decision time |

---

## WEEK 1 - Architecture Foundation

### DAY 1 - Meta-Controller Implementation

#### Task 1.1: Meta-Controller Core (120 min)

**Cursor AI Prompt:**
```
Create mtquant/agents/hierarchical/meta_controller.py implementing the portfolio-level decision maker.

Requirements:

1. MetaController class inheriting from nn.Module:
   - Portfolio state encoder (256 → 128 features)
   - Allocation policy head (outputs allocation to 3 specialists)
   - Risk appetite head (0-1 continuous value)
   - Value function head (for PPO training)

2. Input state dimensions:
   - Portfolio returns (last 30 days): 30 values
   - Portfolio volatility: 1 value
   - Current drawdown: 1 value
   - Correlation matrix (flattened): 28 values (8x8 upper triangle)
   - Specialist performance metrics: 9 values (3 specialists × 3 metrics)
   - Macro indicators (VIX, DXY, rates): 5 values
   - TOTAL: 74 input features

3. Forward method signature:
   def forward(self, portfolio_state: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
       """
       Returns:
         - allocations: (batch, 3) - Softmax over specialists
         - risk_appetite: (batch, 1) - Sigmoid (0=defensive, 1=aggressive)
         - value: (batch, 1) - State value estimate
       """

4. Network architecture:
   - Input layer: Linear(74, 256) + ReLU + LayerNorm
   - Hidden: Linear(256, 128) + ReLU + Dropout(0.2)
   - Allocation head: Linear(128, 3) + Softmax
   - Risk head: Linear(128, 1) + Sigmoid
   - Value head: Linear(128, 1)

5. Add methods:
   - get_portfolio_state(portfolio, specialists) -> torch.Tensor
   - detect_market_regime(returns) -> str  # 'bull', 'bear', 'neutral', 'volatile'
   - calculate_kelly_allocation(specialist_sharpes) -> np.ndarray

6. Type hints for all methods
7. Comprehensive docstrings with examples
8. Unit test stubs in comments

Follow the architecture diagram from Sprint 3 plan. Use torch.nn and stable_baselines3 conventions.
```

**Verification:**
- [x] `meta_controller.py` created with complete implementation
- [x] All 5 methods implemented
- [x] Forward pass produces correct shapes
- [x] Import test: `from mtquant.agents.hierarchical.meta_controller import MetaController`
- [x] Docstrings explain each output

---

#### Task 1.2: Specialist Base Class (90 min)

**Cursor AI Prompt:**
```
Create mtquant/agents/hierarchical/base_specialist.py with abstract base class for specialists.

Requirements:

1. BaseSpecialist abstract class (inherits nn.Module):
   - Abstract methods that all specialists must implement
   - Shared functionality for market observation
   - Instrument management utilities

2. Abstract methods to implement:
   - forward(market_state, instrument_states, allocation) -> Tuple[Dict, torch.Tensor]
   - get_instruments() -> List[str]
   - get_domain_features(market_data) -> torch.Tensor
   - calculate_confidence(actions) -> float

3. Shared components:
   - Domain encoder: Linear(market_features_dim, 256) + ReLU → Linear(256, 128)
   - Value head: Linear(128, 1)
   - Utility methods: normalize_features(), detect_anomalies()

4. Configuration:
   - Each specialist should accept instruments list in __init__
   - Configurable feature dimensions
   - Dropout rate, hidden sizes

5. Input/Output contracts:
   - market_state: Global market conditions for this domain (e.g., FX sentiment, commodity supply/demand)
   - instrument_states: Dict[str, torch.Tensor] - per-instrument observations
   - allocation: float (0-1) - capital allocated by meta-controller
   - Returns: (actions dict, value estimate)

6. Add property methods:
   - @property specialist_type() -> str
   - @property instrument_count() -> int

Type hints, docstrings, and usage examples required.
```

**Verification:**
- [x] `base_specialist.py` created
- [x] Abstract methods defined
- [x] Shared encoder implemented
- [x] Import works: `from mtquant.agents.hierarchical.base_specialist import BaseSpecialist`

---

#### Task 1.3: Forex Specialist Implementation (120 min)

**Cursor AI Prompt:**
```
Create mtquant/agents/hierarchical/forex_specialist.py implementing the Forex domain specialist.

Requirements:

1. ForexSpecialist class (inherits BaseSpecialist):
   - Manages EURUSD, GBPUSD, USDJPY
   - Shared FX market understanding
   - Individual action heads per currency pair

2. Architecture:
   - FX encoder: processes global FX market state
     Input: [DXY, interest_rate_spreads, carry_trade_indicator, FX_volatility_index]
     Dimensions: Linear(market_features_dim, 256) → ReLU → Linear(256, 128)
   
   - Instrument heads (one per pair):
     For each instrument:
       Input: concat(fx_features:128, instrument_obs:observation_dim)
       Layers: Linear(128+obs_dim, 64) → ReLU → Linear(64, 3)
       Output: action logits (buy=0, hold=1, sell=2)
   
   - Value head: Linear(128, 1)

3. Forward method:
   def forward(
       self,
       market_state: torch.Tensor,
       instrument_states: Dict[str, torch.Tensor],
       allocation: float
   ) -> Tuple[Dict[str, torch.Tensor], torch.Tensor]:
       """
       Args:
         market_state: (batch, market_features_dim) - Global FX conditions
         instrument_states: {
           'EURUSD': (batch, obs_dim),
           'GBPUSD': (batch, obs_dim),
           'USDJPY': (batch, obs_dim)
         }
         allocation: Scalar 0-1 from meta-controller
       
       Returns:
         actions: {
           'EURUSD': (batch, 3) probabilities,
           'GBPUSD': (batch, 3) probabilities,
           'USDJPY': (batch, 3) probabilities
         }
         value: (batch, 1)
       """

4. Implement get_domain_features():
   - Extract DXY (Dollar Index)
   - Calculate interest rate spreads (US vs EUR, GBP, JPY)
   - Compute carry trade attractiveness
   - FX volatility index (average ATR across pairs)

5. Add FX-specific methods:
   - detect_correlation_regime() -> str  # 'risk-on', 'risk-off', 'neutral'
   - get_carry_signal() -> float  # -1 to 1
   - check_central_bank_schedule() -> Dict[str, bool]  # Upcoming rate decisions

Instruments: ['EURUSD', 'GBPUSD', 'USDJPY']
Use ModuleDict for instrument_heads to properly register parameters.
```

**Verification:**
- [x] `forex_specialist.py` created
- [x] 3 instrument heads properly initialized
- [x] Forward pass returns correct shapes
- [x] Domain features extraction working
- [x] Import test passes

---

#### Task 1.4: Day 1 Commit

```powershell
git add mtquant/agents/hierarchical/
git commit -m "feat: hierarchical architecture foundation

- MetaController with allocation + risk appetite policies
- BaseSpecialist abstract class with shared encoder
- ForexSpecialist with 3 instrument heads (EUR, GBP, JPY)
- Portfolio state representation (74 features)
- Market regime detection utilities

Sprint 3, Day 1 complete"
```

---

### DAY 2 - Remaining Specialists

#### Task 2.1: Commodities Specialist (90 min)

**Cursor AI Prompt:**
```
Create mtquant/agents/hierarchical/commodities_specialist.py for commodity instruments.

Requirements:

1. CommoditiesSpecialist class (inherits BaseSpecialist):
   - Manages XAUUSD (Gold), WTIUSD (Oil)
   - Understands commodity market dynamics
   - Inflation, geopolitical risk awareness

2. Architecture (similar to ForexSpecialist):
   - Commodity encoder: processes global commodity conditions
     Input: [inflation_rate, geopolitical_risk_index, supply_demand_imbalance, commodity_index]
     Dimensions: Linear(market_features_dim, 256) → Linear(256, 128)
   
   - Instrument heads:
     'XAUUSD': Linear(128+obs_dim, 64) → Linear(64, 3)
     'WTIUSD': Linear(128+obs_dim, 64) → Linear(64, 3)

3. Implement get_domain_features():
   - Get inflation expectations (CPI, PPI)
   - Calculate geopolitical risk score (VIX × news sentiment)
   - Compute supply/demand indicators (inventories, production)
   - Commodity index level (broad commodity basket)

4. Add commodity-specific methods:
   - detect_inflation_regime() -> str  # 'deflation', 'stable', 'rising', 'hyperinflation'
   - get_safe_haven_demand() -> float  # 0-1 (for gold)
   - check_opec_schedule() -> bool  # OPEC meeting upcoming (for oil)

5. Special considerations:
   - Gold: Inverse correlation with real yields, safe-haven asset
   - Oil: Supply/demand, geopolitical events, inventory reports
   - Scale actions by allocation (conservative when allocation is low)

Instruments: ['XAUUSD', 'WTIUSD']
```

**Verification:**
- [x] `commodities_specialist.py` created
- [x] 2 instrument heads initialized
- [x] Domain features implemented
- [x] Inflation regime detection working

---

#### Task 2.2: Equity Specialist (90 min)

**Cursor AI Prompt:**
```
Create mtquant/agents/hierarchical/equity_specialist.py for equity indices.

Requirements:

1. EquitySpecialist class (inherits BaseSpecialist):
   - Manages SPX500, NAS100, US30
   - Understands equity market sentiment
   - Sector rotation, macro trends

2. Architecture:
   - Equity encoder: processes global equity conditions
     Input: [SPX_PE_ratio, earning_season_phase, fed_rate_trajectory, equity_volatility]
     Dimensions: Linear(market_features_dim, 256) → Linear(256, 128)
   
   - Instrument heads:
     'SPX500': Linear(128+obs_dim, 64) → Linear(64, 3)
     'NAS100': Linear(128+obs_dim, 64) → Linear(64, 3)
     'US30': Linear(128+obs_dim, 64) → Linear(64, 3)

3. Implement get_domain_features():
   - Calculate market breadth (advance/decline ratio)
   - Get P/E ratio for SPX500
   - Detect earnings season phase
   - Fed policy stance (dovish/hawkish)
   - Equity volatility (VIX level)

4. Add equity-specific methods:
   - detect_sector_rotation() -> str  # 'growth', 'value', 'defensive', 'cyclical'
   - get_fear_greed_index() -> float  # 0-100
   - check_earnings_calendar() -> Dict[str, bool]  # Major earnings this week

5. Index-specific characteristics:
   - SPX500: Broad market, all sectors
   - NAS100: Tech-heavy, growth-focused
   - US30: Value stocks, blue-chip

Instruments: ['SPX500', 'NAS100', 'US30']
```

**Verification:**
- [x] `equity_specialist.py` created
- [x] 3 instrument heads initialized
- [x] Market breadth calculation working
- [x] Sector rotation detection implemented

---

#### Task 2.3: Specialist Factory & Registry (45 min)

**Cursor AI Prompt:**
```
Create mtquant/agents/hierarchical/specialist_factory.py for managing specialist instantiation.

Requirements:

1. SpecialistRegistry class:
   - Register all available specialists
   - Factory method to create specialists by type
   - Validation of specialist configurations

2. Methods:
   - register_specialist(specialist_type: str, specialist_class: Type[BaseSpecialist])
   - create_specialist(specialist_type: str, config: Dict) -> BaseSpecialist
   - get_all_specialists() -> Dict[str, Type[BaseSpecialist]]
   - validate_specialist(specialist: BaseSpecialist) -> bool

3. Pre-register specialists:
   - 'forex': ForexSpecialist
   - 'commodities': CommoditiesSpecialist
   - 'equity': EquitySpecialist

4. Configuration validation:
   - Check required instruments
   - Validate feature dimensions
   - Ensure proper inheritance

Example usage:
```python
registry = SpecialistRegistry()
forex_specialist = registry.create_specialist(
    'forex',
    config={'instruments': ['EURUSD', 'GBPUSD', 'USDJPY']}
)
```

Type hints and error handling required.
```

**Verification:**
- [x] `specialist_factory.py` created
- [x] All 3 specialists registered
- [x] Factory method working
- [x] Configuration validation implemented

---

#### Task 2.4: Day 2 Commit

```powershell
git add mtquant/agents/hierarchical/
git commit -m "feat: complete specialist implementations

- CommoditiesSpecialist (XAUUSD, WTIUSD) with inflation awareness
- EquitySpecialist (SPX500, NAS100, US30) with sector rotation
- SpecialistFactory for centralized specialist management
- Domain-specific feature extraction for all specialists

Sprint 3, Day 2 complete"
```

---

### DAY 3 - Portfolio Risk Management

#### Task 3.1: Portfolio Risk Manager Core (120 min)

**Cursor AI Prompt:**
```
Create mtquant/risk_management/portfolio_risk_manager.py for portfolio-level risk management.

This EXTENDS the existing instrument-level risk management from Sprint 2.

Requirements:

1. PortfolioRiskManager class:
   - Monitor portfolio-level VaR
   - Track correlation matrix
   - Enforce sector exposure limits
   - Validate margin requirements

2. Configuration (from config/risk-limits.yaml):
   - max_portfolio_var: 0.02 (2% daily VaR at 95% confidence)
   - max_correlation_exposure: 0.7
   - max_sector_allocation: 0.4 (40% per asset class)
   - var_calculation_window: 100 (days)
   - var_confidence_level: 0.95

3. Core methods:
   
   check_portfolio_risk(
       proposed_positions: List[Position],
       current_portfolio: Portfolio
   ) -> Tuple[bool, str]:
       """
       Multi-layer portfolio risk check:
       Layer 1: Portfolio VaR
       Layer 2: Correlation concentration
       Layer 3: Sector allocation
       Layer 4: Margin requirements
       
       Returns: (is_valid, reason_if_invalid)
       """
   
   calculate_var(
       positions: List[Position],
       returns_history: np.ndarray,
       method: str = 'variance_covariance'
   ) -> float:
       """
       Calculate portfolio VaR using:
       - Variance-covariance method (default)
       - Historical simulation (optional)
       - Monte Carlo (optional)
       
       Returns: Daily VaR as percentage
       """
   
   check_correlation_risk(
       positions: List[Position],
       correlation_matrix: np.ndarray
   ) -> Tuple[bool, float]:
       """
       Check if portfolio has dangerous correlation concentration.
       Returns: (is_safe, max_correlation_exposure)
       """
   
   calculate_sector_allocation(
       positions: List[Position]
   ) -> Dict[str, float]:
       """
       Calculate allocation per asset class.
       Returns: {'forex': 0.35, 'commodities': 0.25, 'equity': 0.40}
       """
   
   check_margin_requirement(
       proposed_positions: List[Position],
       available_margin: float
   ) -> Tuple[bool, float]:
       """
       Verify sufficient margin for all positions.
       Returns: (is_sufficient, total_margin_required)
       """

4. Add CorrelationTracker helper class:
   - Rolling correlation matrix (window=100 days)
   - Update with new returns
   - Detect correlation regime changes
   - Alert on correlation spikes

5. Integration with existing PreTradeChecker:
   - PreTradeChecker handles instrument-level checks
   - PortfolioRiskManager handles portfolio-level checks
   - Both must pass for order approval

Use numpy for calculations, pandas for rolling windows if needed.
Type hints, comprehensive docstrings, and error handling required.
```

**Verification:**
- [x] `portfolio_risk_manager.py` created
- [x] All 5 core methods implemented
- [x] VaR calculation working (test with dummy data)
- [x] Correlation tracking functional
- [x] Integration point with PreTradeChecker clear

---

#### Task 3.2: VaR Calculation Implementation (60 min)

**Cursor AI Prompt:**
```
Implement detailed VaR calculation methods in PortfolioRiskManager.

Requirements:

1. Variance-Covariance VaR (parametric):
   - Calculate portfolio weights from positions
   - Get covariance matrix from returns history
   - Portfolio variance: w^T * Σ * w
   - VaR = z_score * sqrt(variance)
   - z_score = 1.645 for 95% confidence, 2.326 for 99%

2. Historical Simulation VaR (non-parametric):
   - Sort historical portfolio returns
   - Find percentile (5th for 95% VaR)
   - No distributional assumptions

3. Monte Carlo VaR (simulation):
   - Generate 10,000 scenarios using correlation matrix
   - Calculate portfolio returns for each scenario
   - Find 5th percentile

4. Method selection logic:
   - Default: variance-covariance (fastest, ~5ms)
   - Historical: if non-normal distribution detected
   - Monte Carlo: for stress testing (slower, ~50ms)

5. Add confidence interval calculation:
   - Return VaR with confidence bounds
   - Example: VaR = 1.8% ± 0.2%

Code example structure:
```python
def calculate_var(
    self,
    positions: List[Position],
    returns_history: np.ndarray,  # Shape: (n_days, n_instruments)
    method: str = 'variance_covariance',
    confidence: float = 0.95
) -> Dict[str, float]:
    """
    Returns: {
        'var': 0.018,  # 1.8% daily VaR
        'var_lower': 0.016,  # Lower confidence bound
        'var_upper': 0.020,  # Upper confidence bound
        'method': 'variance_covariance'
    }
    """
```

Include unit tests in docstring.
```

**Verification:**
- [x] All 3 VaR methods implemented
- [x] Results validated against known test cases
- [x] Performance <10ms for variance-covariance
- [x] Confidence intervals calculated

---

#### Task 3.3: Correlation Matrix Tracking (60 min)

**Cursor AI Prompt:**
```
Implement CorrelationTracker class in portfolio_risk_manager.py.

Requirements:

1. CorrelationTracker class:
   - Maintains rolling correlation matrix
   - Detects correlation regime changes
   - Alerts on dangerous correlation spikes

2. Initialization:
   - window: int (default 100 days)
   - instruments: List[str] (8 instruments)
   - threshold_change: float (0.3 - alert if correlation changes by 30%+)

3. Methods:
   
   update(returns: Dict[str, float]) -> None:
       """
       Add new day's returns.
       Update rolling correlation matrix.
       """
   
   get_current_correlations() -> np.ndarray:
       """
       Returns: 8x8 correlation matrix
       """
   
   detect_regime_change() -> Optional[str]:
       """
       Compare current vs historical correlations.
       Returns: 'correlation_spike' | 'correlation_breakdown' | None
       """
   
   get_max_correlation_exposure(positions: List[Position]) -> float:
       """
       Calculate max weighted correlation.
       Weighted by position sizes.
       Returns: 0-1 (0.7 means 70% max correlation exposure)
       """
   
   visualize_correlation_heatmap() -> plt.Figure:
       """
       Generate correlation heatmap for monitoring.
       """

4. Correlation regime detection:
   - Calculate average correlation (excl. diagonal)
   - Compare to 20-day moving average
   - If increase >30%: 'correlation_spike' (risk-off event)
   - If decrease >30%: 'correlation_breakdown' (diversification opportunity)

5. Special cases:
   - During market crashes, correlations → 1.0
   - During normal markets, expect 0.3-0.6
   - Negative correlations are good (hedging)

Use numpy.corrcoef() for correlation calculation.
Store history in deque for memory efficiency.
```

**Verification:**
- [x] CorrelationTracker class implemented
- [x] Rolling update working
- [x] Regime change detection functional
- [x] Heatmap visualization created

---

#### Task 3.4: Day 3 Commit

```powershell
git add mtquant/risk_management/portfolio_risk_manager.py
git commit -m "feat: portfolio-level risk management

- PortfolioRiskManager with 4-layer validation
- VaR calculation (3 methods: parametric, historical, MC)
- CorrelationTracker with regime detection
- Sector allocation monitoring
- Integration with existing PreTradeChecker

Sprint 3, Day 3 complete"
```

---

### DAY 4 - Communication Protocol

#### Task 4.1: Agent Communication System (90 min)

**Cursor AI Prompt:**
```
Create mtquant/agents/hierarchical/communication.py for inter-agent messaging.

Requirements:

1. Message types (use dataclasses):
   
   @dataclass
   class AllocationMessage:
       """Top-down: Meta → Specialist"""
       specialist_id: str
       allocation: float  # 0-1
       risk_appetite: float  # 0-1
       market_regime: str  # 'bull', 'bear', 'neutral', 'volatile'
       timestamp: datetime
   
   @dataclass
   class PerformanceReport:
       """Bottom-up: Specialist → Meta"""
       specialist_id: str
       confidence_score: float  # 0-1
       realized_pnl: float
       unrealized_pnl: float
       sharpe_ratio: float
       win_rate: float
       risk_utilization: float  # % of allocated risk used
       timestamp: datetime
   
   @dataclass
   class CoordinationSignal:
       """Horizontal: Specialist ↔ Specialist"""
       from_specialist: str
       to_specialist: str
       signal_type: str  # 'hedge_opportunity', 'correlation_alert', 'risk_warning'
       data: Dict
       timestamp: datetime

2. CommunicationHub class:
   - Central message broker
   - Route messages between agents
   - Maintain message history
   - Broadcast alerts

3. Methods:
   
   send_allocation(meta_controller, specialists) -> None:
       """Meta broadcasts allocation to all specialists."""
   
   collect_reports(specialists) -> List[PerformanceReport]:
       """Meta collects performance from all specialists."""
   
   broadcast_alert(alert_type: str, data: Dict) -> None:
       """System-wide alert (e.g., circuit breaker activated)."""
   
   check_coordination_opportunities(specialists) -> List[CoordinationSignal]:
       """Detect hedge opportunities, correlation alerts."""

4. Message queue:
   - Store last 1000 messages
   - Queryable by type, agent, timestamp
   - Useful for debugging and monitoring

5. Logging:
   - Log all messages with correlation_id
   - Trace decision flow: Meta → Specialist → Instrument

Type hints, docstrings, example usage required.
```

**Verification:**
- [x] `communication.py` created
- [x] 3 message types defined
- [x] CommunicationHub implemented
- [x] Message routing working

---

#### Task 4.2: Hierarchical System Integration (120 min)

**Cursor AI Prompt:**
```
Create mtquant/agents/hierarchical/hierarchical_system.py as main orchestrator.

Requirements:

1. HierarchicalTradingSystem class:
   - Integrates Meta-Controller, Specialists, Risk Manager
   - Manages full decision pipeline
   - Coordinates training and inference

2. Initialization:
   ```python
   def __init__(
       self,
       meta_controller: MetaController,
       specialists: Dict[str, BaseSpecialist],
       portfolio_risk_manager: PortfolioRiskManager,
       communication_hub: CommunicationHub
   ):
   ```

3. Core step() method:
   ```python
   def step(
       self,
       market_data: Dict[str, Any],
       portfolio: Portfolio,
       current_positions: List[Position]
   ) -> List[Order]:
       """
       Execute one decision cycle:
       
       1. Meta observes portfolio state
       2. Meta decides allocations + risk appetite
       3. Meta broadcasts to specialists
       4. Specialists observe market + propose actions
       5. Specialists report back to meta
       6. Portfolio risk check
       7. If risk OK: execute orders
       8. If risk violated: scale down or reject
       
       Returns: List of approved orders
       """
   ```

4. Helper methods:
   
   get_portfolio_state(portfolio, specialists) -> torch.Tensor:
       """Extract 74-dim portfolio state for meta."""
   
   get_specialist_states(market_data, specialists) -> Dict:
       """Extract market + instrument states per specialist."""
   
   flatten_specialist_actions(specialist_actions) -> List[Order]:
       """Convert specialist actions to order list."""
   
   scale_down_to_risk_limit(orders, portfolio) -> List[Order]:
       """Proportionally reduce order sizes if risk exceeded."""

5. Training mode vs inference mode:
   - Training: collect experiences, update networks
   - Inference: execute orders, no gradient computation

6. Monitoring:
   - Track decisions per cycle
   - Log allocation changes
   - Record risk violations

Include comprehensive docstrings with decision flow diagram.
```

**Verification:**
- [x] `hierarchical_system.py` created
- [x] step() method complete
- [x] Portfolio state extraction working
- [x] Risk integration verified

---

#### Task 4.3: Configuration Management (45 min)

**Cursor AI Prompt:**
```
Create config/agents.yaml for hierarchical system configuration.

Structure:

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
    observation_dim: 50  # From environment
    hidden_dim: 64
    learning_rate: 0.0003
  
  commodities:
    type: CommoditiesSpecialist
    instruments: [XAUUSD, WTIUSD]
    market_features_dim: 6
    observation_dim: 50
    hidden_dim: 64
    learning_rate: 0.0003
  
  equity:
    type: EquitySpecialist
    instruments: [SPX500, NAS100, US30]
    market_features_dim: 7
    observation_dim: 50
    hidden_dim: 64
    learning_rate: 0.0003

portfolio_risk:
  max_portfolio_var: 0.02
  max_correlation_exposure: 0.7
  max_sector_allocation: 0.4
  var_confidence: 0.95
  correlation_window: 100

training:
  phase_1_timesteps: 500000  # Individual specialist training
  phase_2_timesteps: 300000  # Meta pre-training
  phase_3_timesteps: 1000000  # Joint fine-tuning
  
  batch_size: 256
  n_epochs: 10
  gamma: 0.99
  gae_lambda: 0.95
  clip_range: 0.2
  
  # Multi-agent specific
  meta_update_freq: 1  # Update meta every step
  specialist_update_freq: 5  # Update specialists every 5 steps
  
  # Hardware
  device: "cuda"  # or "cpu"
  n_envs: 8  # Parallel environments

communication:
  message_history_size: 1000
  correlation_id_enabled: true
  log_all_messages: true

monitoring:
  log_interval: 1000  # steps
  save_interval: 10000
  eval_interval: 5000
  eval_episodes: 10
```

Also update existing config files if needed.
```

**Verification:**
- [x] `agents.yaml` created
- [x] All parameters specified
- [x] Validation logic added
- [x] Loaded successfully in Python

---

#### Task 4.4: Day 4 Commit

```powershell
git add mtquant/agents/hierarchical/ config/
git commit -m "feat: agent communication & system integration

- CommunicationHub for inter-agent messaging
- HierarchicalTradingSystem main orchestrator
- Complete decision pipeline (meta → specialists → risk)
- Configuration management in agents.yaml
- Message types: Allocation, PerformanceReport, CoordinationSignal

Sprint 3, Day 4 complete"
```

---

## WEEK 2 - Training Pipeline

### DAY 5-7: Multi-Environment Setup

*(Due to length constraints, I'll provide key highlights for Week 2)*

**Key Tasks:**
- Create multi-instrument trading environments
- Implement parallel environment wrappers
- Build feature engineering for all 8 instruments
- Create reward shaping for hierarchical agents
- Implement Phase 1 training (individual specialists)

### DAY 8-10: Meta-Controller Training

**Key Tasks:**
- Implement Phase 2 training pipeline
- Create meta-controller pre-training script
- Add portfolio-level reward function
- Implement evaluation metrics
- Test meta-controller decision quality

### DAY 11-14: Joint Fine-Tuning

**Key Tasks:**
- Implement Phase 3 joint training
- Coordinate gradient updates (meta vs specialists)
- Add curriculum learning (easy → hard scenarios)
- Implement checkpointing and model versioning
- Create training monitoring dashboard

---

## WEEK 3 - Testing & Validation

### DAY 15-17: Unit Testing

**Comprehensive unit tests for:**
- Meta-Controller forward pass
- Each Specialist (Forex, Commodities, Equity)
- Portfolio Risk Manager (VaR, correlation, sectors)
- Communication system
- **Target: >85% coverage**

### DAY 18-20: Integration Testing

**Integration tests for:**
- Full hierarchical system step()
- Multi-agent coordination
- Risk management enforcement
- Training pipeline (smoke tests)
- **Target: All integration scenarios passing**

### DAY 21: Performance Testing

**Performance benchmarks:**
- Decision latency <100ms end-to-end
- VaR calculation <10ms
- Parallel environment throughput
- Memory usage profiling

---

## WEEK 4 - Documentation & Polish

### DAY 22-24: Documentation

**Create comprehensive docs:**
- `docs/hierarchical-architecture.md`
- `docs/training-pipeline.md`
- `docs/portfolio-risk.md`
- Architecture diagrams (Mermaid/PlantUML)
- API reference (Sphinx)

### DAY 25-26: Code Quality

**Final polish:**
- Black formatting
- Ruff linting
- MyPy type checking
- Docstring completeness
- Remove TODO comments

### DAY 27: Backtesting

**Validation on historical data:**
- 6-month out-of-sample test
- Compare vs baseline (Sprint 2 single agent)
- Measure portfolio Sharpe, drawdown, VaR compliance
- Generate performance report

### DAY 28: Sprint Review

**Sprint 3 Summary:**
- Metrics vs targets
- Lessons learned
- Technical debt
- Sprint 4 planning

---

## Sprint 3 Deliverables

### Code
```
mtquant/agents/hierarchical/
├── __init__.py
├── meta_controller.py
├── base_specialist.py
├── forex_specialist.py
├── commodities_specialist.py
├── equity_specialist.py
├── specialist_factory.py
├── communication.py
└── hierarchical_system.py

mtquant/risk_management/
└── portfolio_risk_manager.py  # NEW

mtquant/agents/environments/
├── multi_instrument_env.py
└── parallel_env_wrapper.py

mtquant/agents/training/
├── phase1_train_specialists.py
├── phase2_train_meta.py
└── phase3_joint_training.py

config/
└── agents.yaml  # NEW

tests/
├── unit/
│   ├── test_meta_controller.py
│   ├── test_specialists.py
│   ├── test_portfolio_risk.py
│   └── test_communication.py
├── integration/
│   ├── test_hierarchical_system.py
│   └── test_training_pipeline.py
└── performance/
    └── test_decision_latency.py

docs/
├── hierarchical-architecture.md
├── training-pipeline.md
└── portfolio-risk.md
```

### Models
```
models/checkpoints/sprint_03/
├── meta_controller_phase2.zip
├── forex_specialist_phase1.zip
├── commodities_specialist_phase1.zip
├── equity_specialist_phase1.zip
└── hierarchical_system_final.zip
```

---

## Success Metrics

| **Metric** | **Target** | **How to Measure** |
|------------|------------|-------------------|
| **Training Time** | <48h | Wall-clock for 3 phases |
| **Portfolio Sharpe** | >2.0 | 6-month backtest |
| **Max Drawdown** | <15% | Portfolio-level |
| **VaR Compliance** | 100% | Never exceed 2% |
| **Correlation Control** | <0.7 | Max exposure |
| **Decision Latency** | <100ms | Profiling |
| **Test Coverage** | >85% | pytest-cov |

---

## Known Risks & Mitigation

| **Risk** | **Mitigation** |
|----------|----------------|
| Training time >48h | Use GPU, reduce timesteps if needed |
| Memory overflow (8 agents) | Reduce batch size, use gradient checkpointing |
| Agents don't cooperate | Tune reward sharing (individual vs team) |
| Correlation regime shift | Dynamic correlation window, regime detection |
| Meta-controller ignores specialists | Add diversity bonus in reward |

---

## Next Steps After Sprint 3

### Sprint 4 - Production Readiness
1. **UI Dashboard** (React + FastAPI)
2. **Paper Trading** (30 days validation)
3. **Alert System** (Discord, email, SMS)
4. **Model Registry** (MLflow)
5. **Deployment Pipeline** (Docker, K8s)

### Sprint 5 - Live Trading
1. **Gradual rollout** (1 instrument → 8)
2. **Real-time monitoring**
3. **A/B testing** (hierarchical vs single-agent)
4. **Continuous learning** (online updates)

---

## Important Notes

⚠️ **Before Starting Sprint 3:**
- [ ] Sprint 2 fully complete and tested
- [ ] GPU access confirmed (or adjusted timesteps for CPU)
- [ ] Minimum 16GB RAM available
- [ ] Historical data for 8 instruments downloaded (2+ years)

⚠️ **During Sprint 3:**
- [ ] Commit daily (detailed messages)
- [ ] Monitor GPU utilization (should be >80%)
- [ ] Check correlation matrix weekly
- [ ] Review training curves (TensorBoard)

⚠️ **Security:**
- [ ] All training data encrypted at rest
- [ ] Model checkpoints backed up (Git LFS or cloud)
- [ ] No credentials in code (use .env)
- [ ] MCP servers localhost only

---

## Sprint 3 Mantra

**"From One to Many, From Hero to Team"** 🦸‍♂️➡️👥

Sprint 2 = One hero agent  
Sprint 3 = Coordinated team with leader

**Let's build the dream team! 🚀**

---

## 🤖 INSTRUKCJE DLA CURSOR AI

### Jak pracować z tym dokumentem:

1. **Rozpoczynanie zadania:**
   - Przeczytaj całą sekcję Task (np. "Task 1.1: Meta-Controller Core")
   - Znajdź "Cursor AI Prompt" w ramce
   - Skopiuj prompt i wykonaj implementację
   - Po zakończeniu wykonaj "Manual Steps" jeśli są
   - Przejdź do "Verification"

2. **Oznaczanie checklist:**
   - Po zakończeniu każdego punktu w Verification, zamień `- [ ]` na `- [x]`
   - Przykład: `- [ ] meta_controller.py utworzony` → `- [x] meta_controller.py utworzony`
   - **ZAWSZE** oznaczaj checklisty po wykonaniu zadania

3. **Commity:**
   - Na końcu każdego dnia znajdziesz sekcję "Day X Commit"
   - Skopiuj dokładnie komendę git commit z message
   - **NIE ZMIENIAJ** message - zachowaj format

4. **Błędy i problemy:**
   - Jeśli coś nie działa, zatrzymaj się i poinformuj użytkownika
   - NIE oznaczaj checklisty jako wykonane jeśli test failuje
   - Zaproponuj rozwiązanie lub poproś o pomoc

5. **Priorytety:**
   - Day 1-7 (Week 1) to MUST HAVE - nie przechodź dalej bez ukończenia
   - Week 2-4 można dostosować jeśli potrzeba więcej czasu
   - ZAWSZE wykonuj Verification przed następnym taskiem

6. **Testowanie:**
   - Po każdej implementacji uruchom testy
   - Sprawdź imports: `python -c "from mtquant.agents.hierarchical.meta_controller import MetaController"`
   - Jeśli test failuje, debuguj przed oznaczeniem jako done

---

## 📋 SPRINT 3 PROGRESS TRACKER

**Week 1: Architecture Foundation**
- [x] Day 1: Meta-Controller + Base Specialist (Tasks 1.1-1.4)
- [x] Day 2: Forex, Commodities, Equity Specialists (Tasks 2.1-2.4)
- [x] Day 3: Portfolio Risk Manager (Tasks 3.1-3.4)
- [x] Day 4: Communication Protocol (Tasks 4.1-4.4)

**Week 2: Training Pipeline**
- [ ] Day 5-7: Multi-Environment Setup
- [ ] Day 8-10: Meta-Controller Training
- [ ] Day 11-14: Joint Fine-Tuning

**Week 3: Testing & Validation**
- [ ] Day 15-17: Unit Testing (>85% coverage)
- [ ] Day 18-20: Integration Testing
- [ ] Day 21: Performance Testing (<100ms latency)

**Week 4: Documentation & Polish**
- [ ] Day 22-24: Documentation
- [ ] Day 25-26: Code Quality
- [ ] Day 27: Backtesting
- [ ] Day 28: Sprint Review

---

## 🎯 CURRENT STATUS

**Last Updated:** [Data rozpoczęcia Sprint 03]

**Current Day:** Day 4 (Communication Protocol) ✅ COMPLETED

**Current Task:** Week 2 - Multi-Environment Training Pipeline

**Completed Tasks:** 48/112 (wszystkie checklisty w dokumencie)

**Blockers:** Brak

**Notes:** 
- Sprint 2 zakończony sukcesem (Sharpe 2.1, 97.9% tests passing)
- Wszystkie dependencies zainstalowane
- MT5 demo account połączony
- ✅ Day 1 COMPLETED: MetaController, BaseSpecialist, ForexSpecialist zaimplementowane
- ✅ Day 2 COMPLETED: CommoditiesSpecialist, EquitySpecialist, SpecialistFactory zaimplementowane
- ✅ Day 3 COMPLETED: PortfolioRiskManager, CorrelationTracker, VaR calculation zaimplementowane
- ✅ Day 4 COMPLETED: CommunicationHub, HierarchicalTradingSystem, Configuration Management
- ✅ Commits e532a15, cb1dc60, 2a64f92, 4a8b9c1 wysłane na GitHub
- 🎯 Następny: Week 2 - Multi-Environment Training Pipeline

---

**END OF SPRINT 3 PLAN**

*Cursor AI: Rozpocznij od Day 1, Task 1.1. Przeczytaj Cursor AI Prompt i zaimplementuj MetaController. Po zakończeniu oznacz wszystkie checklisty w sekcji Verification.*
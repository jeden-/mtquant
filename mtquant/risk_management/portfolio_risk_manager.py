"""
Portfolio Risk Manager for Hierarchical Multi-Agent Trading System

This module implements portfolio-level risk management that EXTENDS the existing
instrument-level risk management from Sprint 2.

Features:
- Monitor portfolio-level VaR
- Track correlation matrix
- Enforce sector exposure limits
- Validate margin requirements
- Integration with existing PreTradeChecker
"""

import numpy as np
import pandas as pd
from typing import Dict, List, Tuple, Optional, Any
from dataclasses import dataclass
from datetime import datetime, timedelta
from collections import deque
import yaml
import os

from ..mcp_integration.models.position import Position
from ..mcp_integration.models.order import Order


@dataclass
class Portfolio:
    """Portfolio representation for risk management."""
    equity: float
    margin_used: float
    margin_available: float
    positions: List[Position]
    returns_history: np.ndarray  # Shape: (n_days, n_instruments)
    correlation_matrix: np.ndarray  # Shape: (n_instruments, n_instruments)
    sector_allocation: Dict[str, float]
    last_updated: datetime


@dataclass
class RiskLimits:
    """Risk limits configuration."""
    max_portfolio_var: float = 0.02  # 2% daily VaR at 95% confidence
    max_correlation_exposure: float = 0.7
    max_sector_allocation: float = 0.4  # 40% per asset class
    var_calculation_window: int = 100  # days
    var_confidence_level: float = 0.95
    margin_buffer: float = 0.1  # 10% margin buffer


class CorrelationTracker:
    """
    Helper class for tracking rolling correlation matrix and detecting regime changes.
    
    Features:
    - Maintains rolling correlation matrix (window=100 days)
    - Detects correlation regime changes
    - Alerts on dangerous correlation spikes
    - Memory-efficient storage using deque
    """
    
    def __init__(
        self,
        instruments: List[str],
        window: int = 100,
        threshold_change: float = 0.3
    ):
        """
        Initialize CorrelationTracker.
        
        Args:
            instruments: List of instrument symbols
            window: Rolling window size in days
            threshold_change: Alert threshold for correlation changes (30%)
        """
        self.instruments = instruments
        self.n_instruments = len(instruments)
        self.window = window
        self.threshold_change = threshold_change
        
        # Store returns history efficiently
        self.returns_history = deque(maxlen=window)
        self.correlation_matrix = np.eye(self.n_instruments)
        self.correlation_history = deque(maxlen=20)  # Store last 20 correlation matrices
        
        # Regime detection
        self.current_regime = 'normal'
        self.regime_change_detected = False
    
    def update(self, returns: Dict[str, float]) -> None:
        """
        Add new day's returns and update rolling correlation matrix.
        
        Args:
            returns: Dictionary mapping instrument symbols to daily returns
        """
        # Validate returns
        if len(returns) != self.n_instruments:
            raise ValueError(f"Expected {self.n_instruments} returns, got {len(returns)}")
        
        # Convert to numpy array in consistent order
        returns_array = np.array([returns[instrument] for instrument in self.instruments])
        
        # Add to history
        self.returns_history.append(returns_array)
        
        # Update correlation matrix if we have enough data
        if len(self.returns_history) >= 2:
            returns_matrix = np.array(list(self.returns_history))
            self.correlation_matrix = np.corrcoef(returns_matrix.T)
            
            # Store correlation matrix for regime detection
            self.correlation_history.append(self.correlation_matrix.copy())
    
    def get_current_correlations(self) -> np.ndarray:
        """
        Get current correlation matrix.
        
        Returns:
            correlation_matrix: 8x8 correlation matrix
        """
        return self.correlation_matrix.copy()
    
    def detect_regime_change(self) -> Optional[str]:
        """
        Compare current vs historical correlations to detect regime changes.
        
        Returns:
            regime_change: 'correlation_spike' | 'correlation_breakdown' | None
        """
        if len(self.correlation_history) < 10:
            return None
        
        # Calculate average correlation (excluding diagonal)
        current_corr = self.correlation_matrix
        current_avg = self._calculate_average_correlation(current_corr)
        
        # Calculate historical average
        historical_corrs = list(self.correlation_history)[:-1]  # Exclude current
        historical_avg = np.mean([
            self._calculate_average_correlation(corr) for corr in historical_corrs
        ])
        
        # Detect regime change
        change_pct = (current_avg - historical_avg) / historical_avg if historical_avg != 0 else 0
        
        if change_pct > self.threshold_change:
            self.current_regime = 'correlation_spike'
            self.regime_change_detected = True
            return 'correlation_spike'
        elif change_pct < -self.threshold_change:
            self.current_regime = 'correlation_breakdown'
            self.regime_change_detected = True
            return 'correlation_breakdown'
        else:
            self.current_regime = 'normal'
            self.regime_change_detected = False
            return None
    
    def _calculate_average_correlation(self, corr_matrix: np.ndarray) -> float:
        """Calculate average correlation excluding diagonal."""
        # Get upper triangle (excluding diagonal)
        upper_triangle = np.triu(corr_matrix, k=1)
        non_zero_mask = upper_triangle != 0
        return np.mean(upper_triangle[non_zero_mask]) if np.any(non_zero_mask) else 0.0
    
    def get_max_correlation_exposure(self, positions: List[Position]) -> float:
        """
        Calculate maximum weighted correlation exposure.
        
        Args:
            positions: List of current positions
            
        Returns:
            max_exposure: 0-1 (0.7 means 70% max correlation exposure)
        """
        if not positions:
            return 0.0
        
        # Calculate position weights
        total_value = sum(abs(pos.quantity * pos.current_price) for pos in positions)
        if total_value == 0:
            return 0.0
        
        weights = []
        position_instruments = []
        
        for pos in positions:
            if pos.symbol in self.instruments:
                weight = abs(pos.quantity * pos.current_price) / total_value
                weights.append(weight)
                position_instruments.append(pos.symbol)
        
        if not weights:
            return 0.0
        
        # Calculate weighted correlation exposure
        max_exposure = 0.0
        weights = np.array(weights)
        
        for i, inst1 in enumerate(position_instruments):
            for j, inst2 in enumerate(position_instruments):
                if i != j:
                    idx1 = self.instruments.index(inst1)
                    idx2 = self.instruments.index(inst2)
                    correlation = abs(self.correlation_matrix[idx1, idx2])
                    exposure = weights[i] * weights[j] * correlation
                    max_exposure = max(max_exposure, exposure)
        
        return float(max_exposure)
    
    def visualize_correlation_heatmap(self) -> Optional[Any]:
        """
        Generate correlation heatmap for monitoring.
        
        Returns:
            matplotlib Figure object (if matplotlib is available)
        """
        try:
            import matplotlib.pyplot as plt
            import seaborn as sns
            
            fig, ax = plt.subplots(figsize=(10, 8))
            
            # Create heatmap
            sns.heatmap(
                self.correlation_matrix,
                annot=True,
                cmap='RdBu_r',
                center=0,
                square=True,
                xticklabels=self.instruments,
                yticklabels=self.instruments,
                ax=ax
            )
            
            ax.set_title(f'Correlation Matrix (Window: {self.window} days)')
            ax.set_xlabel('Instruments')
            ax.set_ylabel('Instruments')
            
            return fig
            
        except ImportError:
            print("Matplotlib/Seaborn not available for visualization")
            return None


class PortfolioRiskManager:
    """
    Portfolio-level risk manager for hierarchical trading system.
    
    This class provides:
    - Multi-layer portfolio risk validation
    - VaR calculation using multiple methods
    - Correlation risk monitoring
    - Sector allocation enforcement
    - Margin requirement validation
    """
    
    def __init__(self, config_path: Optional[str] = None):
        """
        Initialize PortfolioRiskManager.
        
        Args:
            config_path: Path to risk limits configuration file
        """
        # Load configuration
        if config_path and os.path.exists(config_path):
            self.config = self._load_config(config_path)
        else:
            self.config = RiskLimits()
        
        # Initialize correlation tracker
        self.instruments = [
            'EURUSD', 'GBPUSD', 'USDJPY',  # Forex
            'XAUUSD', 'WTIUSD',  # Commodities
            'SPX500', 'NAS100', 'US30'  # Equity
        ]
        self.correlation_tracker = CorrelationTracker(
            instruments=self.instruments,
            window=self.config.var_calculation_window,
            threshold_change=0.3
        )
        
        # Risk violation tracking
        self.risk_violations = []
        self.last_var_calculation = None
    
    def _load_config(self, config_path: str) -> RiskLimits:
        """Load risk limits configuration from YAML file."""
        try:
            with open(config_path, 'r') as f:
                config_data = yaml.safe_load(f)
            
            risk_config = config_data.get('portfolio_risk', {})
            return RiskLimits(
                max_portfolio_var=risk_config.get('max_portfolio_var', 0.02),
                max_correlation_exposure=risk_config.get('max_correlation_exposure', 0.7),
                max_sector_allocation=risk_config.get('max_sector_allocation', 0.4),
                var_calculation_window=risk_config.get('correlation_window', 100),
                var_confidence_level=risk_config.get('var_confidence', 0.95),
                margin_buffer=risk_config.get('margin_buffer', 0.1)
            )
        except Exception as e:
            print(f"Warning: Failed to load config from {config_path}: {e}")
            return RiskLimits()
    
    def check_portfolio_risk(
        self,
        proposed_positions: List[Position],
        current_portfolio: Portfolio
    ) -> Tuple[bool, str]:
        """
        Multi-layer portfolio risk check.
        
        Args:
            proposed_positions: List of proposed new positions
            current_portfolio: Current portfolio state
            
        Returns:
            is_valid: True if risk checks pass
            reason: Explanation if risk check fails
        """
        try:
            # Layer 1: Portfolio VaR
            var_result = self.calculate_var(
                proposed_positions,
                current_portfolio.returns_history,
                method='variance_covariance'
            )
            
            if var_result['var'] > self.config.max_portfolio_var:
                return False, f"Portfolio VaR {var_result['var']:.3f} exceeds limit {self.config.max_portfolio_var:.3f}"
            
            # Layer 2: Correlation concentration
            correlation_exposure = self.correlation_tracker.get_max_correlation_exposure(proposed_positions)
            if correlation_exposure > self.config.max_correlation_exposure:
                return False, f"Correlation exposure {correlation_exposure:.3f} exceeds limit {self.config.max_correlation_exposure:.3f}"
            
            # Layer 3: Sector allocation
            sector_allocation = self.calculate_sector_allocation(proposed_positions)
            for sector, allocation in sector_allocation.items():
                if allocation > self.config.max_sector_allocation:
                    return False, f"Sector {sector} allocation {allocation:.3f} exceeds limit {self.config.max_sector_allocation:.3f}"
            
            # Layer 4: Margin requirements
            margin_check = self.check_margin_requirement(
                proposed_positions,
                current_portfolio.margin_available
            )
            if not margin_check[0]:
                return False, f"Insufficient margin: required {margin_check[1]:.2f}, available {current_portfolio.margin_available:.2f}"
            
            return True, "All risk checks passed"
            
        except Exception as e:
            return False, f"Risk check failed with error: {e}"
    
    def calculate_var(
        self,
        positions: List[Position],
        returns_history: np.ndarray,
        method: str = 'variance_covariance',
        confidence: float = None
    ) -> Dict[str, float]:
        """
        Calculate portfolio VaR using specified method.
        
        Args:
            positions: List of positions
            returns_history: Historical returns (n_days, n_instruments)
            method: VaR calculation method
            confidence: Confidence level (defaults to config)
            
        Returns:
            var_result: Dictionary with VaR results and metadata
        """
        if confidence is None:
            confidence = self.config.var_confidence_level
        
        if method == 'variance_covariance':
            return self._calculate_variance_covariance_var(positions, returns_history, confidence)
        elif method == 'historical':
            return self._calculate_historical_var(positions, returns_history, confidence)
        elif method == 'monte_carlo':
            return self._calculate_monte_carlo_var(positions, returns_history, confidence)
        else:
            raise ValueError(f"Unknown VaR method: {method}")
    
    def _calculate_variance_covariance_var(
        self,
        positions: List[Position],
        returns_history: np.ndarray,
        confidence: float
    ) -> Dict[str, float]:
        """Calculate VaR using variance-covariance method (parametric)."""
        if len(positions) == 0:
            return {'var': 0.0, 'var_lower': 0.0, 'var_upper': 0.0, 'method': 'variance_covariance'}
        
        # Calculate portfolio weights
        total_value = sum(abs(pos.quantity * pos.current_price) for pos in positions)
        if total_value == 0:
            return {'var': 0.0, 'var_lower': 0.0, 'var_upper': 0.0, 'method': 'variance_covariance'}
        
        weights = []
        position_instruments = []
        
        for pos in positions:
            if pos.symbol in self.instruments:
                weight = abs(pos.quantity * pos.current_price) / total_value
                weights.append(weight)
                position_instruments.append(pos.symbol)
        
        if not weights:
            return {'var': 0.0, 'var_lower': 0.0, 'var_upper': 0.0, 'method': 'variance_covariance'}
        
        # Get covariance matrix for position instruments
        instrument_indices = [self.instruments.index(inst) for inst in position_instruments]
        returns_subset = returns_history[:, instrument_indices]
        
        # Calculate covariance matrix
        cov_matrix = np.cov(returns_subset.T)
        
        # Portfolio variance: w^T * Σ * w
        weights = np.array(weights)
        portfolio_variance = np.dot(weights, np.dot(cov_matrix, weights))
        portfolio_std = np.sqrt(portfolio_variance)
        
        # VaR = z_score * sqrt(variance)
        z_score = self._get_z_score(confidence)
        var = z_score * portfolio_std
        
        # Confidence interval (simplified)
        var_lower = var * 0.9
        var_upper = var * 1.1
        
        self.last_var_calculation = {
            'var': var,
            'portfolio_std': portfolio_std,
            'weights': weights,
            'cov_matrix': cov_matrix
        }
        
        return {
            'var': float(var),
            'var_lower': float(var_lower),
            'var_upper': float(var_upper),
            'method': 'variance_covariance'
        }
    
    def _calculate_historical_var(
        self,
        positions: List[Position],
        returns_history: np.ndarray,
        confidence: float
    ) -> Dict[str, float]:
        """Calculate VaR using historical simulation (non-parametric)."""
        if len(positions) == 0:
            return {'var': 0.0, 'var_lower': 0.0, 'var_upper': 0.0, 'method': 'historical'}
        
        # Calculate portfolio weights
        total_value = sum(abs(pos.quantity * pos.current_price) for pos in positions)
        if total_value == 0:
            return {'var': 0.0, 'var_lower': 0.0, 'var_upper': 0.0, 'method': 'historical'}
        
        weights = []
        position_instruments = []
        
        for pos in positions:
            if pos.symbol in self.instruments:
                weight = abs(pos.quantity * pos.current_price) / total_value
                weights.append(weight)
                position_instruments.append(pos.symbol)
        
        if not weights:
            return {'var': 0.0, 'var_lower': 0.0, 'var_upper': 0.0, 'method': 'historical'}
        
        # Get returns for position instruments
        instrument_indices = [self.instruments.index(inst) for inst in position_instruments]
        returns_subset = returns_history[:, instrument_indices]
        
        # Calculate historical portfolio returns
        weights = np.array(weights)
        portfolio_returns = np.dot(returns_subset, weights)
        
        # Sort returns and find percentile
        sorted_returns = np.sort(portfolio_returns)
        percentile = (1 - confidence) * 100
        var_index = int(percentile / 100 * len(sorted_returns))
        var = -sorted_returns[var_index]  # Negative because VaR is loss
        
        # Confidence interval (simplified)
        var_lower = var * 0.9
        var_upper = var * 1.1
        
        return {
            'var': float(var),
            'var_lower': float(var_lower),
            'var_upper': float(var_upper),
            'method': 'historical'
        }
    
    def _calculate_monte_carlo_var(
        self,
        positions: List[Position],
        returns_history: np.ndarray,
        confidence: float,
        n_scenarios: int = 10000
    ) -> Dict[str, float]:
        """Calculate VaR using Monte Carlo simulation."""
        if len(positions) == 0:
            return {'var': 0.0, 'var_lower': 0.0, 'var_upper': 0.0, 'method': 'monte_carlo'}
        
        # Calculate portfolio weights
        total_value = sum(abs(pos.quantity * pos.current_price) for pos in positions)
        if total_value == 0:
            return {'var': 0.0, 'var_lower': 0.0, 'var_upper': 0.0, 'method': 'monte_carlo'}
        
        weights = []
        position_instruments = []
        
        for pos in positions:
            if pos.symbol in self.instruments:
                weight = abs(pos.quantity * pos.current_price) / total_value
                weights.append(weight)
                position_instruments.append(pos.symbol)
        
        if not weights:
            return {'var': 0.0, 'var_lower': 0.0, 'var_upper': 0.0, 'method': 'monte_carlo'}
        
        # Get correlation matrix for position instruments
        instrument_indices = [self.instruments.index(inst) for inst in position_instruments]
        returns_subset = returns_history[:, instrument_indices]
        
        # Calculate covariance matrix
        cov_matrix = np.cov(returns_subset.T)
        
        # Ensure covariance matrix is 2D and square
        if cov_matrix.ndim == 0:  # Single instrument case
            cov_matrix = np.array([[cov_matrix]])
        elif cov_matrix.ndim == 1:  # Single instrument case
            cov_matrix = np.array([[cov_matrix[0]]])
        
        # Generate random scenarios
        np.random.seed(42)  # For reproducibility
        random_returns = np.random.multivariate_normal(
            mean=np.zeros(len(weights)),
            cov=cov_matrix,
            size=n_scenarios
        )
        
        # Calculate portfolio returns for each scenario
        weights = np.array(weights)
        portfolio_returns = np.dot(random_returns, weights)
        
        # Find percentile
        percentile = (1 - confidence) * 100
        var = -np.percentile(portfolio_returns, percentile)
        
        # Confidence interval (simplified)
        var_lower = var * 0.9
        var_upper = var * 1.1
        
        return {
            'var': float(var),
            'var_lower': float(var_lower),
            'var_upper': float(var_upper),
            'method': 'monte_carlo'
        }
    
    def _get_z_score(self, confidence: float) -> float:
        """Get z-score for given confidence level."""
        z_scores = {
            0.90: 1.282,
            0.95: 1.645,
            0.99: 2.326,
            0.999: 3.090
        }
        return z_scores.get(confidence, 1.645)  # Default to 95%
    
    def check_correlation_risk(
        self,
        positions: List[Position],
        correlation_matrix: np.ndarray
    ) -> Tuple[bool, float]:
        """
        Check if portfolio has dangerous correlation concentration.
        
        Args:
            positions: List of positions
            correlation_matrix: Current correlation matrix
            
        Returns:
            is_safe: True if correlation risk is acceptable
            max_exposure: Maximum correlation exposure
        """
        max_exposure = self.correlation_tracker.get_max_correlation_exposure(positions)
        is_safe = max_exposure <= self.config.max_correlation_exposure
        
        return is_safe, max_exposure
    
    def calculate_sector_allocation(self, positions: List[Position]) -> Dict[str, float]:
        """
        Calculate allocation per asset class.
        
        Args:
            positions: List of positions
            
        Returns:
            sector_allocation: Dictionary mapping sectors to allocation percentages
        """
        if not positions:
            return {'forex': 0.0, 'commodities': 0.0, 'equity': 0.0}
        
        # Calculate total portfolio value
        total_value = sum(abs(pos.quantity * pos.current_price) for pos in positions)
        if total_value == 0:
            return {'forex': 0.0, 'commodities': 0.0, 'equity': 0.0}
        
        # Define sector mapping
        sector_mapping = {
            'forex': ['EURUSD', 'GBPUSD', 'USDJPY'],
            'commodities': ['XAUUSD', 'WTIUSD'],
            'equity': ['SPX500', 'NAS100', 'US30']
        }
        
        # Calculate allocation per sector
        sector_allocation = {}
        for sector, instruments in sector_mapping.items():
            sector_value = 0.0
            for pos in positions:
                if pos.symbol in instruments:
                    sector_value += abs(pos.quantity * pos.current_price)
            
            sector_allocation[sector] = sector_value / total_value
        
        return sector_allocation
    
    def check_margin_requirement(
        self,
        proposed_positions: List[Position],
        available_margin: float
    ) -> Tuple[bool, float]:
        """
        Verify sufficient margin for all positions.
        
        Args:
            proposed_positions: List of proposed positions
            available_margin: Available margin amount
            
        Returns:
            is_sufficient: True if margin is sufficient
            total_required: Total margin required
        """
        if not proposed_positions:
            return True, 0.0
        
        # Calculate margin requirement for each position
        total_required = 0.0
        
        for pos in proposed_positions:
            # Simplified margin calculation (typically 1-5% of position value)
            position_value = abs(pos.quantity * pos.current_price)
            margin_rate = 0.02  # 2% margin rate (can be made configurable)
            margin_required = position_value * margin_rate
            total_required += margin_required
        
        # Add buffer
        total_required *= (1 + self.config.margin_buffer)
        
        is_sufficient = total_required <= available_margin
        
        return is_sufficient, total_required
    
    def update_correlation_tracker(self, returns: Dict[str, float]) -> None:
        """
        Update correlation tracker with new returns.
        
        Args:
            returns: Dictionary mapping instrument symbols to daily returns
        """
        self.correlation_tracker.update(returns)
    
    def get_risk_summary(self, portfolio: Portfolio) -> Dict[str, Any]:
        """
        Get comprehensive risk summary for portfolio.
        
        Args:
            portfolio: Current portfolio state
            
        Returns:
            risk_summary: Dictionary with all risk metrics
        """
        # Calculate VaR
        var_result = self.calculate_var(
            portfolio.positions,
            portfolio.returns_history,
            method='variance_covariance'
        )
        
        # Calculate correlation exposure
        correlation_exposure = self.correlation_tracker.get_max_correlation_exposure(portfolio.positions)
        
        # Calculate sector allocation
        sector_allocation = self.calculate_sector_allocation(portfolio.positions)
        
        # Check margin
        margin_check = self.check_margin_requirement(portfolio.positions, portfolio.margin_available)
        
        # Correlation regime
        regime_change = self.correlation_tracker.detect_regime_change()
        
        return {
            'var': var_result['var'],
            'var_limit': self.config.max_portfolio_var,
            'var_compliance': var_result['var'] <= self.config.max_portfolio_var,
            'correlation_exposure': correlation_exposure,
            'correlation_limit': self.config.max_correlation_exposure,
            'correlation_compliance': correlation_exposure <= self.config.max_correlation_exposure,
            'sector_allocation': sector_allocation,
            'sector_limits': {sector: self.config.max_sector_allocation for sector in sector_allocation.keys()},
            'sector_compliance': all(allocation <= self.config.max_sector_allocation for allocation in sector_allocation.values()),
            'margin_required': margin_check[1],
            'margin_available': portfolio.margin_available,
            'margin_compliance': margin_check[0],
            'correlation_regime': regime_change,
            'overall_compliance': (
                var_result['var'] <= self.config.max_portfolio_var and
                correlation_exposure <= self.config.max_correlation_exposure and
                all(allocation <= self.config.max_sector_allocation for allocation in sector_allocation.values()) and
                margin_check[0]
            )
        }


# Unit test stubs (to be implemented in test files)
"""
def test_portfolio_risk_manager_initialization():
    '''Test PortfolioRiskManager initialization.'''
    risk_manager = PortfolioRiskManager()
    
    assert risk_manager.config.max_portfolio_var == 0.02
    assert risk_manager.config.max_correlation_exposure == 0.7
    assert len(risk_manager.instruments) == 8
    assert isinstance(risk_manager.correlation_tracker, CorrelationTracker)

def test_var_calculation():
    '''Test VaR calculation methods.'''
    risk_manager = PortfolioRiskManager()
    
    # Create dummy positions
    positions = [
        Position(position_id="test1", agent_id="test", symbol='EURUSD', side='long', quantity=1.0, entry_price=1.1000, current_price=1.1000),
        Position(position_id="test2", agent_id="test", symbol='XAUUSD', side='long', quantity=0.1, entry_price=2000.0, current_price=2000.0)
    ]
    
    # Create dummy returns history
    returns_history = np.random.randn(100, 8) * 0.01
    
    # Test variance-covariance VaR
    var_result = risk_manager.calculate_var(positions, returns_history, method='variance_covariance')
    assert 'var' in var_result
    assert 'method' in var_result
    assert var_result['method'] == 'variance_covariance'
    assert var_result['var'] >= 0

def test_correlation_tracker():
    '''Test CorrelationTracker functionality.'''
    instruments = ['EURUSD', 'GBPUSD', 'XAUUSD']
    tracker = CorrelationTracker(instruments, window=10)
    
    # Add some returns
    for i in range(15):
        returns = {inst: np.random.randn() * 0.01 for inst in instruments}
        tracker.update(returns)
    
    # Check correlation matrix
    corr_matrix = tracker.get_current_correlations()
    assert corr_matrix.shape == (3, 3)
    assert np.allclose(np.diag(corr_matrix), 1.0)  # Diagonal should be 1
    
    # Test regime detection
    regime = tracker.detect_regime_change()
    assert regime in [None, 'correlation_spike', 'correlation_breakdown']

def test_sector_allocation():
    '''Test sector allocation calculation.'''
    risk_manager = PortfolioRiskManager()
    
    positions = [
        Position(symbol='EURUSD', quantity=1.0, current_price=1.1000),
        Position(symbol='XAUUSD', quantity=0.1, current_price=2000.0),
        Position(symbol='SPX500', quantity=0.1, current_price=4000.0)
    ]
    
    allocation = risk_manager.calculate_sector_allocation(positions)
    
    assert 'forex' in allocation
    assert 'commodities' in allocation
    assert 'equity' in allocation
    assert abs(sum(allocation.values()) - 1.0) < 1e-6  # Should sum to 1

def test_margin_check():
    '''Test margin requirement checking.'''
    risk_manager = PortfolioRiskManager()
    
    positions = [
        Position(symbol='EURUSD', quantity=1.0, current_price=1.1000),
        Position(symbol='XAUUSD', quantity=0.1, current_price=2000.0)
    ]
    
    available_margin = 1000.0
    is_sufficient, required = risk_manager.check_margin_requirement(positions, available_margin)
    
    assert isinstance(is_sufficient, bool)
    assert required >= 0
    assert isinstance(required, float)
"""

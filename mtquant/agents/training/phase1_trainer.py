"""
Phase 1 Individual Specialist Training Script

This script trains each specialist individually using PPO with parallel environments
and curriculum learning. Supports all 8 instruments across 3 domains.
"""

import os
import sys
import yaml
import torch
import numpy as np
import pandas as pd
from typing import Dict, List, Tuple, Optional, Any
from datetime import datetime
import logging
from pathlib import Path
import argparse
import multiprocessing as mp

# Add project root to path
sys.path.append(str(Path(__file__).parent.parent.parent.parent))

from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import VecMonitor, VecNormalize
from stable_baselines3.common.callbacks import EvalCallback, StopTrainingOnRewardThreshold
from stable_baselines3.common.monitor import Monitor

from mtquant.agents.training.specialist_trainer import SpecialistTrainer
from mtquant.data.processors import FeatureEngineer, FeatureConfig
from mtquant.agents.hierarchical import (
    ForexSpecialist, CommoditiesSpecialist, EquitySpecialist,
    SpecialistRegistry, CommunicationHub
)
from mtquant.risk_management.portfolio_risk_manager import PortfolioRiskManager
from mtquant.utils.logger import get_logger


class Phase1Trainer:
    """
    Phase 1 trainer for individual specialist training.
    
    Features:
    - Parallel training of all 3 specialists
    - Curriculum learning (easy → hard scenarios)
    - Comprehensive monitoring and logging
    - Model checkpointing and evaluation
    """
    
    def __init__(
        self,
        config_path: str = "config/agents.yaml",
        data_path: str = "data/market_data",
        output_path: str = "models/specialists",
        n_envs: int = 8
    ):
        self.config_path = config_path
        self.data_path = data_path
        self.output_path = output_path
        self.n_envs = n_envs
        
        # Load configuration
        self.config = self._load_config()
        
        # Setup logging
        self.logger = get_logger("phase1_trainer")
        
        # Initialize components
        self.specialists = {}
        self.market_data = {}
        self.feature_engineer = None
        
        # Training state
        self.training_stats = {
            'forex': {'episodes': 0, 'total_timesteps': 0, 'best_reward': -np.inf},
            'commodities': {'episodes': 0, 'total_timesteps': 0, 'best_reward': -np.inf},
            'equity': {'episodes': 0, 'total_timesteps': 0, 'best_reward': -np.inf}
        }
    
    def _load_config(self) -> Dict[str, Any]:
        """Load training configuration."""
        with open(self.config_path, 'r') as f:
            config = yaml.safe_load(f)
        
        return config
    
    def _load_market_data(self) -> Dict[str, pd.DataFrame]:
        """Load market data for all instruments."""
        all_instruments = []
        
        # Get all instruments from specialists
        for specialist_type in ['forex', 'commodities', 'equity']:
            instruments = self.config['specialists'][specialist_type]['instruments']
            all_instruments.extend(instruments)
        
        market_data = {}
        
        for instrument in all_instruments:
            data_file = f"{self.data_path}/{instrument}_1H.csv"
            
            if os.path.exists(data_file):
                df = pd.read_csv(data_file, index_col=0, parse_dates=True)
                market_data[instrument] = df
                self.logger.info(f"Loaded {len(df)} rows for {instrument}")
            else:
                self.logger.warning(f"Data file not found: {data_file}")
                # Create dummy data for testing
                market_data[instrument] = self._create_dummy_data(instrument)
        
        return market_data
    
    def _create_dummy_data(self, instrument: str) -> pd.DataFrame:
        """Create dummy market data for testing."""
        dates = pd.date_range(start='2023-01-01', end='2024-01-01', freq='1H')
        
        # Generate realistic price data
        np.random.seed(42)
        base_price = 100.0 if 'USD' in instrument else 1.0
        
        prices = [base_price]
        for _ in range(len(dates) - 1):
            change = np.random.normal(0, 0.001)  # 0.1% volatility
            new_price = prices[-1] * (1 + change)
            prices.append(max(new_price, 0.01))  # Prevent negative prices
        
        df = pd.DataFrame({
            'open': prices,
            'high': [p * (1 + abs(np.random.normal(0, 0.005))) for p in prices],
            'low': [p * (1 - abs(np.random.normal(0, 0.005))) for p in prices],
            'close': prices,
            'volume': np.random.uniform(1000, 10000, len(dates))
        }, index=dates)
        
        return df
    
    def _create_feature_engineer(self) -> FeatureEngineer:
        """Create feature engineer for all instruments."""
        
        # Get all instruments
        all_instruments = []
        for specialist_type in ['forex', 'commodities', 'equity']:
            instruments = self.config['specialists'][specialist_type]['instruments']
            all_instruments.extend(instruments)
        
        feature_config = FeatureConfig(
            instruments=all_instruments,
            timeframes=['1H'],
            technical_indicators=True,
            price_features=True,
            volume_features=True,
            volatility_features=True,
            momentum_features=True,
            correlation_features=True,
            lookback_windows=[5, 10, 20, 50],
            normalization='z_score',
            forex_features=True,
            commodity_features=True,
            equity_features=True
        )
        
        return FeatureEngineer(feature_config)
    
    def _create_specialists(self) -> Dict[str, Any]:
        """Create all specialist instances."""
        specialists = {}
        
        # Forex Specialist
        forex_config = self.config['specialists']['forex']
        specialists['forex'] = ForexSpecialist(
            instruments=forex_config['instruments'],
            market_features_dim=forex_config['market_features_dim'],
            observation_dim=forex_config['observation_dim'],
            hidden_dim=forex_config['hidden_dim']
        )
        
        # Commodities Specialist
        commodities_config = self.config['specialists']['commodities']
        specialists['commodities'] = CommoditiesSpecialist(
            instruments=commodities_config['instruments'],
            market_features_dim=commodities_config['market_features_dim'],
            observation_dim=commodities_config['observation_dim'],
            hidden_dim=commodities_config['hidden_dim']
        )
        
        # Equity Specialist
        equity_config = self.config['specialists']['equity']
        specialists['equity'] = EquitySpecialist(
            instruments=equity_config['instruments'],
            market_features_dim=equity_config['market_features_dim'],
            observation_dim=equity_config['observation_dim'],
            hidden_dim=equity_config['hidden_dim']
        )
        
        return specialists
    
    def train_specialist(self, specialist_type: str, total_timesteps: int = None) -> None:
        """Train individual specialist."""
        if total_timesteps is None:
            total_timesteps = self.config['training']['phase_1_timesteps']
        
        self.logger.info(f"Starting Phase 1 training for {specialist_type} specialist")
        self.logger.info(f"Total timesteps: {total_timesteps}")
        
        # Create trainer
        trainer = SpecialistTrainer(
            specialist_type=specialist_type,
            config_path=self.config_path,
            data_path=self.data_path,
            output_path=self.output_path
        )
        
        # Train specialist
        start_time = datetime.now()
        
        try:
            trainer.train(total_timesteps)
            
            # Update training stats
            end_time = datetime.now()
            training_time = (end_time - start_time).total_seconds()
            
            self.training_stats[specialist_type]['total_timesteps'] = total_timesteps
            self.training_stats[specialist_type]['training_time'] = training_time
            
            self.logger.info(f"{specialist_type} specialist training completed in {training_time:.2f} seconds")
            
        except Exception as e:
            self.logger.error(f"Training failed for {specialist_type} specialist: {e}")
            raise
    
    def train_all_specialists(self, total_timesteps: int = None, parallel: bool = True) -> None:
        """Train all specialists."""
        if total_timesteps is None:
            total_timesteps = self.config['training']['phase_1_timesteps']
        
        self.logger.info("Starting Phase 1 training for all specialists")
        self.logger.info(f"Total timesteps per specialist: {total_timesteps}")
        self.logger.info(f"Parallel training: {parallel}")
        
        specialist_types = ['forex', 'commodities', 'equity']
        
        if parallel:
            # Parallel training
            processes = []
            
            for specialist_type in specialist_types:
                p = mp.Process(
                    target=self.train_specialist,
                    args=(specialist_type, total_timesteps)
                )
                processes.append(p)
                p.start()
            
            # Wait for all processes to complete
            for p in processes:
                p.join()
            
            self.logger.info("All specialists training completed")
        
        else:
            # Sequential training
            for specialist_type in specialist_types:
                self.train_specialist(specialist_type, total_timesteps)
    
    def evaluate_specialists(self, n_episodes: int = 10) -> Dict[str, Any]:
        """Evaluate all trained specialists."""
        self.logger.info("Evaluating all specialists")
        
        evaluation_results = {}
        
        for specialist_type in ['forex', 'commodities', 'equity']:
            model_path = f"{self.output_path}/{specialist_type}_final.zip"
            
            if os.path.exists(model_path):
                # Create trainer for evaluation
                trainer = SpecialistTrainer(
                    specialist_type=specialist_type,
                    config_path=self.config_path,
                    data_path=self.data_path,
                    output_path=self.output_path
                )
                
                # Evaluate specialist
                eval_stats = trainer.evaluate(model_path, n_episodes)
                evaluation_results[specialist_type] = eval_stats
                
                self.logger.info(f"{specialist_type} evaluation completed:")
                self.logger.info(f"  Mean reward: {eval_stats['mean_reward']:.4f} ± {eval_stats['std_reward']:.4f}")
                self.logger.info(f"  Mean length: {eval_stats['mean_length']:.2f} ± {eval_stats['std_length']:.2f}")
            else:
                self.logger.warning(f"Model not found for {specialist_type}: {model_path}")
                evaluation_results[specialist_type] = None
        
        return evaluation_results
    
    def create_training_report(self) -> Dict[str, Any]:
        """Create comprehensive training report."""
        report = {
            'phase': 1,
            'phase_name': 'Individual Specialist Training',
            'training_stats': self.training_stats,
            'config': self.config,
            'timestamp': datetime.now().isoformat()
        }
        
        # Add evaluation results if available
        try:
            evaluation_results = self.evaluate_specialists()
            report['evaluation_results'] = evaluation_results
        except Exception as e:
            self.logger.warning(f"Could not generate evaluation results: {e}")
            report['evaluation_results'] = None
        
        return report
    
    def save_training_report(self, report: Dict[str, Any], filename: str = None) -> None:
        """Save training report to file."""
        if filename is None:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            filename = f"training_report_phase1_{timestamp}.yaml"
        
        report_path = f"{self.output_path}/{filename}"
        
        with open(report_path, 'w') as f:
            yaml.dump(report, f, default_flow_style=False)
        
        self.logger.info(f"Training report saved to {report_path}")


def main():
    """Main training script."""
    parser = argparse.ArgumentParser(description='Phase 1 Individual Specialist Training')
    parser.add_argument('--specialist', type=str, choices=['forex', 'commodities', 'equity', 'all'],
                       default='all', help='Specialist type to train')
    parser.add_argument('--timesteps', type=int, default=None,
                       help='Total training timesteps per specialist')
    parser.add_argument('--parallel', action='store_true', default=True,
                       help='Use parallel training for all specialists')
    parser.add_argument('--eval', action='store_true',
                       help='Run evaluation only')
    parser.add_argument('--config', type=str, default='config/agents.yaml',
                       help='Configuration file path')
    parser.add_argument('--data', type=str, default='data/market_data',
                       help='Market data directory')
    parser.add_argument('--output', type=str, default='models/specialists',
                       help='Output directory for models')
    parser.add_argument('--n_envs', type=int, default=8,
                       help='Number of parallel environments')
    
    args = parser.parse_args()
    
    # Create trainer
    trainer = Phase1Trainer(
        config_path=args.config,
        data_path=args.data,
        output_path=args.output,
        n_envs=args.n_envs
    )
    
    if args.eval:
        # Run evaluation
        evaluation_results = trainer.evaluate_specialists()
        print("Evaluation Results:")
        for specialist_type, results in evaluation_results.items():
            if results:
                print(f"{specialist_type}: {results['mean_reward']:.4f} ± {results['std_reward']:.4f}")
            else:
                print(f"{specialist_type}: No model found")
    else:
        # Train specialists
        if args.specialist == 'all':
            trainer.train_all_specialists(args.timesteps, args.parallel)
        else:
            trainer.train_specialist(args.specialist, args.timesteps)
        
        # Create and save training report
        report = trainer.create_training_report()
        trainer.save_training_report(report)
        
        print("Phase 1 training completed successfully!")
        print(f"Training report saved to {args.output}/training_report_phase1_*.yaml")


if __name__ == "__main__":
    main()

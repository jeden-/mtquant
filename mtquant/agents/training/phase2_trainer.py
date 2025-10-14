"""
Phase 2 Meta-Controller Training Script

This script trains the meta-controller to make portfolio-level decisions
by observing specialist performance and allocating capital accordingly.
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

# Add project root to path
sys.path.append(str(Path(__file__).parent.parent.parent.parent))

from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import VecMonitor, VecNormalize
from stable_baselines3.common.callbacks import EvalCallback, StopTrainingOnRewardThreshold
from stable_baselines3.common.monitor import Monitor

from mtquant.agents.environments.meta_controller_training_env import MetaControllerTrainingEnv, MetaControllerConfig
from mtquant.agents.training.portfolio_reward import PortfolioRewardFunction, RewardConfig
from mtquant.agents.hierarchical import (
    MetaController, ForexSpecialist, CommoditiesSpecialist, EquitySpecialist,
    CommunicationHub
)
from mtquant.risk_management.portfolio_risk_manager import PortfolioRiskManager
from mtquant.data.processors import FeatureEngineer, FeatureConfig
from mtquant.utils.logger import get_logger


class Phase2Trainer:
    """
    Phase 2 trainer for meta-controller training.
    
    Features:
    - Meta-controller learns portfolio-level decisions
    - Observes specialist performance and allocates capital
    - Uses sophisticated portfolio-level reward function
    - Comprehensive monitoring and evaluation
    """
    
    def __init__(
        self,
        config_path: str = "config/agents.yaml",
        data_path: str = "data/market_data",
        specialist_models_path: str = "models/specialists",
        output_path: str = "models/meta_controller",
        n_envs: int = 4
    ):
        self.config_path = config_path
        self.data_path = data_path
        self.specialist_models_path = specialist_models_path
        self.output_path = output_path
        self.n_envs = n_envs
        
        # Load configuration
        self.config = self._load_config()
        
        # Setup logging
        self.logger = get_logger("phase2_trainer")
        
        # Initialize components
        self.meta_controller = None
        self.specialists = {}
        self.market_data = {}
        self.feature_engineer = None
        self.portfolio_reward_function = None
        
        # Training state
        self.training_stats = {
            'episodes': 0,
            'total_timesteps': 0,
            'best_reward': -np.inf,
            'training_time': 0
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
    
    def _create_meta_controller(self) -> MetaController:
        """Create meta-controller instance."""
        meta_config = self.config['meta_controller']
        
        return MetaController(
            state_dim=meta_config['state_dim'],
            hidden_dim=meta_config['hidden_dim'],
            hidden_dim_2=meta_config['hidden_dim_2'],
            dropout=meta_config['dropout']
        )
    
    def _create_specialists(self) -> Dict[str, Any]:
        """Create specialist instances."""
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
    
    def _create_meta_controller_config(self) -> MetaControllerConfig:
        """Create meta-controller training configuration."""
        
        return MetaControllerConfig(
            initial_capital=100000.0,
            transaction_cost=0.003,
            max_position_size=0.1,
            max_portfolio_var=self.config['portfolio_risk']['max_portfolio_var'],
            max_correlation_exposure=self.config['portfolio_risk']['max_correlation_exposure'],
            max_sector_allocation=self.config['portfolio_risk']['max_sector_allocation'],
            portfolio_return_weight=1.0,
            risk_penalty_weight=2.0,
            diversification_bonus_weight=0.5,
            allocation_stability_weight=0.3,
            episode_length=1000,
            warmup_steps=50,
            allocation_update_freq=10,
            performance_lookback=20,
            market_regime_detection=True
        )
    
    def _create_reward_config(self) -> RewardConfig:
        """Create reward function configuration."""
        
        return RewardConfig(
            portfolio_return_weight=1.0,
            risk_adjusted_return_weight=2.0,
            diversification_weight=0.5,
            allocation_stability_weight=0.3,
            specialist_coordination_weight=0.4,
            risk_management_weight=3.0,
            transaction_cost_weight=1.0,
            drawdown_penalty_weight=5.0,
            target_sharpe_ratio=2.0,
            max_drawdown_threshold=0.15,
            var_confidence_level=0.95,
            min_specialist_allocation=0.1,
            max_specialist_allocation=0.7,
            target_correlation=0.3,
            allocation_change_threshold=0.2,
            max_allocation_volatility=0.1,
            performance_correlation_threshold=0.8,
            coordination_bonus_threshold=0.6
        )
    
    def _create_training_environment(self) -> MetaControllerTrainingEnv:
        """Create meta-controller training environment."""
        
        # Create meta-controller
        self.meta_controller = self._create_meta_controller()
        
        # Create specialists
        self.specialists = self._create_specialists()
        
        # Create configuration
        meta_config = self._create_meta_controller_config()
        
        # Create communication hub and risk manager
        communication_hub = CommunicationHub()
        portfolio_risk_manager = PortfolioRiskManager()
        
        # Create environment
        env = MetaControllerTrainingEnv(
            config=meta_config,
            market_data=self.market_data,
            meta_controller=self.meta_controller,
            specialists=self.specialists,
            communication_hub=communication_hub,
            portfolio_risk_manager=portfolio_risk_manager
        )
        
        return env
    
    def _create_ppo_model(self, env) -> PPO:
        """Create PPO model for meta-controller training."""
        training_config = self.config['training']
        
        model = PPO(
            "MlpPolicy",
            env,
            learning_rate=training_config['learning_rate'],
            n_steps=2048,
            batch_size=training_config['batch_size'],
            n_epochs=training_config['n_epochs'],
            gamma=training_config['gamma'],
            gae_lambda=training_config['gae_lambda'],
            clip_range=training_config['clip_range'],
            ent_coef=0.01,
            vf_coef=0.5,
            max_grad_norm=0.5,
            verbose=1,
            tensorboard_log=f"logs/meta_controller_phase2"
        )
        
        return model
    
    def _create_callbacks(self, eval_env) -> List:
        """Create training callbacks."""
        callbacks = []
        
        # Evaluation callback
        eval_callback = EvalCallback(
            eval_env,
            best_model_save_path=f"{self.output_path}/meta_controller",
            log_path=f"logs/meta_controller_phase2",
            eval_freq=self.config['training']['eval_interval'],
            n_eval_episodes=self.config['training']['eval_episodes'],
            deterministic=True,
            render=False
        )
        callbacks.append(eval_callback)
        
        # Stop training on reward threshold
        stop_callback = StopTrainingOnRewardThreshold(
            reward_threshold=0.1,  # 10% return threshold
            verbose=1
        )
        callbacks.append(stop_callback)
        
        return callbacks
    
    def train(self, total_timesteps: int = None) -> None:
        """Train the meta-controller."""
        if total_timesteps is None:
            total_timesteps = self.config['training']['phase_2_timesteps']
        
        self.logger.info(f"Starting Phase 2 meta-controller training")
        self.logger.info(f"Total timesteps: {total_timesteps}")
        
        # Load market data
        self.market_data = self._load_market_data()
        
        # Create feature engineer
        self.feature_engineer = self._create_feature_engineer()
        
        # Create training environment
        env = self._create_training_environment()
        
        # Add monitoring
        env = VecMonitor(env)
        env = VecNormalize(env, norm_obs=True, norm_reward=True)
        
        # Create PPO model
        model = self._create_ppo_model(env)
        
        # Create evaluation environment
        eval_env = self._create_training_environment()
        eval_env = VecMonitor(eval_env)
        eval_env = VecNormalize(eval_env, norm_obs=True, norm_reward=True)
        
        # Create callbacks
        callbacks = self._create_callbacks(eval_env)
        
        # Start training
        start_time = datetime.now()
        
        try:
            model.learn(
                total_timesteps=total_timesteps,
                callback=callbacks,
                progress_bar=True
            )
            
            # Save final model
            model.save(f"{self.output_path}/meta_controller_final")
            env.save(f"{self.output_path}/meta_controller_vec_normalize.pkl")
            
            self.logger.info("Meta-controller training completed successfully!")
            
        except KeyboardInterrupt:
            self.logger.info("Training interrupted by user")
            model.save(f"{self.output_path}/meta_controller_interrupted")
        
        finally:
            # Update training stats
            end_time = datetime.now()
            self.training_stats['training_time'] = (end_time - start_time).total_seconds()
            self.training_stats['total_timesteps'] = total_timesteps
            
            # Close environments
            env.close()
            eval_env.close()
            
            self.logger.info(f"Training completed in {self.training_stats['training_time']:.2f} seconds")
    
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
    
    def evaluate(self, model_path: str, n_episodes: int = 10) -> Dict[str, Any]:
        """Evaluate trained meta-controller."""
        self.logger.info(f"Evaluating meta-controller")
        
        # Load market data
        self.market_data = self._load_market_data()
        
        # Create meta-controller
        self.meta_controller = self._create_meta_controller()
        
        # Create specialists
        self.specialists = self._create_specialists()
        
        # Create evaluation environment
        eval_env = self._create_training_environment()
        
        # Load model
        model = PPO.load(model_path)
        
        # Run evaluation
        episode_rewards = []
        episode_lengths = []
        allocation_history = []
        risk_appetite_history = []
        
        for episode in range(n_episodes):
            obs, _ = eval_env.reset()
            episode_reward = 0
            episode_length = 0
            
            done = False
            while not done:
                action, _ = model.predict(obs, deterministic=True)
                obs, reward, done, truncated, _ = eval_env.step(action)
                episode_reward += reward
                episode_length += 1
                
                # Store action for analysis
                allocation_history.append(action[:3])
                risk_appetite_history.append(action[3])
                
                if truncated:
                    done = True
            
            episode_rewards.append(episode_reward)
            episode_lengths.append(episode_length)
            
            self.logger.info(f"Episode {episode + 1}: Reward = {episode_reward:.4f}, Length = {episode_length}")
        
        # Calculate statistics
        eval_stats = {
            'mean_reward': np.mean(episode_rewards),
            'std_reward': np.std(episode_rewards),
            'mean_length': np.mean(episode_lengths),
            'std_length': np.std(episode_lengths),
            'episode_rewards': episode_rewards,
            'episode_lengths': episode_lengths,
            'allocation_history': allocation_history,
            'risk_appetite_history': risk_appetite_history,
            'mean_allocation': np.mean(allocation_history, axis=0) if allocation_history else None,
            'mean_risk_appetite': np.mean(risk_appetite_history) if risk_appetite_history else None
        }
        
        self.logger.info(f"Evaluation completed:")
        self.logger.info(f"Mean reward: {eval_stats['mean_reward']:.4f} ± {eval_stats['std_reward']:.4f}")
        self.logger.info(f"Mean length: {eval_stats['mean_length']:.2f} ± {eval_stats['std_length']:.2f}")
        if eval_stats['mean_allocation'] is not None:
            self.logger.info(f"Mean allocation: {eval_stats['mean_allocation']}")
        if eval_stats['mean_risk_appetite'] is not None:
            self.logger.info(f"Mean risk appetite: {eval_stats['mean_risk_appetite']:.4f}")
        
        eval_env.close()
        return eval_stats
    
    def create_training_report(self) -> Dict[str, Any]:
        """Create comprehensive training report."""
        report = {
            'phase': 2,
            'phase_name': 'Meta-Controller Training',
            'training_stats': self.training_stats,
            'config': self.config,
            'timestamp': datetime.now().isoformat()
        }
        
        # Add evaluation results if available
        try:
            model_path = f"{self.output_path}/meta_controller_final.zip"
            if os.path.exists(model_path):
                evaluation_results = self.evaluate(model_path)
                report['evaluation_results'] = evaluation_results
        except Exception as e:
            self.logger.warning(f"Could not generate evaluation results: {e}")
            report['evaluation_results'] = None
        
        return report
    
    def save_training_report(self, report: Dict[str, Any], filename: str = None) -> None:
        """Save training report to file."""
        if filename is None:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            filename = f"training_report_phase2_{timestamp}.yaml"
        
        report_path = f"{self.output_path}/{filename}"
        
        with open(report_path, 'w') as f:
            yaml.dump(report, f, default_flow_style=False)
        
        self.logger.info(f"Training report saved to {report_path}")


def main():
    """Main training script."""
    parser = argparse.ArgumentParser(description='Phase 2 Meta-Controller Training')
    parser.add_argument('--timesteps', type=int, default=None,
                       help='Total training timesteps')
    parser.add_argument('--eval', action='store_true',
                       help='Run evaluation only')
    parser.add_argument('--model_path', type=str, default=None,
                       help='Model path for evaluation')
    parser.add_argument('--config', type=str, default='config/agents.yaml',
                       help='Configuration file path')
    parser.add_argument('--data', type=str, default='data/market_data',
                       help='Market data directory')
    parser.add_argument('--specialist_models', type=str, default='models/specialists',
                       help='Specialist models directory')
    parser.add_argument('--output', type=str, default='models/meta_controller',
                       help='Output directory for models')
    parser.add_argument('--n_envs', type=int, default=4,
                       help='Number of parallel environments')
    
    args = parser.parse_args()
    
    # Create trainer
    trainer = Phase2Trainer(
        config_path=args.config,
        data_path=args.data,
        specialist_models_path=args.specialist_models,
        output_path=args.output,
        n_envs=args.n_envs
    )
    
    if args.eval:
        # Run evaluation
        if args.model_path is None:
            args.model_path = f"{args.output}/meta_controller_final.zip"
        
        eval_stats = trainer.evaluate(args.model_path)
        print(f"Evaluation results: {eval_stats}")
    else:
        # Train meta-controller
        trainer.train(args.timesteps)
        
        # Create and save training report
        report = trainer.create_training_report()
        trainer.save_training_report(report)
        
        print("Phase 2 training completed successfully!")
        print(f"Training report saved to {args.output}/training_report_phase2_*.yaml")


if __name__ == "__main__":
    main()

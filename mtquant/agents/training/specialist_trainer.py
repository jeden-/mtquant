"""
Individual Specialist Training Scripts

This module provides training scripts for each specialist (Forex, Commodities, Equity)
using PPO with parallel environments and curriculum learning.
"""

import os
import sys
import yaml
import torch
import numpy as np
import pandas as pd
from typing import Dict, List, Tuple, Optional, Any
from datetime import datetime, timedelta
import logging
from pathlib import Path

# Add project root to path
sys.path.append(str(Path(__file__).parent.parent.parent.parent))

from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import VecMonitor, VecNormalize
from stable_baselines3.common.callbacks import EvalCallback, StopTrainingOnRewardThreshold
from stable_baselines3.common.monitor import Monitor

from mtquant.agents.environments import (
    SpecialistEnv, ParallelHierarchicalWrapper, CurriculumLearningWrapper,
    EnvironmentConfig
)
from mtquant.agents.hierarchical import (
    ForexSpecialist, CommoditiesSpecialist, EquitySpecialist,
    SpecialistRegistry, CommunicationHub
)
from mtquant.risk_management.portfolio_risk_manager import PortfolioRiskManager
from mtquant.data.processors.feature_engineering import FeatureEngineer
from mtquant.utils.logger import get_logger


class SpecialistTrainer:
    """
    Trainer for individual specialists using PPO with parallel environments.
    
    Features:
    - Parallel environment training (8+ environments)
    - Curriculum learning (easy → hard scenarios)
    - Comprehensive logging and monitoring
    - Model checkpointing and evaluation
    """
    
    def __init__(
        self,
        specialist_type: str,
        config_path: str = "config/agents.yaml",
        data_path: str = "data/market_data",
        output_path: str = "models/specialists"
    ):
        self.specialist_type = specialist_type
        self.config_path = config_path
        self.data_path = data_path
        self.output_path = output_path
        
        # Load configuration
        self.config = self._load_config()
        
        # Setup logging
        self.logger = get_logger(f"specialist_trainer_{specialist_type}")
        
        # Initialize components
        self.specialist = None
        self.market_data = {}
        self.feature_engineer = None
        self.parallel_wrapper = None
        self.curriculum_wrapper = None
        
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
        
        # Get specialist-specific config
        specialist_config = config['specialists'][self.specialist_type]
        
        return {
            'specialist': specialist_config,
            'specialists': config['specialists'],  # Add this line
            'training': config['training'],
            'portfolio_risk': config['portfolio_risk'],
            'communication': config.get('communication', {})
        }
    
    def _load_market_data(self) -> Dict[str, pd.DataFrame]:
        """Load market data for specialist instruments."""
        instruments = self.config['specialist']['instruments']
        market_data = {}
        
        for instrument in instruments:
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
        """Create feature engineer for specialist instruments."""
        instruments = self.config['specialist']['instruments']
        
        feature_config = {
            'instruments': instruments,
            'timeframes': ['1H'],
            'features': {
                'technical_indicators': True,
                'price_features': True,
                'volume_features': True,
                'volatility_features': True,
                'momentum_features': True
            },
            'lookback_windows': [5, 10, 20, 50],
            'normalization': 'z_score'
        }
        
        return FeatureEngineer(feature_config)
    
    def _create_specialist(self) -> Any:
        """Create specialist instance."""
        specialist_config = self.config['specialist']
        
        if self.specialist_type == 'forex':
            return ForexSpecialist(
                instruments=specialist_config['instruments'],
                market_features_dim=specialist_config['market_features_dim'],
                observation_dim=specialist_config['observation_dim'],
                hidden_dim=specialist_config['hidden_dim']
            )
        elif self.specialist_type == 'commodities':
            return CommoditiesSpecialist(
                instruments=specialist_config['instruments'],
                market_features_dim=specialist_config['market_features_dim'],
                observation_dim=specialist_config['observation_dim'],
                hidden_dim=specialist_config['hidden_dim']
            )
        elif self.specialist_type == 'equity':
            return EquitySpecialist(
                instruments=specialist_config['instruments'],
                market_features_dim=specialist_config['market_features_dim'],
                observation_dim=specialist_config['observation_dim'],
                hidden_dim=specialist_config['hidden_dim']
            )
        else:
            raise ValueError(f"Unknown specialist type: {self.specialist_type}")
    
    def _create_environment_config(self) -> EnvironmentConfig:
        """Create environment configuration."""
        instruments = self.config['specialist']['instruments']
        
        return EnvironmentConfig(
            instruments=instruments,
            timeframe="1H",
            lookback_window=100,
            initial_capital=100000.0,
            transaction_cost=0.003,
            max_position_size=0.1,
            max_portfolio_var=self.config['portfolio_risk']['max_portfolio_var'],
            max_correlation_exposure=self.config['portfolio_risk']['max_correlation_exposure'],
            stop_loss_pct=0.02,
            risk_penalty_weight=1.0,
            transaction_cost_weight=1.0,
            diversification_bonus_weight=0.5,
            episode_length=1000,
            warmup_steps=50
        )
    
    def _create_parallel_wrapper(self) -> ParallelHierarchicalWrapper:
        """Create parallel environment wrapper."""
        # Create specialist
        self.specialist = self._create_specialist()
        
        # Create environment config
        env_config = self._create_environment_config()
        
        # Create environment configs dict
        env_configs = {
            self.specialist_type: env_config
        }
        
        # Create communication hub and risk manager
        communication_hub = CommunicationHub()
        portfolio_risk_manager = PortfolioRiskManager()
        
        # Create parallel wrapper
        return ParallelHierarchicalWrapper(
            env_configs=env_configs,
            market_data=self.market_data,
            meta_controller=None,  # Not needed for individual training
            specialists={self.specialist_type: self.specialist},
            communication_hub=communication_hub,
            portfolio_risk_manager=portfolio_risk_manager,
            n_envs=self.config['training']['n_envs'],
            async_envs=True
        )
    
    def _create_curriculum_wrapper(self) -> CurriculumLearningWrapper:
        """Create curriculum learning wrapper."""
        curriculum_config = {
            'phases': {
                1: {'difficulty': 'easy', 'scenarios': ['low_volatility', 'trending_market']},
                2: {'difficulty': 'medium', 'scenarios': ['moderate_volatility', 'ranging_market']},
                3: {'difficulty': 'hard', 'scenarios': ['high_volatility', 'crisis_market']}
            },
            'phase_transitions': [0.4, 0.7, 1.0]
        }
        
        return CurriculumLearningWrapper(
            parallel_wrapper=self.parallel_wrapper,
            curriculum_config=curriculum_config
        )
    
    def _create_ppo_model(self, env) -> PPO:
        """Create PPO model for training."""
        training_config = self.config['training']
        specialist_config = self.config['specialists'][self.specialist_type]
        
        model = PPO(
            "MlpPolicy",
            env,
            learning_rate=specialist_config['learning_rate'],
            n_steps=2048,
            batch_size=training_config.get('batch_size', 64),
            n_epochs=training_config.get('n_epochs', 10),
            gamma=training_config.get('gamma', 0.99),
            gae_lambda=training_config.get('gae_lambda', 0.95),
            clip_range=training_config.get('clip_range', 0.2),
            ent_coef=0.01,
            vf_coef=0.5,
            max_grad_norm=0.5,
            verbose=1,
            tensorboard_log=f"logs/{self.specialist_type}_specialist"
        )
        
        return model
    
    def _create_callbacks(self, eval_env) -> List:
        """Create training callbacks."""
        callbacks = []
        
        # Evaluation callback
        eval_callback = EvalCallback(
            eval_env,
            best_model_save_path=f"{self.output_path}/{self.specialist_type}",
            log_path=f"logs/{self.specialist_type}_specialist",
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
        """Train the specialist."""
        if total_timesteps is None:
            total_timesteps = self.config['training']['phase_1_timesteps']
        
        self.logger.info(f"Starting training for {self.specialist_type} specialist")
        self.logger.info(f"Total timesteps: {total_timesteps}")
        
        # Load market data
        self.market_data = self._load_market_data()
        
        # Create feature engineer
        self.feature_engineer = self._create_feature_engineer()
        
        # Create parallel wrapper
        self.parallel_wrapper = self._create_parallel_wrapper()
        
        # Create curriculum wrapper
        self.curriculum_wrapper = self._create_curriculum_wrapper()
        
        # Create specialist environments
        specialist_envs = self.parallel_wrapper.create_specialist_envs()
        env = specialist_envs[self.specialist_type]
        
        # Add monitoring - use DummyVecEnv to avoid compatibility issues
        from stable_baselines3.common.vec_env import DummyVecEnv
        # Don't use VecNormalize with SyncVectorEnv due to compatibility issues
        # env = VecNormalize(env, norm_obs=True, norm_reward=True)
        
        # Create PPO model
        model = self._create_ppo_model(env)
        
        # Create evaluation environment
        eval_env = self._create_eval_environment()
        eval_env = DummyVecEnv([lambda: eval_env])
        # Don't use VecNormalize with DummyVecEnv due to compatibility issues
        # eval_env = VecNormalize(eval_env, norm_obs=True, norm_reward=True)
        
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
            model.save(f"{self.output_path}/{self.specialist_type}_final")
            env.save(f"{self.output_path}/{self.specialist_type}_vec_normalize.pkl")
            
            self.logger.info("Training completed successfully!")
            
        except KeyboardInterrupt:
            self.logger.info("Training interrupted by user")
            model.save(f"{self.output_path}/{self.specialist_type}_interrupted")
        
        finally:
            # Update training stats
            end_time = datetime.now()
            self.training_stats['training_time'] = (end_time - start_time).total_seconds()
            self.training_stats['total_timesteps'] = total_timesteps
            
            # Close environments
            env.close()
            eval_env.close()
            
            self.logger.info(f"Training completed in {self.training_stats['training_time']:.2f} seconds")
    
    def _create_eval_environment(self):
        """Create evaluation environment."""
        # Create single environment for evaluation
        env_config = self._create_environment_config()
        
        eval_env = SpecialistEnv(
            config=env_config,
            market_data=self.market_data,
            specialist=self.specialist,
            communication_hub=CommunicationHub(),
            portfolio_risk_manager=PortfolioRiskManager()
        )
        
        return eval_env
    
    def evaluate(self, model_path: str, n_episodes: int = 10) -> Dict[str, Any]:
        """Evaluate trained specialist."""
        self.logger.info(f"Evaluating {self.specialist_type} specialist")
        
        # Load market data
        self.market_data = self._load_market_data()
        
        # Create specialist
        self.specialist = self._create_specialist()
        
        # Create evaluation environment
        eval_env = self._create_eval_environment()
        
        # Load model
        model = PPO.load(model_path)
        
        # Run evaluation
        episode_rewards = []
        episode_lengths = []
        
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
            'episode_lengths': episode_lengths
        }
        
        self.logger.info(f"Evaluation completed:")
        self.logger.info(f"Mean reward: {eval_stats['mean_reward']:.4f} ± {eval_stats['std_reward']:.4f}")
        self.logger.info(f"Mean length: {eval_stats['mean_length']:.2f} ± {eval_stats['std_length']:.2f}")
        
        eval_env.close()
        return eval_stats


def main():
    """Main training script."""
    import argparse
    
    parser = argparse.ArgumentParser(description='Train individual specialist')
    parser.add_argument('--specialist', type=str, required=True,
                       choices=['forex', 'commodities', 'equity'],
                       help='Specialist type to train')
    parser.add_argument('--timesteps', type=int, default=None,
                       help='Total training timesteps')
    parser.add_argument('--eval', action='store_true',
                       help='Run evaluation only')
    parser.add_argument('--model_path', type=str, default=None,
                       help='Model path for evaluation')
    
    args = parser.parse_args()
    
    # Create trainer
    trainer = SpecialistTrainer(args.specialist)
    
    if args.eval:
        if args.model_path is None:
            args.model_path = f"models/specialists/{args.specialist}_final.zip"
        
        eval_stats = trainer.evaluate(args.model_path)
        print(f"Evaluation results: {eval_stats}")
    else:
        # Train specialist
        trainer.train(args.timesteps)


if __name__ == "__main__":
    main()

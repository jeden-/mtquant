"""
PPO training script for MTQuant RL agents.

Trains PPO agent using Stable Baselines3 with evaluation.
"""

import os
import json
import yaml
import numpy as np
import pandas as pd
from typing import Dict, Optional, Tuple
from datetime import datetime
import warnings

from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import DummyVecEnv
from stable_baselines3.common.callbacks import EvalCallback, StopTrainingOnRewardThreshold
from stable_baselines3.common.monitor import Monitor

from mtquant.agents.environments.base_trading_env import MTQuantTradingEnv
from mtquant.data.processors.feature_engineering import create_sample_data, prepare_training_data
from mtquant.risk_management.position_sizer import PositionSizer
from mtquant.utils.logger import get_logger


def load_config(config_path: str = "config/agents.yaml") -> Dict:
    """Load agent configuration."""
    logger = get_logger(__name__)
    
    try:
        with open(config_path, 'r') as f:
            config = yaml.safe_load(f)
        
        logger.info(f"Configuration loaded from {config_path}")
        return config
        
    except Exception as e:
        logger.error(f"Failed to load config: {e}")
        # Return default configuration
        return {
            'ppo_agent': {
                'initial_capital': 10000,
                'transaction_cost': 0.003,
                'learning_rate': 0.0003,
                'n_steps': 2048,
                'batch_size': 64,
                'gamma': 0.99,
                'gae_lambda': 0.95,
                'ent_coef': 0.01,
                'vf_coef': 0.5,
                'max_grad_norm': 0.5,
                'clip_range': 0.2,
                'n_epochs': 10
            },
            'position_sizing': {
                'volatility': {
                    'risk_per_trade': 0.02,
                    'atr_multiplier': 2.0,
                    'max_position_pct': 0.05
                }
            }
        }


def prepare_data(symbol: str = "XAUUSD", data_path: Optional[str] = None, seed: Optional[int] = None) -> pd.DataFrame:
    """
    Prepare training data for RL agent.
    
    Args:
        symbol: Trading symbol
        data_path: Path to historical data file (optional)
        
    Returns:
        Prepared DataFrame with features
    """
    logger = get_logger(__name__)
    
    try:
        if data_path and os.path.exists(data_path):
            # Load from file
            logger.info(f"Loading data from {data_path}")
            data = pd.read_csv(data_path, index_col=0, parse_dates=True)
        else:
            # Generate sample data
            logger.info(f"Generating sample data for {symbol}")
            data = create_sample_data(symbol, periods=2000, seed=seed)
        
        # Prepare features
        prepared_data = prepare_training_data(data, symbol)
        
        # Validate data quality
        if len(prepared_data) < 100:
            raise ValueError(f"Insufficient data: {len(prepared_data)} rows")
        
        # Check for missing values
        missing_count = prepared_data.isnull().sum().sum()
        if missing_count > 0:
            logger.warning(f"Found {missing_count} missing values, filling with forward fill")
            prepared_data = prepared_data.fillna(method='ffill').fillna(method='bfill')
        
        logger.info(f"Data prepared: {len(prepared_data)} rows, {missing_count} missing values filled")
        return prepared_data
        
    except Exception as e:
        logger.error(f"Data preparation failed: {e}")
        # Fallback to sample data
        return create_sample_data(symbol, periods=1000)


def create_env(data: pd.DataFrame, config: Dict, symbol: str = "XAUUSD") -> MTQuantTradingEnv:
    """Create trading environment."""
    logger = get_logger(__name__)
    
    try:
        # Create position sizer
        position_sizer = PositionSizer(config.get('position_sizing', {}))
        
        # Validate environment parameters
        if len(data) < 100:
            raise ValueError(f"Insufficient data for environment: {len(data)} rows")
        
        # Create environment
        env = MTQuantTradingEnv(
            data=data,
            symbol=symbol,
            initial_capital=config['ppo_agent']['initial_capital'],
            transaction_cost=config['ppo_agent']['transaction_cost'],
            position_sizer=position_sizer,
            config=config
        )
        
        # Test environment
        obs = env.reset()
        if obs is None or len(obs) == 0:
            raise ValueError("Environment reset failed")
        
        logger.info(f"Environment created and validated for {symbol}")
        return env
        
    except Exception as e:
        logger.error(f"Environment creation failed: {e}")
        raise


def train_ppo_agent(
    symbol: str = "XAUUSD",
    data_path: Optional[str] = None,
    config_path: str = "config/agents.yaml",
    total_timesteps: int = 200000,
    save_path: str = "models/checkpoints",
    seed: Optional[int] = None
) -> Tuple[PPO, Dict]:
    """
    Train PPO agent for trading.
    
    Args:
        symbol: Trading symbol
        data_path: Path to training data
        config_path: Path to configuration file
        total_timesteps: Total training timesteps
        save_path: Path to save model
        
    Returns:
        Tuple of (trained_model, training_info)
    """
    logger = get_logger(__name__)
    
    try:
        # Load configuration
        config = load_config(config_path)
        
        # Prepare data
        data = prepare_data(symbol, data_path, seed=seed)
        
        # Split data (80% train, 20% test)
        split_idx = int(len(data) * 0.8)
        train_data = data.iloc[:split_idx]
        test_data = data.iloc[split_idx:]
        
        logger.info(f"Data split: {len(train_data)} train, {len(test_data)} test")
        
        # Create environments
        train_env = create_env(train_data, config, symbol)
        test_env = create_env(test_data, config, symbol)
        
        # Wrap in vectorized environment
        train_env = DummyVecEnv([lambda: Monitor(train_env)])
        test_env = DummyVecEnv([lambda: Monitor(test_env)])
        
        # Create PPO model with optimized parameters
        model = PPO(
            "MlpPolicy",
            train_env,
            learning_rate=config['ppo_agent']['learning_rate'],
            n_steps=config['ppo_agent']['n_steps'],
            batch_size=config['ppo_agent']['batch_size'],
            gamma=config['ppo_agent']['gamma'],
            gae_lambda=config['ppo_agent']['gae_lambda'],
            ent_coef=config['ppo_agent']['ent_coef'],
            vf_coef=config['ppo_agent']['vf_coef'],
            max_grad_norm=config['ppo_agent']['max_grad_norm'],
            clip_range=config['ppo_agent']['clip_range'],
            n_epochs=config['ppo_agent']['n_epochs'],
            verbose=1,
            tensorboard_log=f"logs/tensorboard/{symbol}"
        )
        
        # Setup callbacks
        os.makedirs(save_path, exist_ok=True)
        
        # Early stopping callback
        stop_callback = StopTrainingOnRewardThreshold(reward_threshold=1000, verbose=1)
        
        eval_callback = EvalCallback(
            test_env,
            best_model_save_path=save_path,
            log_path=f"logs/eval/{symbol}",
            eval_freq=10000,
            deterministic=True,
            render=False,
            callback_on_new_best=stop_callback
        )
        
        # Train model
        logger.info(f"Starting training for {symbol} ({total_timesteps} timesteps)")
        start_time = datetime.now()
        
        model.learn(
            total_timesteps=total_timesteps,
            callback=eval_callback,
            tb_log_name=f"PPO_{symbol}_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
        )
        
        training_time = datetime.now() - start_time
        
        # Save final model
        final_model_path = os.path.join(save_path, f"{symbol}_ppo_final.zip")
        model.save(final_model_path)
        
        # Evaluate final model
        logger.info("Evaluating final model...")
        final_metrics = evaluate_agent(model, test_env, n_episodes=10)
        
        # Training info
        training_info = {
            'symbol': symbol,
            'total_timesteps': total_timesteps,
            'training_time': str(training_time),
            'model_path': final_model_path,
            'train_data_size': len(train_data),
            'test_data_size': len(test_data),
            'final_metrics': final_metrics,
            'config': config,
            'timestamp': datetime.now().isoformat()
        }
        
        # Save training summary
        summary_path = f"logs/training/{symbol}_training_summary.json"
        os.makedirs(os.path.dirname(summary_path), exist_ok=True)
        
        with open(summary_path, 'w') as f:
            json.dump(training_info, f, indent=2, default=str)
        
        logger.info(f"Training completed in {training_time}")
        logger.info(f"Final metrics: {final_metrics}")
        logger.info(f"Model saved to {final_model_path}")
        logger.info(f"Training summary saved to {summary_path}")
        
        return model, training_info
        
    except Exception as e:
        logger.error(f"Training failed: {e}")
        raise


def evaluate_agent(
    model: PPO,
    test_env: MTQuantTradingEnv,
    n_episodes: int = 10
) -> Dict:
    """
    Evaluate trained agent.
    
    Args:
        model: Trained PPO model
        test_env: Test environment (can be VecEnv or Gym env)
        n_episodes: Number of episodes to evaluate
        
    Returns:
        Evaluation results
    """
    logger = get_logger(__name__)
    
    try:
        episode_rewards = []
        episode_metrics = []
        
        # Check if VecEnv or Gym env
        is_vecenv = hasattr(test_env, 'envs')
        
        for episode in range(n_episodes):
            # Reset environment
            if is_vecenv:
                obs = test_env.reset()
            else:
                obs, info = test_env.reset()
            
            episode_reward = 0
            done = False
            step_count = 0
            
            while not done:
                action, _ = model.predict(obs, deterministic=False)  # Allow exploration during evaluation
                
                # Debug: log first few actions
                if step_count < 5:
                    action_val = float(action[0]) if isinstance(action, np.ndarray) else float(action)
                    logger.debug(f"Evaluation Episode {episode+1}, Step {step_count}: Action = {action_val:.4f}")
                
                # Execute action
                if is_vecenv:
                    obs, reward, done, info = test_env.step(action)
                else:
                    obs, reward, done, truncated, info = test_env.step(action)
                    done = done or truncated
                
                # Extract scalar reward from array if needed
                if isinstance(reward, np.ndarray):
                    reward = reward.item() if reward.size == 1 else reward[0]
                
                # Debug: log first few rewards
                if step_count < 5:
                    logger.debug(f"Evaluation Episode {episode+1}, Step {step_count}: Reward = {reward:.4f}")
                
                episode_reward += reward
                step_count += 1
            
            # Extract scalar episode_reward if needed
            if isinstance(episode_reward, np.ndarray):
                episode_reward_scalar = episode_reward.item() if episode_reward.size == 1 else episode_reward[0]
            else:
                episode_reward_scalar = float(episode_reward)
            
            episode_rewards.append(episode_reward_scalar)
            
            # Get metrics for this episode BEFORE any reset
            episode_metrics_obj = None
            if is_vecenv and len(test_env.envs) > 0:
                env = test_env.envs[0]
                if hasattr(env, 'env'):
                    env = env.env  # Unwrap Monitor
                if hasattr(env, 'get_episode_metrics'):
                    episode_metrics_obj = env.get_episode_metrics()
            elif hasattr(test_env, 'get_episode_metrics'):
                episode_metrics_obj = test_env.get_episode_metrics()
            
            if episode_metrics_obj is not None:
                if hasattr(episode_metrics_obj, '__dict__'):
                    episode_metrics_dict = episode_metrics_obj.__dict__
                else:
                    episode_metrics_dict = episode_metrics_obj
                episode_metrics.append(episode_metrics_dict)
                logger.info(f"Episode {episode + 1}: Reward = {episode_reward_scalar:.2f}, "
                           f"Trades = {episode_metrics_dict.get('total_trades', 0)}, "
                           f"P&L = {episode_metrics_dict.get('total_pnl', 0):.2f}")
            else:
                # Default metrics if no metrics available
                episode_metrics.append({
                    'total_trades': 0, 
                    'win_rate': 0.0, 
                    'sharpe_ratio': 0.0,
                    'total_pnl': 0.0
                })
                logger.info(f"Episode {episode + 1}: Reward = {episode_reward_scalar:.2f}, "
                           f"Trades = 0, P&L = 0.00")
            
        
        # Calculate summary statistics
        mean_reward = np.mean(episode_rewards)
        std_reward = np.std(episode_rewards)
        mean_sharpe = np.mean([m.get('sharpe_ratio', 0.0) for m in episode_metrics])
        mean_win_rate = np.mean([m.get('win_rate', 0.0) for m in episode_metrics])
        
        evaluation_results = {
            'n_episodes': n_episodes,
            'mean_reward': float(mean_reward),
            'std_reward': float(std_reward),
            'mean_sharpe_ratio': float(mean_sharpe),
            'mean_win_rate': float(mean_win_rate),
            'episode_rewards': [float(r) for r in episode_rewards],
            'episode_metrics': episode_metrics
        }
        
        logger.info(f"Evaluation completed:")
        logger.info(f"  Mean Reward: {mean_reward:.2f} ± {std_reward:.2f}")
        logger.info(f"  Mean Sharpe: {mean_sharpe:.2f}")
        logger.info(f"  Mean Win Rate: {mean_win_rate:.2f}")
        
        return evaluation_results
        
    except Exception as e:
        logger.error(f"Evaluation failed: {e}")
        raise


def main():
    """Main training script."""
    logger = get_logger(__name__)
    
    try:
        # Training parameters
        symbol = "XAUUSD"
        total_timesteps = 500000  # Increased for better learning
        n_eval_episodes = 20      # More episodes for better evaluation
        
        # Use random seed for data diversity
        import random
        seed = random.randint(1, 1000)
        
        logger.info(f"Starting PPO training for {symbol}")
        
        # Train agent
        model, training_info = train_ppo_agent(
            symbol=symbol,
            total_timesteps=total_timesteps,
            seed=seed
        )
        
        # Evaluate agent
        logger.info("Evaluating trained agent...")
        
        # Create test environment (use DummyVecEnv for consistency with training)
        config = training_info['config']
        test_data = prepare_data(symbol, None, seed=seed)
        test_env_raw = create_env(test_data, config, symbol)
        
        # Use raw environment for evaluation to avoid VecEnv auto-reset issues
        # DummyVecEnv automatically calls reset() when done=True, which clears metrics
        evaluation_results = evaluate_agent(model, test_env_raw, n_eval_episodes)
        
        # Print final results
        print("\n" + "="*50)
        print("TRAINING COMPLETED")
        print("="*50)
        print(f"Symbol: {symbol}")
        print(f"Training Time: {training_info['training_time']}")
        print(f"Total Timesteps: {total_timesteps}")
        print(f"Model Path: {training_info['model_path']}")
        print("\nEvaluation Results:")
        print(f"  Mean Reward: {evaluation_results['mean_reward']:.2f}")
        print(f"  Mean Sharpe: {evaluation_results['mean_sharpe_ratio']:.2f}")
        print(f"  Mean Win Rate: {evaluation_results['mean_win_rate']:.2f}")
        print("="*50)
        
    except Exception as e:
        logger.error(f"Main training script failed: {e}")
        raise


if __name__ == "__main__":
    main()

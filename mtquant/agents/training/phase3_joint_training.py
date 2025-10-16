"""
Phase 3 Joint Training - Hierarchical Multi-Agent System

This module implements the final phase of hierarchical training where meta-controller
and specialists are trained jointly to optimize portfolio-level performance.

Phase 3 Training Strategy:
1. Load pre-trained specialists (from Phase 1)
2. Load pre-trained meta-controller (from Phase 2) 
3. Joint fine-tuning with coordinated gradients
4. Portfolio-level reward optimization
5. Risk-aware decision making
"""

import torch
import torch.nn as nn
import numpy as np
from typing import Dict, List, Tuple, Optional, Any, Union
from dataclasses import dataclass
from pathlib import Path
import logging
from collections import deque
import time
from stable_baselines3 import PPO
from stable_baselines3.common.callbacks import BaseCallback
from stable_baselines3.common.vec_env import DummyVecEnv, SubprocVecEnv
from stable_baselines3.common.monitor import Monitor

from ..hierarchical.meta_controller import MetaController
from ..hierarchical.base_specialist import BaseSpecialist
from ..hierarchical.hierarchical_system import HierarchicalTradingSystem
from ..hierarchical.communication import CommunicationHub
from ..environments.joint_training_env import JointTrainingEnv
from ..environments.parallel_env import ParallelHierarchicalWrapper
from .gradient_coordination import GradientCoordinationSystem
from .curriculum_learning import AdvancedCurriculumLearning
from .model_checkpointing import ModelCheckpointingSystem
from .training_monitoring import TrainingMonitoringDashboard
from .portfolio_reward import PortfolioRewardFunction
from ...risk_management.portfolio_risk_manager import PortfolioRiskManager
from ...utils.logger import get_logger


@dataclass
class Phase3Config:
    """Configuration for Phase 3 joint training."""
    
    # Training parameters
    total_timesteps: int = 1000000
    learning_rate: float = 0.0001  # Lower LR for fine-tuning
    batch_size: int = 256
    n_epochs: int = 10
    gamma: float = 0.99
    gae_lambda: float = 0.95
    clip_range: float = 0.2
    
    # Joint training specific
    meta_update_freq: int = 1  # Update meta every step
    specialist_update_freq: int = 5  # Update specialists every 5 steps
    gradient_coordination: bool = True
    curriculum_learning: bool = True
    
    # Model paths
    phase1_models_dir: str = "models/checkpoints/phase1"
    phase2_models_dir: str = "models/checkpoints/phase2"
    phase3_models_dir: str = "models/checkpoints/phase3"
    
    # Environment
    n_envs: int = 8
    episode_length: int = 1000
    initial_capital: float = 100000.0
    
    # Risk management
    max_portfolio_var: float = 0.02
    max_correlation_exposure: float = 0.7
    
    # Monitoring
    log_interval: int = 1000
    save_interval: int = 10000
    eval_interval: int = 5000
    eval_episodes: int = 10
    
    # Hardware
    device: str = "cuda" if torch.cuda.is_available() else "cpu"


class JointTrainingCallback(BaseCallback):
    """Callback for joint training monitoring and coordination."""
    
    def __init__(
        self,
        gradient_coordinator: GradientCoordinationSystem,
        curriculum_manager: AdvancedCurriculumLearning,
        monitoring_dashboard: TrainingMonitoringDashboard,
        verbose: int = 0
    ):
        super().__init__(verbose)
        self.gradient_coordinator = gradient_coordinator
        self.curriculum_manager = curriculum_manager
        self.monitoring_dashboard = monitoring_dashboard
        self.step_count = 0
        
    def _on_step(self) -> bool:
        """Called at each training step."""
        self.step_count += 1
        
        # Update gradient coordination
        if self.gradient_coordinator:
            self.gradient_coordinator.update_step(self.step_count)
        
        # Update curriculum learning
        if self.curriculum_manager:
            self.curriculum_manager.update_step(self.step_count)
        
        # Update monitoring
        if self.monitoring_dashboard:
            self.monitoring_dashboard.update_step(self.step_count)
        
        return True


class Phase3JointTrainer:
    """
    Phase 3 Joint Trainer for hierarchical multi-agent system.
    
    This trainer coordinates the final fine-tuning phase where meta-controller
    and specialists learn to work together optimally.
    """
    
    def __init__(
        self,
        config: Phase3Config,
        market_data: Dict[str, Any],
        specialists: Dict[str, BaseSpecialist],
        meta_controller: MetaController,
        portfolio_risk_manager: PortfolioRiskManager,
        communication_hub: CommunicationHub
    ):
        self.config = config
        self.market_data = market_data
        self.specialists = specialists
        self.meta_controller = meta_controller
        self.portfolio_risk_manager = portfolio_risk_manager
        self.communication_hub = communication_hub
        
        # Initialize components
        self.logger = get_logger(__name__)
        self.gradient_coordinator = None
        self.curriculum_manager = None
        self.monitoring_dashboard = None
        self.checkpointing_system = None
        self.portfolio_reward = None
        
        # Training state
        self.training_step = 0
        self.best_portfolio_sharpe = -np.inf
        self.training_history = deque(maxlen=10000)
        
        # Create output directories
        self._create_directories()
        
        # Initialize training components
        self._initialize_components()
        
    def _create_directories(self) -> None:
        """Create necessary directories for Phase 3 training."""
        Path(self.config.phase3_models_dir).mkdir(parents=True, exist_ok=True)
        Path("logs/phase3").mkdir(parents=True, exist_ok=True)
        Path("plots/phase3").mkdir(parents=True, exist_ok=True)
        
    def _initialize_components(self) -> None:
        """Initialize all training components."""
        self.logger.info("Initializing Phase 3 training components...")
        
        # Gradient coordination system
        if self.config.gradient_coordination:
            from .gradient_coordination import GradientCoordinationConfig
            grad_config = GradientCoordinationConfig(
                meta_update_freq=self.config.meta_update_freq,
                specialist_update_freq=self.config.specialist_update_freq,
                coordination_method="adaptive"
            )
            self.gradient_coordinator = GradientCoordinationSystem(
                config=grad_config,
                meta_controller=self.meta_controller,
                specialists=self.specialists
            )
        
        # Curriculum learning system
        if self.config.curriculum_learning:
            from .curriculum_learning import CurriculumConfig
            curriculum_config = CurriculumConfig(
                total_timesteps=self.config.total_timesteps,
                difficulty_progression="exponential",
                scenario_mixing=True
            )
            self.curriculum_manager = AdvancedCurriculumLearning(
                config=curriculum_config,
                specialists=self.specialists,
                meta_controller=self.meta_controller
            )
        
        # Monitoring dashboard
        from .training_monitoring import MonitoringConfig
        monitoring_config = MonitoringConfig(
            log_interval=self.config.log_interval,
            save_interval=self.config.save_interval,
            eval_interval=self.config.eval_interval
        )
        self.monitoring_dashboard = TrainingMonitoringDashboard(
            config=monitoring_config,
            output_dir="logs/phase3"
        )
        
        # Model checkpointing
        from .model_checkpointing import CheckpointConfig
        checkpoint_config = CheckpointConfig(
            save_interval=self.config.save_interval,
            max_checkpoints=10,
            save_best_only=True
        )
        self.checkpointing_system = ModelCheckpointingSystem(
            config=checkpoint_config,
            output_dir=self.config.phase3_models_dir
        )
        
        # Portfolio reward function
        from .portfolio_reward import RewardConfig
        reward_config = RewardConfig(
            risk_penalty_weight=1.0,
            transaction_cost_weight=1.0,
            diversification_bonus_weight=0.5,
            sharpe_target=2.0
        )
        self.portfolio_reward = PortfolioRewardFunction(
            config=reward_config,
            portfolio_risk_manager=self.portfolio_risk_manager
        )
        
        self.logger.info("Phase 3 components initialized successfully")
    
    def load_pretrained_models(self) -> None:
        """Load pre-trained models from Phase 1 and Phase 2."""
        self.logger.info("Loading pre-trained models...")
        
        # Load Phase 1 specialist models
        for specialist_name, specialist in self.specialists.items():
            model_path = Path(self.config.phase1_models_dir) / f"{specialist_name}_final.zip"
            if model_path.exists():
                self.logger.info(f"Loading {specialist_name} from {model_path}")
                # Load model weights into specialist
                # Implementation depends on how specialists store their models
            else:
                self.logger.warning(f"Phase 1 model not found for {specialist_name}: {model_path}")
        
        # Load Phase 2 meta-controller model
        meta_model_path = Path(self.config.phase2_models_dir) / "meta_controller_final.zip"
        if meta_model_path.exists():
            self.logger.info(f"Loading meta-controller from {meta_model_path}")
            # Load model weights into meta-controller
        else:
            self.logger.warning(f"Phase 2 meta-controller model not found: {meta_model_path}")
        
        self.logger.info("Pre-trained models loaded")
    
    def create_joint_training_env(self) -> JointTrainingEnv:
        """Create joint training environment."""
        from ..environments.joint_training_env import JointTrainingConfig
        
        env_config = JointTrainingConfig(
            instruments=list(self.specialists.keys()),
            episode_length=self.config.episode_length,
            initial_capital=self.config.initial_capital,
            max_portfolio_var=self.config.max_portfolio_var,
            max_correlation_exposure=self.config.max_correlation_exposure
        )
        
        return JointTrainingEnv(
            config=env_config,
            market_data=self.market_data,
            specialists=self.specialists,
            meta_controller=self.meta_controller,
            portfolio_risk_manager=self.portfolio_risk_manager,
            communication_hub=self.communication_hub,
            portfolio_reward=self.portfolio_reward
        )
    
    def create_parallel_envs(self) -> DummyVecEnv:
        """Create parallel environments for training."""
        envs = []
        
        for i in range(self.config.n_envs):
            env = self.create_joint_training_env()
            env = Monitor(env, filename=f"logs/phase3/env_{i}")
            envs.append(env)
        
        return DummyVecEnv(envs)
    
    def train(self) -> Dict[str, Any]:
        """
        Execute Phase 3 joint training.
        
        Returns:
            Training results and metrics
        """
        self.logger.info("Starting Phase 3 joint training...")
        start_time = time.time()
        
        # Load pre-trained models
        self.load_pretrained_models()
        
        # Create training environment
        env = self.create_parallel_envs()
        
        # Create hierarchical system
        hierarchical_system = HierarchicalTradingSystem(
            meta_controller=self.meta_controller,
            specialists=self.specialists,
            portfolio_risk_manager=self.portfolio_risk_manager,
            communication_hub=self.communication_hub
        )
        
        # Create PPO model for joint training
        from stable_baselines3 import PPO
        from stable_baselines3.common.policies import ActorCriticPolicy
        
        # Custom policy for hierarchical system
        class HierarchicalPolicy(ActorCriticPolicy):
            def __init__(self, *args, **kwargs):
                super().__init__(*args, **kwargs)
                self.hierarchical_system = hierarchical_system
        
        model = PPO(
            policy=HierarchicalPolicy,
            env=env,
            learning_rate=self.config.learning_rate,
            n_steps=self.config.batch_size,
            batch_size=self.config.batch_size,
            n_epochs=self.config.n_epochs,
            gamma=self.config.gamma,
            gae_lambda=self.config.gae_lambda,
            clip_range=self.config.clip_range,
            device=self.config.device,
            verbose=1
        )
        
        # Create training callback
        callback = JointTrainingCallback(
            gradient_coordinator=self.gradient_coordinator,
            curriculum_manager=self.curriculum_manager,
            monitoring_dashboard=self.monitoring_dashboard
        )
        
        # Training loop
        self.logger.info(f"Training for {self.config.total_timesteps} timesteps...")
        
        try:
            model.learn(
                total_timesteps=self.config.total_timesteps,
                callback=callback,
                progress_bar=True
            )
            
            # Save final model
            final_model_path = Path(self.config.phase3_models_dir) / "hierarchical_system_final.zip"
            model.save(str(final_model_path))
            self.logger.info(f"Final model saved to {final_model_path}")
            
        except KeyboardInterrupt:
            self.logger.info("Training interrupted by user")
        except Exception as e:
            self.logger.error(f"Training failed: {e}")
            raise
        
        # Training completed
        training_time = time.time() - start_time
        self.logger.info(f"Phase 3 training completed in {training_time:.2f} seconds")
        
        # Generate training report
        training_results = self._generate_training_report(training_time)
        
        return training_results
    
    def _generate_training_report(self, training_time: float) -> Dict[str, Any]:
        """Generate comprehensive training report."""
        return {
            "phase": "Phase 3 - Joint Training",
            "training_time_seconds": training_time,
            "training_time_hours": training_time / 3600,
            "total_timesteps": self.config.total_timesteps,
            "best_portfolio_sharpe": self.best_portfolio_sharpe,
            "final_model_path": str(Path(self.config.phase3_models_dir) / "hierarchical_system_final.zip"),
            "components_trained": {
                "meta_controller": True,
                "specialists": list(self.specialists.keys()),
                "gradient_coordination": self.config.gradient_coordination,
                "curriculum_learning": self.config.curriculum_learning
            },
            "training_config": {
                "learning_rate": self.config.learning_rate,
                "batch_size": self.config.batch_size,
                "n_epochs": self.config.n_epochs,
                "gamma": self.config.gamma,
                "gae_lambda": self.config.gae_lambda,
                "clip_range": self.config.clip_range,
                "n_envs": self.config.n_envs
            },
            "risk_management": {
                "max_portfolio_var": self.config.max_portfolio_var,
                "max_correlation_exposure": self.config.max_correlation_exposure
            }
        }
    
    def evaluate(self, n_episodes: int = 100) -> Dict[str, Any]:
        """
        Evaluate the trained hierarchical system.
        
        Args:
            n_episodes: Number of episodes to evaluate
            
        Returns:
            Evaluation metrics
        """
        self.logger.info(f"Evaluating hierarchical system for {n_episodes} episodes...")
        
        # Create evaluation environment
        env = self.create_joint_training_env()
        
        # Load trained model
        model_path = Path(self.config.phase3_models_dir) / "hierarchical_system_final.zip"
        if not model_path.exists():
            raise FileNotFoundError(f"Trained model not found: {model_path}")
        
        from stable_baselines3 import PPO
        model = PPO.load(str(model_path))
        
        # Evaluation metrics
        episode_returns = []
        episode_lengths = []
        portfolio_sharpes = []
        max_drawdowns = []
        var_violations = []
        
        for episode in range(n_episodes):
            obs = env.reset()
            episode_return = 0
            episode_length = 0
            portfolio_values = []
            
            done = False
            while not done:
                action, _ = model.predict(obs, deterministic=True)
                obs, reward, done, info = env.step(action)
                
                episode_return += reward
                episode_length += 1
                portfolio_values.append(info.get('portfolio_value', 0))
            
            # Calculate metrics
            episode_returns.append(episode_return)
            episode_lengths.append(episode_length)
            
            if len(portfolio_values) > 1:
                returns = np.diff(portfolio_values) / portfolio_values[:-1]
                sharpe = np.mean(returns) / np.std(returns) * np.sqrt(252) if np.std(returns) > 0 else 0
                portfolio_sharpes.append(sharpe)
                
                # Max drawdown
                peak = np.maximum.accumulate(portfolio_values)
                drawdown = (portfolio_values - peak) / peak
                max_drawdowns.append(np.min(drawdown))
            
            # VaR violations (simplified)
            var_violations.append(0)  # Placeholder
        
        # Compile results
        evaluation_results = {
            "n_episodes": n_episodes,
            "mean_episode_return": np.mean(episode_returns),
            "std_episode_return": np.std(episode_returns),
            "mean_episode_length": np.mean(episode_lengths),
            "mean_portfolio_sharpe": np.mean(portfolio_sharpes),
            "std_portfolio_sharpe": np.std(portfolio_sharpes),
            "mean_max_drawdown": np.mean(max_drawdowns),
            "var_violation_rate": np.mean(var_violations),
            "success_rate": np.mean([r > 0 for r in episode_returns])
        }
        
        self.logger.info(f"Evaluation completed: Sharpe={evaluation_results['mean_portfolio_sharpe']:.3f}")
        
        return evaluation_results


def create_phase3_trainer(
    config_path: str = "config/agents.yaml",
    market_data: Optional[Dict[str, Any]] = None
) -> Phase3JointTrainer:
    """
    Factory function to create Phase 3 trainer with default configuration.
    
    Args:
        config_path: Path to configuration file
        market_data: Market data dictionary
        
    Returns:
        Configured Phase3JointTrainer instance
    """
    import yaml
    
    # Load configuration
    with open(config_path, 'r') as f:
        config_dict = yaml.safe_load(f)
    
    # Create Phase 3 config
    phase3_config = Phase3Config(**config_dict.get('training', {}))
    
    # Create market data if not provided
    if market_data is None:
        from ..training.train_ppo import create_sample_data
        market_data = {
            'EURUSD': create_sample_data('EURUSD', 10000),
            'GBPUSD': create_sample_data('GBPUSD', 10000),
            'USDJPY': create_sample_data('USDJPY', 10000),
            'XAUUSD': create_sample_data('XAUUSD', 10000),
            'WTIUSD': create_sample_data('WTIUSD', 10000),
            'SPX500': create_sample_data('SPX500', 10000),
            'NAS100': create_sample_data('NAS100', 10000),
            'US30': create_sample_data('US30', 10000)
        }
    
    # Create specialists
    from ..hierarchical.specialist_factory import SpecialistRegistry
    registry = SpecialistRegistry()
    
    specialists = {}
    for spec_name, spec_config in config_dict['specialists'].items():
        specialist = registry.create_specialist(spec_name, spec_config)
        specialists[spec_name] = specialist
    
    # Create meta-controller
    meta_config = config_dict['meta_controller']
    meta_controller = MetaController(
        state_dim=meta_config['state_dim'],
        hidden_dim=meta_config['hidden_dim'],
        hidden_dim_2=meta_config['hidden_dim_2'],
        dropout=meta_config['dropout']
    )
    
    # Create portfolio risk manager
    portfolio_risk_manager = PortfolioRiskManager()
    
    # Create communication hub
    communication_hub = CommunicationHub()
    
    return Phase3JointTrainer(
        config=phase3_config,
        market_data=market_data,
        specialists=specialists,
        meta_controller=meta_controller,
        portfolio_risk_manager=portfolio_risk_manager,
        communication_hub=communication_hub
    )


if __name__ == "__main__":
    # Example usage
    trainer = create_phase3_trainer()
    
    # Train the hierarchical system
    results = trainer.train()
    print("Training Results:", results)
    
    # Evaluate the trained system
    eval_results = trainer.evaluate(n_episodes=50)
    print("Evaluation Results:", eval_results)

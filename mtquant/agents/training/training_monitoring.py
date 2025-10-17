"""
Training Monitoring Dashboard

This module provides comprehensive training monitoring and visualization
for the hierarchical multi-agent training pipeline.
"""

import os
import json
import time
from typing import Dict, List, Tuple, Optional, Any, Union
from dataclasses import dataclass, asdict
from datetime import datetime, timedelta
from pathlib import Path
import logging
import threading
import queue
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from collections import deque, defaultdict

from ..hierarchical.meta_controller import MetaController
from ..hierarchical.base_specialist import BaseSpecialist


@dataclass
class TrainingMetrics:
    """Training metrics data structure."""
    
    # Basic metrics
    episode: int
    step: int
    timestep: int
    phase: int
    
    # Performance metrics
    joint_reward: float
    meta_reward: float
    specialist_rewards: Dict[str, float]
    
    # Training metrics
    learning_rate: float
    gradient_norm: float
    loss: float
    
    # System metrics
    memory_usage: float
    gpu_usage: float
    training_time: float
    
    # Coordination metrics
    allocation_efficiency: float
    specialist_synchronization: float
    risk_coordination: float
    
    # Curriculum metrics
    current_difficulty: str
    current_scenario: str
    curriculum_progress: float
    
    # Timestamp
    timestamp: datetime
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for serialization."""
        data = asdict(self)
        data['timestamp'] = self.timestamp.isoformat()
        return data
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'TrainingMetrics':
        """Create from dictionary."""
        data['timestamp'] = datetime.fromisoformat(data['timestamp'])
        return cls(**data)


@dataclass
class MonitoringConfig:
    """Configuration for training monitoring."""
    
    # Monitoring intervals
    metrics_interval: float = 1.0  # seconds
    plot_interval: float = 60.0    # seconds
    save_interval: float = 300.0   # seconds
    
    # Data retention
    max_metrics_history: int = 10000
    max_plot_history: int = 1000
    
    # Output directories
    logs_dir: str = "logs/training"
    plots_dir: str = "plots/training"
    metrics_dir: str = "metrics/training"
    
    # Plotting configuration
    plot_style: str = "seaborn"
    figure_size: Tuple[int, int] = (12, 8)
    dpi: int = 100
    
    # Real-time monitoring
    enable_realtime: bool = True
    websocket_port: int = 8765
    
    # Alerting
    enable_alerts: bool = True
    performance_threshold: float = -0.1
    memory_threshold: float = 0.9
    gpu_threshold: float = 0.95


class TrainingMonitoringDashboard:
    """
    Comprehensive training monitoring dashboard.
    
    Features:
    - Real-time metrics collection
    - Performance visualization
    - System resource monitoring
    - Alert system
    - Historical data analysis
    """
    
    def __init__(self, config: MonitoringConfig):
        self.config = config
        self.logger = logging.getLogger(self.__class__.__name__)
        
        # Create directories
        self._create_directories()
        
        # Metrics storage
        self.metrics_history: deque = deque(maxlen=config.max_metrics_history)
        self.plot_history: deque = deque(maxlen=config.max_plot_history)
        
        # Real-time monitoring
        self.monitoring_active = False
        self.monitoring_thread = None
        self.metrics_queue = queue.Queue()
        
        # Performance tracking
        self.performance_alerts: List[Dict[str, Any]] = []
        self.system_alerts: List[Dict[str, Any]] = []
        
        # Plotting setup
        try:
            plt.style.use(config.plot_style)
        except OSError:
            # Fallback to default style if seaborn is not available
            plt.style.use('default')
        sns.set_palette("husl")
        
        # Start monitoring
        if config.enable_realtime:
            self.start_monitoring()
    
    def _create_directories(self) -> None:
        """Create necessary directories."""
        
        directories = [
            self.config.logs_dir,
            self.config.plots_dir,
            self.config.metrics_dir
        ]
        
        for directory in directories:
            Path(directory).mkdir(parents=True, exist_ok=True)
    
    def start_monitoring(self) -> None:
        """Start real-time monitoring."""
        
        if self.monitoring_active:
            return
        
        self.monitoring_active = True
        self.monitoring_thread = threading.Thread(target=self._monitoring_loop, daemon=True)
        self.monitoring_thread.start()
        
        self.logger.info("Training monitoring started")
    
    def stop_monitoring(self) -> None:
        """Stop real-time monitoring."""
        
        self.monitoring_active = False
        if self.monitoring_thread:
            self.monitoring_thread.join(timeout=5.0)
        
        self.logger.info("Training monitoring stopped")
    
    def _monitoring_loop(self) -> None:
        """Main monitoring loop."""
        
        last_plot_time = time.time()
        last_save_time = time.time()
        
        while self.monitoring_active:
            try:
                # Process metrics queue
                self._process_metrics_queue()
                
                # Generate plots
                current_time = time.time()
                if current_time - last_plot_time >= self.config.plot_interval:
                    self._generate_plots()
                    last_plot_time = current_time
                
                # Save metrics
                if current_time - last_save_time >= self.config.save_interval:
                    self._save_metrics()
                    last_save_time = current_time
                
                # Check alerts
                self._check_alerts()
                
                time.sleep(self.config.metrics_interval)
                
            except Exception as e:
                self.logger.error(f"Error in monitoring loop: {e}")
                time.sleep(self.config.metrics_interval)
    
    def _process_metrics_queue(self) -> None:
        """Process metrics from queue."""
        
        while not self.metrics_queue.empty():
            try:
                metrics = self.metrics_queue.get_nowait()
                self.metrics_history.append(metrics)
                self.plot_history.append(metrics)
            except queue.Empty:
                break
    
    def log_metrics(self, metrics: TrainingMetrics) -> None:
        """Log training metrics."""
        
        # Add to queue for processing
        if self.config.enable_realtime:
            self.metrics_queue.put(metrics)
        
        # Store directly
        self.metrics_history.append(metrics)
        self.plot_history.append(metrics)
        
        # Log to file
        self._log_to_file(metrics)
    
    def _log_to_file(self, metrics: TrainingMetrics) -> None:
        """Log metrics to file."""
        
        log_file = Path(self.config.logs_dir) / f"training_metrics_{datetime.now().strftime('%Y%m%d')}.jsonl"
        
        with open(log_file, 'a') as f:
            f.write(json.dumps(metrics.to_dict()) + '\n')
    
    def _generate_plots(self) -> None:
        """Generate training plots."""
        
        if len(self.plot_history) < 2:
            return
        
        try:
            # Convert to DataFrame
            df = pd.DataFrame([metrics.to_dict() for metrics in self.plot_history])
            df['timestamp'] = pd.to_datetime(df['timestamp'])
            
            # Create plots
            self._plot_performance_metrics(df)
            self._plot_training_metrics(df)
            self._plot_coordination_metrics(df)
            self._plot_system_metrics(df)
            self._plot_curriculum_progress(df)
            
        except Exception as e:
            self.logger.error(f"Error generating plots: {e}")
    
    def _plot_performance_metrics(self, df: pd.DataFrame) -> None:
        """Plot performance metrics."""
        
        fig, axes = plt.subplots(2, 2, figsize=self.config.figure_size)
        fig.suptitle('Performance Metrics', fontsize=16)
        
        # Joint reward
        axes[0, 0].plot(df['timestamp'], df['joint_reward'], label='Joint Reward')
        axes[0, 0].set_title('Joint Reward')
        axes[0, 0].set_ylabel('Reward')
        axes[0, 0].grid(True)
        axes[0, 0].legend()
        
        # Meta reward
        axes[0, 1].plot(df['timestamp'], df['meta_reward'], label='Meta Reward', color='orange')
        axes[0, 1].set_title('Meta-Controller Reward')
        axes[0, 1].set_ylabel('Reward')
        axes[0, 1].grid(True)
        axes[0, 1].legend()
        
        # Specialist rewards
        specialist_columns = [col for col in df.columns if col.startswith('specialist_rewards_')]
        if specialist_columns:
            for col in specialist_columns:
                specialist_name = col.replace('specialist_rewards_', '')
                axes[1, 0].plot(df['timestamp'], df[col], label=specialist_name)
            axes[1, 0].set_title('Specialist Rewards')
            axes[1, 0].set_ylabel('Reward')
            axes[1, 0].grid(True)
            axes[1, 0].legend()
        
        # Reward distribution
        axes[1, 1].hist(df['joint_reward'], bins=50, alpha=0.7, label='Joint Reward')
        axes[1, 1].set_title('Reward Distribution')
        axes[1, 1].set_xlabel('Reward')
        axes[1, 1].set_ylabel('Frequency')
        axes[1, 1].grid(True)
        axes[1, 1].legend()
        
        plt.tight_layout()
        plt.savefig(Path(self.config.plots_dir) / 'performance_metrics.png', dpi=self.config.dpi)
        plt.close()
    
    def _plot_training_metrics(self, df: pd.DataFrame) -> None:
        """Plot training metrics."""
        
        fig, axes = plt.subplots(2, 2, figsize=self.config.figure_size)
        fig.suptitle('Training Metrics', fontsize=16)
        
        # Learning rate
        axes[0, 0].plot(df['timestamp'], df['learning_rate'], label='Learning Rate')
        axes[0, 0].set_title('Learning Rate')
        axes[0, 0].set_ylabel('Learning Rate')
        axes[0, 0].grid(True)
        axes[0, 0].legend()
        
        # Gradient norm
        axes[0, 1].plot(df['timestamp'], df['gradient_norm'], label='Gradient Norm', color='red')
        axes[0, 1].set_title('Gradient Norm')
        axes[0, 1].set_ylabel('Gradient Norm')
        axes[0, 1].grid(True)
        axes[0, 1].legend()
        
        # Loss
        axes[1, 0].plot(df['timestamp'], df['loss'], label='Loss', color='green')
        axes[1, 0].set_title('Training Loss')
        axes[1, 0].set_ylabel('Loss')
        axes[1, 0].grid(True)
        axes[1, 0].legend()
        
        # Training time
        axes[1, 1].plot(df['timestamp'], df['training_time'], label='Training Time', color='purple')
        axes[1, 1].set_title('Training Time')
        axes[1, 1].set_ylabel('Time (seconds)')
        axes[1, 1].grid(True)
        axes[1, 1].legend()
        
        plt.tight_layout()
        plt.savefig(Path(self.config.plots_dir) / 'training_metrics.png', dpi=self.config.dpi)
        plt.close()
    
    def _plot_coordination_metrics(self, df: pd.DataFrame) -> None:
        """Plot coordination metrics."""
        
        fig, axes = plt.subplots(2, 2, figsize=self.config.figure_size)
        fig.suptitle('Coordination Metrics', fontsize=16)
        
        # Allocation efficiency
        axes[0, 0].plot(df['timestamp'], df['allocation_efficiency'], label='Allocation Efficiency')
        axes[0, 0].set_title('Allocation Efficiency')
        axes[0, 0].set_ylabel('Efficiency')
        axes[0, 0].grid(True)
        axes[0, 0].legend()
        
        # Specialist synchronization
        axes[0, 1].plot(df['timestamp'], df['specialist_synchronization'], label='Synchronization', color='orange')
        axes[0, 1].set_title('Specialist Synchronization')
        axes[0, 1].set_ylabel('Synchronization')
        axes[0, 1].grid(True)
        axes[0, 1].legend()
        
        # Risk coordination
        axes[1, 0].plot(df['timestamp'], df['risk_coordination'], label='Risk Coordination', color='red')
        axes[1, 0].set_title('Risk Coordination')
        axes[1, 0].set_ylabel('Coordination')
        axes[1, 0].grid(True)
        axes[1, 0].legend()
        
        # Coordination correlation
        coordination_metrics = ['allocation_efficiency', 'specialist_synchronization', 'risk_coordination']
        correlation_matrix = df[coordination_metrics].corr()
        sns.heatmap(correlation_matrix, annot=True, cmap='coolwarm', center=0, ax=axes[1, 1])
        axes[1, 1].set_title('Coordination Metrics Correlation')
        
        plt.tight_layout()
        plt.savefig(Path(self.config.plots_dir) / 'coordination_metrics.png', dpi=self.config.dpi)
        plt.close()
    
    def _plot_system_metrics(self, df: pd.DataFrame) -> None:
        """Plot system metrics."""
        
        fig, axes = plt.subplots(2, 2, figsize=self.config.figure_size)
        fig.suptitle('System Metrics', fontsize=16)
        
        # Memory usage
        axes[0, 0].plot(df['timestamp'], df['memory_usage'], label='Memory Usage')
        axes[0, 0].set_title('Memory Usage')
        axes[0, 0].set_ylabel('Memory Usage (%)')
        axes[0, 0].grid(True)
        axes[0, 0].legend()
        
        # GPU usage
        axes[0, 1].plot(df['timestamp'], df['gpu_usage'], label='GPU Usage', color='orange')
        axes[0, 1].set_title('GPU Usage')
        axes[0, 1].set_ylabel('GPU Usage (%)')
        axes[0, 1].grid(True)
        axes[0, 1].legend()
        
        # Training time distribution
        axes[1, 0].hist(df['training_time'], bins=50, alpha=0.7, label='Training Time')
        axes[1, 0].set_title('Training Time Distribution')
        axes[1, 0].set_xlabel('Time (seconds)')
        axes[1, 0].set_ylabel('Frequency')
        axes[1, 0].grid(True)
        axes[1, 0].legend()
        
        # System resources correlation
        system_metrics = ['memory_usage', 'gpu_usage', 'training_time']
        correlation_matrix = df[system_metrics].corr()
        sns.heatmap(correlation_matrix, annot=True, cmap='coolwarm', center=0, ax=axes[1, 1])
        axes[1, 1].set_title('System Metrics Correlation')
        
        plt.tight_layout()
        plt.savefig(Path(self.config.plots_dir) / 'system_metrics.png', dpi=self.config.dpi)
        plt.close()
    
    def _plot_curriculum_progress(self, df: pd.DataFrame) -> None:
        """Plot curriculum learning progress."""
        
        fig, axes = plt.subplots(2, 2, figsize=self.config.figure_size)
        fig.suptitle('Curriculum Learning Progress', fontsize=16)
        
        # Difficulty progression
        difficulty_map = {'easy': 1, 'medium': 2, 'hard': 3, 'expert': 4}
        df['difficulty_numeric'] = df['current_difficulty'].map(difficulty_map)
        axes[0, 0].plot(df['timestamp'], df['difficulty_numeric'], label='Difficulty Level')
        axes[0, 0].set_title('Difficulty Progression')
        axes[0, 0].set_ylabel('Difficulty Level')
        axes[0, 0].grid(True)
        axes[0, 0].legend()
        
        # Curriculum progress
        axes[0, 1].plot(df['timestamp'], df['curriculum_progress'], label='Curriculum Progress', color='green')
        axes[0, 1].set_title('Curriculum Progress')
        axes[0, 1].set_ylabel('Progress')
        axes[0, 1].grid(True)
        axes[0, 1].legend()
        
        # Scenario distribution
        scenario_counts = df['current_scenario'].value_counts()
        axes[1, 0].pie(scenario_counts.values, labels=scenario_counts.index, autopct='%1.1f%%')
        axes[1, 0].set_title('Scenario Distribution')
        
        # Performance by difficulty
        difficulty_performance = df.groupby('current_difficulty')['joint_reward'].mean()
        axes[1, 1].bar(difficulty_performance.index, difficulty_performance.values)
        axes[1, 1].set_title('Performance by Difficulty')
        axes[1, 1].set_ylabel('Mean Reward')
        axes[1, 1].grid(True)
        
        plt.tight_layout()
        plt.savefig(Path(self.config.plots_dir) / 'curriculum_progress.png', dpi=self.config.dpi)
        plt.close()
    
    def _check_alerts(self) -> None:
        """Check for alert conditions."""
        
        if not self.config.enable_alerts or len(self.metrics_history) == 0:
            return
        
        latest_metrics = self.metrics_history[-1]
        
        # Performance alerts
        if latest_metrics.joint_reward < self.config.performance_threshold:
            alert = {
                'type': 'performance',
                'message': f'Low performance: {latest_metrics.joint_reward:.4f}',
                'timestamp': latest_metrics.timestamp,
                'severity': 'warning'
            }
            self.performance_alerts.append(alert)
            self.logger.warning(alert['message'])
        
        # System alerts
        if latest_metrics.memory_usage > self.config.memory_threshold:
            alert = {
                'type': 'system',
                'message': f'High memory usage: {latest_metrics.memory_usage:.2%}',
                'timestamp': latest_metrics.timestamp,
                'severity': 'warning'
            }
            self.system_alerts.append(alert)
            self.logger.warning(alert['message'])
        
        if latest_metrics.gpu_usage > self.config.gpu_threshold:
            alert = {
                'type': 'system',
                'message': f'High GPU usage: {latest_metrics.gpu_usage:.2%}',
                'timestamp': latest_metrics.timestamp,
                'severity': 'warning'
            }
            self.system_alerts.append(alert)
            self.logger.warning(alert['message'])
    
    def _save_metrics(self) -> None:
        """Save metrics to file."""
        
        if len(self.metrics_history) == 0:
            return
        
        # Save recent metrics
        recent_metrics = list(self.metrics_history)[-1000:]  # Last 1000 metrics
        
        metrics_file = Path(self.config.metrics_dir) / f"training_metrics_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
        
        with open(metrics_file, 'w') as f:
            json.dump([metrics.to_dict() for metrics in recent_metrics], f, indent=2)
        
        self.logger.info(f"Metrics saved to: {metrics_file}")
    
    def get_dashboard_stats(self) -> Dict[str, Any]:
        """Get dashboard statistics."""
        
        stats = {
            'total_metrics': len(self.metrics_history),
            'monitoring_active': self.monitoring_active,
            'performance_alerts': len(self.performance_alerts),
            'system_alerts': len(self.system_alerts),
            'latest_metrics': self.metrics_history[-1].to_dict() if self.metrics_history else None,
            'plot_files': list(Path(self.config.plots_dir).glob('*.png')),
            'log_files': list(Path(self.config.logs_dir).glob('*.jsonl')),
            'metrics_files': list(Path(self.config.metrics_dir).glob('*.json'))
        }
        
        return stats
    
    def generate_report(self) -> Dict[str, Any]:
        """Generate comprehensive training report."""
        
        if len(self.metrics_history) == 0:
            return {'error': 'No metrics available'}
        
        # Convert to DataFrame
        df = pd.DataFrame([metrics.to_dict() for metrics in self.metrics_history])
        df['timestamp'] = pd.to_datetime(df['timestamp'])
        
        # Calculate statistics
        report = {
            'training_summary': {
                'total_episodes': df['episode'].max(),
                'total_steps': df['step'].max(),
                'total_timesteps': df['timestep'].max(),
                'training_duration': (df['timestamp'].max() - df['timestamp'].min()).total_seconds(),
                'current_phase': df['phase'].iloc[-1] if len(df) > 0 else 0
            },
            'performance_summary': {
                'mean_joint_reward': df['joint_reward'].mean(),
                'std_joint_reward': df['joint_reward'].std(),
                'best_joint_reward': df['joint_reward'].max(),
                'worst_joint_reward': df['joint_reward'].min(),
                'mean_meta_reward': df['meta_reward'].mean(),
                'mean_specialist_rewards': {
                    col.replace('specialist_rewards_', ''): df[col].mean()
                    for col in df.columns if col.startswith('specialist_rewards_')
                }
            },
            'training_summary': {
                'mean_learning_rate': df['learning_rate'].mean(),
                'mean_gradient_norm': df['gradient_norm'].mean(),
                'mean_loss': df['loss'].mean(),
                'mean_training_time': df['training_time'].mean()
            },
            'coordination_summary': {
                'mean_allocation_efficiency': df['allocation_efficiency'].mean(),
                'mean_specialist_synchronization': df['specialist_synchronization'].mean(),
                'mean_risk_coordination': df['risk_coordination'].mean()
            },
            'system_summary': {
                'mean_memory_usage': df['memory_usage'].mean(),
                'mean_gpu_usage': df['gpu_usage'].mean(),
                'max_memory_usage': df['memory_usage'].max(),
                'max_gpu_usage': df['gpu_usage'].max()
            },
            'curriculum_summary': {
                'current_difficulty': df['current_difficulty'].iloc[-1] if len(df) > 0 else 'unknown',
                'current_scenario': df['current_scenario'].iloc[-1] if len(df) > 0 else 'unknown',
                'mean_curriculum_progress': df['curriculum_progress'].mean(),
                'difficulty_distribution': df['current_difficulty'].value_counts().to_dict(),
                'scenario_distribution': df['current_scenario'].value_counts().to_dict()
            },
            'alerts_summary': {
                'performance_alerts': len(self.performance_alerts),
                'system_alerts': len(self.system_alerts),
                'recent_alerts': self.performance_alerts[-5:] + self.system_alerts[-5:]
            }
        }
        
        return report
    
    def cleanup(self) -> None:
        """Clean up monitoring resources."""
        
        self.stop_monitoring()
        
        # Save final metrics
        self._save_metrics()
        
        # Generate final plots
        self._generate_plots()
        
        # Generate final report
        report = self.generate_report()
        report_file = Path(self.config.metrics_dir) / f"training_report_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
        
        with open(report_file, 'w') as f:
            json.dump(report, f, indent=2)
        
        self.logger.info(f"Final report saved to: {report_file}")

"""
Extended unit tests for training_monitoring.py to increase coverage.
"""

import pytest
import os
import json
import tempfile
import shutil
from unittest.mock import Mock, patch, MagicMock
from pathlib import Path
from datetime import datetime
from collections import deque

from mtquant.agents.training.training_monitoring import (
    TrainingMetrics,
    MonitoringConfig,
    TrainingMonitoringDashboard
)


class TestTrainingMetrics:
    """Test TrainingMetrics dataclass."""
    
    def test_training_metrics_creation(self):
        """Test TrainingMetrics creation."""
        metrics = TrainingMetrics(
            episode=100,
            step=1000,
            timestep=50000,
            phase=2,
            joint_reward=0.85,
            meta_reward=0.75,
            specialist_rewards={"forex": 0.8, "commodities": 0.9},
            learning_rate=0.001,
            gradient_norm=0.5,
            loss=0.1,
            memory_usage=0.6,
            gpu_usage=0.4,
            training_time=3600,
            allocation_efficiency=0.8,
            specialist_synchronization=0.7,
            risk_coordination=0.9,
            current_difficulty="medium",
            current_scenario="bull_market",
            curriculum_progress=0.5,
            timestamp=datetime.now()
        )
        
        assert metrics.episode == 100
        assert metrics.step == 1000
        assert metrics.timestep == 50000
        assert metrics.phase == 2
        assert metrics.joint_reward == 0.85
        assert metrics.meta_reward == 0.75
        assert metrics.specialist_rewards["forex"] == 0.8
        assert metrics.specialist_rewards["commodities"] == 0.9
        assert metrics.learning_rate == 0.001
        assert metrics.gradient_norm == 0.5
        assert metrics.loss == 0.1
        assert metrics.memory_usage == 0.6
        assert metrics.gpu_usage == 0.4
        assert metrics.training_time == 3600
        assert metrics.allocation_efficiency == 0.8
        assert metrics.specialist_synchronization == 0.7
        assert metrics.risk_coordination == 0.9
        assert metrics.current_difficulty == "medium"
        assert metrics.current_scenario == "bull_market"
        assert metrics.curriculum_progress == 0.5
        assert metrics.timestamp is not None
    
    def test_training_metrics_defaults(self):
        """Test TrainingMetrics with default values."""
        metrics = TrainingMetrics(
            episode=50,
            step=500,
            timestep=25000,
            phase=1,
            joint_reward=0.7,
            meta_reward=0.6,
            specialist_rewards={},
            learning_rate=0.002,
            gradient_norm=0.3,
            loss=0.2,
            memory_usage=0.5,
            gpu_usage=0.3,
            training_time=1800,
            allocation_efficiency=0.6,
            specialist_synchronization=0.5,
            risk_coordination=0.7,
            current_difficulty="easy",
            current_scenario="stable_market",
            curriculum_progress=0.3,
            timestamp=datetime.now()
        )
        
        assert metrics.episode == 50
        assert metrics.step == 500
        assert metrics.timestep == 25000
        assert metrics.phase == 1
        assert metrics.joint_reward == 0.7
        assert metrics.meta_reward == 0.6
        assert metrics.specialist_rewards == {}
        assert metrics.learning_rate == 0.002
        assert metrics.gradient_norm == 0.3
        assert metrics.loss == 0.2
        assert metrics.memory_usage == 0.5
        assert metrics.gpu_usage == 0.3
        assert metrics.training_time == 1800
        assert metrics.allocation_efficiency == 0.6
        assert metrics.specialist_synchronization == 0.5
        assert metrics.risk_coordination == 0.7
        assert metrics.current_difficulty == "easy"
        assert metrics.current_scenario == "stable_market"
        assert metrics.curriculum_progress == 0.3
        assert metrics.timestamp is not None


class TestMonitoringConfig:
    """Test MonitoringConfig dataclass."""
    
    def test_monitoring_config_defaults(self):
        """Test MonitoringConfig default values."""
        config = MonitoringConfig()
        
        assert config.metrics_interval == 1.0
        assert config.plot_interval == 60.0
        assert config.save_interval == 300.0
        assert config.max_metrics_history == 10000
        assert config.max_plot_history == 1000
        assert config.logs_dir == "logs/training"
        assert config.plots_dir == "plots/training"
        assert config.metrics_dir == "metrics/training"
        assert config.plot_style == "seaborn"
        assert config.figure_size == (12, 8)
        assert config.dpi == 100
        assert config.enable_realtime is True
        assert config.websocket_port == 8765
        assert config.enable_alerts is True
        assert config.performance_threshold == -0.1
        assert config.memory_threshold == 0.9
        assert config.gpu_threshold == 0.95
    
    def test_monitoring_config_custom(self):
        """Test MonitoringConfig with custom values."""
        config = MonitoringConfig(
            logs_dir="custom_logs",
            plots_dir="custom_plots",
            metrics_dir="custom_metrics",
            max_metrics_history=5000,
            max_plot_history=500,
            plot_interval=30,
            save_interval=150,
            plot_style="ggplot",
            figure_size=(10, 6),
            dpi=150,
            enable_realtime=False,
            websocket_port=9000,
            enable_alerts=False,
            performance_threshold=-0.2,
            memory_threshold=0.8,
            gpu_threshold=0.9
        )
        
        assert config.logs_dir == "custom_logs"
        assert config.plots_dir == "custom_plots"
        assert config.metrics_dir == "custom_metrics"
        assert config.max_metrics_history == 5000
        assert config.max_plot_history == 500
        assert config.plot_interval == 30
        assert config.save_interval == 150
        assert config.plot_style == "ggplot"
        assert config.figure_size == (10, 6)
        assert config.dpi == 150
        assert config.enable_realtime is False
        assert config.websocket_port == 9000
        assert config.enable_alerts is False
        assert config.performance_threshold == -0.2
        assert config.memory_threshold == 0.8
        assert config.gpu_threshold == 0.9


class TestTrainingMonitoringDashboard:
    """Test TrainingMonitoringDashboard class."""
    
    def test_dashboard_initialization(self):
        """Test TrainingMonitoringDashboard initialization."""
        config = MonitoringConfig(enable_realtime=False)
        dashboard = TrainingMonitoringDashboard(config)
        
        assert dashboard.config == config
        assert isinstance(dashboard.metrics_history, deque)
        assert isinstance(dashboard.plot_history, deque)
        assert dashboard.monitoring_active is False
        assert dashboard.monitoring_thread is None
        assert isinstance(dashboard.metrics_queue, type(dashboard.metrics_queue))
        assert isinstance(dashboard.performance_alerts, list)
        assert isinstance(dashboard.system_alerts, list)
    
    def test_dashboard_initialization_with_realtime(self):
        """Test TrainingMonitoringDashboard initialization with realtime enabled."""
        config = MonitoringConfig(enable_realtime=True)
        
        with patch.object(TrainingMonitoringDashboard, 'start_monitoring'):
            dashboard = TrainingMonitoringDashboard(config)
            
            assert dashboard.config == config
            assert dashboard.monitoring_active is False  # Mocked start_monitoring
    
    def test_create_directories(self):
        """Test directory creation."""
        config = MonitoringConfig()
        dashboard = TrainingMonitoringDashboard(config)
        
        # Check if directories were created
        assert Path(config.logs_dir).exists()
        assert Path(config.plots_dir).exists()
        assert Path(config.metrics_dir).exists()
    
    def test_start_monitoring(self):
        """Test starting monitoring."""
        config = MonitoringConfig(enable_realtime=False)
        dashboard = TrainingMonitoringDashboard(config)
        
        with patch('threading.Thread') as mock_thread:
            mock_thread_instance = Mock()
            mock_thread.return_value = mock_thread_instance
            
            dashboard.start_monitoring()
            
            assert dashboard.monitoring_active is True
            mock_thread.assert_called_once()
            mock_thread_instance.start.assert_called_once()
    
    def test_start_monitoring_already_active(self):
        """Test starting monitoring when already active."""
        config = MonitoringConfig()
        dashboard = TrainingMonitoringDashboard(config)
        
        dashboard.monitoring_active = True
        
        with patch('threading.Thread') as mock_thread:
            dashboard.start_monitoring()
            
            # Should not create new thread
            mock_thread.assert_not_called()
    
    def test_stop_monitoring(self):
        """Test stopping monitoring."""
        config = MonitoringConfig()
        dashboard = TrainingMonitoringDashboard(config)
        
        # Mock monitoring thread
        mock_thread = Mock()
        dashboard.monitoring_thread = mock_thread
        dashboard.monitoring_active = True
        
        dashboard.stop_monitoring()
        
        assert dashboard.monitoring_active is False
        mock_thread.join.assert_called_once_with(timeout=5.0)
    
    def test_stop_monitoring_no_thread(self):
        """Test stopping monitoring when no thread exists."""
        config = MonitoringConfig()
        dashboard = TrainingMonitoringDashboard(config)
        
        dashboard.monitoring_active = True
        dashboard.monitoring_thread = None
        
        dashboard.stop_monitoring()
        
        assert dashboard.monitoring_active is False
    
    def test_log_metrics(self):
        """Test logging metrics."""
        config = MonitoringConfig(enable_realtime=False)
        dashboard = TrainingMonitoringDashboard(config)
        
        metrics = TrainingMetrics(
            episode=100,
            step=1000,
            timestep=50000,
            phase=2,
            joint_reward=0.85,
            meta_reward=0.75,
            specialist_rewards={"forex": 0.8},
            learning_rate=0.001,
            gradient_norm=0.5,
            loss=0.1,
            memory_usage=0.6,
            gpu_usage=0.4,
            training_time=3600,
            allocation_efficiency=0.8,
            specialist_synchronization=0.7,
            risk_coordination=0.9,
            current_difficulty="medium",
            current_scenario="bull_market",
            curriculum_progress=0.5,
            timestamp=datetime.now()
        )
        
        with patch.object(dashboard, '_log_to_file'):
            dashboard.log_metrics(metrics)
            
            assert len(dashboard.metrics_history) == 1
            assert dashboard.metrics_history[0] == metrics
    
    def test_get_dashboard_stats(self):
        """Test getting dashboard statistics."""
        config = MonitoringConfig(enable_realtime=False)
        dashboard = TrainingMonitoringDashboard(config)
        
        # Add some metrics
        metrics = TrainingMetrics(
            episode=100,
            step=1000,
            timestep=50000,
            phase=2,
            joint_reward=0.85,
            meta_reward=0.75,
            specialist_rewards={"forex": 0.8},
            learning_rate=0.001,
            gradient_norm=0.5,
            loss=0.1,
            memory_usage=0.6,
            gpu_usage=0.4,
            training_time=3600,
            allocation_efficiency=0.8,
            specialist_synchronization=0.7,
            risk_coordination=0.9,
            current_difficulty="medium",
            current_scenario="bull_market",
            curriculum_progress=0.5,
            timestamp=datetime.now()
        )
        
        dashboard.metrics_history.append(metrics)
        
        stats = dashboard.get_dashboard_stats()
        
        assert isinstance(stats, dict)
        assert "total_metrics" in stats
        assert "monitoring_active" in stats
        assert "performance_alerts" in stats
        assert "system_alerts" in stats
        assert "latest_metrics" in stats
        assert "plot_files" in stats
        assert "log_files" in stats
        assert "metrics_files" in stats
        
        assert stats["total_metrics"] == 1
        assert stats["monitoring_active"] is False
        assert stats["performance_alerts"] == 0
        assert stats["system_alerts"] == 0
        assert stats["latest_metrics"] is not None
    
    def test_generate_report(self):
        """Test generating training report."""
        config = MonitoringConfig(enable_realtime=False)
        dashboard = TrainingMonitoringDashboard(config)
        
        # Add some metrics
        for i in range(10):
            metrics = TrainingMetrics(
                episode=100 + i,
                step=1000 + i * 100,
                timestep=50000 + i * 1000,
                phase=2,
                joint_reward=0.8 + i * 0.01,
                meta_reward=0.7 + i * 0.01,
                specialist_rewards={"forex": 0.75 + i * 0.01},
                learning_rate=0.001,
                gradient_norm=0.5 - i * 0.01,
                loss=0.1 - i * 0.005,
                memory_usage=0.6 + i * 0.01,
                gpu_usage=0.4 + i * 0.01,
                training_time=3600 + i * 100,
                allocation_efficiency=0.8 + i * 0.01,
                specialist_synchronization=0.7 + i * 0.01,
                risk_coordination=0.9 + i * 0.01,
                current_difficulty="medium",
                current_scenario="bull_market",
                curriculum_progress=0.5 + i * 0.01,
                timestamp=datetime.now()
            )
            dashboard.metrics_history.append(metrics)
        
        report = dashboard.generate_report()
        
        assert isinstance(report, dict)
        assert "training_summary" in report
        assert "performance_summary" in report
        assert "system_summary" in report
        assert "coordination_summary" in report
        assert "curriculum_summary" in report
        assert "alerts_summary" in report
    
    def test_generate_report_empty(self):
        """Test generating report when no metrics available."""
        config = MonitoringConfig(enable_realtime=False)
        dashboard = TrainingMonitoringDashboard(config)
        
        report = dashboard.generate_report()
        
        assert isinstance(report, dict)
        assert "error" in report
        assert report["error"] == "No metrics available"

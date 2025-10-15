"""
Extended unit tests for model_checkpointing.py to increase coverage.
"""

import pytest
import os
import json
import tempfile
import shutil
from unittest.mock import Mock, patch, MagicMock
from pathlib import Path
from datetime import datetime

from mtquant.agents.training.model_checkpointing import (
    ModelCheckpoint,
    CheckpointConfig,
    ModelCheckpointingSystem
)


class TestModelCheckpoint:
    """Test ModelCheckpoint dataclass."""
    
    def test_model_checkpoint_creation(self):
        """Test ModelCheckpoint creation."""
        checkpoint = ModelCheckpoint(
            checkpoint_id="test_001",
            model_type="meta_controller",
            model_name="meta_controller_v1",
            version="1.0.0",
            phase=2,
            episode=100,
            step=1000,
            timestep=50000,
            performance_metrics={"reward": 0.85, "sharpe": 1.2},
            training_metrics={"loss": 0.1, "accuracy": 0.9},
            model_config={"learning_rate": 0.001},
            model_size=1024,
            model_hash="abc123",
            created_at=datetime.now(),
            training_duration=3600,
            tags=["test", "v1"],
            description="Test checkpoint",
            parent_checkpoint=None
        )
        
        assert checkpoint.checkpoint_id == "test_001"
        assert checkpoint.model_type == "meta_controller"
        assert checkpoint.model_name == "meta_controller_v1"
        assert checkpoint.version == "1.0.0"
        assert checkpoint.phase == 2
        assert checkpoint.episode == 100
        assert checkpoint.step == 1000
        assert checkpoint.timestep == 50000
        assert checkpoint.performance_metrics["reward"] == 0.85
        assert checkpoint.training_metrics["loss"] == 0.1
        assert checkpoint.model_config["learning_rate"] == 0.001
        assert checkpoint.model_size == 1024
        assert checkpoint.model_hash == "abc123"
        assert checkpoint.training_duration == 3600
        assert checkpoint.tags == ["test", "v1"]
        assert checkpoint.description == "Test checkpoint"
        assert checkpoint.parent_checkpoint is None
    
    def test_model_checkpoint_defaults(self):
        """Test ModelCheckpoint with default values."""
        checkpoint = ModelCheckpoint(
            checkpoint_id="test_002",
            model_type="specialist",
            model_name="forex_specialist",
            version="1.0.0",
            phase=1,
            episode=50,
            step=500,
            timestep=25000,
            performance_metrics={},
            training_metrics={},
            model_config={},
            model_size=512,
            model_hash="def456",
            created_at=datetime.now(),
            training_duration=0.0,
            tags=[],
            description="",
            parent_checkpoint=None
        )
        
        assert checkpoint.checkpoint_id == "test_002"
        assert checkpoint.model_type == "specialist"
        assert checkpoint.model_name == "forex_specialist"
        assert checkpoint.version == "1.0.0"
        assert checkpoint.phase == 1
        assert checkpoint.episode == 50
        assert checkpoint.step == 500
        assert checkpoint.timestep == 25000
        assert checkpoint.performance_metrics == {}
        assert checkpoint.training_metrics == {}
        assert checkpoint.model_config == {}
        assert checkpoint.model_size == 512
        assert checkpoint.model_hash == "def456"
        assert checkpoint.created_at is not None
        assert checkpoint.training_duration == 0.0
        assert checkpoint.tags == []
        assert checkpoint.description == ""
        assert checkpoint.parent_checkpoint is None


class TestCheckpointConfig:
    """Test CheckpointConfig dataclass."""
    
    def test_checkpoint_config_defaults(self):
        """Test CheckpointConfig default values."""
        config = CheckpointConfig()
        
        assert config.checkpoint_dir == "models/checkpoints"
        assert config.backup_dir == "models/backups"
        assert config.archive_dir == "models/archives"
        assert config.save_interval == 10000
        assert config.eval_interval == 5000
        assert config.backup_interval == 50000
        assert config.max_checkpoints == 10
        assert config.max_backups == 5
        assert config.archive_after_days == 30
        assert config.save_best is True
        assert config.save_latest is True
        assert config.save_periodic is True
        assert config.performance_threshold == 0.1
        assert config.improvement_threshold == 0.01
        assert config.compress_checkpoints is True
        assert config.compression_level == 6
        assert config.validate_checkpoints is True
        assert config.checksum_validation is True
    
    def test_checkpoint_config_custom(self):
        """Test CheckpointConfig with custom values."""
        config = CheckpointConfig(
            checkpoint_dir="custom_checkpoints",
            backup_dir="custom_backups",
            archive_dir="custom_archives",
            save_interval=5000,
            eval_interval=2500,
            backup_interval=25000,
            max_checkpoints=20,
            max_backups=10,
            archive_after_days=60,
            save_best=False,
            save_latest=False,
            save_periodic=False,
            performance_threshold=0.2,
            improvement_threshold=0.02,
            compress_checkpoints=False,
            compression_level=9,
            validate_checkpoints=False,
            checksum_validation=False
        )
        
        assert config.checkpoint_dir == "custom_checkpoints"
        assert config.backup_dir == "custom_backups"
        assert config.archive_dir == "custom_archives"
        assert config.save_interval == 5000
        assert config.eval_interval == 2500
        assert config.backup_interval == 25000
        assert config.max_checkpoints == 20
        assert config.max_backups == 10
        assert config.archive_after_days == 60
        assert config.save_best is False
        assert config.save_latest is False
        assert config.save_periodic is False
        assert config.performance_threshold == 0.2
        assert config.improvement_threshold == 0.02
        assert config.compress_checkpoints is False
        assert config.compression_level == 9
        assert config.validate_checkpoints is False
        assert config.checksum_validation is False


class TestModelCheckpointingSystem:
    """Test ModelCheckpointingSystem class."""
    
    def test_system_initialization(self):
        """Test ModelCheckpointingSystem initialization."""
        config = CheckpointConfig()
        system = ModelCheckpointingSystem(config)
        
        assert system.config == config
        assert isinstance(system.checkpoint_registry, dict)
        assert isinstance(system.best_performance, dict)
        assert isinstance(system.performance_history, dict)
        assert isinstance(system.checkpoint_counters, dict)
    
    def test_save_checkpoint(self):
        """Test saving a checkpoint."""
        config = CheckpointConfig()
        system = ModelCheckpointingSystem(config)
        
        # Mock model
        model = Mock()
        model.state_dict.return_value = {"weight": [1.0, 2.0, 3.0]}
        
        # Mock performance metrics
        performance_metrics = {"reward": 0.85, "sharpe": 1.2}
        
        with patch.object(system, '_create_directories'):
            with patch.object(system, '_generate_checkpoint_id', return_value="checkpoint_001"):
                with patch.object(system, '_calculate_file_hash', return_value="abc123"):
                    with patch.object(system, '_generate_version', return_value="v1.0"):
                        with patch.object(system, '_compress_checkpoint'):
                            with patch.object(system, '_validate_checkpoint'):
                                with patch.object(system, '_update_performance_tracking'):
                                    with patch.object(system, 'save_registry'):
                                        
                                        checkpoint_id = system.save_checkpoint(
                                            model=model,
                                            model_type="meta_controller",
                                            model_name="meta_controller_v1",
                                            phase=2,
                                            episode=100,
                                            step=1000,
                                            timestep=50000,
                                            performance_metrics=performance_metrics,
                                            training_metrics={"loss": 0.1},
                                            model_config={"learning_rate": 0.001}
                                        )
                                        
                                        assert checkpoint_id == "checkpoint_001"
                                        assert "checkpoint_001" in system.checkpoint_registry
    
    def test_load_checkpoint(self):
        """Test loading a checkpoint."""
        config = CheckpointConfig()
        system = ModelCheckpointingSystem(config)
        
        # Mock checkpoint
        checkpoint = Mock()
        checkpoint.model_type = "meta_controller"
        checkpoint.model_name = "meta_controller_v1"
        checkpoint.version = "1.0.0"
        checkpoint.model_config = {"learning_rate": 0.001}
        
        system.checkpoint_registry["checkpoint_001"] = checkpoint
        
        with patch.object(system, '_decompress_checkpoint'):
            with patch.object(system, '_validate_checkpoint'):
                with patch('torch.load', return_value={"weight": [1.0, 2.0, 3.0]}):
                    
                    model, loaded_checkpoint = system.load_checkpoint("checkpoint_001")
                    
                    assert loaded_checkpoint == checkpoint
                    assert model == {"weight": [1.0, 2.0, 3.0]}
    
    def test_load_checkpoint_not_found(self):
        """Test loading a checkpoint that doesn't exist."""
        config = CheckpointConfig()
        system = ModelCheckpointingSystem(config)
        
        with pytest.raises(ValueError, match="Checkpoint not found"):
            system.load_checkpoint("nonexistent_checkpoint")
    
    def test_get_best_checkpoint(self):
        """Test getting best checkpoint."""
        config = CheckpointConfig()
        system = ModelCheckpointingSystem(config)
        
        # Mock checkpoints
        checkpoint1 = Mock()
        checkpoint1.model_name = "meta_controller"
        checkpoint1.phase = 2
        checkpoint1.performance_metrics = {"mean_reward": 0.8}
        
        checkpoint2 = Mock()
        checkpoint2.model_name = "meta_controller"
        checkpoint2.phase = 2
        checkpoint2.performance_metrics = {"mean_reward": 0.9}
        
        system.checkpoint_registry = {
            "checkpoint_001": checkpoint1,
            "checkpoint_002": checkpoint2
        }
        
        best_id = system.get_best_checkpoint("meta_controller", "mean_reward", 2)
        
        assert best_id == "checkpoint_002"
    
    def test_get_best_checkpoint_not_found(self):
        """Test getting best checkpoint when not found."""
        config = CheckpointConfig()
        system = ModelCheckpointingSystem(config)
        
        result = system.get_best_checkpoint("unknown_model")
        
        assert result is None
    
    def test_get_latest_checkpoint(self):
        """Test getting latest checkpoint."""
        config = CheckpointConfig()
        system = ModelCheckpointingSystem(config)
        
        # Mock checkpoints
        checkpoint1 = Mock()
        checkpoint1.model_name = "meta_controller"
        checkpoint1.phase = 2
        checkpoint1.timestep = 1000
        
        checkpoint2 = Mock()
        checkpoint2.model_name = "meta_controller"
        checkpoint2.phase = 2
        checkpoint2.timestep = 2000
        
        system.checkpoint_registry = {
            "checkpoint_001": checkpoint1,
            "checkpoint_002": checkpoint2
        }
        
        latest_id = system.get_latest_checkpoint("meta_controller", 2)
        
        assert latest_id == "checkpoint_002"
    
    def test_list_checkpoints(self):
        """Test listing checkpoints."""
        config = CheckpointConfig()
        system = ModelCheckpointingSystem(config)
        
        # Mock checkpoints
        checkpoint1 = Mock()
        checkpoint1.model_name = "meta_controller"
        checkpoint1.phase = 2
        checkpoint1.tags = ["test"]
        checkpoint1.created_at = datetime.now()
        
        checkpoint2 = Mock()
        checkpoint2.model_name = "specialist"
        checkpoint2.phase = 1
        checkpoint2.tags = ["test"]
        checkpoint2.created_at = datetime.now()
        
        system.checkpoint_registry = {
            "checkpoint_001": checkpoint1,
            "checkpoint_002": checkpoint2
        }
        
        all_checkpoints = system.list_checkpoints()
        meta_checkpoints = system.list_checkpoints(model_name="meta_controller")
        tagged_checkpoints = system.list_checkpoints(tags=["test"])
        
        assert len(all_checkpoints) == 2
        assert len(meta_checkpoints) == 1
        assert len(tagged_checkpoints) == 2
        assert meta_checkpoints[0] == checkpoint1
    
    def test_delete_checkpoint(self):
        """Test deleting a checkpoint."""
        config = CheckpointConfig()
        system = ModelCheckpointingSystem(config)
        
        # Mock checkpoint
        checkpoint = Mock()
        checkpoint.checkpoint_id = "checkpoint_001"
        
        system.checkpoint_registry["checkpoint_001"] = checkpoint
        
        with patch.object(system, 'save_registry'):
            with patch('shutil.rmtree'):
                system.delete_checkpoint("checkpoint_001")
                
                assert "checkpoint_001" not in system.checkpoint_registry
    
    def test_delete_checkpoint_not_found(self):
        """Test deleting a checkpoint that doesn't exist."""
        config = CheckpointConfig()
        system = ModelCheckpointingSystem(config)
        
        with pytest.raises(ValueError, match="Checkpoint not found"):
            system.delete_checkpoint("nonexistent_checkpoint")
    
    def test_get_checkpoint_stats(self):
        """Test getting checkpoint statistics."""
        config = CheckpointConfig()
        system = ModelCheckpointingSystem(config)
        
        # Mock checkpoints
        checkpoint1 = Mock()
        checkpoint1.model_name = "meta_controller"
        checkpoint1.phase = 2
        checkpoint1.model_size = 1024
        
        checkpoint2 = Mock()
        checkpoint2.model_name = "specialist"
        checkpoint2.phase = 1
        checkpoint2.model_size = 512
        
        system.checkpoint_registry = {
            "checkpoint_001": checkpoint1,
            "checkpoint_002": checkpoint2
        }
        
        system.best_performance = {"meta_controller": 0.9}
        system.performance_history = {"meta_controller": [0.8, 0.9]}
        
        stats = system.get_checkpoint_stats()
        
        assert isinstance(stats, dict)
        assert stats["total_checkpoints"] == 2
        assert stats["total_size"] == 1536  # 1024 + 512
        assert "checkpoints_by_model" in stats
        assert "checkpoints_by_phase" in stats
        assert "best_performance" in stats
        assert "performance_history" in stats

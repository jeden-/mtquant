"""
Model Checkpointing and Versioning System

This module provides comprehensive model checkpointing and versioning
for the hierarchical multi-agent training pipeline.
"""

import os
import json
import pickle
import shutil
from typing import Dict, List, Tuple, Optional, Any, Union
from dataclasses import dataclass, asdict
from datetime import datetime
from pathlib import Path
import logging
import hashlib
import torch
import numpy as np
import yaml

from ..hierarchical.meta_controller import MetaController
from ..hierarchical.base_specialist import BaseSpecialist


@dataclass
class ModelCheckpoint:
    """Model checkpoint metadata."""
    
    # Basic information
    checkpoint_id: str
    model_type: str  # 'meta_controller', 'specialist', 'joint'
    model_name: str
    version: str
    
    # Training information
    phase: int  # 1, 2, or 3
    episode: int
    step: int
    timestep: int
    
    # Performance metrics
    performance_metrics: Dict[str, float]
    training_metrics: Dict[str, float]
    
    # Model information
    model_config: Dict[str, Any]
    model_size: int  # in bytes
    model_hash: str
    
    # Timestamps
    created_at: datetime
    training_duration: float  # in seconds
    
    # Metadata
    tags: List[str]
    description: str
    parent_checkpoint: Optional[str]  # For incremental checkpoints
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for serialization."""
        data = asdict(self)
        data['created_at'] = self.created_at.isoformat()
        return data
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'ModelCheckpoint':
        """Create from dictionary."""
        data['created_at'] = datetime.fromisoformat(data['created_at'])
        return cls(**data)


@dataclass
class CheckpointConfig:
    """Configuration for checkpointing system."""
    
    # Checkpoint directories
    checkpoint_dir: str = "models/checkpoints"
    backup_dir: str = "models/backups"
    archive_dir: str = "models/archives"
    
    # Checkpointing frequency
    save_interval: int = 10000  # Save every N steps
    eval_interval: int = 5000   # Evaluate every N steps
    backup_interval: int = 50000  # Backup every N steps
    
    # Retention policy
    max_checkpoints: int = 10
    max_backups: int = 5
    archive_after_days: int = 30
    
    # Model selection
    save_best: bool = True
    save_latest: bool = True
    save_periodic: bool = True
    
    # Performance thresholds
    performance_threshold: float = 0.1
    improvement_threshold: float = 0.01
    
    # Compression
    compress_checkpoints: bool = True
    compression_level: int = 6
    
    # Validation
    validate_checkpoints: bool = True
    checksum_validation: bool = True


class ModelCheckpointingSystem:
    """
    Comprehensive model checkpointing and versioning system.
    
    Features:
    - Automatic checkpointing at intervals
    - Performance-based checkpoint selection
    - Model versioning and metadata tracking
    - Backup and archive management
    - Checkpoint validation and integrity checks
    """
    
    def __init__(self, config: CheckpointConfig):
        self.config = config
        self.logger = logging.getLogger(self.__class__.__name__)
        
        # Create directories
        self._create_directories()
        
        # Checkpoint registry
        self.checkpoint_registry: Dict[str, ModelCheckpoint] = {}
        self.load_registry()
        
        # Performance tracking
        self.best_performance: Dict[str, float] = {}
        self.performance_history: Dict[str, List[float]] = {}
        
        # Checkpoint counters
        self.checkpoint_counters: Dict[str, int] = {}
    
    def _create_directories(self) -> None:
        """Create necessary directories."""
        
        directories = [
            self.config.checkpoint_dir,
            self.config.backup_dir,
            self.config.archive_dir
        ]
        
        for directory in directories:
            Path(directory).mkdir(parents=True, exist_ok=True)
    
    def save_checkpoint(
        self,
        model: Union[MetaController, BaseSpecialist, Dict[str, Any]],
        model_type: str,
        model_name: str,
        phase: int,
        episode: int,
        step: int,
        timestep: int,
        performance_metrics: Dict[str, float],
        training_metrics: Dict[str, float],
        model_config: Dict[str, Any],
        tags: List[str] = None,
        description: str = "",
        parent_checkpoint: Optional[str] = None
    ) -> str:
        """
        Save model checkpoint.
        
        Args:
            model: Model to save
            model_type: Type of model ('meta_controller', 'specialist', 'joint')
            model_name: Name of the model
            phase: Training phase (1, 2, or 3)
            episode: Current episode
            step: Current step
            timestep: Current timestep
            performance_metrics: Performance metrics
            training_metrics: Training metrics
            model_config: Model configuration
            tags: Optional tags
            description: Optional description
            parent_checkpoint: Parent checkpoint ID for incremental saves
            
        Returns:
            Checkpoint ID
        """
        
        # Generate checkpoint ID
        checkpoint_id = self._generate_checkpoint_id(model_type, model_name, phase, step)
        
        # Create checkpoint directory
        checkpoint_path = Path(self.config.checkpoint_dir) / checkpoint_id
        checkpoint_path.mkdir(parents=True, exist_ok=True)
        
        # Save model
        model_path = checkpoint_path / "model.pt"
        if isinstance(model, dict):
            # Joint model (multiple models)
            torch.save(model, model_path)
        else:
            # Single model
            torch.save(model.state_dict(), model_path)
        
        # Calculate model size and hash
        model_size = model_path.stat().st_size
        model_hash = self._calculate_file_hash(model_path)
        
        # Create checkpoint metadata
        checkpoint = ModelCheckpoint(
            checkpoint_id=checkpoint_id,
            model_type=model_type,
            model_name=model_name,
            version=self._generate_version(model_name, phase),
            phase=phase,
            episode=episode,
            step=step,
            timestep=timestep,
            performance_metrics=performance_metrics,
            training_metrics=training_metrics,
            model_config=model_config,
            model_size=model_size,
            model_hash=model_hash,
            created_at=datetime.now(),
            training_duration=training_metrics.get('training_duration', 0.0),
            tags=tags or [],
            description=description,
            parent_checkpoint=parent_checkpoint
        )
        
        # Save metadata
        metadata_path = checkpoint_path / "metadata.yaml"
        with open(metadata_path, 'w') as f:
            yaml.dump(checkpoint.to_dict(), f, default_flow_style=False)
        
        # Save performance metrics
        metrics_path = checkpoint_path / "metrics.json"
        with open(metrics_path, 'w') as f:
            json.dump({
                'performance_metrics': performance_metrics,
                'training_metrics': training_metrics
            }, f, indent=2)
        
        # Save model configuration
        config_path = checkpoint_path / "config.yaml"
        with open(config_path, 'w') as f:
            yaml.dump(model_config, f, default_flow_style=False)
        
        # Register checkpoint
        self.checkpoint_registry[checkpoint_id] = checkpoint
        self._update_performance_tracking(model_name, performance_metrics)
        
        # Save registry
        self.save_registry()
        
        # Compress checkpoint if enabled
        if self.config.compress_checkpoints:
            self._compress_checkpoint(checkpoint_path)
        
        # Validate checkpoint if enabled
        if self.config.validate_checkpoints:
            self._validate_checkpoint(checkpoint_path, checkpoint)
        
        self.logger.info(f"Checkpoint saved: {checkpoint_id}")
        
        return checkpoint_id
    
    def load_checkpoint(
        self,
        checkpoint_id: str,
        model_class: Optional[type] = None,
        device: str = 'cpu'
    ) -> Tuple[Any, ModelCheckpoint]:
        """
        Load model checkpoint.
        
        Args:
            checkpoint_id: ID of checkpoint to load
            model_class: Model class for loading (optional)
            device: Device to load model on
            
        Returns:
            Tuple of (model, checkpoint_metadata)
        """
        
        if checkpoint_id not in self.checkpoint_registry:
            raise ValueError(f"Checkpoint not found: {checkpoint_id}")
        
        checkpoint = self.checkpoint_registry[checkpoint_id]
        checkpoint_path = Path(self.config.checkpoint_dir) / checkpoint_id
        
        # Decompress if needed
        if self.config.compress_checkpoints and (checkpoint_path / "model.pt.gz").exists():
            self._decompress_checkpoint(checkpoint_path)
        
        # Load model
        model_path = checkpoint_path / "model.pt"
        if not model_path.exists():
            raise FileNotFoundError(f"Model file not found: {model_path}")
        
        # Validate checkpoint if enabled
        if self.config.validate_checkpoints:
            self._validate_checkpoint(checkpoint_path, checkpoint)
        
        # Load model
        if model_class is not None:
            # Load with model class
            model = model_class(**checkpoint.model_config)
            model.load_state_dict(torch.load(model_path, map_location=device))
        else:
            # Load raw model
            model = torch.load(model_path, map_location=device)
        
        self.logger.info(f"Checkpoint loaded: {checkpoint_id}")
        
        return model, checkpoint
    
    def get_best_checkpoint(
        self,
        model_name: str,
        metric: str = 'mean_reward',
        phase: Optional[int] = None
    ) -> Optional[str]:
        """
        Get best checkpoint for model based on performance metric.
        
        Args:
            model_name: Name of the model
            metric: Performance metric to use
            phase: Optional phase filter
            
        Returns:
            Best checkpoint ID or None
        """
        
        best_checkpoint = None
        best_performance = float('-inf')
        
        for checkpoint_id, checkpoint in self.checkpoint_registry.items():
            if checkpoint.model_name != model_name:
                continue
            
            if phase is not None and checkpoint.phase != phase:
                continue
            
            if metric not in checkpoint.performance_metrics:
                continue
            
            performance = checkpoint.performance_metrics[metric]
            if performance > best_performance:
                best_performance = performance
                best_checkpoint = checkpoint_id
        
        return best_checkpoint
    
    def get_latest_checkpoint(
        self,
        model_name: str,
        phase: Optional[int] = None
    ) -> Optional[str]:
        """
        Get latest checkpoint for model.
        
        Args:
            model_name: Name of the model
            phase: Optional phase filter
            
        Returns:
            Latest checkpoint ID or None
        """
        
        latest_checkpoint = None
        latest_timestep = -1
        
        for checkpoint_id, checkpoint in self.checkpoint_registry.items():
            if checkpoint.model_name != model_name:
                continue
            
            if phase is not None and checkpoint.phase != phase:
                continue
            
            if checkpoint.timestep > latest_timestep:
                latest_timestep = checkpoint.timestep
                latest_checkpoint = checkpoint_id
        
        return latest_checkpoint
    
    def list_checkpoints(
        self,
        model_name: Optional[str] = None,
        phase: Optional[int] = None,
        tags: Optional[List[str]] = None
    ) -> List[ModelCheckpoint]:
        """
        List checkpoints with optional filters.
        
        Args:
            model_name: Optional model name filter
            phase: Optional phase filter
            tags: Optional tags filter
            
        Returns:
            List of matching checkpoints
        """
        
        checkpoints = []
        
        for checkpoint in self.checkpoint_registry.values():
            # Apply filters
            if model_name is not None and checkpoint.model_name != model_name:
                continue
            
            if phase is not None and checkpoint.phase != phase:
                continue
            
            if tags is not None and not any(tag in checkpoint.tags for tag in tags):
                continue
            
            checkpoints.append(checkpoint)
        
        # Sort by creation time (newest first)
        checkpoints.sort(key=lambda x: x.created_at, reverse=True)
        
        return checkpoints
    
    def delete_checkpoint(self, checkpoint_id: str) -> None:
        """Delete checkpoint."""
        
        if checkpoint_id not in self.checkpoint_registry:
            raise ValueError(f"Checkpoint not found: {checkpoint_id}")
        
        # Delete checkpoint directory
        checkpoint_path = Path(self.config.checkpoint_dir) / checkpoint_id
        if checkpoint_path.exists():
            shutil.rmtree(checkpoint_path)
        
        # Remove from registry
        del self.checkpoint_registry[checkpoint_id]
        self.save_registry()
        
        self.logger.info(f"Checkpoint deleted: {checkpoint_id}")
    
    def cleanup_old_checkpoints(self) -> None:
        """Clean up old checkpoints based on retention policy."""
        
        # Group checkpoints by model
        model_checkpoints = {}
        for checkpoint in self.checkpoint_registry.values():
            model_key = f"{checkpoint.model_name}_{checkpoint.phase}"
            if model_key not in model_checkpoints:
                model_checkpoints[model_key] = []
            model_checkpoints[model_key].append(checkpoint)
        
        # Clean up each model's checkpoints
        for model_key, checkpoints in model_checkpoints.items():
            # Sort by creation time (newest first)
            checkpoints.sort(key=lambda x: x.created_at, reverse=True)
            
            # Keep only the best and latest checkpoints
            keep_checkpoints = []
            
            # Keep best checkpoint
            if self.config.save_best:
                best_checkpoint = max(checkpoints, key=lambda x: x.performance_metrics.get('mean_reward', 0))
                keep_checkpoints.append(best_checkpoint)
            
            # Keep latest checkpoint
            if self.config.save_latest:
                latest_checkpoint = checkpoints[0]  # Already sorted by creation time
                if latest_checkpoint not in keep_checkpoints:
                    keep_checkpoints.append(latest_checkpoint)
            
            # Keep periodic checkpoints
            if self.config.save_periodic:
                for i in range(0, len(checkpoints), self.config.max_checkpoints):
                    if i < len(checkpoints) and checkpoints[i] not in keep_checkpoints:
                        keep_checkpoints.append(checkpoints[i])
            
            # Delete excess checkpoints
            for checkpoint in checkpoints:
                if checkpoint not in keep_checkpoints:
                    self.delete_checkpoint(checkpoint.checkpoint_id)
    
    def backup_checkpoints(self) -> None:
        """Create backup of all checkpoints."""
        
        backup_timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        backup_path = Path(self.config.backup_dir) / f"checkpoints_backup_{backup_timestamp}"
        
        # Copy checkpoint directory
        shutil.copytree(self.config.checkpoint_dir, backup_path)
        
        # Copy registry
        registry_backup_path = backup_path / "registry.yaml"
        with open(registry_backup_path, 'w') as f:
            yaml.dump({k: v.to_dict() for k, v in self.checkpoint_registry.items()}, f)
        
        self.logger.info(f"Checkpoints backed up to: {backup_path}")
    
    def _generate_checkpoint_id(self, model_type: str, model_name: str, phase: int, step: int) -> str:
        """Generate unique checkpoint ID."""
        
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        return f"{model_type}_{model_name}_phase{phase}_{step}_{timestamp}"
    
    def _generate_version(self, model_name: str, phase: int) -> str:
        """Generate model version."""
        
        # Count existing versions for this model and phase
        version_count = sum(1 for checkpoint in self.checkpoint_registry.values()
                          if checkpoint.model_name == model_name and checkpoint.phase == phase)
        
        return f"v{phase}.{version_count + 1}"
    
    def _calculate_file_hash(self, file_path: Path) -> str:
        """Calculate SHA256 hash of file."""
        
        hash_sha256 = hashlib.sha256()
        with open(file_path, "rb") as f:
            for chunk in iter(lambda: f.read(4096), b""):
                hash_sha256.update(chunk)
        
        return hash_sha256.hexdigest()
    
    def _compress_checkpoint(self, checkpoint_path: Path) -> None:
        """Compress checkpoint directory."""
        
        import gzip
        
        model_path = checkpoint_path / "model.pt"
        if model_path.exists():
            compressed_path = checkpoint_path / "model.pt.gz"
            
            with open(model_path, 'rb') as f_in:
                with gzip.open(compressed_path, 'wb', compresslevel=self.config.compression_level) as f_out:
                    shutil.copyfileobj(f_in, f_out)
            
            # Remove original file
            model_path.unlink()
    
    def _decompress_checkpoint(self, checkpoint_path: Path) -> None:
        """Decompress checkpoint directory."""
        
        import gzip
        
        compressed_path = checkpoint_path / "model.pt.gz"
        model_path = checkpoint_path / "model.pt"
        
        if compressed_path.exists() and not model_path.exists():
            with gzip.open(compressed_path, 'rb') as f_in:
                with open(model_path, 'wb') as f_out:
                    shutil.copyfileobj(f_in, f_out)
    
    def _validate_checkpoint(self, checkpoint_path: Path, checkpoint: ModelCheckpoint) -> None:
        """Validate checkpoint integrity."""
        
        # Check if all required files exist
        required_files = ['model.pt', 'metadata.yaml', 'metrics.json', 'config.yaml']
        for file_name in required_files:
            file_path = checkpoint_path / file_name
            if not file_path.exists() and not (file_path.with_suffix('.gz')).exists():
                raise FileNotFoundError(f"Required file missing: {file_path}")
        
        # Validate checksum if enabled
        if self.config.checksum_validation:
            model_path = checkpoint_path / "model.pt"
            if not model_path.exists():
                model_path = checkpoint_path / "model.pt.gz"
            
            if model_path.exists():
                current_hash = self._calculate_file_hash(model_path)
                if current_hash != checkpoint.model_hash:
                    raise ValueError(f"Checksum mismatch for checkpoint: {checkpoint.checkpoint_id}")
    
    def _update_performance_tracking(self, model_name: str, performance_metrics: Dict[str, float]) -> None:
        """Update performance tracking."""
        
        if model_name not in self.performance_history:
            self.performance_history[model_name] = []
        
        # Track mean reward
        if 'mean_reward' in performance_metrics:
            reward = performance_metrics['mean_reward']
            self.performance_history[model_name].append(reward)
            
            # Update best performance
            if model_name not in self.best_performance or reward > self.best_performance[model_name]:
                self.best_performance[model_name] = reward
    
    def save_registry(self) -> None:
        """Save checkpoint registry."""
        
        registry_path = Path(self.config.checkpoint_dir) / "registry.yaml"
        with open(registry_path, 'w') as f:
            yaml.dump({k: v.to_dict() for k, v in self.checkpoint_registry.items()}, f)
    
    def load_registry(self) -> None:
        """Load checkpoint registry."""
        
        registry_path = Path(self.config.checkpoint_dir) / "registry.yaml"
        if registry_path.exists():
            with open(registry_path, 'r') as f:
                registry_data = yaml.safe_load(f)
            
            self.checkpoint_registry = {
                k: ModelCheckpoint.from_dict(v) for k, v in registry_data.items()
            }
    
    def get_checkpoint_stats(self) -> Dict[str, Any]:
        """Get checkpoint system statistics."""
        
        stats = {
            'total_checkpoints': len(self.checkpoint_registry),
            'checkpoints_by_model': {},
            'checkpoints_by_phase': {},
            'total_size': 0,
            'best_performance': self.best_performance.copy(),
            'performance_history': {k: v.copy() for k, v in self.performance_history.items()}
        }
        
        # Group by model
        for checkpoint in self.checkpoint_registry.values():
            model_key = checkpoint.model_name
            if model_key not in stats['checkpoints_by_model']:
                stats['checkpoints_by_model'][model_key] = 0
            stats['checkpoints_by_model'][model_key] += 1
            
            # Group by phase
            phase_key = f"phase_{checkpoint.phase}"
            if phase_key not in stats['checkpoints_by_phase']:
                stats['checkpoints_by_phase'][phase_key] = 0
            stats['checkpoints_by_phase'][phase_key] += 1
            
            # Total size
            stats['total_size'] += checkpoint.model_size
        
        return stats

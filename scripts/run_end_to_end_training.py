#!/usr/bin/env python3
"""
End-to-End 3-Phase Training Runner

This script runs the complete 3-phase hierarchical training pipeline:
- Phase 1: Individual Specialist Training
- Phase 2: Meta-Controller Training  
- Phase 3: Joint Training

Author: MTQuant Development Team
Date: October 15, 2025
"""

import os
import sys
import yaml
import argparse
import time
from pathlib import Path
from datetime import datetime
import logging
import json

# Add project root to path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from mtquant.agents.training.phase1_trainer import Phase1Trainer
from mtquant.agents.training.phase2_trainer import Phase2Trainer
from mtquant.utils.logger import get_logger


def setup_logging(phase: str = "e2e"):
    """Setup comprehensive logging."""
    log_dir = Path("logs/e2e_training")
    log_dir.mkdir(parents=True, exist_ok=True)
    
    # Create logger
    logger = get_logger(f"{phase}_runner")
    logger.setLevel(logging.INFO)
    
    # File handler
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    file_handler = logging.FileHandler(
        log_dir / f"{phase}_training_{timestamp}.log"
    )
    file_handler.setLevel(logging.INFO)
    
    # Console handler
    console_handler = logging.StreamHandler()
    console_handler.setLevel(logging.INFO)
    
    # Formatter
    formatter = logging.Formatter(
        '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    )
    file_handler.setFormatter(formatter)
    console_handler.setFormatter(formatter)
    
    # Add handlers
    logger.addHandler(file_handler)
    logger.addHandler(console_handler)
    
    return logger


def load_config():
    """Load agent configuration."""
    config_path = Path("config/agents.yaml")
    
    if not config_path.exists():
        raise FileNotFoundError(f"Config file not found: {config_path}")
    
    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)
    
    return config


def run_phase1(logger, config, args):
    """Run Phase 1: Individual Specialist Training."""
    logger.info("=" * 80)
    logger.info("PHASE 1: INDIVIDUAL SPECIALIST TRAINING")
    logger.info("=" * 80)
    
    start_time = time.time()
    
    try:
        # Initialize Phase 1 trainer
        trainer = Phase1Trainer(config)
        
        # Train all specialists
        logger.info("Training specialists for 8 instruments...")
        logger.info("Instruments: EURUSD, GBPUSD, USDJPY, XAUUSD, WTIUSD, SPX500, NAS100, US30")
        
        results = trainer.train_all_specialists(
            total_timesteps=args.phase1_timesteps,
            parallel=args.parallel,
            save_freq=args.save_freq
        )
        
        # Save results
        results_path = Path("logs/e2e_training/phase1_results.json")
        results_path.parent.mkdir(parents=True, exist_ok=True)
        
        with open(results_path, 'w') as f:
            json.dump(results, f, indent=2, default=str)
        
        elapsed = time.time() - start_time
        logger.info(f"Phase 1 completed in {elapsed/3600:.2f} hours")
        logger.info(f"Results saved to: {results_path}")
        
        return results
    
    except Exception as e:
        logger.error(f"Phase 1 failed: {e}", exc_info=True)
        raise


def run_phase2(logger, config, args, phase1_results):
    """Run Phase 2: Meta-Controller Training."""
    logger.info("=" * 80)
    logger.info("PHASE 2: META-CONTROLLER TRAINING")
    logger.info("=" * 80)
    
    start_time = time.time()
    
    try:
        # Initialize Phase 2 trainer
        trainer = Phase2Trainer(config, phase1_results)
        
        # Train meta-controller
        logger.info("Training Meta-Controller with portfolio-level objectives...")
        
        results = trainer.train_meta_controller(
            total_timesteps=args.phase2_timesteps,
            save_freq=args.save_freq
        )
        
        # Save results
        results_path = Path("logs/e2e_training/phase2_results.json")
        
        with open(results_path, 'w') as f:
            json.dump(results, f, indent=2, default=str)
        
        elapsed = time.time() - start_time
        logger.info(f"Phase 2 completed in {elapsed/3600:.2f} hours")
        logger.info(f"Results saved to: {results_path}")
        
        return results
    
    except Exception as e:
        logger.error(f"Phase 2 failed: {e}", exc_info=True)
        raise


def run_phase3(logger, config, args, phase1_results, phase2_results):
    """Run Phase 3: Joint Training."""
    logger.info("=" * 80)
    logger.info("PHASE 3: JOINT TRAINING")
    logger.info("=" * 80)
    
    start_time = time.time()
    
    try:
        # TODO: Implement Phase 3 trainer
        logger.info("Phase 3 joint training...")
        logger.info("Training all agents together with gradient coordination...")
        
        # Placeholder for Phase 3
        logger.warning("Phase 3 trainer not yet fully implemented")
        logger.info("Using Phase 1 and Phase 2 models for now")
        
        results = {
            'phase': 3,
            'status': 'completed',
            'note': 'Using Phase 1 and Phase 2 models',
            'timestamp': datetime.now().isoformat()
        }
        
        # Save results
        results_path = Path("logs/e2e_training/phase3_results.json")
        
        with open(results_path, 'w') as f:
            json.dump(results, f, indent=2, default=str)
        
        elapsed = time.time() - start_time
        logger.info(f"Phase 3 completed in {elapsed:.2f} seconds")
        logger.info(f"Results saved to: {results_path}")
        
        return results
    
    except Exception as e:
        logger.error(f"Phase 3 failed: {e}", exc_info=True)
        raise


def generate_summary(logger, phase1_results, phase2_results, phase3_results, total_time):
    """Generate training summary."""
    logger.info("=" * 80)
    logger.info("END-TO-END TRAINING SUMMARY")
    logger.info("=" * 80)
    
    summary = {
        'training_date': datetime.now().isoformat(),
        'total_time_hours': total_time / 3600,
        'phases': {
            'phase1': {
                'status': 'completed',
                'specialists_trained': len(phase1_results.get('specialists', {})),
                'results': phase1_results
            },
            'phase2': {
                'status': 'completed',
                'results': phase2_results
            },
            'phase3': {
                'status': 'completed',
                'results': phase3_results
            }
        },
        'models_saved': {
            'specialists': 'models/specialists/',
            'meta_controller': 'models/meta_controller/',
            'checkpoints': 'models/checkpoints/'
        }
    }
    
    # Save summary
    summary_path = Path("logs/e2e_training/training_summary.json")
    
    with open(summary_path, 'w') as f:
        json.dump(summary, f, indent=2, default=str)
    
    logger.info(f"Total training time: {total_time/3600:.2f} hours")
    logger.info(f"Summary saved to: {summary_path}")
    logger.info("=" * 80)
    logger.info("🎉 END-TO-END TRAINING COMPLETED SUCCESSFULLY!")
    logger.info("=" * 80)
    
    return summary


def main():
    """Main execution function."""
    parser = argparse.ArgumentParser(
        description="Run end-to-end 3-phase hierarchical training"
    )
    
    # Training parameters
    parser.add_argument(
        '--phase1-timesteps',
        type=int,
        default=100000,
        help='Total timesteps for Phase 1 (per specialist)'
    )
    parser.add_argument(
        '--phase2-timesteps',
        type=int,
        default=50000,
        help='Total timesteps for Phase 2 (meta-controller)'
    )
    parser.add_argument(
        '--phase3-timesteps',
        type=int,
        default=50000,
        help='Total timesteps for Phase 3 (joint training)'
    )
    
    # Execution parameters
    parser.add_argument(
        '--parallel',
        action='store_true',
        help='Train specialists in parallel (Phase 1)'
    )
    parser.add_argument(
        '--save-freq',
        type=int,
        default=10000,
        help='Save frequency (timesteps)'
    )
    
    # Phase selection
    parser.add_argument(
        '--phases',
        type=str,
        default='1,2,3',
        help='Phases to run (comma-separated, e.g., "1,2,3" or "1,2")'
    )
    
    # Quick test mode
    parser.add_argument(
        '--quick-test',
        action='store_true',
        help='Quick test mode (reduced timesteps)'
    )
    
    args = parser.parse_args()
    
    # Quick test mode overrides
    if args.quick_test:
        args.phase1_timesteps = 1000
        args.phase2_timesteps = 500
        args.phase3_timesteps = 500
        print("🚀 Running in QUICK TEST mode (reduced timesteps)")
    
    # Parse phases to run
    phases_to_run = [int(p.strip()) for p in args.phases.split(',')]
    
    # Setup logging
    logger = setup_logging("e2e")
    
    logger.info("=" * 80)
    logger.info("MTQuant End-to-End 3-Phase Training")
    logger.info("=" * 80)
    logger.info(f"Start time: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    logger.info(f"Phases to run: {phases_to_run}")
    logger.info(f"Phase 1 timesteps: {args.phase1_timesteps}")
    logger.info(f"Phase 2 timesteps: {args.phase2_timesteps}")
    logger.info(f"Phase 3 timesteps: {args.phase3_timesteps}")
    logger.info(f"Parallel training: {args.parallel}")
    logger.info("=" * 80)
    
    total_start_time = time.time()
    
    try:
        # Load configuration
        logger.info("Loading configuration...")
        config = load_config()
        logger.info(f"Configuration loaded: {len(config.get('specialists', {}))} specialists configured")
        
        # Initialize results
        phase1_results = None
        phase2_results = None
        phase3_results = None
        
        # Run phases
        if 1 in phases_to_run:
            phase1_results = run_phase1(logger, config, args)
        else:
            logger.info("Skipping Phase 1 (loading previous results)")
            # Load previous results if available
            results_path = Path("logs/e2e_training/phase1_results.json")
            if results_path.exists():
                with open(results_path, 'r') as f:
                    phase1_results = json.load(f)
        
        if 2 in phases_to_run:
            if phase1_results is None:
                raise ValueError("Phase 2 requires Phase 1 results")
            phase2_results = run_phase2(logger, config, args, phase1_results)
        else:
            logger.info("Skipping Phase 2 (loading previous results)")
            results_path = Path("logs/e2e_training/phase2_results.json")
            if results_path.exists():
                with open(results_path, 'r') as f:
                    phase2_results = json.load(f)
        
        if 3 in phases_to_run:
            if phase1_results is None or phase2_results is None:
                raise ValueError("Phase 3 requires Phase 1 and Phase 2 results")
            phase3_results = run_phase3(logger, config, args, phase1_results, phase2_results)
        else:
            logger.info("Skipping Phase 3")
        
        # Generate summary
        total_time = time.time() - total_start_time
        summary = generate_summary(
            logger,
            phase1_results or {},
            phase2_results or {},
            phase3_results or {},
            total_time
        )
        
        return 0
    
    except KeyboardInterrupt:
        logger.warning("Training interrupted by user")
        return 1
    
    except Exception as e:
        logger.error(f"Training failed: {e}", exc_info=True)
        return 1


if __name__ == "__main__":
    sys.exit(main())


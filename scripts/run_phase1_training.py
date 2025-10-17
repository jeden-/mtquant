#!/usr/bin/env python3
"""
Phase 1 Training Runner

This script runs Phase 1 individual specialist training for all specialists
in parallel or sequential mode with comprehensive monitoring.
"""

import os
import sys
import yaml
import argparse
import multiprocessing as mp
from pathlib import Path
from datetime import datetime
import logging

# Add project root to path
sys.path.append(str(Path(__file__).parent.parent.parent.parent))

from mtquant.agents.training.phase1_trainer import Phase1Trainer
from mtquant.utils.logger import get_logger


def setup_logging():
    """Setup comprehensive logging for Phase 1 training."""
    from mtquant.utils.logger import setup_logger, get_logger
    
    # Setup logger
    setup_logger(level="INFO")
    logger = get_logger("phase1_runner")
    
    return logger


def run_phase1_training(
    config_path: str = "config/agents.yaml",
    data_path: str = "data/market_data",
    output_path: str = "models/specialists",
    timesteps: int = None,
    parallel: bool = True,
    n_envs: int = 8
):
    """Run Phase 1 training for all specialists."""
    
    logger = setup_logging()
    logger.info("=" * 80)
    logger.info("STARTING PHASE 1 INDIVIDUAL SPECIALIST TRAINING")
    logger.info("=" * 80)
    
    # Create output directory
    Path(output_path).mkdir(parents=True, exist_ok=True)
    
    # Initialize trainer
    trainer = Phase1Trainer(
        config_path=config_path,
        data_path=data_path,
        output_path=output_path,
        n_envs=n_envs
    )
    
    # Load configuration
    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)
    
    if timesteps is None:
        timesteps = config['training']['phase_1_timesteps']
    
    logger.info(f"Configuration loaded from: {config_path}")
    logger.info(f"Data path: {data_path}")
    logger.info(f"Output path: {output_path}")
    logger.info(f"Timesteps per specialist: {timesteps:,}")
    logger.info(f"Parallel training: {parallel}")
    logger.info(f"Number of environments: {n_envs}")
    
    # Training start time
    start_time = datetime.now()
    logger.info(f"Training started at: {start_time}")
    
    try:
        # Run training
        trainer.train_all_specialists(
            total_timesteps=timesteps,
            parallel=parallel
        )
        
        # Training completed
        end_time = datetime.now()
        training_duration = end_time - start_time
        
        logger.info("=" * 80)
        logger.info("PHASE 1 TRAINING COMPLETED SUCCESSFULLY")
        logger.info("=" * 80)
        logger.info(f"Training duration: {training_duration}")
        logger.info(f"End time: {end_time}")
        
        # Generate training report
        logger.info("Generating training report...")
        report = trainer.create_training_report()
        trainer.save_training_report(report)
        
        # Run evaluation
        logger.info("Running evaluation...")
        evaluation_results = trainer.evaluate_specialists(n_episodes=10)
        
        # Log evaluation results
        logger.info("EVALUATION RESULTS:")
        logger.info("-" * 40)
        for specialist_type, results in evaluation_results.items():
            if results:
                logger.info(f"{specialist_type.upper()}:")
                logger.info(f"  Mean reward: {results['mean_reward']:.4f} ± {results['std_reward']:.4f}")
                logger.info(f"  Mean length: {results['mean_length']:.2f} ± {results['std_length']:.2f}")
            else:
                logger.warning(f"{specialist_type.upper()}: No model found")
        
        logger.info("=" * 80)
        logger.info("PHASE 1 TRAINING SUMMARY")
        logger.info("=" * 80)
        logger.info(f"✅ All specialists trained successfully")
        logger.info(f"✅ Training duration: {training_duration}")
        logger.info(f"✅ Models saved to: {output_path}")
        logger.info(f"✅ Training report generated")
        logger.info(f"✅ Evaluation completed")
        
        return True
        
    except Exception as e:
        logger.error(f"Phase 1 training failed: {e}")
        logger.exception("Full traceback:")
        return False


def main():
    """Main entry point for Phase 1 training."""
    parser = argparse.ArgumentParser(description='Phase 1 Individual Specialist Training')
    parser.add_argument('--config', type=str, default='config/agents.yaml',
                       help='Configuration file path')
    parser.add_argument('--data', type=str, default='data/market_data',
                       help='Market data directory')
    parser.add_argument('--output', type=str, default='models/specialists',
                       help='Output directory for models')
    parser.add_argument('--timesteps', type=int, default=None,
                       help='Total training timesteps per specialist')
    parser.add_argument('--sequential', action='store_true',
                       help='Use sequential training instead of parallel')
    parser.add_argument('--n_envs', type=int, default=8,
                       help='Number of parallel environments')
    parser.add_argument('--eval-only', action='store_true',
                       help='Run evaluation only (skip training)')
    
    args = parser.parse_args()
    
    # Validate paths
    if not os.path.exists(args.config):
        print(f"Error: Configuration file not found: {args.config}")
        sys.exit(1)
    
    if not os.path.exists(args.data):
        print(f"Warning: Data directory not found: {args.data}")
        print("Will create dummy data for testing")
    
    # Create output directory
    Path(args.output).mkdir(parents=True, exist_ok=True)
    
    if args.eval_only:
        # Evaluation only
        logger = setup_logging()
        logger.info("Running evaluation only...")
        
        trainer = Phase1Trainer(
            config_path=args.config,
            data_path=args.data,
            output_path=args.output
        )
        
        evaluation_results = trainer.evaluate_specialists(n_episodes=10)
        
        print("\nEVALUATION RESULTS:")
        print("=" * 50)
        for specialist_type, results in evaluation_results.items():
            if results:
                print(f"{specialist_type.upper()}:")
                print(f"  Mean reward: {results['mean_reward']:.4f} ± {results['std_reward']:.4f}")
                print(f"  Mean length: {results['mean_length']:.2f} ± {results['std_length']:.2f}")
            else:
                print(f"{specialist_type.upper()}: No model found")
        
        return
    
    # Run training
    success = run_phase1_training(
        config_path=args.config,
        data_path=args.data,
        output_path=args.output,
        timesteps=args.timesteps,
        parallel=not args.sequential,
        n_envs=args.n_envs
    )
    
    if success:
        print("\n🎉 Phase 1 training completed successfully!")
        print(f"Models saved to: {args.output}")
        print("Ready for Phase 2: Meta-Controller Training")
    else:
        print("\n❌ Phase 1 training failed!")
        sys.exit(1)


if __name__ == "__main__":
    main()

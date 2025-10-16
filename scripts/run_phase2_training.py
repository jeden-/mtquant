#!/usr/bin/env python3
"""
Phase 2 Training Script - Meta-Controller Pre-training

This script trains the meta-controller to make portfolio-level decisions
using pre-trained specialists from Phase 1.

Usage:
    python scripts/run_phase2_training.py [--config config/agents.yaml] [--timesteps 300000]
"""

import argparse
import sys
import os
from pathlib import Path
import yaml
import logging
from datetime import datetime

# Add project root to path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from mtquant.agents.training.phase2_trainer import Phase2Trainer, create_phase2_trainer
from mtquant.agents.training.train_ppo import create_sample_data
from mtquant.utils.logger import get_logger


def main():
    """Main training function."""
    parser = argparse.ArgumentParser(description="Phase 2 Meta-Controller Training")
    parser.add_argument(
        "--config", 
        type=str, 
        default="config/agents.yaml",
        help="Path to configuration file"
    )
    parser.add_argument(
        "--timesteps", 
        type=int, 
        default=300000,
        help="Number of training timesteps"
    )
    parser.add_argument(
        "--output-dir", 
        type=str, 
        default="models/checkpoints/phase2",
        help="Output directory for models"
    )
    parser.add_argument(
        "--log-dir", 
        type=str, 
        default="logs/phase2",
        help="Log directory"
    )
    parser.add_argument(
        "--device", 
        type=str, 
        default="auto",
        help="Device to use (cuda/cpu/auto)"
    )
    parser.add_argument(
        "--verbose", 
        action="store_true",
        help="Verbose logging"
    )
    
    args = parser.parse_args()
    
    # Setup logging
    log_level = logging.DEBUG if args.verbose else logging.INFO
    logger = get_logger(__name__, level=log_level)
    
    logger.info("=" * 60)
    logger.info("PHASE 2 TRAINING - META-CONTROLLER PRE-TRAINING")
    logger.info("=" * 60)
    logger.info(f"Config: {args.config}")
    logger.info(f"Timesteps: {args.timesteps:,}")
    logger.info(f"Output: {args.output_dir}")
    logger.info(f"Device: {args.device}")
    
    try:
        # Load configuration
        with open(args.config, 'r') as f:
            config = yaml.safe_load(f)
        
        # Override timesteps if specified
        if 'training' not in config:
            config['training'] = {}
        config['training']['phase_2_timesteps'] = args.timesteps
        
        # Create output directories
        Path(args.output_dir).mkdir(parents=True, exist_ok=True)
        Path(args.log_dir).mkdir(parents=True, exist_ok=True)
        
        # Create market data
        logger.info("Creating market data...")
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
        
        # Create Phase 2 trainer
        logger.info("Creating Phase 2 trainer...")
        trainer = create_phase2_trainer(
            config_path=args.config,
            market_data=market_data,
            output_dir=args.output_dir,
            log_dir=args.log_dir,
            device=args.device
        )
        
        # Start training
        logger.info("Starting Phase 2 training...")
        start_time = datetime.now()
        
        results = trainer.train()
        
        end_time = datetime.now()
        training_duration = end_time - start_time
        
        # Log results
        logger.info("=" * 60)
        logger.info("PHASE 2 TRAINING COMPLETED")
        logger.info("=" * 60)
        logger.info(f"Training Duration: {training_duration}")
        logger.info(f"Final Portfolio Sharpe: {results.get('final_portfolio_sharpe', 'N/A')}")
        logger.info(f"Best Model Path: {results.get('best_model_path', 'N/A')}")
        logger.info(f"Training Steps: {results.get('training_steps', 'N/A')}")
        
        # Save training summary
        summary_path = Path(args.output_dir) / "training_summary.json"
        import json
        with open(summary_path, 'w') as f:
            json.dump({
                'phase': 'Phase 2 - Meta-Controller Pre-training',
                'start_time': start_time.isoformat(),
                'end_time': end_time.isoformat(),
                'duration_seconds': training_duration.total_seconds(),
                'config': config,
                'results': results
            }, f, indent=2)
        
        logger.info(f"Training summary saved to: {summary_path}")
        
        return 0
        
    except KeyboardInterrupt:
        logger.info("Training interrupted by user")
        return 1
    except Exception as e:
        logger.error(f"Training failed: {e}")
        if args.verbose:
            import traceback
            traceback.print_exc()
        return 1


if __name__ == "__main__":
    exit_code = main()
    sys.exit(exit_code)

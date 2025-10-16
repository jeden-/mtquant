#!/usr/bin/env python3
"""
Phase 3 Training Script - Joint Fine-tuning

This script performs the final joint training phase where meta-controller
and specialists are trained together to optimize portfolio performance.

Usage:
    python scripts/run_phase3_training.py [--config config/agents.yaml] [--timesteps 1000000]
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

from mtquant.agents.training.phase3_joint_training import Phase3JointTrainer, create_phase3_trainer
from mtquant.agents.training.train_ppo import create_sample_data
from mtquant.utils.logger import get_logger


def main():
    """Main training function."""
    parser = argparse.ArgumentParser(description="Phase 3 Joint Training")
    parser.add_argument(
        "--config", 
        type=str, 
        default="config/agents.yaml",
        help="Path to configuration file"
    )
    parser.add_argument(
        "--timesteps", 
        type=int, 
        default=1000000,
        help="Number of training timesteps"
    )
    parser.add_argument(
        "--output-dir", 
        type=str, 
        default="models/checkpoints/phase3",
        help="Output directory for models"
    )
    parser.add_argument(
        "--log-dir", 
        type=str, 
        default="logs/phase3",
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
    parser.add_argument(
        "--eval-episodes", 
        type=int, 
        default=100,
        help="Number of episodes for evaluation"
    )
    
    args = parser.parse_args()
    
    # Setup logging
    log_level = logging.DEBUG if args.verbose else logging.INFO
    logger = get_logger(__name__, level=log_level)
    
    logger.info("=" * 60)
    logger.info("PHASE 3 TRAINING - JOINT FINE-TUNING")
    logger.info("=" * 60)
    logger.info(f"Config: {args.config}")
    logger.info(f"Timesteps: {args.timesteps:,}")
    logger.info(f"Output: {args.output_dir}")
    logger.info(f"Device: {args.device}")
    logger.info(f"Evaluation Episodes: {args.eval_episodes}")
    
    try:
        # Load configuration
        with open(args.config, 'r') as f:
            config = yaml.safe_load(f)
        
        # Override timesteps if specified
        if 'training' not in config:
            config['training'] = {}
        config['training']['phase_3_timesteps'] = args.timesteps
        
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
        
        # Create Phase 3 trainer
        logger.info("Creating Phase 3 trainer...")
        trainer = create_phase3_trainer(
            config_path=args.config,
            market_data=market_data
        )
        
        # Override output directories
        trainer.config.phase3_models_dir = args.output_dir
        
        # Start training
        logger.info("Starting Phase 3 joint training...")
        start_time = datetime.now()
        
        results = trainer.train()
        
        end_time = datetime.now()
        training_duration = end_time - start_time
        
        # Evaluate trained model
        logger.info("Evaluating trained model...")
        eval_results = trainer.evaluate(n_episodes=args.eval_episodes)
        
        # Log results
        logger.info("=" * 60)
        logger.info("PHASE 3 TRAINING COMPLETED")
        logger.info("=" * 60)
        logger.info(f"Training Duration: {training_duration}")
        logger.info(f"Final Portfolio Sharpe: {results.get('best_portfolio_sharpe', 'N/A')}")
        logger.info(f"Final Model Path: {results.get('final_model_path', 'N/A')}")
        logger.info(f"Training Steps: {results.get('total_timesteps', 'N/A')}")
        
        # Evaluation results
        logger.info("=" * 40)
        logger.info("EVALUATION RESULTS")
        logger.info("=" * 40)
        logger.info(f"Mean Portfolio Sharpe: {eval_results.get('mean_portfolio_sharpe', 'N/A'):.3f}")
        logger.info(f"Mean Max Drawdown: {eval_results.get('mean_max_drawdown', 'N/A'):.3f}")
        logger.info(f"Success Rate: {eval_results.get('success_rate', 'N/A'):.3f}")
        logger.info(f"VaR Violation Rate: {eval_results.get('var_violation_rate', 'N/A'):.3f}")
        
        # Save training summary
        summary_path = Path(args.output_dir) / "training_summary.json"
        import json
        with open(summary_path, 'w') as f:
            json.dump({
                'phase': 'Phase 3 - Joint Fine-tuning',
                'start_time': start_time.isoformat(),
                'end_time': end_time.isoformat(),
                'duration_seconds': training_duration.total_seconds(),
                'config': config,
                'training_results': results,
                'evaluation_results': eval_results
            }, f, indent=2)
        
        logger.info(f"Training summary saved to: {summary_path}")
        
        # Check if targets are met
        mean_sharpe = eval_results.get('mean_portfolio_sharpe', 0)
        mean_drawdown = abs(eval_results.get('mean_max_drawdown', 0))
        
        logger.info("=" * 40)
        logger.info("TARGET ACHIEVEMENT")
        logger.info("=" * 40)
        logger.info(f"Portfolio Sharpe > 2.0: {'✅' if mean_sharpe > 2.0 else '❌'} ({mean_sharpe:.3f})")
        logger.info(f"Max Drawdown < 15%: {'✅' if mean_drawdown < 0.15 else '❌'} ({mean_drawdown:.1%})")
        logger.info(f"VaR Compliance: {'✅' if eval_results.get('var_violation_rate', 1) == 0 else '❌'}")
        
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

#!/usr/bin/env python3
"""
End-to-End Training Script - Complete 3-Phase Training Pipeline

This script executes the complete hierarchical multi-agent training pipeline:
1. Phase 1: Individual specialist training
2. Phase 2: Meta-controller pre-training  
3. Phase 3: Joint fine-tuning

Usage:
    python scripts/run_end_to_end_training.py [--config config/agents.yaml] [--skip-phase1] [--skip-phase2]
"""

import argparse
import sys
import os
from pathlib import Path
import yaml
import logging
from datetime import datetime
import subprocess
import json

# Add project root to path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from mtquant.utils.logger import get_logger


def run_phase1_training(config_path: str, output_dir: str, log_dir: str, device: str, verbose: bool) -> bool:
    """Run Phase 1 training (individual specialists)."""
    logger = get_logger(__name__)
    logger.info("Starting Phase 1 training...")
    
    cmd = [
        sys.executable, "scripts/run_phase1_training.py",
        "--config", config_path,
        "--output-dir", output_dir,
        "--log-dir", log_dir,
        "--device", device
    ]
    
    if verbose:
        cmd.append("--verbose")
    
    try:
        result = subprocess.run(cmd, check=True, capture_output=True, text=True)
        logger.info("Phase 1 training completed successfully")
        return True
    except subprocess.CalledProcessError as e:
        logger.error(f"Phase 1 training failed: {e}")
        logger.error(f"STDOUT: {e.stdout}")
        logger.error(f"STDERR: {e.stderr}")
        return False


def run_phase2_training(config_path: str, output_dir: str, log_dir: str, device: str, verbose: bool) -> bool:
    """Run Phase 2 training (meta-controller pre-training)."""
    logger = get_logger(__name__)
    logger.info("Starting Phase 2 training...")
    
    cmd = [
        sys.executable, "scripts/run_phase2_training.py",
        "--config", config_path,
        "--output-dir", output_dir,
        "--log-dir", log_dir,
        "--device", device
    ]
    
    if verbose:
        cmd.append("--verbose")
    
    try:
        result = subprocess.run(cmd, check=True, capture_output=True, text=True)
        logger.info("Phase 2 training completed successfully")
        return True
    except subprocess.CalledProcessError as e:
        logger.error(f"Phase 2 training failed: {e}")
        logger.error(f"STDOUT: {e.stdout}")
        logger.error(f"STDERR: {e.stderr}")
        return False


def run_phase3_training(config_path: str, output_dir: str, log_dir: str, device: str, verbose: bool) -> bool:
    """Run Phase 3 training (joint fine-tuning)."""
    logger = get_logger(__name__)
    logger.info("Starting Phase 3 training...")
    
    cmd = [
        sys.executable, "scripts/run_phase3_training.py",
        "--config", config_path,
        "--output-dir", output_dir,
        "--log-dir", log_dir,
        "--device", device
    ]
    
    if verbose:
        cmd.append("--verbose")
    
    try:
        result = subprocess.run(cmd, check=True, capture_output=True, text=True)
        logger.info("Phase 3 training completed successfully")
        return True
    except subprocess.CalledProcessError as e:
        logger.error(f"Phase 3 training failed: {e}")
        logger.error(f"STDOUT: {e.stdout}")
        logger.error(f"STDERR: {e.stderr}")
        return False


def main():
    """Main training function."""
    parser = argparse.ArgumentParser(description="End-to-End Hierarchical Training Pipeline")
    parser.add_argument(
        "--config", 
        type=str, 
        default="config/agents.yaml",
        help="Path to configuration file"
    )
    parser.add_argument(
        "--output-dir", 
        type=str, 
        default="models/checkpoints",
        help="Base output directory for all phases"
    )
    parser.add_argument(
        "--log-dir", 
        type=str, 
        default="logs",
        help="Base log directory for all phases"
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
        "--skip-phase1", 
        action="store_true",
        help="Skip Phase 1 training (use existing models)"
    )
    parser.add_argument(
        "--skip-phase2", 
        action="store_true",
        help="Skip Phase 2 training (use existing models)"
    )
    parser.add_argument(
        "--phase1-timesteps", 
        type=int, 
        default=500000,
        help="Phase 1 training timesteps"
    )
    parser.add_argument(
        "--phase2-timesteps", 
        type=int, 
        default=300000,
        help="Phase 2 training timesteps"
    )
    parser.add_argument(
        "--phase3-timesteps", 
        type=int, 
        default=1000000,
        help="Phase 3 training timesteps"
    )
    
    args = parser.parse_args()
    
    # Setup logging
    log_level = logging.DEBUG if args.verbose else logging.INFO
    logger = get_logger(__name__, level=log_level)
    
    logger.info("=" * 80)
    logger.info("END-TO-END HIERARCHICAL MULTI-AGENT TRAINING PIPELINE")
    logger.info("=" * 80)
    logger.info(f"Config: {args.config}")
    logger.info(f"Output Directory: {args.output_dir}")
    logger.info(f"Log Directory: {args.log_dir}")
    logger.info(f"Device: {args.device}")
    logger.info(f"Phase 1 Timesteps: {args.phase1_timesteps:,}")
    logger.info(f"Phase 2 Timesteps: {args.phase2_timesteps:,}")
    logger.info(f"Phase 3 Timesteps: {args.phase3_timesteps:,}")
    logger.info(f"Skip Phase 1: {args.skip_phase1}")
    logger.info(f"Skip Phase 2: {args.skip_phase2}")
    
    # Create directories
    Path(args.output_dir).mkdir(parents=True, exist_ok=True)
    Path(args.log_dir).mkdir(parents=True, exist_ok=True)
    
    # Training results
    training_results = {
        'start_time': datetime.now().isoformat(),
        'phases': {},
        'total_duration': 0,
        'success': False
    }
    
    overall_start_time = datetime.now()
    
    try:
        # Phase 1: Individual Specialist Training
        if not args.skip_phase1:
            logger.info("=" * 60)
            logger.info("PHASE 1: INDIVIDUAL SPECIALIST TRAINING")
            logger.info("=" * 60)
            
            phase1_start = datetime.now()
            success = run_phase1_training(
                config_path=args.config,
                output_dir=f"{args.output_dir}/phase1",
                log_dir=f"{args.log_dir}/phase1",
                device=args.device,
                verbose=args.verbose
            )
            phase1_duration = (datetime.now() - phase1_start).total_seconds()
            
            training_results['phases']['phase1'] = {
                'success': success,
                'duration_seconds': phase1_duration,
                'timesteps': args.phase1_timesteps
            }
            
            if not success:
                logger.error("Phase 1 training failed. Aborting pipeline.")
                return 1
        else:
            logger.info("Skipping Phase 1 training (using existing models)")
            training_results['phases']['phase1'] = {
                'success': True,
                'duration_seconds': 0,
                'timesteps': 0,
                'skipped': True
            }
        
        # Phase 2: Meta-Controller Pre-training
        if not args.skip_phase2:
            logger.info("=" * 60)
            logger.info("PHASE 2: META-CONTROLLER PRE-TRAINING")
            logger.info("=" * 60)
            
            phase2_start = datetime.now()
            success = run_phase2_training(
                config_path=args.config,
                output_dir=f"{args.output_dir}/phase2",
                log_dir=f"{args.log_dir}/phase2",
                device=args.device,
                verbose=args.verbose
            )
            phase2_duration = (datetime.now() - phase2_start).total_seconds()
            
            training_results['phases']['phase2'] = {
                'success': success,
                'duration_seconds': phase2_duration,
                'timesteps': args.phase2_timesteps
            }
            
            if not success:
                logger.error("Phase 2 training failed. Aborting pipeline.")
                return 1
        else:
            logger.info("Skipping Phase 2 training (using existing models)")
            training_results['phases']['phase2'] = {
                'success': True,
                'duration_seconds': 0,
                'timesteps': 0,
                'skipped': True
            }
        
        # Phase 3: Joint Fine-tuning
        logger.info("=" * 60)
        logger.info("PHASE 3: JOINT FINE-TUNING")
        logger.info("=" * 60)
        
        phase3_start = datetime.now()
        success = run_phase3_training(
            config_path=args.config,
            output_dir=f"{args.output_dir}/phase3",
            log_dir=f"{args.log_dir}/phase3",
            device=args.device,
            verbose=args.verbose
        )
        phase3_duration = (datetime.now() - phase3_start).total_seconds()
        
        training_results['phases']['phase3'] = {
            'success': success,
            'duration_seconds': phase3_duration,
            'timesteps': args.phase3_timesteps
        }
        
        if not success:
            logger.error("Phase 3 training failed.")
            return 1
        
        # Pipeline completed successfully
        overall_duration = (datetime.now() - overall_start_time).total_seconds()
        training_results['end_time'] = datetime.now().isoformat()
        training_results['total_duration'] = overall_duration
        training_results['success'] = True
        
        # Log final results
        logger.info("=" * 80)
        logger.info("END-TO-END TRAINING PIPELINE COMPLETED SUCCESSFULLY")
        logger.info("=" * 80)
        logger.info(f"Total Duration: {overall_duration / 3600:.2f} hours")
        logger.info(f"Phase 1 Duration: {training_results['phases']['phase1']['duration_seconds'] / 3600:.2f} hours")
        logger.info(f"Phase 2 Duration: {training_results['phases']['phase2']['duration_seconds'] / 3600:.2f} hours")
        logger.info(f"Phase 3 Duration: {training_results['phases']['phase3']['duration_seconds'] / 3600:.2f} hours")
        
        # Save comprehensive training report
        report_path = Path(args.output_dir) / "end_to_end_training_report.json"
        with open(report_path, 'w') as f:
            json.dump(training_results, f, indent=2)
        
        logger.info(f"Training report saved to: {report_path}")
        
        # Check if targets are met (from Phase 3 evaluation)
        phase3_summary_path = Path(args.output_dir) / "phase3" / "training_summary.json"
        if phase3_summary_path.exists():
            with open(phase3_summary_path, 'r') as f:
                phase3_summary = json.load(f)
            
            eval_results = phase3_summary.get('evaluation_results', {})
            mean_sharpe = eval_results.get('mean_portfolio_sharpe', 0)
            mean_drawdown = abs(eval_results.get('mean_max_drawdown', 0))
            
            logger.info("=" * 40)
            logger.info("FINAL TARGET ACHIEVEMENT")
            logger.info("=" * 40)
            logger.info(f"Portfolio Sharpe > 2.0: {'✅' if mean_sharpe > 2.0 else '❌'} ({mean_sharpe:.3f})")
            logger.info(f"Max Drawdown < 15%: {'✅' if mean_drawdown < 0.15 else '❌'} ({mean_drawdown:.1%})")
            logger.info(f"VaR Compliance: {'✅' if eval_results.get('var_violation_rate', 1) == 0 else '❌'}")
        
        return 0
        
    except KeyboardInterrupt:
        logger.info("Training pipeline interrupted by user")
        training_results['end_time'] = datetime.now().isoformat()
        training_results['interrupted'] = True
        return 1
    except Exception as e:
        logger.error(f"Training pipeline failed: {e}")
        if args.verbose:
            import traceback
            traceback.print_exc()
        training_results['end_time'] = datetime.now().isoformat()
        training_results['error'] = str(e)
        return 1
    finally:
        # Save training results even if interrupted
        report_path = Path(args.output_dir) / "end_to_end_training_report.json"
        with open(report_path, 'w') as f:
            json.dump(training_results, f, indent=2)


if __name__ == "__main__":
    exit_code = main()
    sys.exit(exit_code)
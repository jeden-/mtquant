#!/usr/bin/env python3
"""
MTQuant Training Pipeline - Complete Hierarchical Multi-Agent Training

This script provides a unified interface for the complete 3-phase training pipeline:
1. Phase 1: Individual specialist training (500K timesteps)
2. Phase 2: Meta-controller pre-training (300K timesteps)  
3. Phase 3: Joint fine-tuning (1M timesteps)

Total training time: ~48 hours (estimated)

Usage:
    python scripts/training_pipeline.py --mode [train|eval|resume] [options]
"""

import argparse
import sys
import os
from pathlib import Path
import yaml
import logging
from datetime import datetime
import json
import subprocess
from typing import Dict, List, Optional, Any

# Add project root to path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from mtquant.utils.logger import get_logger, setup_logger


class TrainingPipeline:
    """
    Complete training pipeline coordinator for hierarchical multi-agent system.
    
    This class manages the entire 3-phase training process, including:
    - Phase 1: Individual specialist training
    - Phase 2: Meta-controller pre-training
    - Phase 3: Joint fine-tuning
    - Comprehensive monitoring and evaluation
    """
    
    def __init__(
        self,
        config_path: str = "config/agents.yaml",
        output_dir: str = "models/checkpoints",
        log_dir: str = "logs",
        device: str = "auto"
    ):
        self.config_path = config_path
        self.output_dir = Path(output_dir)
        self.log_dir = Path(log_dir)
        self.device = device
        
        # Setup logging
        self.logger = get_logger(__name__)
        
        # Create directories
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.log_dir.mkdir(parents=True, exist_ok=True)
        
        # Load configuration
        with open(config_path, 'r') as f:
            self.config = yaml.safe_load(f)
        
        # Training state
        self.training_state = {
            'start_time': None,
            'end_time': None,
            'current_phase': None,
            'phases_completed': [],
            'phases_failed': [],
            'total_duration': 0,
            'success': False
        }
        
        # Phase configurations
        self.phase_configs = {
            'phase1': {
                'name': 'Individual Specialist Training',
                'script': 'scripts/run_phase1_training.py',
                'timesteps': self.config.get('training', {}).get('phase_1_timesteps', 500000),
                'output_dir': self.output_dir / 'phase1',
                'log_dir': self.log_dir / 'phase1',
                'estimated_hours': 16
            },
            'phase2': {
                'name': 'Meta-Controller Pre-training',
                'script': 'scripts/run_phase2_training.py',
                'timesteps': self.config.get('training', {}).get('phase_2_timesteps', 300000),
                'output_dir': self.output_dir / 'phase2',
                'log_dir': self.log_dir / 'phase2',
                'estimated_hours': 12
            },
            'phase3': {
                'name': 'Joint Fine-tuning',
                'script': 'scripts/run_phase3_training.py',
                'timesteps': self.config.get('training', {}).get('phase_3_timesteps', 1000000),
                'output_dir': self.output_dir / 'phase3',
                'log_dir': self.log_dir / 'phase3',
                'estimated_hours': 20
            }
        }
    
    def run_phase(self, phase: str, skip_if_exists: bool = True) -> bool:
        """
        Run a specific training phase.
        
        Args:
            phase: Phase name ('phase1', 'phase2', 'phase3')
            skip_if_exists: Skip if phase already completed
            
        Returns:
            True if successful, False otherwise
        """
        if phase not in self.phase_configs:
            self.logger.error(f"Unknown phase: {phase}")
            return False
        
        phase_config = self.phase_configs[phase]
        
        # Check if phase already completed
        if skip_if_exists and self._is_phase_completed(phase):
            self.logger.info(f"Phase {phase} already completed, skipping...")
            return True
        
        self.logger.info("=" * 80)
        self.logger.info(f"STARTING {phase.upper()}: {phase_config['name']}")
        self.logger.info("=" * 80)
        self.logger.info(f"Script: {phase_config['script']}")
        self.logger.info(f"Timesteps: {phase_config['timesteps']:,}")
        self.logger.info(f"Output: {phase_config['output_dir']}")
        self.logger.info(f"Estimated Duration: {phase_config['estimated_hours']} hours")
        
        # Create phase directories
        phase_config['output_dir'].mkdir(parents=True, exist_ok=True)
        phase_config['log_dir'].mkdir(parents=True, exist_ok=True)
        
        # Build command
        cmd = [
            sys.executable, phase_config['script'],
            '--config', self.config_path,
            '--timesteps', str(phase_config['timesteps']),
            '--output', str(phase_config['output_dir'])
        ]
        
        # Run phase
        start_time = datetime.now()
        self.training_state['current_phase'] = phase
        
        try:
            self.logger.info(f"Executing: {' '.join(cmd)}")
            result = subprocess.run(cmd, check=True, capture_output=True, text=True)
            
            end_time = datetime.now()
            duration = (end_time - start_time).total_seconds()
            
            self.logger.info(f"Phase {phase} completed successfully in {duration / 3600:.2f} hours")
            self.training_state['phases_completed'].append(phase)
            
            # Save phase results
            self._save_phase_results(phase, duration, True)
            
            return True
            
        except subprocess.CalledProcessError as e:
            end_time = datetime.now()
            duration = (end_time - start_time).total_seconds()
            
            self.logger.error(f"Phase {phase} failed after {duration / 3600:.2f} hours")
            self.logger.error(f"STDOUT: {e.stdout}")
            self.logger.error(f"STDERR: {e.stderr}")
            
            self.training_state['phases_failed'].append(phase)
            self._save_phase_results(phase, duration, False)
            
            return False
    
    def _is_phase_completed(self, phase: str) -> bool:
        """Check if a phase has been completed successfully."""
        phase_config = self.phase_configs[phase]
        
        # Check for final model
        if phase == 'phase1':
            # Check for specialist models
            specialist_models = list(phase_config['output_dir'].glob('*_final.zip'))
            return len(specialist_models) >= 3  # At least 3 specialists
        elif phase == 'phase2':
            # Check for meta-controller model
            meta_model = phase_config['output_dir'] / 'meta_controller_final.zip'
            return meta_model.exists()
        elif phase == 'phase3':
            # Check for hierarchical system model
            hierarchical_model = phase_config['output_dir'] / 'hierarchical_system_final.zip'
            return hierarchical_model.exists()
        
        return False
    
    def _save_phase_results(self, phase: str, duration: float, success: bool) -> None:
        """Save phase results to file."""
        results = {
            'phase': phase,
            'name': self.phase_configs[phase]['name'],
            'duration_seconds': duration,
            'duration_hours': duration / 3600,
            'success': success,
            'timestamp': datetime.now().isoformat(),
            'config': self.phase_configs[phase]
        }
        
        results_path = self.log_dir / f'{phase}_results.json'
        with open(results_path, 'w') as f:
            json.dump(results, f, indent=2)
    
    def train_all_phases(self, resume: bool = True) -> bool:
        """
        Train all phases in sequence.
        
        Args:
            resume: Resume from last completed phase
            
        Returns:
            True if all phases completed successfully
        """
        self.logger.info("=" * 80)
        self.logger.info("STARTING COMPLETE TRAINING PIPELINE")
        self.logger.info("=" * 80)
        self.logger.info(f"Config: {self.config_path}")
        self.logger.info(f"Output: {self.output_dir}")
        self.logger.info(f"Device: {self.device}")
        self.logger.info(f"Resume: {resume}")
        
        # Calculate total estimated time
        total_estimated_hours = sum(config['estimated_hours'] for config in self.phase_configs.values())
        self.logger.info(f"Total Estimated Time: {total_estimated_hours} hours")
        
        self.training_state['start_time'] = datetime.now()
        
        # Run phases in sequence
        phases = ['phase1', 'phase2', 'phase3']
        
        for phase in phases:
            if not self.run_phase(phase, skip_if_exists=resume):
                self.logger.error(f"Training pipeline failed at {phase}")
                self.training_state['end_time'] = datetime.now()
                self.training_state['success'] = False
                self._save_training_state()
                return False
        
        # All phases completed successfully
        self.training_state['end_time'] = datetime.now()
        self.training_state['success'] = True
        self.training_state['total_duration'] = (
            self.training_state['end_time'] - self.training_state['start_time']
        ).total_seconds()
        
        self.logger.info("=" * 80)
        self.logger.info("TRAINING PIPELINE COMPLETED SUCCESSFULLY")
        self.logger.info("=" * 80)
        self.logger.info(f"Total Duration: {self.training_state['total_duration'] / 3600:.2f} hours")
        self.logger.info(f"Phases Completed: {', '.join(self.training_state['phases_completed'])}")
        
        # Generate final report
        self._generate_final_report()
        
        return True
    
    def evaluate_final_model(self, n_episodes: int = 100) -> Dict[str, Any]:
        """
        Evaluate the final trained model.
        
        Args:
            n_episodes: Number of episodes for evaluation
            
        Returns:
            Evaluation results
        """
        self.logger.info(f"Evaluating final model with {n_episodes} episodes...")
        
        # Run Phase 3 evaluation
        phase3_config = self.phase_configs['phase3']
        cmd = [
            sys.executable, phase3_config['script'],
            '--config', self.config_path,
            '--output-dir', str(phase3_config['output_dir']),
            '--eval-episodes', str(n_episodes)
        ]
        
        try:
            result = subprocess.run(cmd, check=True, capture_output=True, text=True)
            self.logger.info("Final model evaluation completed")
            
            # Parse evaluation results from Phase 3 summary
            summary_path = phase3_config['output_dir'] / 'training_summary.json'
            if summary_path.exists():
                with open(summary_path, 'r') as f:
                    summary = json.load(f)
                return summary.get('evaluation_results', {})
            
        except subprocess.CalledProcessError as e:
            self.logger.error(f"Evaluation failed: {e}")
        
        return {}
    
    def _save_training_state(self) -> None:
        """Save current training state."""
        state_path = self.log_dir / 'training_state.json'
        with open(state_path, 'w') as f:
            json.dump(self.training_state, f, indent=2, default=str)
    
    def _generate_final_report(self) -> None:
        """Generate comprehensive final training report."""
        report = {
            'pipeline_info': {
                'start_time': self.training_state['start_time'].isoformat(),
                'end_time': self.training_state['end_time'].isoformat(),
                'total_duration_hours': self.training_state['total_duration'] / 3600,
                'success': self.training_state['success'],
                'phases_completed': self.training_state['phases_completed'],
                'phases_failed': self.training_state['phases_failed']
            },
            'phase_details': {},
            'final_evaluation': {},
            'config': self.config
        }
        
        # Add phase details
        for phase in self.training_state['phases_completed']:
            results_path = self.log_dir / f'{phase}_results.json'
            if results_path.exists():
                with open(results_path, 'r') as f:
                    report['phase_details'][phase] = json.load(f)
        
        # Add final evaluation
        eval_results = self.evaluate_final_model()
        report['final_evaluation'] = eval_results
        
        # Save report
        report_path = self.output_dir / 'final_training_report.json'
        with open(report_path, 'w') as f:
            json.dump(report, f, indent=2)
        
        self.logger.info(f"Final training report saved to: {report_path}")
        
        # Print summary
        self._print_final_summary(report)
    
    def _print_final_summary(self, report: Dict[str, Any]) -> None:
        """Print final training summary."""
        self.logger.info("=" * 60)
        self.logger.info("FINAL TRAINING SUMMARY")
        self.logger.info("=" * 60)
        
        pipeline_info = report['pipeline_info']
        self.logger.info(f"Total Duration: {pipeline_info['total_duration_hours']:.2f} hours")
        self.logger.info(f"Success: {'✅' if pipeline_info['success'] else '❌'}")
        self.logger.info(f"Phases Completed: {len(pipeline_info['phases_completed'])}/3")
        
        # Phase breakdown
        for phase in pipeline_info['phases_completed']:
            phase_info = report['phase_details'].get(phase, {})
            duration = phase_info.get('duration_hours', 0)
            self.logger.info(f"  {phase}: {duration:.2f} hours")
        
        # Final evaluation results
        eval_results = report.get('final_evaluation', {})
        if eval_results:
            self.logger.info("=" * 40)
            self.logger.info("FINAL EVALUATION RESULTS")
            self.logger.info("=" * 40)
            
            mean_sharpe = eval_results.get('mean_portfolio_sharpe', 0)
            mean_drawdown = abs(eval_results.get('mean_max_drawdown', 0))
            success_rate = eval_results.get('success_rate', 0)
            
            self.logger.info(f"Portfolio Sharpe: {mean_sharpe:.3f}")
            self.logger.info(f"Max Drawdown: {mean_drawdown:.1%}")
            self.logger.info(f"Success Rate: {success_rate:.1%}")
            
            # Target achievement
            self.logger.info("=" * 40)
            self.logger.info("TARGET ACHIEVEMENT")
            self.logger.info("=" * 40)
            self.logger.info(f"Sharpe > 2.0: {'✅' if mean_sharpe > 2.0 else '❌'} ({mean_sharpe:.3f})")
            self.logger.info(f"Drawdown < 15%: {'✅' if mean_drawdown < 0.15 else '❌'} ({mean_drawdown:.1%})")
            self.logger.info(f"Success Rate > 60%: {'✅' if success_rate > 0.6 else '❌'} ({success_rate:.1%})")


def main():
    """Main function."""
    parser = argparse.ArgumentParser(description="MTQuant Training Pipeline")
    parser.add_argument(
        '--mode', 
        choices=['train', 'eval', 'resume'], 
        default='train',
        help='Pipeline mode'
    )
    parser.add_argument(
        '--config', 
        type=str, 
        default='config/agents.yaml',
        help='Configuration file path'
    )
    parser.add_argument(
        '--output-dir', 
        type=str, 
        default='models/checkpoints',
        help='Output directory'
    )
    parser.add_argument(
        '--log-dir', 
        type=str, 
        default='logs',
        help='Log directory'
    )
    parser.add_argument(
        '--device', 
        type=str, 
        default='auto',
        help='Device to use (cuda/cpu/auto)'
    )
    parser.add_argument(
        '--phase', 
        choices=['phase1', 'phase2', 'phase3'], 
        help='Run specific phase only'
    )
    parser.add_argument(
        '--eval-episodes', 
        type=int, 
        default=100,
        help='Number of episodes for evaluation'
    )
    parser.add_argument(
        '--verbose', 
        action='store_true',
        help='Verbose logging'
    )
    
    args = parser.parse_args()
    
    # Setup logging
    log_level = "DEBUG" if args.verbose else "INFO"
    setup_logger(level=log_level)
    logger = get_logger(__name__)
    
    # Create pipeline
    pipeline = TrainingPipeline(
        config_path=args.config,
        output_dir=args.output_dir,
        log_dir=args.log_dir,
        device=args.device
    )
    
    try:
        if args.mode == 'train':
            if args.phase:
                # Run specific phase
                success = pipeline.run_phase(args.phase, skip_if_exists=False)
            else:
                # Run all phases
                success = pipeline.train_all_phases(resume=False)
            
            return 0 if success else 1
            
        elif args.mode == 'eval':
            # Evaluate final model
            eval_results = pipeline.evaluate_final_model(args.eval_episodes)
            logger.info("Evaluation Results:")
            for key, value in eval_results.items():
                logger.info(f"  {key}: {value}")
            return 0
            
        elif args.mode == 'resume':
            # Resume training from last completed phase
            success = pipeline.train_all_phases(resume=True)
            return 0 if success else 1
            
    except KeyboardInterrupt:
        logger.info("Training pipeline interrupted by user")
        return 1
    except Exception as e:
        logger.error(f"Training pipeline failed: {e}")
        if args.verbose:
            import traceback
            traceback.print_exc()
        return 1


if __name__ == "__main__":
    exit_code = main()
    sys.exit(exit_code)

#!/usr/bin/env python3
"""
Standardized File Naming and Organization Utilities for LNN Trading System
Save as: src/utils/file_naming.py

This module provides consistent file naming and folder organization across
all components of the LNN trading system workflow.
"""

import os
import json
import shutil
from datetime import datetime, timedelta
from typing import Dict, Optional, Tuple
from pathlib import Path

class FileNamingStandard:
    """
    Centralized file naming and organization system for the entire LNN workflow.
    Ensures consistent naming conventions across all scripts and stages.
    """
    
    def __init__(self, base_results_dir: str = "results"):
        """
        Initialize the file naming system.
        
        Args:
            base_results_dir: Base directory for all results (default: "results")
        """
        self.base_dir = Path(base_results_dir)
        self.ensure_directory_structure()
        
    def ensure_directory_structure(self):
        """Create the standardized directory structure if it doesn't exist."""
        
        directories = [
            "01_training/models",
            "01_training/analysis_reports", 
            "01_training/batch_results",
            "01_training/plots",
            "02_evaluation/model_performance",
            "02_evaluation/comparative_analysis", 
            "02_evaluation/plots",
            "03_backtesting/individual_backtests",
            "03_backtesting/qualified_models",
            "03_backtesting/qualified_models/champion_models",
            "03_backtesting/period_analysis",
            "04_options/forecasts",
            "04_options/recommendations",
            "04_options/analysis",
            "archive"
        ]
        
        for directory in directories:
            (self.base_dir / directory).mkdir(parents=True, exist_ok=True)
            
        print(f"✓ Ensured directory structure in {self.base_dir}")
    
    # ==== STAGE 1: TRAINING & BATCH ANALYSIS ====
    
    def create_model_filename(self, ticker: str, hidden_size: int, sequence_length: int, 
                            timestamp: Optional[str] = None) -> str:
        """
        Create standardized model filename.
        
        Args:
            ticker: Stock ticker symbol
            hidden_size: Number of hidden units
            sequence_length: Input sequence length
            timestamp: Optional timestamp (if None, uses current time)
            
        Returns:
            Standardized model filename
        """
        if timestamp is None:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        
        return f"{ticker}_{hidden_size}H_{sequence_length}S_{timestamp}.pth"
    
    def create_training_paths(self, ticker: str, hidden_size: int, sequence_length: int,
                            experiment_name: Optional[str] = None) -> Dict[str, str]:
        """
        Create all file paths for a training run.
        
        Returns:
            Dictionary with all standardized paths for this training run
        """
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        
        # Model filename
        model_filename = self.create_model_filename(ticker, hidden_size, sequence_length, timestamp)
        
        # Base name for related files
        base_name = f"{ticker}_{hidden_size}H_{sequence_length}S_{timestamp}"
        
        paths = {
            # Model file
            'model_path': str(self.base_dir / "01_training" / "models" / model_filename),
            
            # Analysis files
            'analysis_json': str(self.base_dir / "01_training" / "analysis_reports" / f"analysis_{base_name}.json"),
            'analysis_report': str(self.base_dir / "01_training" / "analysis_reports" / f"comprehensive_report_{base_name}.txt"),
            
            # Plot files
            'training_plot': str(self.base_dir / "01_training" / "plots" / f"training_{base_name}.png"),
            'features_plot': str(self.base_dir / "01_training" / "plots" / f"features_{base_name}.png"),
            'performance_plot': str(self.base_dir / "01_training" / "plots" / f"performance_{base_name}.png"),
            
            # Metadata
            'experiment_name': experiment_name or f"training_{base_name}",
            'timestamp': timestamp,
            'base_name': base_name
        }
        
        return paths
    
    def create_batch_analysis_paths(self, portfolio_name: str) -> Dict[str, str]:
        """Create file paths for batch analysis results."""
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        
        return {
            'summary_csv': str(self.base_dir / "01_training" / "batch_results" / f"batch_summary_{portfolio_name}_{timestamp}.csv"),
            'detailed_json': str(self.base_dir / "01_training" / "batch_results" / f"batch_detailed_{portfolio_name}_{timestamp}.json"),
            'comparison_csv': str(self.base_dir / "01_training" / "batch_results" / f"batch_comparison_{portfolio_name}_{timestamp}.csv"),
            'timestamp': timestamp
        }
    
    # ==== STAGE 2: EVALUATION & ANALYSIS ====
    
    def create_evaluation_paths(self, model_path: str) -> Dict[str, str]:
        """
        Create evaluation file paths based on a model path.
        
        Args:
            model_path: Path to the trained model file
            
        Returns:
            Dictionary with evaluation file paths
        """
        # Extract model info from filename
        model_filename = Path(model_path).stem  # Remove .pth extension
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        
        return {
            'evaluation_json': str(self.base_dir / "02_evaluation" / "model_performance" / f"evaluation_{model_filename}_{timestamp}.json"),
            'trading_performance': str(self.base_dir / "02_evaluation" / "model_performance" / f"trading_performance_{model_filename}_{timestamp}.json"),
            'prediction_plot': str(self.base_dir / "02_evaluation" / "plots" / f"prediction_accuracy_{model_filename}_{timestamp}.png"),
            'trading_plot': str(self.base_dir / "02_evaluation" / "plots" / f"trading_performance_{model_filename}_{timestamp}.png"),
            'model_filename': model_filename,
            'timestamp': timestamp
        }
    
    def create_comparative_analysis_paths(self) -> Dict[str, str]:
        """Create paths for comparative model analysis."""
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        
        return {
            'rankings_csv': str(self.base_dir / "02_evaluation" / "comparative_analysis" / f"model_rankings_{timestamp}.csv"),
            'performance_summary': str(self.base_dir / "02_evaluation" / "comparative_analysis" / f"performance_summary_{timestamp}.json"),
            'comparison_plot': str(self.base_dir / "02_evaluation" / "plots" / f"model_comparison_{timestamp}.png"),
            'timestamp': timestamp
        }
    
    # ==== STAGE 3: BACKTESTING ====
    
    def create_backtest_paths(self, model_path: str) -> Dict[str, str]:
        """Create backtest file paths."""
        model_filename = Path(model_path).stem
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        
        return {
            'backtest_json': str(self.base_dir / "03_backtesting" / "individual_backtests" / f"backtest_{model_filename}_{timestamp}.json"),
            'period_analysis': str(self.base_dir / "03_backtesting" / "period_analysis" / f"periods_{model_filename}_{timestamp}.json"),
            'model_filename': model_filename,
            'timestamp': timestamp
        }
    
    def create_qualified_models_paths(self) -> Dict[str, str]:
        """Create paths for qualified models summary."""
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        
        return {
            'qualified_csv': str(self.base_dir / "03_backtesting" / "qualified_models" / f"qualified_models_{timestamp}.csv"),
            'qualification_report': str(self.base_dir / "03_backtesting" / "qualified_models" / f"qualification_report_{timestamp}.txt"),
            'timestamp': timestamp
        }
    
    def promote_to_champion(self, model_path: str, performance_score: float) -> str:
        """
        Copy a high-performing model to the champion models directory.
        
        Args:
            model_path: Path to the original model
            performance_score: Performance score for this model
            
        Returns:
            Path to the champion model copy
        """
        model_filename = Path(model_path).name
        champion_filename = model_filename.replace('.pth', f'_champion_score{performance_score:.2f}.pth')
        champion_path = self.base_dir / "03_backtesting" / "qualified_models" / "champion_models" / champion_filename
        
        # Copy the model file
        shutil.copy2(model_path, champion_path)
        
        print(f"✓ Promoted model to champion: {champion_path}")
        return str(champion_path)
    
    # ==== STAGE 4: OPTIONS ANALYSIS ====
    
    def create_options_paths(self, ticker: str, forecast_periods: list = None) -> Dict[str, Dict[str, str]]:
        """Create option analysis file paths."""
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        
        if forecast_periods is None:
            forecast_periods = ['2w', '4w', '6m']
        
        paths = {
            'recommendations': {
                'options_json': str(self.base_dir / "04_options" / "recommendations" / f"options_{ticker}_{timestamp}.json"),
                'recommendations_txt': str(self.base_dir / "04_options" / "recommendations" / f"option_recommendations_{ticker}_{timestamp}.txt"),
            },
            'analysis': {
                'chain_analysis': str(self.base_dir / "04_options" / "analysis" / f"option_chain_analysis_{ticker}_{timestamp}.json"),
                'iv_analysis': str(self.base_dir / "04_options" / "analysis" / f"iv_analysis_{ticker}_{timestamp}.json"),
            },
            'forecasts': {},
            'timestamp': timestamp
        }
        
        # Add forecast paths for each period
        for period in forecast_periods:
            paths['forecasts'][period] = str(self.base_dir / "04_options" / "forecasts" / f"forecast_{ticker}_{period}_{timestamp}.json")
        
        return paths
    
    # ==== UTILITY FUNCTIONS ====
    
    def parse_model_filename(self, model_filename: str) -> Dict[str, str]:
        """
        Parse a standardized model filename to extract metadata.
        
        Args:
            model_filename: Filename to parse (e.g., "LOW_450H_45S_20241223_143022.pth")
            
        Returns:
            Dictionary with parsed components
        """
        try:
            # Remove .pth extension if present
            base_name = model_filename.replace('.pth', '')
            
            # Split into components
            parts = base_name.split('_')
            
            if len(parts) >= 5:
                ticker = parts[0]
                hidden_size = int(parts[1].replace('H', ''))
                sequence_length = int(parts[2].replace('S', ''))
                date_part = parts[3]
                time_part = parts[4]
                
                return {
                    'ticker': ticker,
                    'hidden_size': hidden_size,
                    'sequence_length': sequence_length,
                    'date': date_part,
                    'time': time_part,
                    'timestamp': f"{date_part}_{time_part}",
                    'datetime': datetime.strptime(f"{date_part}_{time_part}", "%Y%m%d_%H%M%S")
                }
            else:
                raise ValueError(f"Filename format not recognized: {model_filename}")
                
        except Exception as e:
            print(f"Warning: Could not parse model filename {model_filename}: {e}")
            return {
                'ticker': 'UNKNOWN',
                'hidden_size': 0,
                'sequence_length': 0,
                'date': 'unknown',
                'time': 'unknown',
                'timestamp': 'unknown'
            }
    
    def find_related_files(self, model_path: str) -> Dict[str, list]:
        """
        Find all files related to a specific model across all stages.
        
        Args:
            model_path: Path to the model file
            
        Returns:
            Dictionary organizing related files by stage
        """
        model_info = self.parse_model_filename(Path(model_path).name)
        base_pattern = f"{model_info['ticker']}_{model_info['hidden_size']}H_{model_info['sequence_length']}S_{model_info['timestamp']}"
        
        related_files = {
            'training': [],
            'evaluation': [],
            'backtesting': [],
            'options': []
        }
        
        # Search each stage directory
        for stage_dir, stage_name in [
            ("01_training", "training"),
            ("02_evaluation", "evaluation"), 
            ("03_backtesting", "backtesting"),
            ("04_options", "options")
        ]:
            stage_path = self.base_dir / stage_dir
            if stage_path.exists():
                for root, dirs, files in os.walk(stage_path):
                    for file in files:
                        if base_pattern in file or model_info['ticker'] in file:
                            file_path = os.path.join(root, file)
                            related_files[stage_name].append(file_path)
        
        return related_files
    
    def archive_old_results(self, days_old: int = 30):
        """
        Archive results older than specified days.
        
        Args:
            days_old: Number of days after which to archive results
        """
        cutoff_date = datetime.now() - timedelta(days=days_old)
        archive_date_dir = self.base_dir / "archive" / cutoff_date.strftime("%Y_%m_%d")
        archive_date_dir.mkdir(parents=True, exist_ok=True)
        
        archived_count = 0
        
        # Check each main directory
        for stage_dir in ["01_training", "02_evaluation", "03_backtesting", "04_options"]:
            stage_path = self.base_dir / stage_dir
            if not stage_path.exists():
                continue
                
            for root, dirs, files in os.walk(stage_path):
                for file in files:
                    file_path = Path(root) / file
                    if file_path.stat().st_mtime < cutoff_date.timestamp():
                        # Move to archive
                        relative_path = file_path.relative_to(self.base_dir)
                        archive_path = archive_date_dir / relative_path
                        archive_path.parent.mkdir(parents=True, exist_ok=True)
                        
                        shutil.move(str(file_path), str(archive_path))
                        archived_count += 1
        
        print(f"✓ Archived {archived_count} files older than {days_old} days to {archive_date_dir}")
    
    def get_workflow_status(self) -> Dict[str, int]:
        """
        Get a summary of files in each workflow stage.
        
        Returns:
            Dictionary with file counts by stage
        """
        status = {}
        
        stage_dirs = {
            "01_training": "Training & Batch Analysis",
            "02_evaluation": "Model Evaluation", 
            "03_backtesting": "Walk-Forward Backtesting",
            "04_options": "Option Analysis"
        }
        
        for stage_dir, stage_name in stage_dirs.items():
            stage_path = self.base_dir / stage_dir
            file_count = 0
            
            if stage_path.exists():
                for root, dirs, files in os.walk(stage_path):
                    file_count += len(files)
            
            status[stage_name] = file_count
        
        return status

# Global instance for easy import
file_namer = FileNamingStandard()

# Convenience functions for quick access
def create_model_paths(ticker: str, hidden_size: int, sequence_length: int, experiment_name: str = None) -> Dict[str, str]:
    """Quick access to create standardized training paths."""
    return file_namer.create_training_paths(ticker, hidden_size, sequence_length, experiment_name)

def create_evaluation_paths(model_path: str) -> Dict[str, str]:
    """Quick access to create evaluation paths."""
    return file_namer.create_evaluation_paths(model_path)

def parse_model_info(model_filename: str) -> Dict[str, str]:
    """Quick access to parse model filename."""
    return file_namer.parse_model_filename(model_filename)

def promote_champion_model(model_path: str, performance_score: float) -> str:
    """Quick access to promote a model to champion status."""
    return file_namer.promote_to_champion(model_path, performance_score)

if __name__ == "__main__":
    # Test the file naming system
    print("Testing LNN File Naming System")
    print("=" * 50)
    
    namer = FileNamingStandard()
    
    # Test model filename creation
    model_filename = namer.create_model_filename("AAPL", 450, 45)
    print(f"Model filename: {model_filename}")
    
    # Test path creation
    paths = namer.create_training_paths("AAPL", 450, 45, "test_experiment")
    print(f"\nTraining paths:")
    for key, path in paths.items():
        print(f"  {key}: {path}")
    
    # Test filename parsing
    parsed = namer.parse_model_filename(model_filename)
    print(f"\nParsed model info:")
    for key, value in parsed.items():
        print(f"  {key}: {value}")
    
    # Test workflow status
    status = namer.get_workflow_status()
    print(f"\nWorkflow status:")
    for stage, count in status.items():
        print(f"  {stage}: {count} files")

import json
import datetime
import os
import pandas as pd
from typing import Dict, Any, Optional, List
import numpy as np

class ExperimentTracker:
    """
    Track machine learning experiments with configurations, metrics, and results.
    Helps compare different model configurations and track improvements over time.
    """
    
    def __init__(self, experiment_dir: str = "experiments"):
        """
        Initialize experiment tracker.
        
        Args:
            experiment_dir: Directory to store experiment logs
        """
        self.experiment_dir = experiment_dir
        self.experiment_file = os.path.join(experiment_dir, "experiments.json")
        
        # Create experiment directory if it doesn't exist
        os.makedirs(experiment_dir, exist_ok=True)
        
        # Initialize experiment file if it doesn't exist
        if not os.path.exists(self.experiment_file):
            with open(self.experiment_file, 'w') as f:
                f.write("")  # Empty file
    
    def log_experiment(self, 
                      experiment_name: str,
                      config: Dict[str, Any],
                      metrics: Dict[str, float],
                      model_path: Optional[str] = None,
                      notes: str = "",
                      tags: Optional[List[str]] = None) -> str:
        """
        Log a new experiment.
        
        Args:
            experiment_name: Name/description of the experiment
            config: Configuration parameters used
            metrics: Performance metrics achieved
            model_path: Path to saved model (optional)
            notes: Additional notes about the experiment
            tags: Tags for categorizing experiments
        
        Returns:
            Experiment ID (timestamp-based)
        """
        # Generate unique experiment ID
        experiment_id = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
        
        # Convert numpy types to Python types for JSON serialization
        config = self._convert_numpy_types(config)
        metrics = self._convert_numpy_types(metrics)
        
        experiment_log = {
            "experiment_id": experiment_id,
            "experiment_name": experiment_name,
            "timestamp": datetime.datetime.now().isoformat(),
            "config": config,
            "metrics": metrics,
            "model_path": model_path,
            "notes": notes,
            "tags": tags or []
        }
        
        # Append to experiment file
        with open(self.experiment_file, "a") as f:
            f.write(json.dumps(experiment_log) + "\n")
        
        print(f"Logged experiment: {experiment_name} (ID: {experiment_id})")
        return experiment_id
    
    def _convert_numpy_types(self, obj: Any) -> Any:
        """Convert numpy types to Python types for JSON serialization."""
        if isinstance(obj, dict):
            return {key: self._convert_numpy_types(value) for key, value in obj.items()}
        elif isinstance(obj, list):
            return [self._convert_numpy_types(item) for item in obj]
        elif isinstance(obj, np.integer):
            return int(obj)
        elif isinstance(obj, np.floating):
            return float(obj)
        elif isinstance(obj, np.ndarray):
            return obj.tolist()
        else:
            return obj
    
    def load_experiments(self) -> List[Dict]:
        """
        Load all logged experiments.
        
        Returns:
            List of experiment dictionaries
        """
        experiments = []
        
        if os.path.exists(self.experiment_file):
            with open(self.experiment_file, 'r') as f:
                for line in f:
                    line = line.strip()
                    if line:
                        try:
                            experiment = json.loads(line)
                            experiments.append(experiment)
                        except json.JSONDecodeError:
                            print(f"Warning: Could not parse line: {line}")
        
        return experiments
    
    def get_experiments_dataframe(self) -> pd.DataFrame:
        """
        Get experiments as a pandas DataFrame for easy analysis.
        
        Returns:
            DataFrame with experiment data
        """
        experiments = self.load_experiments()
        
        if not experiments:
            return pd.DataFrame()
        
        # Flatten experiment data for DataFrame
        flattened_data = []
        for exp in experiments:
            row = {
                'experiment_id': exp.get('experiment_id', ''),
                'experiment_name': exp.get('experiment_name', ''),
                'timestamp': exp.get('timestamp', ''),
                'notes': exp.get('notes', ''),
                'tags': ','.join(exp.get('tags', [])),
                'model_path': exp.get('model_path', '')
            }
            
            # Add config parameters
            config = exp.get('config', {})
            for key, value in config.items():
                row[f'config_{key}'] = value
            
            # Add metrics
            metrics = exp.get('metrics', {})
            for key, value in metrics.items():
                row[f'metric_{key}'] = value
            
            flattened_data.append(row)
        
        return pd.DataFrame(flattened_data)
    
    def get_best_experiments(self, metric_name: str, top_k: int = 5, 
                           higher_is_better: bool = True) -> pd.DataFrame:
        """
        Get top experiments based on a specific metric.
        
        Args:
            metric_name: Name of the metric to sort by
            top_k: Number of top experiments to return
            higher_is_better: Whether higher values are better for this metric
        
        Returns:
            DataFrame with top experiments
        """
        df = self.get_experiments_dataframe()
        
        if df.empty:
            return df
        
        metric_col = f'metric_{metric_name}'
        if metric_col not in df.columns:
            print(f"Warning: Metric '{metric_name}' not found in experiments")
            return pd.DataFrame()
        
        # Sort by metric
        df_sorted = df.sort_values(metric_col, ascending=not higher_is_better)
        
        return df_sorted.head(top_k)
    
    def compare_experiments(self, experiment_ids: List[str]) -> pd.DataFrame:
        """
        Compare specific experiments side by side.
        
        Args:
            experiment_ids: List of experiment IDs to compare
        
        Returns:
            DataFrame comparing the experiments
        """
        df = self.get_experiments_dataframe()
        
        if df.empty:
            return df
        
        # Filter for specified experiments
        comparison_df = df[df['experiment_id'].isin(experiment_ids)]
        
        return comparison_df
    
    def get_experiment_trends(self, metric_name: str) -> pd.DataFrame:
        """
        Get trends for a specific metric over time.
        
        Args:
            metric_name: Name of the metric to track
        
        Returns:
            DataFrame with metric trends over time
        """
        df = self.get_experiments_dataframe()
        
        if df.empty:
            return df
        
        metric_col = f'metric_{metric_name}'
        if metric_col not in df.columns:
            print(f"Warning: Metric '{metric_name}' not found in experiments")
            return pd.DataFrame()
        
        # Sort by timestamp and select relevant columns
        trend_df = df[['timestamp', 'experiment_name', metric_col]].copy()
        trend_df['timestamp'] = pd.to_datetime(trend_df['timestamp'])
        trend_df = trend_df.sort_values('timestamp')
        
        return trend_df
    
    def search_experiments(self, search_term: str, search_fields: List[str] = None) -> pd.DataFrame:
        """
        Search experiments by name, notes, or tags.
        
        Args:
            search_term: Term to search for
            search_fields: Fields to search in (default: name, notes, tags)
        
        Returns:
            DataFrame with matching experiments
        """
        df = self.get_experiments_dataframe()
        
        if df.empty:
            return df
        
        if search_fields is None:
            search_fields = ['experiment_name', 'notes', 'tags']
        
        # Create search mask
        search_mask = pd.Series([False] * len(df))
        
        for field in search_fields:
            if field in df.columns:
                field_mask = df[field].astype(str).str.contains(search_term, case=False, na=False)
                search_mask = search_mask | field_mask
        
        return df[search_mask]
    
    def export_experiments(self, filename: str, format: str = 'csv'):
        """
        Export experiments to file.
        
        Args:
            filename: Output filename
            format: Export format ('csv', 'excel', 'json')
        """
        df = self.get_experiments_dataframe()
        
        if df.empty:
            print("No experiments to export")
            return
        
        filepath = os.path.join(self.experiment_dir, filename)
        
        if format.lower() == 'csv':
            df.to_csv(filepath, index=False)
        elif format.lower() == 'excel':
            df.to_excel(filepath, index=False)
        elif format.lower() == 'json':
            experiments = self.load_experiments()
            with open(filepath, 'w') as f:
                json.dump(experiments, f, indent=2)
        else:
            raise ValueError(f"Unsupported format: {format}")
        
        print(f"Experiments exported to {filepath}")
    
    def delete_experiment(self, experiment_id: str):
        """
        Delete a specific experiment (removes from log file).
        
        Args:
            experiment_id: ID of experiment to delete
        """
        experiments = self.load_experiments()
        
        # Filter out the experiment to delete
        filtered_experiments = [exp for exp in experiments if exp.get('experiment_id') != experiment_id]
        
        if len(filtered_experiments) == len(experiments):
            print(f"Warning: Experiment {experiment_id} not found")
            return
        
        # Rewrite the file without the deleted experiment
        with open(self.experiment_file, 'w') as f:
            for exp in filtered_experiments:
                f.write(json.dumps(exp) + '\n')
        
        print(f"Deleted experiment: {experiment_id}")
    
    def get_summary_stats(self) -> Dict[str, Any]:
        """
        Get summary statistics about all experiments.
        
        Returns:
            Dictionary with summary statistics
        """
        experiments = self.load_experiments()
        
        if not experiments:
            return {'total_experiments': 0}
        
        df = self.get_experiments_dataframe()
        
        # Count experiments by tags
        all_tags = []
        for exp in experiments:
            all_tags.extend(exp.get('tags', []))
        
        tag_counts = pd.Series(all_tags).value_counts().to_dict() if all_tags else {}
        
        # Get metric columns
        metric_cols = [col for col in df.columns if col.startswith('metric_')]
        metric_stats = {}
        
        for col in metric_cols:
            if df[col].dtype in ['int64', 'float64']:
                metric_name = col.replace('metric_', '')
                metric_stats[metric_name] = {
                    'count': df[col].count(),
                    'mean': df[col].mean(),
                    'std': df[col].std(),
                    'min': df[col].min(),
                    'max': df[col].max()
                }
        
        return {
            'total_experiments': len(experiments),
            'date_range': {
                'first': df['timestamp'].min() if 'timestamp' in df.columns else None,
                'last': df['timestamp'].max() if 'timestamp' in df.columns else None
            },
            'tag_counts': tag_counts,
            'metric_statistics': metric_stats
        }

def log_experiment_simple(experiment_name: str, config: Dict, metrics: Dict, 
                         experiment_dir: str = "experiments") -> str:
    """
    Simple function to log an experiment without creating a tracker instance.
    
    Args:
        experiment_name: Name of the experiment
        config: Configuration parameters
        metrics: Performance metrics
        experiment_dir: Directory to store logs
    
    Returns:
        Experiment ID
    """
    tracker = ExperimentTracker(experiment_dir)
    return tracker.log_experiment(experiment_name, config, metrics)

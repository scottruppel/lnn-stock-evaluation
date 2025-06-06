"""
Utility functions for loading YAML configurations
"""

import yaml
from datetime import datetime
import os
from typing import Dict, Any

def load_config(config_name: str) -> Dict[str, Any]:
    """
    Load a YAML configuration file
    
    Args:
        config_name: Name of config file (e.g., 'economic_config')
        
    Returns:
        Dictionary with configuration
    """
    config_path = f"config/{config_name}.yaml"
    
    if not os.path.exists(config_path):
        raise FileNotFoundError(f"Configuration file not found: {config_path}")
    
    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)
    
    return config

def save_config(config: Dict[str, Any], config_name: str):
    """
    Save configuration to YAML file
    
    Args:
        config: Configuration dictionary
        config_name: Name of config file (e.g., 'economic_config')
    """
    config_path = f"config/{config_name}.yaml"
    
    with open(config_path, 'w') as f:
        yaml.dump(config, f, default_flow_style=False, indent=2)

def update_config_value(config_name: str, key_path: str, value: Any):
    """
    Update a specific value in a config file
    
    Args:
        config_name: Name of config file
        key_path: Dot-separated path to key (e.g., 'processing.lookback_months')
        value: New value
    """
    config = load_config(config_name)
    
    # Navigate to the key
    keys = key_path.split('.')
    current = config
    for key in keys[:-1]:
        current = current[key]
    
    # Update the value
    current[keys[-1]] = value
    
    # Save back
    save_config(config, config_name)

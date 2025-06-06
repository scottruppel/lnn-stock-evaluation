#!/usr/bin/env python3
"""
Training script for Liquid Neural Network stock prediction model.
This script orchestrates the entire training pipeline from data loading to model saving.

Usage:
    python scripts/train_model.py
    python scripts/train_model.py --config config/custom_config.yaml
    python scripts/train_model.py --experiment-name "my_experiment"
"""

import os
import sys
import argparse
import yaml
import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import pandas as pd
from datetime import datetime
import warnings
warnings.filterwarnings('ignore')

# Add project root to Python path
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))

# Import our custom modules
from data.data_loader import StockDataLoader
from data.preprocessor import StockDataPreprocessor, prepare_model_data
from models.lnn_model import LiquidNetwork, ModelConfig, create_sequences
from analysis.feature_engineering import AdvancedFeatureEngineer
from utils.metrics import StockPredictionMetrics, MetricTracker
from utils.experiment_tracker import ExperimentTracker

class LNNTrainer:
    """
    Main trainer class that orchestrates the entire training process.
    Handles data preparation, model training, evaluation, and experiment logging.
    """
    
    def __init__(self, config_path: str = "config/config.yaml"):
        """
        Initialize trainer with configuration.
        
        Args:
            config_path: Path to configuration file
        """
        self.config = self.load_config(config_path)
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        print(f"Using device: {self.device}")
        
        # Initialize components
        self.data_loader = None
        self.preprocessor = None
        self.model = None
        self.optimizer = None
        self.criterion = nn.MSELoss()
        self.metric_tracker = MetricTracker()
        self.experiment_tracker = ExperimentTracker()
        
        # Training data storage
        self.train_data = None
        self.val_data = None
        self.test_data = None
        self.scalers = None
        
    def load_config(self, config_path: str) -> dict:
        """Load configuration from YAML file."""
        try:
            with open(config_path, 'r') as f:
                config = yaml.safe_load(f)
            print(f"Loaded configuration from {config_path}")
            return config
        except FileNotFoundError:
            print(f"Config file {config_path} not found. Using default configuration.")
            return self.get_default_config()
    
    def get_default_config(self) -> dict:
        """Provide default configuration if config file is missing."""
        return {
            'data': {
                'tickers': ['^GSPC', 'AGG', 'QQQ', 'AAPL'],
                'start_date': '2020-01-01',
                'end_date': '2024-12-31',
                'target_ticker': 'AAPL'
            },
            'model': {
                'sequence_length': 30,
                'hidden_size': 50,
                'learning_rate': 0.001,
                'batch_size': 32,
                'num_epochs': 100,
                'patience': 10
            },
            'analysis': {
                'use_advanced_features': True,
                'n_components_pca': 10
            }
        }
    
    def prepare_data(self):
        """
        Load and prepare all data for training.
        This is where we orchestrate data loading, preprocessing, and feature engineering.
        """
        print("=" * 50)
        print("STEP 1: DATA PREPARATION")
        print("=" * 50)
        
        # 1. Load raw stock data
        print("Loading stock data...")
        self.data_loader = StockDataLoader(
            tickers=self.config['data']['tickers'],
            start_date=self.config['data']['start_date'],
            end_date=self.config['data']['end_date']
        )
        
        raw_data = self.data_loader.download_data()
        price_data = self.prepare_data_with_enhanced_features()
        
        def debug_data_flow(self, price_data):
            """Debug what's happening to your enhanced features."""
    
            print("\n🔍 DEBUGGING DATA FLOW:")
            print("=" * 50)
    
            target_ticker = self.config['data']['target_ticker']
    
            print(f"Target ticker: {target_ticker}")
            print(f"Price data keys: {list(price_data.keys())}")
    
            for ticker, data in price_data.items():
                print(f"{ticker}: shape = {data.shape}, type = {type(data)}")
        
                if ticker == target_ticker and len(data.shape) > 1:
                    print(f"  ✅ {ticker} has enhanced features: {data.shape[1]} features")
                elif ticker == target_ticker:
                    print(f"  ❌ {ticker} still has basic data: {data.shape}")
        
                # Check first few values
                if hasattr(data, 'flatten'):
                    flat_data = data.flatten()
                    print(f"  Sample values: [{flat_data[0]:.3f}, {flat_data[1]:.3f}, ...]")
    
            return price_data
        
        print(f"Loaded data for {len(price_data)} tickers")
        print(f"Data shape for each ticker: {next(iter(price_data.values())).shape}")
        
        # 2. Check if enhanced features are already created
        if self.config.get('analysis', {}).get('use_advanced_features', False):
            print("Creating advanced features...")
            engineer = AdvancedFeatureEngineer()
            
            enhanced_price_data = {}
            for ticker, prices in price_data.items():
                if ticker != self.config['data']['target_ticker']:
                    # Create OHLCV data structure (using close for all since we only have close)
                    ohlcv_data = {
                        'close': prices,
                        'high': prices,  # Approximation
                        'low': prices,   # Approximation
                        'open': prices,  # Approximation
                        'volume': np.ones_like(prices)  # Dummy volume
                    }
                    
                    # Generate features
                    features, feature_names = engineer.create_comprehensive_features(
                        ohlcv_data, include_advanced=False  # Start simple
                    )
                    enhanced_price_data[ticker] = features
                else:
                    enhanced_price_data[ticker] = prices
            
            price_data = enhanced_price_data
            print(f"Enhanced features created. New feature counts: {[v.shape[1] for v in price_data.values() if len(v.shape) > 1]}")
        
        # 3. Preprocessing and scaling
        print("Preprocessing and scaling data...")
        self.preprocessor = StockDataPreprocessor(
            scaling_method='minmax',
            feature_range=(-1, 1)
        )
        
        scaled_data = self.preprocessor.fit_transform(price_data)
        
        # 4. Prepare sequences for model training
        print("Creating sequences for training...")
        X_train, y_train, X_test, y_test = prepare_model_data(
            price_data=scaled_data,
            target_ticker=self.config['data']['target_ticker'],
            sequence_length=self.config['model']['sequence_length'],
            train_ratio=0.8,
            add_features=False  # We already added features above if needed
        )
        
        # 5. Create validation split from training data
        val_split = 0.2
        val_size = int(len(X_train) * val_split)
        
        X_val = X_train[-val_size:]
        y_val = y_train[-val_size:]
        X_train = X_train[:-val_size]
        y_train = y_train[:-val_size]
        
        # 6. Convert to PyTorch tensors
        self.train_data = {
            'X': torch.tensor(X_train, dtype=torch.float32),
            'y': torch.tensor(y_train, dtype=torch.float32)
        }
        
        self.val_data = {
            'X': torch.tensor(X_val, dtype=torch.float32),
            'y': torch.tensor(y_val, dtype=torch.float32)
        }
        
        self.test_data = {
            'X': torch.tensor(X_test, dtype=torch.float32),
            'y': torch.tensor(y_test, dtype=torch.float32)
        }
        
        print(f"Data preparation complete!")
        print(f"Training set: {self.train_data['X'].shape}")
        print(f"Validation set: {self.val_data['X'].shape}")
        print(f"Test set: {self.test_data['X'].shape}")
        
    def prepare_data_with_enhanced_features(self):
        """Enhanced version with extensive debugging."""
    
        print("🔍 ENHANCED FEATURE DEBUG")
        print("=" * 50)
    
        # Get basic price data first
        price_data = self.data_loader.get_closing_prices()
        target_ticker = self.config['data']['target_ticker']
    
        print(f"🔍 STEP 1: Original price data")
        for ticker, data in price_data.items():
            print(f"   {ticker}: {data.shape}")
    
        # Check if enhanced features are enabled
        use_enhanced = self.config.get('analysis', {}).get('use_advanced_features', False)
        use_abstractions = self.config.get('analysis', {}).get('use_abstractions', False)
    
        print(f"🔍 STEP 2: Feature flags")
        print(f"   use_advanced_features: {use_enhanced}")
        print(f"   use_abstractions: {use_abstractions}")
    
        if use_enhanced or use_abstractions:
            try:
                print("🔍 STEP 3: Creating enhanced features...")
            
                from analysis.market_abstraction_pipeline import EnhancedFeatureEngineer
                enhanced_engineer = EnhancedFeatureEngineer(use_abstractions=use_abstractions)
            
                # Create OHLCV approximation
                target_prices = price_data[target_ticker]
                print(f"   Target prices shape: {target_prices.shape}")
            
                ohlcv_data = {
                    'close': target_prices,
                    'high': target_prices * 1.02,
                    'low': target_prices * 0.98,
                    'open': target_prices,
                    'volume': np.ones_like(target_prices) * 1000000
                }
            
                # Create enhanced features
                features, feature_names = enhanced_engineer.create_features_with_abstractions(
                    price_data=price_data,
                    target_ticker=target_ticker,
                    ohlcv_data=ohlcv_data
                )
            
                print(f"🔍 STEP 4: Enhanced features created")
                print(f"   Features shape: {features.shape}")
                print(f"   Feature count: {len(feature_names)}")
            
                # CRITICAL: Replace ONLY the target ticker with enhanced features
                enhanced_price_data = {}
                for ticker, data in price_data.items():
                    if ticker == target_ticker:
                        print(f"🔍 STEP 5: Replacing {ticker} data")
                        print(f"   Original: {data.shape}")
                        print(f"   Enhanced: {features.shape}")
                        enhanced_price_data[ticker] = features
                    else:
                        enhanced_price_data[ticker] = data
            
                print(f"🔍 STEP 6: Final enhanced_price_data")
                for ticker, data in enhanced_price_data.items():
                    print(f"   {ticker}: {data.shape}")
            
                return enhanced_price_data
            
            except Exception as e:
                print(f"❌ Error creating enhanced features: {e}")
                import traceback
                traceback.print_exc()
                print("   Falling back to basic features...")
                return price_data
    
        else:
            print("🔍 STEP 3: Using basic price features (enhanced disabled)")
            return price_data
            
    def debug_final_data_shapes(self):
        """Debug the final data shapes before training."""
    
        print("\n🔍 FINAL DATA SHAPES DEBUG:")
        print("=" * 50)
    
        print(f"Train data: {self.train_data['X'].shape}")
        print(f"Val data: {self.val_data['X'].shape}")
        print(f"Test data: {self.test_data['X'].shape}")
    
        print(f"Model expects input size: {self.train_data['X'].shape[2]}")
    
        # Check if this makes sense
        expected_samples = 752 - 45  # Original data - sequence length
        actual_samples = self.train_data['X'].shape[0] + self.val_data['X'].shape[0]
    
        print(f"Expected samples: ~{expected_samples}")
        print(f"Actual samples: {actual_samples}")
    
        if actual_samples > expected_samples * 10:
            print("❌ WARNING: Data explosion detected!")
            print("   The preprocessing is creating too many samples")
    
        return True
    
    def initialize_model(self):
        """Initialize the LNN model and optimizer."""
        print("=" * 50)
        print("STEP 2: MODEL INITIALIZATION")
        print("=" * 50)
        
        # Get input size from data
        input_size = self.train_data['X'].shape[2]  # [batch, sequence, features]
        output_size = self.train_data['y'].shape[1] if len(self.train_data['y'].shape) > 1 else 1
        
        print(f"Model input size: {input_size}")
        print(f"Model output size: {output_size}")
        
        # Create model configuration
        model_config = ModelConfig(
            input_size=input_size,
            hidden_size=self.config['model']['hidden_size'],
            output_size=output_size,
            sequence_length=self.config['model']['sequence_length'],
            learning_rate=self.config['model']['learning_rate'],
            batch_size=self.config['model']['batch_size'],
            num_epochs=self.config['model']['num_epochs'],
            patience=self.config['model']['patience']
        )
        
        # Initialize model
        self.model = LiquidNetwork(
            input_size=input_size,
            hidden_size=model_config.hidden_size,
            output_size=output_size
        ).to(self.device)
        
        # Initialize optimizer
        self.optimizer = optim.Adam(
            self.model.parameters(),
            lr=model_config.learning_rate,
            weight_decay=1e-5  # Small L2 regularization
        )
        
        # Print model summary
        total_params = sum(p.numel() for p in self.model.parameters() if p.requires_grad)
        print(f"Model initialized with {total_params:,} trainable parameters")
        
        return model_config
    
    def train_epoch(self, epoch: int) -> float:
        """Train for one epoch."""
        self.model.train()
        total_loss = 0
        batch_size = self.config['model']['batch_size']
        
        # Create random batch indices
        num_samples = len(self.train_data['X'])
        indices = torch.randperm(num_samples)
        
        num_batches = 0
        for start_idx in range(0, num_samples, batch_size):
            # Get batch
            batch_indices = indices[start_idx:start_idx + batch_size]
            batch_x = self.train_data['X'][batch_indices].to(self.device)
            batch_y = self.train_data['y'][batch_indices].to(self.device)
            
            # Forward pass
            self.optimizer.zero_grad()
            outputs = self.model(batch_x)
            
            # Ensure output shape matches target shape
            if len(batch_y.shape) == 1:
                batch_y = batch_y.unsqueeze(1)
            
            loss = self.criterion(outputs, batch_y)
            
            # Backward pass
            loss.backward()
            
            # Gradient clipping to prevent exploding gradients
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=1.0)
            
            self.optimizer.step()
            
            total_loss += loss.item()
            num_batches += 1
        
        return total_loss / num_batches
    
    def validate_epoch(self) -> float:
        """Validate for one epoch."""
        self.model.eval()
        total_loss = 0
        
        with torch.no_grad():
            val_x = self.val_data['X'].to(self.device)
            val_y = self.val_data['y'].to(self.device)
            
            if len(val_y.shape) == 1:
                val_y = val_y.unsqueeze(1)
            
            outputs = self.model(val_x)
            loss = self.criterion(outputs, val_y)
            total_loss = loss.item()
        
        return total_loss
    
    def train_model(self, experiment_name: str = None):
        """
        Main training loop with early stopping and progress tracking.
        
        Args:
            experiment_name: Name for this training experiment
        """
        print("=" * 50)
        print("STEP 3: MODEL TRAINING")
        print("=" * 50)
        
        model_config = self.initialize_model()
        
        # Initialize model_save_path at the start
        model_save_path = None
        
        # Training parameters
        num_epochs = self.config['model']['num_epochs']
        patience = self.config['model']['patience']
        best_val_loss = float('inf')
        patience_counter = 0
        
        # Create model save directory
        os.makedirs('models/saved_models', exist_ok=True)
        
        print(f"Starting training for {num_epochs} epochs...")
        print(f"Early stopping patience: {patience}")
        
        for epoch in range(num_epochs):
            # Train and validate
            train_loss = self.train_epoch(epoch)
            val_loss = self.validate_epoch()
            
            # Update metric tracker
            self.metric_tracker.update(epoch + 1, train_loss, val_loss)
            
            # Print progress
            if (epoch + 1) % 10 == 0 or epoch == 0:
                print(f"Epoch {epoch + 1:3d}/{num_epochs} | "
                      f"Train Loss: {train_loss:.6f} | "
                      f"Val Loss: {val_loss:.6f}")
            
            # Early stopping check
            if val_loss < best_val_loss:
                best_val_loss = val_loss
                patience_counter = 0
                
                # Save best model
                model_save_path = f'models/saved_models/best_lnn_model_{datetime.now().strftime("%Y%m%d_%H%M%S")}.pth'
                torch.save({
                    'model_state_dict': self.model.state_dict(),
                    'optimizer_state_dict': self.optimizer.state_dict(),
                    'epoch': epoch + 1,
                    'val_loss': val_loss,
                    'config': self.config
                }, model_save_path)
                
            else:
                patience_counter += 1
                if patience_counter >= patience:
                    print(f"\nEarly stopping triggered at epoch {epoch + 1}")
                    print(f"Best validation loss: {best_val_loss:.6f}")
                    break
        
        # Training completed - print summary
        print(f"\nTraining completed!")
        print(f"Final training loss: {train_loss:.6f}")
        print(f"Best validation loss: {best_val_loss:.6f}")
        
        # Ensure we always have a model saved
        if model_save_path is None:
            model_save_path = f'models/saved_models/final_lnn_model_{datetime.now().strftime("%Y%m%d_%H%M%S")}.pth'
            torch.save({
                'model_state_dict': self.model.state_dict(),
                'optimizer_state_dict': self.optimizer.state_dict(),
                'epoch': epoch + 1,
                'val_loss': best_val_loss,
                'config': self.config
            }, model_save_path)
            print(f"Saved final model to: {model_save_path}")
        
        # Log experiment if requested
        if experiment_name:
            self.log_experiment(experiment_name, model_config, model_save_path)
        else:
            print("✅ Training completed successfully (no experiment logging)")
        
        return model_save_path
    
    def log_experiment(self, experiment_name: str, model_config: ModelConfig, model_path: str):
        """Log the training experiment."""
        print("=" * 50)
        print("STEP 4: LOGGING EXPERIMENT")
        print("=" * 50)
        
        # Get final metrics
        training_summary = self.metric_tracker.get_training_summary()
        
        # Prepare configuration for logging
        config_dict = {
            'data_config': self.config['data'],
            'model_config': {
                'input_size': model_config.input_size,
                'hidden_size': model_config.hidden_size,
                'output_size': model_config.output_size,
                'sequence_length': model_config.sequence_length,
                'learning_rate': model_config.learning_rate,
                'batch_size': model_config.batch_size,
                'num_epochs': model_config.num_epochs,
                'patience': model_config.patience
            },
            'training_config': {
                'device': str(self.device),
                'total_parameters': sum(p.numel() for p in self.model.parameters()),
                'preprocessing': 'minmax_scaling'
            }
        }
        
        # Log experiment
        experiment_id = self.experiment_tracker.log_experiment(
            experiment_name=experiment_name,
            config=config_dict,
            metrics=training_summary,
            model_path=model_path,
            notes=f"LNN training run on {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}"
        )
        
        print(f"Experiment logged with ID: {experiment_id}")
        return experiment_id

def main():
    """Main function to run training."""
    parser = argparse.ArgumentParser(description='Train Liquid Neural Network for stock prediction')
    parser.add_argument('--config', type=str, default='config/config.yaml',
                      help='Path to configuration file')
    parser.add_argument('--experiment-name', type=str, default=None,
                      help='Name for this experiment')
    parser.add_argument('--gpu', action='store_true',
                      help='Force GPU usage (will fail if not available)')
    
    args = parser.parse_args()
    
    # Set up CUDA if requested
    if args.gpu and not torch.cuda.is_available():
        print("ERROR: GPU requested but CUDA not available!")
        sys.exit(1)
    
    print("="*60)
    print("LIQUID NEURAL NETWORK TRAINING PIPELINE")
    print("="*60)
    print(f"Start time: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print(f"Configuration: {args.config}")
    print(f"Experiment name: {args.experiment_name or 'Unnamed'}")
    
    try:
        # Initialize trainer
        trainer = LNNTrainer(config_path=args.config)
        
        # Run training pipeline
        trainer.prepare_data()
        model_path = trainer.train_model(experiment_name=args.experiment_name)
        
        print("="*60)
        print("TRAINING COMPLETED SUCCESSFULLY!")
        print(f"Model saved to: {model_path}")
        print(f"End time: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
        print("="*60)
        
    except Exception as e:
        print(f"ERROR during training: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)

if __name__ == "__main__":
    main()

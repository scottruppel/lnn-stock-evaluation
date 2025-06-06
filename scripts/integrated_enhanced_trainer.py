#!/usr/bin/env python3
"""
Compatible Integrated Enhanced Trainer
This version works with your existing evaluate_model.py and MetricTracker.
"""

import os
import sys
import yaml
import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
from datetime import datetime
import time
from sklearn.preprocessing import MinMaxScaler

# Add src to path
sys.path.append('src')

from data.data_loader import StockDataLoader
from analysis.market_abstraction_pipeline import EnhancedFeatureEngineer
from models.lnn_model import LiquidNetwork
from utils.experiment_tracker import ExperimentTracker

class CompatibleIntegratedTrainer:
    """
    Integrated trainer that's compatible with your existing codebase.
    Uses only methods/attributes that actually exist in your system.
    """
    
    def __init__(self, config_path: str = None, config: dict = None):
        """Initialize with either config path or config dict."""
        
        if config_path:
            with open(config_path, 'r') as f:
                self.config = yaml.safe_load(f)
        elif config:
            self.config = config
        else:
            raise ValueError("Must provide either config_path or config dict")
        
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.experiment_tracker = ExperimentTracker()
        
        # Simple metrics tracking (compatible with your system)
        self.training_history = {
            'train_losses': [],
            'val_losses': [],
            'epochs': []
        }
        
        # Data containers
        self.enhanced_features = None
        self.feature_names = None
        self.scaler = None
        self.model = None
        self.model_path = None
        self.best_val_loss = float('inf')
        
        print(f"🚀 Compatible Integrated Trainer initialized")
        print(f"   Device: {self.device}")
    
    def create_enhanced_features(self):
        """Create enhanced features using the working pipeline."""
        print("\n🧠 CREATING ENHANCED FEATURES")
        print("=" * 55)
        
        # Load data
        self.data_loader = StockDataLoader(
            tickers=self.config['data']['tickers'],
            start_date=self.config['data']['start_date'],
            end_date=self.config['data']['end_date']
        )
        
        raw_data = self.data_loader.download_data()
        price_data = self.data_loader.get_closing_prices()
        target_ticker = self.config['data']['target_ticker']
        
        print(f"✅ Loaded {len(price_data)} assets for {target_ticker}")
        
        # Determine feature engineering approach
        use_enhanced = self.config.get('analysis', {}).get('use_advanced_features', False)
        use_abstractions = self.config.get('analysis', {}).get('use_abstractions', False)
        
        if use_enhanced or use_abstractions:
            print("🔧 Creating enhanced features with abstractions...")
            
            # Create enhanced features
            enhanced_engineer = EnhancedFeatureEngineer(use_abstractions=use_abstractions)
            
            # Create OHLCV approximation
            target_prices = price_data[target_ticker]
            ohlcv_data = {
                'close': target_prices,
                'high': target_prices * 1.02,
                'low': target_prices * 0.98,
                'open': target_prices,
                'volume': np.ones_like(target_prices) * 1000000
            }
            
            # Generate enhanced features
            features, feature_names = enhanced_engineer.create_features_with_abstractions(
                price_data=price_data,
                target_ticker=target_ticker,
                ohlcv_data=ohlcv_data
            )
            
            print(f"✅ Enhanced features: {features.shape}")
            
        else:
            print("📊 Using basic price features...")
            
            # Use basic price features (multiple assets)
            asset_prices = []
            feature_names = []
            
            for ticker in self.config['data']['tickers']:
                if ticker in price_data:
                    prices = price_data[ticker].flatten()
                    asset_prices.append(prices)
                    feature_names.append(f'{ticker}_price')
            
            # Combine all asset prices
            min_length = min(len(p) for p in asset_prices)
            aligned_prices = np.column_stack([p[:min_length] for p in asset_prices])
            
            # Add basic technical features
            features = self._add_basic_technical_features(aligned_prices, feature_names)
            
            print(f"✅ Basic features: {features.shape}")
        
        # Create target (future returns for target ticker)
        target_prices = price_data[target_ticker].flatten()
        target_returns = np.diff(target_prices) / target_prices[:-1]
        
        # Align features with returns
        min_length = min(len(features), len(target_returns))
        features_aligned = features[-min_length:]
        target_aligned = target_returns[-min_length:]
        
        print(f"✅ Data aligned: features={features_aligned.shape}, target={target_aligned.shape}")
        
        # Store for later use
        self.enhanced_features = features_aligned
        self.target_returns = target_aligned
        self.feature_names = feature_names
        
        return features_aligned, target_aligned, feature_names
    
    def _add_basic_technical_features(self, price_matrix, base_names):
        """Add basic technical features to price matrix."""
        
        features = []
        
        # Add original prices
        features.append(price_matrix)
        
        # Add returns for each asset
        for i in range(price_matrix.shape[1]):
            prices = price_matrix[:, i]
            returns = np.diff(prices) / prices[:-1]
            returns = np.concatenate([[0], returns])  # Pad to same length
            features.append(returns.reshape(-1, 1))
        
        # Add simple moving averages
        for i in range(price_matrix.shape[1]):
            prices = price_matrix[:, i]
            ma_5 = np.convolve(prices, np.ones(5)/5, mode='same')
            ma_20 = np.convolve(prices, np.ones(20)/20, mode='same')
            features.extend([ma_5.reshape(-1, 1), ma_20.reshape(-1, 1)])
        
        combined_features = np.concatenate(features, axis=1)
        
        return combined_features
    
    def prepare_sequences(self):
        """Prepare sequences for training."""
        print("\n📊 PREPARING SEQUENCES")
        print("=" * 55)
        
        if self.enhanced_features is None:
            raise ValueError("Must create features before preparing sequences")
        
        sequence_length = self.config['model']['sequence_length']
        
        # Scale features
        print("🔧 Scaling features...")
        self.scaler = MinMaxScaler(feature_range=(-1, 1))
        features_scaled = self.scaler.fit_transform(self.enhanced_features)
        
        # Create sequences
        X_sequences = []
        y_sequences = []
        
        for i in range(sequence_length, len(features_scaled)):
            X_seq = features_scaled[i-sequence_length:i]  # (seq_len, n_features)
            y_seq = self.target_returns[i]
            
            X_sequences.append(X_seq)
            y_sequences.append(y_seq)
        
        X = np.array(X_sequences)  # (n_samples, seq_len, n_features)
        y = np.array(y_sequences)  # (n_samples,)
        
        print(f"✅ Sequences created: X={X.shape}, y={y.shape}")
        
        # Train/val/test split
        train_size = int(0.8 * len(X))
        val_size = int(0.1 * len(X))
        
        self.X_train = torch.FloatTensor(X[:train_size]).to(self.device)
        self.y_train = torch.FloatTensor(y[:train_size]).unsqueeze(1).to(self.device)
        self.X_val = torch.FloatTensor(X[train_size:train_size+val_size]).to(self.device)
        self.y_val = torch.FloatTensor(y[train_size:train_size+val_size]).unsqueeze(1).to(self.device)
        self.X_test = torch.FloatTensor(X[train_size+val_size:]).to(self.device)
        self.y_test = torch.FloatTensor(y[train_size+val_size:]).unsqueeze(1).to(self.device)
        
        print(f"✅ Data splits:")
        print(f"   Train: {self.X_train.shape}")
        print(f"   Val:   {self.X_val.shape}")
        print(f"   Test:  {self.X_test.shape}")
        
        return X.shape[2]  # Return input size
    
    def train_model(self, experiment_name: str = None):
        """Train the LNN model."""
        print("\n🎯 TRAINING MODEL")
        print("=" * 55)

        if experiment_name is None:
            experiment_name = f"compatible_enhanced_{datetime.now().strftime('%Y%m%d_%H%M%S')}"

        # Import standardized file naming
        from utils.file_naming import create_model_paths

        # Create standardized paths once at beginning
        model_config = self.config.get('model', {})
        data_config = self.config.get('data', {})
        hidden_size = model_config.get('hidden_size', 50)
        sequence_length = model_config.get('sequence_length', 30) 
        target_ticker = data_config.get('target_ticker', 'UNKNOWN')

        # Create standardized paths once
        file_paths = create_model_paths(target_ticker, hidden_size, sequence_length, experiment_name)
        self.model_path = file_paths['model_path']

        print(f"📁 Model will be saved to: {self.model_path}")

        # Create enhanced features and prepare sequences
        self.create_enhanced_features()
        input_size = self.prepare_sequences()

        # Create model
        hidden_size = self.config['model']['hidden_size']
        self.model = LiquidNetwork(
            input_size=input_size,
            hidden_size=hidden_size,
            output_size=1
        ).to(self.device)

        param_count = sum(p.numel() for p in self.model.parameters() if p.requires_grad)
        print(f"✅ Model: {input_size} → {hidden_size} → 1 ({param_count:,} params)")

        # Setup training
        criterion = nn.MSELoss()
        optimizer = optim.Adam(
            self.model.parameters(),
            lr=self.config['model']['learning_rate'],
            weight_decay=self.config['model'].get('weight_decay', 1e-5)
        )

        # Training parameters
        num_epochs = self.config['model']['num_epochs']
        patience = self.config['model']['patience']
        batch_size = self.config['model']['batch_size']

        print(f"📋 Training config: epochs={num_epochs}, lr={self.config['model']['learning_rate']}, batch_size={batch_size}")

        # Initialize training variables
        self.best_val_loss = float('inf')
        patience_counter = 0
        training_start = time.time()

        # Training loop
        for epoch in range(num_epochs):
            # Training phase
            self.model.train()
            train_losses = []

            # Batch training
            for i in range(0, len(self.X_train), batch_size):
                batch_X = self.X_train[i:i+batch_size]
                batch_y = self.y_train[i:i+batch_size]

                optimizer.zero_grad()
                outputs = self.model(batch_X)
                loss = criterion(outputs, batch_y)
                loss.backward()
                optimizer.step()

                train_losses.append(loss.item())

            avg_train_loss = np.mean(train_losses)

            # Validation phase
            self.model.eval()
            with torch.no_grad():
                val_outputs = self.model(self.X_val)
                val_loss = criterion(val_outputs, self.y_val).item()

            # Track metrics (compatible with your system)
            self.training_history['train_losses'].append(avg_train_loss)
            self.training_history['val_losses'].append(val_loss)
            self.training_history['epochs'].append(epoch + 1)

            # Print progress
            if epoch == 0 or (epoch + 1) % 10 == 0:
                print(f"Epoch {epoch+1:3d}/{num_epochs} | Train: {avg_train_loss:.6f} | Val: {val_loss:.6f}")

            # Early stopping check (INSIDE the training loop)
            if val_loss < self.best_val_loss:
                self.best_val_loss = val_loss
                patience_counter = 0

                # Save best model to predetermined path
                torch.save({
                    'model_state_dict': self.model.state_dict(),
                    'optimizer_state_dict': optimizer.state_dict(),
                    'epoch': epoch + 1,
                    'val_loss': val_loss,
                    'config': self.config,
                    'feature_names': self.feature_names,
                    'training_history': self.training_history,
                    'scaler_params': {
                        'data_min_': self.scaler.data_min_,
                        'data_max_': self.scaler.data_max_,
                        'data_range_': self.scaler.data_range_,
                        'scale_': self.scaler.scale_,
                        'min_': self.scaler.min_
                    }
                }, self.model_path)

            else:
                patience_counter += 1
                if patience_counter >= patience:
                    print(f"\nEarly stopping at epoch {epoch+1}")
                    break

        # Training completed
        training_time = time.time() - training_start
        print(f"\n✅ Training completed!")
        print(f"   Time: {training_time:.1f} seconds")
        print(f"   Best val loss: {self.best_val_loss:.6f}")
        print(f"   Model saved: {self.model_path}")
  
        # Log experiment (using your existing ExperimentTracker)
        experiment_id = self.experiment_tracker.log_experiment(
            experiment_name=experiment_name,
            config=self.config,
            metrics={'best_val_loss': self.best_val_loss, 'training_time': training_time},
            model_path=self.model_path,
            notes=f"Compatible enhanced training with {input_size} features"
        )

        return self.model_path
    
    def get_training_summary(self):
        """Get training summary compatible with run_analysis.py expectations."""
        return {
            'model_path': self.model_path,
            'final_val_loss': self.best_val_loss,
            'feature_count': len(self.feature_names) if self.feature_names else 0,
            'training_completed': True,
            'training_history': self.training_history
        }

# Compatibility wrapper for existing run_analysis.py code
class LNNTrainer:
    """Wrapper class that provides compatibility with existing run_analysis.py expectations."""
    
    def __init__(self, config_path: str):
        self.compatible_trainer = CompatibleIntegratedTrainer(config_path=config_path)
        # Create a dummy metric tracker for compatibility
        self.metric_tracker = type('MockMetricTracker', (), {
            'get_training_summary': lambda: self.compatible_trainer.get_training_summary()
        })()
    
    def prepare_data(self):
        """Compatibility method - data preparation is handled in train_model."""
        print("✅ Enhanced data preparation will be handled during training...")
        pass
    
    def train_model(self, experiment_name: str = None):
        """Train model using compatible enhanced trainer."""
        return self.compatible_trainer.train_model(experiment_name)
    
    def get_training_summary(self):
        """Get training summary."""
        return self.compatible_trainer.get_training_summary()

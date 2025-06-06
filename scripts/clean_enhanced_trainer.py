#!/usr/bin/env python3
"""
Clean enhanced feature trainer that bypasses the conflicting pipeline.
This creates enhanced features and trains the model without pipeline conflicts.

Usage:
    python scripts/clean_enhanced_trainer.py --config config/config.yaml
"""

import os
import sys
import argparse
import yaml
import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
from datetime import datetime
import time

# Add src to path
sys.path.append('src')

from data.data_loader import StockDataLoader
from analysis.market_abstraction_pipeline import EnhancedFeatureEngineer
from models.lnn_model import LiquidNetwork
from sklearn.preprocessing import MinMaxScaler

class CleanEnhancedTrainer:
    """Clean trainer that properly handles enhanced features without conflicts."""
    
    def __init__(self, config_path):
        self.config_path = config_path
        self.config = self.load_config()
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        
        print(f"🚀 Clean Enhanced Trainer initialized")
        print(f"   Device: {self.device}")
        print(f"   Config: {config_path}")
    
    def load_config(self):
        """Load configuration."""
        with open(self.config_path, 'r') as f:
            return yaml.safe_load(f)
    
    def create_enhanced_features(self):
        """Create enhanced features cleanly without conflicts."""
        print("\n🧠 CREATING ENHANCED FEATURES")
        print("=" * 50)
        
        # Load data
        data_loader = StockDataLoader(
            tickers=self.config['data']['tickers'],
            start_date=self.config['data']['start_date'],
            end_date=self.config['data']['end_date']
        )
        
        raw_data = data_loader.download_data()
        price_data = data_loader.get_closing_prices()
        target_ticker = self.config['data']['target_ticker']
        
        print(f"Loaded {len(price_data)} assets")
        for ticker, data in price_data.items():
            print(f"  {ticker}: {data.shape}")
        
        # Create enhanced features
        use_abstractions = self.config.get('analysis', {}).get('use_abstractions', False)
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
        
        print(f"✅ Enhanced features created:")
        print(f"   Shape: {features.shape}")
        print(f"   Features: {len(feature_names)}")
        
        # Create target (future returns)
        target_returns = np.diff(target_prices.flatten()) / target_prices.flatten()[:-1]
        
        # Align features with returns
        min_length = min(len(features), len(target_returns))
        features_aligned = features[-min_length:]
        target_aligned = target_returns[-min_length:]
        
        print(f"✅ Data aligned:")
        print(f"   Features: {features_aligned.shape}")
        print(f"   Target: {target_aligned.shape}")
        
        return features_aligned, target_aligned, feature_names
    
    def create_sequences(self, features, target, sequence_length):
        """Create sequences properly for enhanced features."""
        print(f"\n📊 CREATING SEQUENCES")
        print("=" * 50)
        
        X_sequences = []
        y_sequences = []
        
        print(f"Input features shape: {features.shape}")
        print(f"Target shape: {target.shape}")
        print(f"Sequence length: {sequence_length}")
        
        # Create sequences correctly
        for i in range(sequence_length, len(features)):
            # Input: sequence of feature vectors
            X_seq = features[i-sequence_length:i]  # Shape: (sequence_length, n_features)
            # Target: next return
            y_seq = target[i]
            
            X_sequences.append(X_seq)
            y_sequences.append(y_seq)
        
        X = np.array(X_sequences)  # Shape: (n_samples, sequence_length, n_features)
        y = np.array(y_sequences)  # Shape: (n_samples,)
        
        print(f"✅ Sequences created:")
        print(f"   X: {X.shape}")
        print(f"   y: {y.shape}")
        print(f"   Features per timestep: {X.shape[2]}")
        
        return X, y
    
    def prepare_data(self):
        """Prepare data for training."""
        print("\n🔧 PREPARING TRAINING DATA")
        print("=" * 50)
        
        # Create enhanced features
        features, target, feature_names = self.create_enhanced_features()
        
        # Scale features
        print("Scaling features...")
        self.scaler = MinMaxScaler(feature_range=(-1, 1))
        features_scaled = self.scaler.fit_transform(features)
        
        # Create sequences
        sequence_length = self.config['model']['sequence_length']
        X, y = self.create_sequences(features_scaled, target, sequence_length)
        
        # Train/test split
        train_size = int(0.8 * len(X))
        val_size = int(0.1 * len(X))
        
        X_train = X[:train_size]
        y_train = y[:train_size]
        X_val = X[train_size:train_size+val_size]
        y_val = y[train_size:train_size+val_size]
        X_test = X[train_size+val_size:]
        y_test = y[train_size+val_size:]
        
        # Convert to tensors
        self.X_train = torch.FloatTensor(X_train).to(self.device)
        self.y_train = torch.FloatTensor(y_train).unsqueeze(1).to(self.device)
        self.X_val = torch.FloatTensor(X_val).to(self.device)
        self.y_val = torch.FloatTensor(y_val).unsqueeze(1).to(self.device)
        self.X_test = torch.FloatTensor(X_test).to(self.device)
        self.y_test = torch.FloatTensor(y_test).unsqueeze(1).to(self.device)
        
        print(f"✅ Final data preparation:")
        print(f"   Train: X={self.X_train.shape}, y={self.y_train.shape}")
        print(f"   Val:   X={self.X_val.shape}, y={self.y_val.shape}")
        print(f"   Test:  X={self.X_test.shape}, y={self.y_test.shape}")
        
        return self.X_train.shape[2]  # Return input size
    
    def create_model(self, input_size):
        """Create the LNN model."""
        print(f"\n🤖 CREATING MODEL")
        print("=" * 50)
        
        hidden_size = self.config['model']['hidden_size']
        
        self.model = LiquidNetwork(
            input_size=input_size,
            hidden_size=hidden_size,
            output_size=1
        ).to(self.device)
        
        param_count = sum(p.numel() for p in self.model.parameters() if p.requires_grad)
        
        print(f"✅ Model created:")
        print(f"   Architecture: {input_size} → {hidden_size} → 1")
        print(f"   Parameters: {param_count:,}")
        print(f"   Device: {self.device}")
        
        return self.model
    
    def train_model(self):
        """Train the model cleanly."""
        print(f"\n🎯 TRAINING MODEL")
        print("=" * 50)
        
        # Prepare data
        input_size = self.prepare_data()
        
        # Create model
        model = self.create_model(input_size)
        
        # Setup training
        criterion = nn.MSELoss()
        optimizer = optim.Adam(
            model.parameters(),
            lr=self.config['model']['learning_rate']
        )
        
        num_epochs = self.config['model']['num_epochs']
        patience = self.config['model']['patience']
        
        print(f"Training parameters:")
        print(f"  Epochs: {num_epochs}")
        print(f"  Learning rate: {self.config['model']['learning_rate']}")
        print(f"  Batch size: {self.config['model']['batch_size']}")
        print(f"  Patience: {patience}")
        
        # Training loop
        best_val_loss = float('inf')
        patience_counter = 0
        
        print(f"\n🚀 Starting training...")
        start_time = time.time()
        
        for epoch in range(num_epochs):
            # Training
            model.train()
            train_loss = 0.0
            
            # Simple batch processing (no data loader to avoid complications)
            batch_size = self.config['model']['batch_size']
            n_batches = len(self.X_train) // batch_size
            
            for i in range(0, len(self.X_train), batch_size):
                batch_X = self.X_train[i:i+batch_size]
                batch_y = self.y_train[i:i+batch_size]
                
                optimizer.zero_grad()
                outputs = model(batch_X)
                loss = criterion(outputs, batch_y)
                loss.backward()
                optimizer.step()
                
                train_loss += loss.item()
            
            train_loss /= n_batches
            
            # Validation
            model.eval()
            with torch.no_grad():
                val_outputs = model(self.X_val)
                val_loss = criterion(val_outputs, self.y_val).item()
            
            # Print progress
            if epoch == 0 or (epoch + 1) % 10 == 0:
                print(f"Epoch {epoch+1:3d}/{num_epochs} | Train Loss: {train_loss:.6f} | Val Loss: {val_loss:.6f}")
            
            # Early stopping
            if val_loss < best_val_loss:
                best_val_loss = val_loss
                patience_counter = 0
                # Save best model
                torch.save({
                    'model_state_dict': model.state_dict(),
                    'optimizer_state_dict': optimizer.state_dict(),
                    'epoch': epoch,
                    'val_loss': val_loss,
                    'config': self.config
                }, f'models/saved_models/clean_enhanced_model_{datetime.now().strftime("%Y%m%d_%H%M%S")}.pth')
            else:
                patience_counter += 1
                if patience_counter >= patience:
                    print(f"\nEarly stopping triggered at epoch {epoch+1}")
                    break
        
        training_time = time.time() - start_time
        print(f"\n✅ Training completed!")
        print(f"   Time: {training_time:.1f} seconds")
        print(f"   Best validation loss: {best_val_loss:.6f}")
        
        return model
    
    def evaluate_model(self, model):
        """Quick evaluation."""
        print(f"\n📊 QUICK EVALUATION")
        print("=" * 50)
        
        model.eval()
        with torch.no_grad():
            test_outputs = model(self.X_test)
            test_loss = nn.MSELoss()(test_outputs, self.y_test)
            
            # Calculate some basic metrics
            predictions = test_outputs.cpu().numpy().flatten()
            actuals = self.y_test.cpu().numpy().flatten()
            
            mae = np.mean(np.abs(predictions - actuals))
            mape = np.mean(np.abs(predictions - actuals) / (np.abs(actuals) + 1e-8)) * 100
            
            # Directional accuracy
            pred_direction = np.sign(predictions)
            actual_direction = np.sign(actuals)
            directional_accuracy = np.mean(pred_direction == actual_direction)
            
        print(f"✅ Test Results:")
        print(f"   Test Loss: {test_loss:.6f}")
        print(f"   MAE: {mae:.6f}")
        print(f"   MAPE: {mape:.2f}%")
        print(f"   Directional Accuracy: {directional_accuracy:.1%}")
        
        return {
            'test_loss': test_loss.item(),
            'mae': mae,
            'mape': mape,
            'directional_accuracy': directional_accuracy
        }

def main():
    """Main training function."""
    parser = argparse.ArgumentParser(description='Clean Enhanced Feature Trainer')
    parser.add_argument('--config', type=str, default='config/config.yaml',
                       help='Path to configuration file')
    
    args = parser.parse_args()
    
    print("🚀 CLEAN ENHANCED FEATURE TRAINER")
    print("=" * 60)
    print(f"Config: {args.config}")
    print(f"Time: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    
    try:
        # Create trainer
        trainer = CleanEnhancedTrainer(args.config)
        
        # Train model
        model = trainer.train_model()
        
        # Evaluate
        results = trainer.evaluate_model(model)
        
        print(f"\n🎉 SUCCESS! Clean enhanced training completed!")
        
    except Exception as e:
        print(f"\n❌ ERROR: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    main()

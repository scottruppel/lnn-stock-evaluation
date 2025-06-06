import torch
import torch.nn as nn
import numpy as np
from typing import Optional, Tuple

class LiquidTimeCell(nn.Module):
    """
    Enhanced Liquid Time Cell with financial-specific dynamics.
    Maintains full compatibility with existing pipeline.
    """
    def __init__(self, input_size: int, hidden_size: int):
        super().__init__()
        self.input_size = input_size
        self.hidden_size = hidden_size
        
        # Core parameters (same as original)
        self.input_weights = nn.Parameter(torch.randn(input_size, hidden_size) * 0.01)
        self.input_bias = nn.Parameter(torch.zeros(hidden_size))
        self.tau = nn.Parameter(torch.ones(hidden_size) * 0.5 + 0.1)
        
        # Enhanced financial parameters
        self.volatility_sensitivity = nn.Parameter(torch.ones(hidden_size) * 0.1)
        self.momentum_decay = nn.Parameter(torch.ones(hidden_size) * 0.02)
        self.volume_sensitivity = nn.Parameter(torch.ones(hidden_size) * 0.3)
        
    def forward(self, x: torch.Tensor, h_prev: Optional[torch.Tensor] = None) -> torch.Tensor:
        """
        Enhanced forward pass with financial dynamics.
        
        Args:
            x: Input tensor of shape [batch_size, input_size]
            h_prev: Previous hidden state of shape [batch_size, hidden_size]
        
        Returns:
            h_new: Updated hidden state of shape [batch_size, hidden_size]
        """
        # Initialize hidden state if none provided
        if h_prev is None:
            h_prev = torch.zeros(x.size(0), self.hidden_size, device=x.device)
        
        # Calculate input contribution
        input_contrib = torch.matmul(x, self.input_weights) + self.input_bias
        
        # Enhanced market-aware dynamics
        # Estimate volatility from input magnitude
        volatility_est = torch.norm(x, dim=1, keepdim=True).expand(-1, self.hidden_size)
        volatility_est = torch.clamp(volatility_est, 0.01, 2.0)
        
        # Volatility-adjusted time constants
        effective_tau = self.tau.unsqueeze(0) * (1 + self.volatility_sensitivity.unsqueeze(0) * volatility_est)
        effective_tau = torch.clamp(effective_tau, 0.1, 2.0)
        
        # Enhanced LNN dynamics
        base_dynamics = (-h_prev + torch.tanh(input_contrib)) / effective_tau
        
        # Add momentum component for financial time series
        momentum_component = self.momentum_decay.unsqueeze(0) * h_prev * torch.tanh(input_contrib)
        
        # Combine dynamics
        dh = base_dynamics + momentum_component
        
        # Euler integration step
        h_new = h_prev + dh
        
        # Keep outputs bounded for stability
        h_new = torch.tanh(h_new)
        
        return h_new

class LiquidNetwork(nn.Module):
    """
    Enhanced Liquid Neural Network for financial sequence processing.
    Maintains full compatibility with existing pipeline.
    """
    def __init__(self, input_size: int, hidden_size: int, output_size: int, dropout_rate: float = 0.1):
        super().__init__()
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.output_size = output_size
        
        # Core components (same interface as original)
        self.liquid_cell = LiquidTimeCell(input_size, hidden_size)
        self.dropout = nn.Dropout(dropout_rate)
        self.output_layer = nn.Linear(hidden_size, output_size)
        
        # Initialize output layer weights
        nn.init.xavier_uniform_(self.output_layer.weight)
        nn.init.zeros_(self.output_layer.bias)
        
    def forward(self, x_seq: torch.Tensor) -> torch.Tensor:
        """
        Forward pass through the entire liquid network.
        
        Args:
            x_seq: Input sequence of shape [batch_size, seq_len, input_size]
        
        Returns:
            output: Predictions of shape [batch_size, output_size]
        """
        batch_size, seq_len, _ = x_seq.shape
        
        # Initialize hidden state
        h = None
        
        # Process each time step in the sequence
        for t in range(seq_len):
            x_t = x_seq[:, t, :]  # Current time step input
            h = self.liquid_cell(x_t, h)
        
        # Apply dropout to final hidden state
        h = self.dropout(h)
        
        # Project to output space
        output = self.output_layer(h)
        
        return output
    
    def get_hidden_states(self, x_seq: torch.Tensor) -> torch.Tensor:
        """
        Return all hidden states throughout the sequence (useful for analysis).
        
        Args:
            x_seq: Input sequence of shape [batch_size, seq_len, input_size]
        
        Returns:
            hidden_states: All hidden states of shape [batch_size, seq_len, hidden_size]
        """
        batch_size, seq_len, _ = x_seq.shape
        hidden_states = []
        
        h = None
        for t in range(seq_len):
            x_t = x_seq[:, t, :]
            h = self.liquid_cell(x_t, h)
            hidden_states.append(h.unsqueeze(1))  # Add time dimension
        
        return torch.cat(hidden_states, dim=1)
    
    def reset_parameters(self):
        """Reset all parameters to initial values."""
        self.liquid_cell.input_weights.data = torch.randn_like(self.liquid_cell.input_weights) * 0.01
        self.liquid_cell.input_bias.data.zero_()
        self.liquid_cell.tau.data = torch.ones_like(self.liquid_cell.tau) * 0.5 + 0.1
        
        # Reset enhanced parameters
        self.liquid_cell.volatility_sensitivity.data = torch.ones_like(self.liquid_cell.volatility_sensitivity) * 0.1
        self.liquid_cell.momentum_decay.data = torch.ones_like(self.liquid_cell.momentum_decay) * 0.02
        self.liquid_cell.volume_sensitivity.data = torch.ones_like(self.liquid_cell.volume_sensitivity) * 0.3
        
        # Reset output layer
        nn.init.xavier_uniform_(self.output_layer.weight)
        nn.init.zeros_(self.output_layer.bias)

class ModelConfig:
    """Configuration class for LNN model parameters."""
    def __init__(self, 
                 input_size: int,
                 hidden_size: int = 50,
                 output_size: int = 1,
                 dropout_rate: float = 0.1,
                 sequence_length: int = 30,
                 learning_rate: float = 0.001,
                 batch_size: int = 32,
                 num_epochs: int = 100,
                 patience: int = 10):
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.output_size = output_size
        self.dropout_rate = dropout_rate
        self.sequence_length = sequence_length
        self.learning_rate = learning_rate
        self.batch_size = batch_size
        self.num_epochs = num_epochs
        self.patience = patience

def create_sequences(data: np.ndarray, target: np.ndarray, seq_length: int) -> Tuple[np.ndarray, np.ndarray]:
    """
    Create sequential data for training the LNN.
    
    Args:
        data: Input features of shape [n_samples, n_features]
        target: Target values of shape [n_samples, n_targets]
        seq_length: Length of input sequences
    
    Returns:
        xs: Input sequences of shape [n_sequences, seq_length, n_features]
        ys: Target values of shape [n_sequences, n_targets]
    """
    xs, ys = [], []
    
    for i in range(len(data) - seq_length):
        x = data[i:(i + seq_length)]
        y = target[i + seq_length]
        xs.append(x)
        ys.append(y)
    
    return np.array(xs), np.array(ys)

def count_parameters(model: nn.Module) -> int:
    """Count the total number of trainable parameters in the model."""
    return sum(p.numel() for p in model.parameters() if p.requires_grad)

def get_model_summary(model: LiquidNetwork) -> dict:
    """Get a summary of the model architecture and parameters."""
    summary = {
        'input_size': model.input_size,
        'hidden_size': model.hidden_size,
        'output_size': model.output_size,
        'total_parameters': count_parameters(model),
        'tau_range': {
            'min': model.liquid_cell.tau.min().item(),
            'max': model.liquid_cell.tau.max().item(),
            'mean': model.liquid_cell.tau.mean().item()
        }
    }
    
    # Add enhanced parameter info if available
    if hasattr(model.liquid_cell, 'volatility_sensitivity'):
        summary['enhanced_parameters'] = {
            'volatility_sensitivity': {
                'min': model.liquid_cell.volatility_sensitivity.min().item(),
                'max': model.liquid_cell.volatility_sensitivity.max().item(),
                'mean': model.liquid_cell.volatility_sensitivity.mean().item()
            },
            'momentum_decay': {
                'min': model.liquid_cell.momentum_decay.min().item(),
                'max': model.liquid_cell.momentum_decay.max().item(),
                'mean': model.liquid_cell.momentum_decay.mean().item()
            }
        }
    
    return summary

# Import aliases for backward compatibility
LiquidNeuralNetwork = LiquidNetwork  # Alias for compatibility

# Export all important classes
__all__ = [
    'LiquidTimeCell',
    'LiquidNetwork', 
    'LiquidNeuralNetwork',  # Alias
    'ModelConfig',
    'create_sequences',
    'count_parameters',
    'get_model_summary'
]

# Print debug info when imported
if __name__ != "__main__":
    import sys
    if 'debug' in sys.argv or '--debug' in sys.argv:
        print(f"✓ Enhanced LNN model imported successfully")
        print(f"  Available classes: {__all__}")
        print(f"  Enhanced features: volatility adaptation, momentum dynamics")
        print(f"  Full backward compatibility maintained")

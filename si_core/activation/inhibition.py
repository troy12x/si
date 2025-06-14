import torch
import torch.nn as nn
from typing import Optional

class CompetitiveInhibition(nn.Module):
    """Implements competitive inhibition between concepts."""
    def __init__(self, embedding_dim: int):
        super().__init__()
        self.embedding_dim = embedding_dim
        
        # Inhibition strength parameters
        self.inhibition_strength = 0.5
        self.self_inhibition = 0.1
        
        # Inhibition network
        self.inhibition_network = nn.Sequential(
            nn.Linear(embedding_dim, embedding_dim),
            nn.ReLU(),
            nn.Linear(embedding_dim, embedding_dim)
        )
        
    def forward(self, state: torch.Tensor, gate_values: torch.Tensor) -> torch.Tensor:
        """Apply competitive inhibition to the state."""
        # Calculate inhibition signals
        inhibition_signals = self.inhibition_network(state)
        
        # Apply self-inhibition
        self_inhibition = inhibition_signals * self.self_inhibition
        
        # Apply competitive inhibition
        competitive_inhibition = inhibition_signals * self.inhibition_strength
        competitive_inhibition = competitive_inhibition * gate_values
        
        # Combine inhibition signals
        total_inhibition = self_inhibition + competitive_inhibition
        
        # Apply inhibition to state
        inhibited_state = state - total_inhibition
        
        # Ensure activations remain positive
        inhibited_state = torch.relu(inhibited_state)
        
        return inhibited_state

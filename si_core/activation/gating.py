import torch
import torch.nn as nn
from typing import Optional
from .inhibition import CompetitiveInhibition

class ActivationGate(nn.Module):
    """Controls the flow of activation between concepts."""
    def __init__(self, embedding_dim: int):
        super().__init__()
        self.embedding_dim = embedding_dim
        
        # Gating components
        self.gate = nn.Sequential(
            nn.Linear(embedding_dim, embedding_dim),
            nn.Tanh(),
            nn.Linear(embedding_dim, embedding_dim),
            nn.Sigmoid()
        )
        
        self.inhibition = CompetitiveInhibition(embedding_dim)
        
    def forward(self, state: torch.Tensor) -> torch.Tensor:
        """Apply gating to the state."""
        # Calculate gate values
        gate_values = self.gate(state)
        
        # Apply competitive inhibition
        inhibited_state = self.inhibition(state, gate_values)
        
        # Apply final gating
        gated_state = state * gate_values
        
        return gated_state

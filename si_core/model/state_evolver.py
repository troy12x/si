import torch
import torch.nn as nn
from typing import Optional

class StateEvolver(nn.Module):
    """Module that evolves the neural state over time."""
    def __init__(self, state_dim: int, embedding_dim: int, device: str = 'cuda' if torch.cuda.is_available() else 'cpu'):
        super().__init__()
        self.state_dim = state_dim
        self.embedding_dim = embedding_dim
        self.device = device
        
        # Define neural dynamics components
        self.state_update = nn.Sequential(
            nn.Linear(2 * embedding_dim, 128),  # Reduced intermediate layer
            nn.ReLU(),
            nn.Linear(128, embedding_dim),  # Output matches embedding_dim
            nn.Tanh()
        ).to(device)
        
        self.memory_modulator = nn.Sequential(
            nn.Linear(embedding_dim, 64),  # Reduced intermediate layer
            nn.ReLU(),
            nn.Linear(64, embedding_dim),  # Match embedding_dim for element-wise multiplication
            nn.Sigmoid()
        ).to(device)
        
        # Recurrent connection
        self.recurrent = nn.GRUCell(embedding_dim, embedding_dim).to(device)
        
    def forward(self, current_state: torch.Tensor, input_embedding: torch.Tensor) -> torch.Tensor:
        """Evolve the state based on input and current state."""
        # Ensure inputs are 2D tensors
        if current_state.dim() == 1:
            current_state = current_state.unsqueeze(0)
        if input_embedding.dim() == 1:
            input_embedding = input_embedding.unsqueeze(0)
        
        # Combine state and input
        combined = torch.cat([current_state, input_embedding], dim=1)
        
        # Update state
        state_update = self.state_update(combined)
        
        # Apply recurrent connection
        next_state = self.recurrent(state_update, current_state)
        
        # Modulate with memory
        memory_modulation = self.memory_modulator(next_state)
        modulated_state = next_state * memory_modulation
        
        # Squeeze back to 1D
        return modulated_state.squeeze(0)
        
    def to(self, device: str):
        """Override to ensure all components are moved to the correct device."""
        super().to(device)
        self.device = device
        self.state_update = self.state_update.to(device)
        self.memory_modulator = self.memory_modulator.to(device)
        self.recurrent = self.recurrent.to(device)
        return self

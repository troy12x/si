import torch
import torch.nn as nn
from typing import Dict, List, Optional
from ..memory.memory_graph import MemoryGraph
from ..activation.gating import ActivationGate
from ..utils.text_encoder import TextEncoder
from ..utils.tokenizer import Tokenizer
from .state_evolver import StateEvolver
from .response_generator import ResponseGenerator

class SI_Core(nn.Module):
    """Main SI-Core model implementation."""
    def __init__(self, 
                 state_dim: int = 256,  # Reduced state dimension
                 embedding_dim: int = 128,  # Reduced embedding dimension
                 num_concepts: int = 5000,  # Reduced number of concepts
                 device: str = 'cuda' if torch.cuda.is_available() else 'cpu'):
        super().__init__()
        self.state_dim = state_dim
        self.embedding_dim = embedding_dim
        self.device = device
        
        # Core components
        self.state_evolver = StateEvolver(state_dim, embedding_dim).to(device)
        self.memory_graph = MemoryGraph(embedding_dim, num_concepts)
        self.activation_gate = ActivationGate(embedding_dim).to(device)
        self.text_encoder = TextEncoder(embedding_dim, device=device)
        self.tokenizer = Tokenizer()
        
        # Initialize state
        self.state = torch.zeros(embedding_dim).to(device)
        self.persona = torch.randn(embedding_dim).to(device)
        
        # Training parameters
        self.learning_rate = 1e-4
        self.optimizer = torch.optim.Adam(self.parameters(), lr=self.learning_rate)
        
    def to(self, device: str):
        """Override to ensure all components are moved to the correct device."""
        super().to(device)
        self.device = device
        self.state_evolver = self.state_evolver.to(device)
        self.activation_gate = self.activation_gate.to(device)
        self.text_encoder = TextEncoder(self.embedding_dim, device=device)
        self.state = self.state.to(device)
        self.persona = self.persona.to(device)
        return self
        
    def forward(self, input_text: str) -> str:
        """Process input and generate response."""
        # Convert text to embedding
        input_embedding = self.text_encoder.encode(input_text)
        
        # Ensure proper tensor dimensions
        if input_embedding.dim() == 1:
            input_embedding = input_embedding.unsqueeze(0)
        
        # Update state
        self.state = self.state_evolver(self.state, input_embedding)
        
        # Update memory
        self.memory_graph.update_memory(input_embedding.squeeze(0), self.state)
        
        # Apply activation gating
        gated_state = self.activation_gate(self.state)
        
        # Generate response
        response = self._generate_response(gated_state)
        return response
        
    def _generate_response(self, state: torch.Tensor) -> str:
        """Generate response from state."""
        # Get active concepts
        active_concepts = self.memory_graph.get_active_concepts()
        if not active_concepts:
            return "I'm not sure how to respond to that."
            
        # Combine concept embeddings
        response_embedding = torch.zeros(self.embedding_dim).to(self.device)
        for concept in active_concepts:
            response_embedding += concept.embedding * concept.confidence
        
        # Decode to text
        return self.text_encoder.decode(response_embedding)
        
    def train_step(self, input_text: str, target_text: str) -> float:
        """Perform a single training step."""
        self.train()
        self.optimizer.zero_grad()
        
        # Forward pass
        response = self(input_text)
        
        # Compute loss
        target_embedding = self.text_encoder.encode(target_text)
        response_embedding = self.text_encoder.encode(response)
        loss = self._compute_loss(response_embedding, target_embedding)
        
        # Backward pass
        loss.backward()
        self.optimizer.step()
        
        return loss.item()
        
    def _compute_loss(self, response_emb: torch.Tensor, target_emb: torch.Tensor) -> torch.Tensor:
        """Compute loss between response and target embeddings."""
        return torch.nn.functional.cosine_embedding_loss(
            response_emb.unsqueeze(0),
            target_emb.unsqueeze(0),
            torch.tensor([1.0]).to(self.device)
        )

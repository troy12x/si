import torch
import torch.nn as nn
from typing import List, Dict, Optional

class ConceptNode:
    """Represents a single concept node in the semantic graph."""
    def __init__(self, 
                 embedding_dim: int = 512,
                 activation_threshold: float = 0.5,
                 decay_rate: float = 0.95):
        self.embedding = nn.Parameter(torch.randn(embedding_dim))
        self.activation = 0.0
        self.relevance_score = 0.0
        self.history_buffer = []  # type: List[float]
        self.threshold = activation_threshold
        self.decay_rate = decay_rate
        self.confidence = 0.0
        
    def update_activation(self, input_signal: torch.Tensor, context: torch.Tensor) -> None:
        """Update activation based on input and context."""
        # Calculate similarity
        similarity = torch.nn.functional.cosine_similarity(
            input_signal, self.embedding, dim=0
        )
        
        # Update activation with decay
        self.activation = self.decay_rate * self.activation + (1 - self.decay_rate) * similarity.item()
        
        # Update confidence
        self.confidence = self._calculate_confidence()
        
    def _calculate_confidence(self) -> float:
        """Calculate confidence score based on activation and history."""
        history_factor = 1 - np.exp(-len(self.history_buffer))
        return self.activation * history_factor
        
    def add_to_history(self, strength: float) -> None:
        """Add activation strength to history buffer."""
        self.history_buffer.append(strength)
        if len(self.history_buffer) > 100:  # Limit history length
            self.history_buffer.pop(0)
        
    def reset(self) -> None:
        """Reset node state."""
        self.activation = 0.0
        self.relevance_score = 0.0
        self.history_buffer = []
        self.confidence = 0.0

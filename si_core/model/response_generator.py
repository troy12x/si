import torch
import torch.nn as nn
from typing import List, Dict
from ..memory.concept_node import ConceptNode

class ResponseGenerator(nn.Module):
    """Generates coherent responses based on the neural state."""
    def __init__(self, embedding_dim: int, vocab_size: int = 30000):
        super().__init__()
        self.embedding_dim = embedding_dim
        self.vocab_size = vocab_size
        
        # Response generation components
        self.decoder = nn.Sequential(
            nn.Linear(embedding_dim, embedding_dim * 2),
            nn.ReLU(),
            nn.Linear(embedding_dim * 2, vocab_size)
        )
        
        self.softmax = nn.Softmax(dim=0)
        self.temperature = 0.7
        
    def generate_response(self, state: torch.Tensor, memory_graph: 'MemoryGraph') -> str:
        """Generate a response based on the current state and active concepts."""
        # Get active concepts
        active_concepts = self._get_active_concepts(memory_graph)
        if not active_concepts:
            return "I'm not sure how to respond to that."
            
        # Create response embedding
        response_embedding = self._create_response_embedding(active_concepts)
        
        # Generate final response
        return self._decode_response(response_embedding)
        
    def _get_active_concepts(self, memory_graph: 'MemoryGraph') -> List[ConceptNode]:
        """Get nodes above activation threshold."""
        return [node for node in memory_graph.semantic_nodes.values() 
               if node.activation > node.threshold]
        
    def _create_response_embedding(self, concepts: List[ConceptNode]) -> torch.Tensor:
        """Combine concept embeddings to create response embedding."""
        embedding = torch.zeros(self.embedding_dim)
        for concept in concepts:
            embedding += concept.embedding * concept.confidence
        return embedding / len(concepts)
        
    def _decode_response(self, embedding: torch.Tensor) -> str:
        """Convert embedding to text response."""
        # Generate logits
        logits = self.decoder(embedding)
        
        # Apply temperature
        probabilities = self.softmax(logits / self.temperature)
        
        # Sample top-k tokens
        top_k = 50
        top_tokens = torch.topk(probabilities, k=top_k).indices
        
        # Construct sentence
        return self._construct_sentence(top_tokens)
        
    def _construct_sentence(self, token_indices: torch.Tensor) -> str:
        """Convert token indices to a coherent sentence."""
        # This is a placeholder - in practice you'd use a tokenizer
        return " ".join(str(i) for i in token_indices[:10])

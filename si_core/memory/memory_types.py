import torch
from typing import Dict, List, Tuple, Optional
from .concept_node import ConceptNode

class EpisodicMemory:
    """Handles temporary contextual memory with decay."""
    def __init__(self, embedding_dim: int, max_size: int = 1000):
        self.embedding_dim = embedding_dim
        self.max_size = max_size
        self.memory = []  # type: List[Tuple[torch.Tensor, float]]
        self.decay_rate = 0.95
        
    def update(self, input_embedding: torch.Tensor, context: torch.Tensor) -> None:
        """Add new memory with initial strength."""
        self.memory.append((input_embedding, 1.0))
        self._apply_decay()
        self._prune_if_needed()
        
    def _apply_decay(self) -> None:
        """Apply exponential decay to all memories."""
        for i in range(len(self.memory)):
            _, strength = self.memory[i]
            self.memory[i] = (self.memory[i][0], strength * self.decay_rate)
        
    def _prune_if_needed(self) -> None:
        """Prune oldest memories if we exceed max size."""
        if len(self.memory) > self.max_size:
            self.memory = self.memory[-self.max_size:]

class SemanticMemory:
    """Handles core concept nodes and their relationships."""
    def __init__(self, embedding_dim: int, num_concepts: int):
        self.embedding_dim = embedding_dim
        self.num_concepts = num_concepts
        self.concepts = {}  # type: Dict[str, ConceptNode]
        self.relations = {}  # type: Dict[Tuple[str, str], float]
        
    def update(self, input_embedding: torch.Tensor, context: torch.Tensor) -> None:
        """Update concept activations and create new concepts if needed."""
        # Update existing concepts
        for concept in self.concepts.values():
            concept.update_activation(input_embedding, context)
            
        # Create new concepts if needed
        self._create_new_concepts(input_embedding)
        
    def _create_new_concepts(self, input_embedding: torch.Tensor) -> None:
        """Create new concepts based on input similarity."""
        # Placeholder - in practice would use clustering
        pass
        
    def get_active_concepts(self) -> List[ConceptNode]:
        """Get all concepts above activation threshold."""
        return [node for node in self.concepts.values() 
               if node.activation > node.threshold]
        
    def consolidate(self) -> None:
        """Consolidate memory by pruning weak concepts."""
        # Prune concepts with low confidence
        self.concepts = {cid: node for cid, node in self.concepts.items() 
                        if node.confidence > 0.1}
        
        # Prune weak relations
        self.relations = {rel: strength for rel, strength in self.relations.items() 
                         if strength > 0.1}

class ProceduralMemory:
    """Handles action sequences and procedural knowledge."""
    def __init__(self, embedding_dim: int):
        self.embedding_dim = embedding_dim
        self.sequences = []  # type: List[List[str]]
        self.sequence_strengths = []  # type: List[float]
        
    def update(self, input_embedding: torch.Tensor, context: torch.Tensor) -> None:
        """Update procedural sequences based on input."""
        # Placeholder - in practice would use sequence learning
        pass
        
    def consolidate(self) -> None:
        """Consolidate procedural memory by pruning weak sequences."""
        # Prune weak sequences
        self.sequences = [seq for seq, strength in zip(self.sequences, self.sequence_strengths) 
                         if strength > 0.1]
        self.sequence_strengths = [strength for strength in self.sequence_strengths 
                                  if strength > 0.1]

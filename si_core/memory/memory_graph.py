import torch
from typing import Dict, List, Tuple, Optional
from .concept_node import ConceptNode
from .memory_types import EpisodicMemory, SemanticMemory, ProceduralMemory

class MemoryGraph:
    """Graph-based memory system with semantic, episodic, and procedural components."""
    def __init__(self, 
                 embedding_dim: int = 512,
                 num_concepts: int = 10000,
                 decay_rate: float = 0.95):
        self.embedding_dim = embedding_dim
        self.num_concepts = num_concepts
        self.decay_rate = decay_rate
        
        # Initialize memory components
        self.semantic_memory = SemanticMemory(embedding_dim, num_concepts)
        self.episodic_memory = EpisodicMemory(embedding_dim)
        self.procedural_memory = ProceduralMemory(embedding_dim)
        
    def update_memory(self, input_embedding: torch.Tensor, context: torch.Tensor) -> None:
        """Update all memory components."""
        # Update semantic memory
        self.semantic_memory.update(input_embedding, context)
        
        # Update episodic memory with decay
        self.episodic_memory.update(input_embedding, context)
        
        # Update procedural memory if applicable
        self.procedural_memory.update(input_embedding, context)
        
    def get_active_concepts(self) -> List[ConceptNode]:
        """Get all active concepts from semantic memory."""
        return self.semantic_memory.get_active_concepts()
        
    def consolidate_memory(self) -> None:
        """Consolidate memory by pruning weak connections."""
        # Prune episodic memory
        self.episodic_memory.prune()
        
        # Consolidate semantic concepts
        self.semantic_memory.consolidate()
        
        # Update procedural sequences
        self.procedural_memory.consolidate()
        
    def reset(self) -> None:
        """Reset all memory components."""
        self.semantic_memory.reset()
        self.episodic_memory.reset()
        self.procedural_memory.reset()

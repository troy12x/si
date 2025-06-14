import torch
import torch.nn as nn
from typing import List, Dict, Optional
from ..memory.concept_node import ConceptNode

class ConceptPropagation(nn.Module):
    """Handles the propagation of activation between concepts."""
    def __init__(self, embedding_dim: int):
        super().__init__()
        self.embedding_dim = embedding_dim
        
        # Propagation components
        self.propagation_network = nn.Sequential(
            nn.Linear(embedding_dim, embedding_dim),
            nn.ReLU(),
            nn.Linear(embedding_dim, embedding_dim)
        )
        
        self.damping_factor = 0.8
        self.max_hops = 3
        
    def propagate(self, 
                 source_concept: ConceptNode, 
                 memory_graph: 'MemoryGraph',
                 max_hops: int = None) -> List[ConceptNode]:
        """Propagate activation from source concept through the graph."""
        if max_hops is None:
            max_hops = self.max_hops
            
        activated_concepts = [source_concept]
        current_hops = 0
        
        while current_hops < max_hops:
            new_activations = []
            for concept in activated_concepts:
                neighbors = self._get_neighbors(concept, memory_graph)
                for neighbor in neighbors:
                    if neighbor not in activated_concepts:
                        new_activations.append(neighbor)
            
            if not new_activations:
                break
                
            activated_concepts.extend(new_activations)
            current_hops += 1
            
        return activated_concepts
        
    def _get_neighbors(self, 
                      concept: ConceptNode, 
                      memory_graph: 'MemoryGraph') -> List[ConceptNode]:
        """Get neighboring concepts based on relations."""
        neighbors = []
        for rel, strength in memory_graph.relations.items():
            if concept.id in rel:
                other_id = rel[0] if rel[1] == concept.id else rel[1]
                if other_id in memory_graph.concepts:
                    neighbors.append(memory_graph.concepts[other_id])
        return neighbors
        
    def _calculate_propagation_strength(self, 
                                       source: ConceptNode, 
                                       target: ConceptNode) -> float:
        """Calculate strength of propagation between concepts."""
        similarity = torch.nn.functional.cosine_similarity(
            source.embedding, target.embedding, dim=0
        )
        return similarity.item() * self.damping_factor

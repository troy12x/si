import torch
import torch.nn as nn
from transformers import AutoModel, AutoTokenizer
from typing import Optional

class TextEncoder(nn.Module):
    """Handles text-to-embedding conversion and decoding."""
    def __init__(self, embedding_dim: int = 128, device: str = 'cuda' if torch.cuda.is_available() else 'cpu'):
        super().__init__()
        self.embedding_dim = embedding_dim
        self.device = device
        
        # Initialize lightweight transformer-based encoder on CPU
        self.model = AutoModel.from_pretrained('sentence-transformers/paraphrase-distilroberta-base-v1')
        self.tokenizer = AutoTokenizer.from_pretrained('sentence-transformers/paraphrase-distilroberta-base-v1')
        
    def encode(self, text: str) -> torch.Tensor:
        """Convert text to embedding."""
        # Move inputs to CPU
        inputs = self.tokenizer(text, return_tensors='pt', padding=True, truncation=True)
        
        # Only use no_grad during inference
        if not self.training:
            with torch.no_grad():
                outputs = self.model(**inputs)
        else:
            outputs = self.model(**inputs)
        
        embedding = outputs.last_hidden_state.mean(dim=1)
        
        # Ensure embedding has correct dimension
        if embedding.size(-1) != self.embedding_dim:
            embedding = torch.nn.functional.adaptive_avg_pool1d(
                embedding.unsqueeze(0), self.embedding_dim
            ).squeeze(0)
        
        # Handle batch dimension
        if embedding.dim() == 3:  # If batch size is 1
            embedding = embedding.squeeze(0)
        
        # Move final embedding to target device
        return embedding.to(self.device)
        
    def decode(self, embedding: torch.Tensor) -> str:
        """Convert embedding to text."""
        # Placeholder - in practice would use a decoder model
        return "[DECODING NOT IMPLEMENTED]"
        
    def get_embedding_dim(self) -> int:
        """Get the embedding dimension."""
        return self.embedding_dim
        
    def train(self, mode: bool = True):
        """Set module in training mode."""
        super().train(mode)
        self.model.train(mode)
        return self
        
    def eval(self):
        """Set module in evaluation mode."""
        super().eval()
        self.model.eval()
        return self

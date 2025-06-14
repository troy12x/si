import torch
import torch.nn as nn
from transformers import AutoTokenizer
from typing import List, Dict, Optional

class Tokenizer:
    """Handles text tokenization and detokenization."""
    def __init__(self, model_name: str = 'sentence-transformers/all-MiniLM-L6-v2'):
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.pad_token_id = self.tokenizer.pad_token_id
        
    def tokenize(self, text: str) -> Dict[str, torch.Tensor]:
        """Convert text to tokenized format."""
        return self.tokenizer(
            text,
            padding=True,
            truncation=True,
            return_tensors='pt'
        )
        
    def detokenize(self, tokens: torch.Tensor) -> str:
        """Convert tokens back to text."""
        return self.tokenizer.decode(
            tokens,
            skip_special_tokens=True,
            clean_up_tokenization_spaces=True
        )
        
    def encode(self, text: str) -> torch.Tensor:
        """Encode text to tensor."""
        tokens = self.tokenize(text)
        return tokens['input_ids']
        
    def decode(self, tokens: torch.Tensor) -> str:
        """Decode tensor to text."""
        return self.detokenize(tokens)
        
    def get_vocab_size(self) -> int:
        """Get vocabulary size."""
        return len(self.tokenizer)
        
    def get_special_tokens(self) -> Dict[str, str]:
        """Get special tokens."""
        return {
            'pad_token': self.tokenizer.pad_token,
            'bos_token': self.tokenizer.bos_token,
            'eos_token': self.tokenizer.eos_token
        }

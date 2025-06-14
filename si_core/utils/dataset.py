import torch
import json
from typing import List, Dict, Tuple
from torch.utils.data import Dataset, DataLoader

class SI_Dataset(Dataset):
    """Dataset for SI-Core training."""
    def __init__(self, 
                 data_path: str,
                 max_length: int = 512,
                 device: str = 'cuda' if torch.cuda.is_available() else 'cpu'):
        self.device = device
        self.max_length = max_length
        
        # Load data
        with open(data_path, 'r') as f:
            self.data = json.load(f)
        
    def __len__(self) -> int:
        return len(self.data)
        
    def __getitem__(self, idx: int) -> Tuple[str, str]:
        item = self.data[idx]
        return item['input'], item['target']
        
    @staticmethod
    def collate_fn(batch):
        """Collate function for DataLoader."""
        inputs = [item[0] for item in batch]
        targets = [item[1] for item in batch]
        return inputs, targets

class DataProcessor:
    """Processes and prepares training data."""
    def __init__(self, 
                 train_path: str,
                 val_path: str,
                 batch_size: int = 32,
                 num_workers: int = 4):
        self.train_dataset = SI_Dataset(train_path)
        self.val_dataset = SI_Dataset(val_path)
        
        self.train_loader = DataLoader(
            self.train_dataset,
            batch_size=batch_size,
            shuffle=True,
            num_workers=num_workers,
            collate_fn=SI_Dataset.collate_fn
        )
        
        self.val_loader = DataLoader(
            self.val_dataset,
            batch_size=batch_size,
            shuffle=False,
            num_workers=num_workers,
            collate_fn=SI_Dataset.collate_fn
        )
        
    def get_train_loader(self) -> DataLoader:
        return self.train_loader
        
    def get_val_loader(self) -> DataLoader:
        return self.val_loader

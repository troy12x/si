import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.tensorboard import SummaryWriter
import numpy as np
from datetime import datetime
import os
from typing import Dict, Any
from si_core.model import SI_Core
from si_core.utils.dataset import DataProcessor
from si_core.utils.text_encoder import TextEncoder

class Trainer:
    """Trainer for SI-Core model."""
    def __init__(self,
                 model: SI_Core,
                 train_path: str,
                 val_path: str,
                 batch_size: int = 10,
                 learning_rate: float = 1e-4,
                 device: str = 'cuda' if torch.cuda.is_available() else 'cpu'):
        self.model = model.to(device)
        self.device = device
        self.data_processor = DataProcessor(train_path, val_path, batch_size)
        
        # Optimizer and scheduler
        self.optimizer = optim.AdamW(self.model.parameters(), lr=learning_rate)
        self.scheduler = optim.lr_scheduler.ReduceLROnPlateau(
            self.optimizer,
            mode='min',
            factor=0.5,
            patience=3,
            verbose=True
        )
        
        # Loss function
        self.loss_fn = nn.CosineEmbeddingLoss()
        
        # TensorBoard logger
        self.writer = SummaryWriter(f'runs/si_core_{datetime.now().strftime("%Y%m%d_%H%M%S")}')
        
        # Text encoder for embeddings
        self.text_encoder = TextEncoder(device=device)
        
    def train_epoch(self, epoch: int) -> float:
        """Train for one epoch."""
        self.model.train()
        train_loader = self.data_processor.get_train_loader()
        
        total_loss = 0
        for batch_idx, (inputs, targets) in enumerate(train_loader):
            # Convert to embeddings
            input_embs = []
            target_embs = []
            
            for input_text, target_text in zip(inputs, targets):
                # Encode input
                input_emb = self.text_encoder.encode(input_text)
                if input_emb.dim() == 1:
                    input_emb = input_emb.unsqueeze(0)
                input_embs.append(input_emb)
                
                # Encode target
                target_emb = self.text_encoder.encode(target_text)
                if target_emb.dim() == 1:
                    target_emb = target_emb.unsqueeze(0)
                target_embs.append(target_emb)
            
            # Convert to tensors
            input_embs = torch.stack(input_embs).squeeze(1).to(self.device)
            target_embs = torch.stack(target_embs).squeeze(1).to(self.device)
            
            # Zero gradients
            self.optimizer.zero_grad()
            
            # Forward pass
            responses = []
            for input_text in inputs:
                response = self.model(input_text)
                response_emb = self.text_encoder.encode(response)
                if response_emb.dim() == 1:
                    response_emb = response_emb.unsqueeze(0)
                responses.append(response_emb)
            
            response_embs = torch.stack(responses).squeeze(1).to(self.device)
            
            # Calculate loss
            loss = self.loss_fn(response_embs, target_embs, 
                              torch.ones(len(response_embs)).to(self.device))
            
            # Backward pass
            loss.backward()
            self.optimizer.step()
            
            # Update total loss
            total_loss += loss.item()
            
            # Log to TensorBoard
            self.writer.add_scalar('Training Loss', loss.item(), 
                                 epoch * len(train_loader) + batch_idx)
            
            if batch_idx % 10 == 0:
                print(f'Epoch {epoch}, Batch {batch_idx}/{len(train_loader)}: Loss = {loss.item():.4f}')
        
        return total_loss / len(train_loader)
        
    def validate(self, epoch: int) -> float:
        """Validate model performance."""
        self.model.eval()
        val_loader = self.data_processor.get_val_loader()
        
        total_loss = 0
        with torch.no_grad():
            for batch_idx, (inputs, targets) in enumerate(val_loader):
                # Convert to embeddings
                input_embs = [self.text_encoder.encode(input_text) for input_text in inputs]
                target_embs = [self.text_encoder.encode(target_text) for target_text in targets]
                
                # Convert to tensors
                input_embs = torch.stack(input_embs).to(self.device)
                target_embs = torch.stack(target_embs).to(self.device)
                
                # Forward pass
                responses = [self.model(input_text) for input_text in inputs]
                response_embs = [self.text_encoder.encode(response) for response in responses]
                response_embs = torch.stack(response_embs).to(self.device)
                
                # Calculate loss
                loss = self.loss_fn(response_embs, target_embs, 
                                  torch.ones(len(response_embs)).to(self.device))
                total_loss += loss.item()
                
                # Log example response
                if batch_idx == 0:
                    self.writer.add_text('Example Response', 
                                       f'Input: {inputs[0]}\n'
                                       f'Target: {targets[0]}\n'
                                       f'Response: {responses[0]}',
                                       epoch)
        
        avg_loss = total_loss / len(val_loader)
        self.writer.add_scalar('Validation Loss', avg_loss, epoch)
        return avg_loss
        
    def train(self, 
             num_epochs: int = 10,
             save_path: str = 'checkpoints/si_core.pt'):
        """Train the model."""
        best_val_loss = float('inf')
        
        for epoch in range(num_epochs):
            print(f'\nEpoch {epoch + 1}/{num_epochs}')
            
            # Train
            train_loss = self.train_epoch(epoch)
            print(f'Train Loss: {train_loss:.4f}')
            
            # Validate
            val_loss = self.validate(epoch)
            print(f'Validation Loss: {val_loss:.4f}')
            
            # Update learning rate
            self.scheduler.step(val_loss)
            
            # Save best model
            if val_loss < best_val_loss:
                best_val_loss = val_loss
                torch.save(self.model.state_dict(), save_path)
                print(f'Best model saved at {save_path}')
        
        self.writer.close()
        print('\nTraining complete!')

def main():
    # Initialize model with smaller dimensions
    model = SI_Core(state_dim=256, embedding_dim=128)
    
    # Create trainer
    trainer = Trainer(
        model=model,
        train_path='data/train.json',
        val_path='data/val.json',
        batch_size=100,
        learning_rate=1e-4
    )
    
    # Train model for just one epoch
    trainer.train(num_epochs=1)

if __name__ == '__main__':
    main()

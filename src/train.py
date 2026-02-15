"""
Training utilities for sequence models.
"""

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from typing import Dict, Any, Optional
import logging
from pathlib import Path
import yaml

from utils import setup_logging, save_checkpoint, load_checkpoint


class Trainer:
    """Trainer class for sequence models."""
    
    def __init__(self, model: nn.Module, config: Dict[str, Any], 
                 train_loader: DataLoader, val_loader: Optional[DataLoader] = None):
        """
        Initialize trainer.
        
        Args:
            model: Model to train
            config: Training configuration
            train_loader: Training data loader
            val_loader: Validation data loader
        """
        self.model = model
        self.config = config
        self.train_loader = train_loader
        self.val_loader = val_loader
        
        # Setup device
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.model.to(self.device)
        
        # Setup optimizer
        self.optimizer = self._setup_optimizer()
        
        # Setup scheduler
        self.scheduler = self._setup_scheduler()
        
        # Setup loss function
        self.criterion = nn.CrossEntropyLoss(ignore_index=0)  # Ignore padding token
        
        # Setup logging
        self.logger = setup_logging(config.get('log_dir', 'outputs/logs'))
        
        # Training state
        self.current_epoch = 0
        self.best_val_loss = float('inf')
        
    def _setup_optimizer(self) -> optim.Optimizer:
        """Setup optimizer based on config."""
        optimizer_name = self.config.get('optimizer', 'adam').lower()
        learning_rate = self.config.get('learning_rate', 0.001)
        weight_decay = self.config.get('weight_decay', 0)
        
        if optimizer_name == 'adam':
            return optim.Adam(self.model.parameters(), lr=learning_rate, weight_decay=weight_decay)
        elif optimizer_name == 'adamw':
            return optim.AdamW(self.model.parameters(), lr=learning_rate, weight_decay=weight_decay)
        elif optimizer_name == 'sgd':
            return optim.SGD(self.model.parameters(), lr=learning_rate, momentum=0.9)
        else:
            raise ValueError(f"Unknown optimizer: {optimizer_name}")
    
    def _setup_scheduler(self) -> Optional[optim.lr_scheduler._LRScheduler]:
        """Setup learning rate scheduler based on config."""
        scheduler_name = self.config.get('scheduler', None)
        
        if scheduler_name is None:
            return None
        elif scheduler_name == 'step':
            return optim.lr_scheduler.StepLR(self.optimizer, step_size=10, gamma=0.1)
        elif scheduler_name == 'cosine':
            return optim.lr_scheduler.CosineAnnealingLR(self.optimizer, T_max=self.config.get('epochs', 100))
        elif scheduler_name == 'plateau':
            return optim.lr_scheduler.ReduceLROnPlateau(self.optimizer, mode='min', patience=5)
        else:
            raise ValueError(f"Unknown scheduler: {scheduler_name}")
    
    def train_epoch(self) -> float:
        """Train for one epoch."""
        self.model.train()
        total_loss = 0.0
        num_batches = len(self.train_loader)
        
        for batch_idx, batch in enumerate(self.train_loader):
            # Move batch to device
            input_ids = batch['input_ids'].to(self.device)
            attention_mask = batch['attention_mask'].to(self.device)
            labels = batch['labels'].to(self.device)
            
            # Forward pass
            self.optimizer.zero_grad()
            outputs = self.model(input_ids=input_ids, attention_mask=attention_mask)
            
            # Calculate loss
            if hasattr(outputs, 'logits'):
                logits = outputs.logits
            else:
                logits = outputs
            
            # Reshape for loss calculation
            loss = self.criterion(logits.view(-1, logits.size(-1)), labels.view(-1))
            
            # Backward pass
            loss.backward()
            self.optimizer.step()
            
            total_loss += loss.item()
            
            # Log progress
            if batch_idx % 100 == 0:
                self.logger.info(f'Epoch {self.current_epoch}, Batch {batch_idx}/{num_batches}, Loss: {loss.item():.4f}')
        
        avg_loss = total_loss / num_batches
        return avg_loss
    
    def validate(self) -> float:
        """Validate the model."""
        if self.val_loader is None:
            return float('inf')
        
        self.model.eval()
        total_loss = 0.0
        num_batches = len(self.val_loader)
        
        with torch.no_grad():
            for batch in self.val_loader:
                input_ids = batch['input_ids'].to(self.device)
                attention_mask = batch['attention_mask'].to(self.device)
                labels = batch['labels'].to(self.device)
                
                outputs = self.model(input_ids=input_ids, attention_mask=attention_mask)
                
                if hasattr(outputs, 'logits'):
                    logits = outputs.logits
                else:
                    logits = outputs
                
                loss = self.criterion(logits.view(-1, logits.size(-1)), labels.view(-1))
                total_loss += loss.item()
        
        avg_loss = total_loss / num_batches
        return avg_loss
    
    def train(self) -> Dict[str, list]:
        """Train the model."""
        epochs = self.config.get('epochs', 100)
        train_losses = []
        val_losses = []
        
        self.logger.info(f"Starting training for {epochs} epochs")
        
        for epoch in range(epochs):
            self.current_epoch = epoch
            
            # Train epoch
            train_loss = self.train_epoch()
            train_losses.append(train_loss)
            
            # Validate
            val_loss = self.validate()
            val_losses.append(val_loss)
            
            # Update scheduler
            if self.scheduler is not None:
                if isinstance(self.scheduler, optim.lr_scheduler.ReduceLROnPlateau):
                    self.scheduler.step(val_loss)
                else:
                    self.scheduler.step()
            
            # Log epoch results
            self.logger.info(f'Epoch {epoch}: Train Loss = {train_loss:.4f}, Val Loss = {val_loss:.4f}')
            
            # Save checkpoint if best model
            if val_loss < self.best_val_loss:
                self.best_val_loss = val_loss
                save_checkpoint({
                    'epoch': epoch,
                    'model_state_dict': self.model.state_dict(),
                    'optimizer_state_dict': self.optimizer.state_dict(),
                    'loss': val_loss,
                    'config': self.config
                }, f"outputs/checkpoints/best_model.pth")
        
        self.logger.info("Training completed")
        return {'train_losses': train_losses, 'val_losses': val_losses}

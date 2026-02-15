"""
Experiment script for Decoder-only LLM model.
"""

import torch
import torch.nn as nn
from torch.utils.data import random_split
from pathlib import Path
import sys
import os

# Add parent directory to path for imports
sys.path.append(str(Path(__file__).parent.parent))

from models.decoder_only_llm import DecoderOnlyLLM
from dataset import SequenceDataset, create_dataloaders
from tokenizer import SimpleTokenizer
from train import Trainer
from evaluate import Evaluator
from utils import load_config, set_seed, setup_logging


class LanguageModelTrainer:
    """Custom trainer for language modeling tasks."""
    
    def __init__(self, model: nn.Module, config: dict, train_loader, val_loader=None):
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
        
        # Setup logging
        self.logger = setup_logging()
        
        # Training state
        self.current_epoch = 0
        self.best_val_loss = float('inf')
        
    def _setup_optimizer(self):
        optimizer_name = self.config.get('optimizer', 'adamw').lower()
        learning_rate = self.config.get('learning_rate', 0.0001)
        weight_decay = self.config.get('weight_decay', 0.01)
        
        if optimizer_name == 'adamw':
            return torch.optim.AdamW(self.model.parameters(), lr=learning_rate, weight_decay=weight_decay)
        elif optimizer_name == 'adam':
            return torch.optim.Adam(self.model.parameters(), lr=learning_rate, weight_decay=weight_decay)
        else:
            raise ValueError(f"Unknown optimizer: {optimizer_name}")
    
    def _setup_scheduler(self):
        scheduler_name = self.config.get('scheduler', None)
        
        if scheduler_name == 'cosine':
            return torch.optim.lr_scheduler.CosineAnnealingLR(
                self.optimizer, T_max=self.config.get('epochs', 100)
            )
        elif scheduler_name == 'step':
            return torch.optim.lr_scheduler.StepLR(self.optimizer, step_size=10, gamma=0.1)
        else:
            return None
    
    def train_epoch(self):
        self.model.train()
        total_loss = 0.0
        num_batches = len(self.train_loader)
        
        for batch_idx, batch in enumerate(self.train_loader):
            input_ids = batch['input_ids'].to(self.device)
            attention_mask = batch['attention_mask'].to(self.device)
            
            self.optimizer.zero_grad()
            
            # Compute language modeling loss
            loss = self.model.compute_loss(input_ids, attention_mask)
            
            loss.backward()
            
            # Gradient clipping
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=1.0)
            
            self.optimizer.step()
            
            total_loss += loss.item()
            
            if batch_idx % 50 == 0:
                self.logger.info(f'Epoch {self.current_epoch}, Batch {batch_idx}/{num_batches}, Loss: {loss.item():.4f}')
        
        avg_loss = total_loss / num_batches
        return avg_loss
    
    def validate(self):
        if self.val_loader is None:
            return float('inf')
        
        self.model.eval()
        total_loss = 0.0
        num_batches = len(self.val_loader)
        
        with torch.no_grad():
            for batch in self.val_loader:
                input_ids = batch['input_ids'].to(self.device)
                attention_mask = batch['attention_mask'].to(self.device)
                
                loss = self.model.compute_loss(input_ids, attention_mask)
                total_loss += loss.item()
        
        avg_loss = total_loss / num_batches
        return avg_loss
    
    def train(self):
        epochs = self.config.get('epochs', 100)
        train_losses = []
        val_losses = []
        
        self.logger.info(f"Starting language model training for {epochs} epochs")
        
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
                self.scheduler.step()
            
            # Log epoch results
            self.logger.info(f'Epoch {epoch}: Train Loss = {train_loss:.4f}, Val Loss = {val_loss:.4f}')
            
            # Save checkpoint if best model
            if val_loss < self.best_val_loss:
                self.best_val_loss = val_loss
                torch.save({
                    'epoch': epoch,
                    'model_state_dict': self.model.state_dict(),
                    'optimizer_state_dict': self.optimizer.state_dict(),
                    'loss': val_loss,
                    'config': self.config
                }, f"outputs/checkpoints/best_llm.pth")
        
        self.logger.info("Training completed")
        return {'train_losses': train_losses, 'val_losses': val_losses}


def prepare_dummy_data(vocab_size: int = 1000, num_samples: int = 1000, seq_length: int = 128):
    """Prepare dummy data for language modeling."""
    import random
    
    # Generate random sequences for language modeling
    sequences = []
    
    for _ in range(num_samples):
        # Random sequence
        seq = [random.randint(3, vocab_size - 1) for _ in range(seq_length)]
        sequences.append(seq)
    
    return sequences


def main():
    """Main experiment function."""
    # Load configuration
    config_path = Path(__file__).parent.parent.parent / "configs" / "llm.yaml"
    config = load_config(str(config_path))
    
    # Set random seed
    set_seed(42)
    
    # Setup logging
    logger = setup_logging()
    logger.info("Starting LLM experiment")
    
    # Prepare data
    logger.info("Preparing data...")
    sequences = prepare_dummy_data(
        vocab_size=config['data']['vocab_size'],
        num_samples=2000,
        seq_length=config['data']['seq_length']
    )
    
    # Create tokenizer
    tokenizer = SimpleTokenizer(vocab_size=config['data']['vocab_size'])
    
    # Create datasets (for language modeling, targets are same as inputs)
    full_dataset = SequenceDataset(sequences, sequences, max_length=config['data']['seq_length'])
    
    # Split dataset
    train_size = int(config['data']['train_split'] * len(full_dataset))
    val_size = int(config['data']['val_split'] * len(full_dataset))
    test_size = len(full_dataset) - train_size - val_size
    
    train_dataset, val_dataset, test_dataset = random_split(
        full_dataset, [train_size, val_size, test_size]
    )
    
    # Create dataloaders
    train_loader, val_loader = create_dataloaders(
        train_dataset, val_dataset,
        batch_size=config['training']['batch_size']
    )
    
    logger.info(f"Train dataset size: {len(train_dataset)}")
    logger.info(f"Validation dataset size: {len(val_dataset)}")
    logger.info(f"Test dataset size: {len(test_dataset)}")
    
    # Create model
    logger.info("Creating model...")
    model = DecoderOnlyLLM(
        vocab_size=config['data']['vocab_size'],
        d_model=config['model']['d_model'],
        num_heads=config['model']['num_heads'],
        num_layers=config['model']['num_layers'],
        d_ff=config['model']['d_ff'],
        max_length=config['model']['max_seq_length'],
        dropout=config['model']['dropout']
    )
    
    logger.info(f"Model parameters: {model.get_num_parameters()}")
    
    # Create custom trainer
    trainer = LanguageModelTrainer(model, config['training'], train_loader, val_loader)
    
    # Train model
    logger.info("Starting training...")
    training_history = trainer.train()
    
    # Evaluate model
    logger.info("Evaluating model...")
    evaluator = Evaluator(model)
    
    # Create test dataloader
    from torch.utils.data import DataLoader
    test_loader = DataLoader(test_dataset, batch_size=config['training']['batch_size'])
    
    metrics = evaluator.evaluate(test_loader)
    logger.info(f"Test metrics: {metrics}")
    
    # Compute perplexity
    perplexity = evaluator.compute_perplexity(test_loader)
    logger.info(f"Test perplexity: {perplexity:.4f}")
    
    # Test text generation
    logger.info("Testing text generation...")
    sample_input = torch.randint(3, config['data']['vocab_size'], (1, 10))
    
    # Greedy generation
    generated_greedy = model.generate(sample_input, max_length=50, do_sample=False)
    logger.info(f"Greedy generation shape: {generated_greedy.shape}")
    
    # Sampling generation
    generated_sample = model.generate(sample_input, max_length=50, do_sample=True, temperature=0.8)
    logger.info(f"Sample generation shape: {generated_sample.shape}")
    
    # Save results
    results = {
        'config': config,
        'training_history': training_history,
        'test_metrics': metrics,
        'test_perplexity': perplexity,
        'model_parameters': model.get_num_parameters(),
        'generation_greedy': generated_greedy.tolist(),
        'generation_sample': generated_sample.tolist()
    }
    
    import json
    results_path = Path(__file__).parent.parent.parent / "outputs" / "llm_results.json"
    with open(results_path, 'w') as f:
        json.dump(results, f, indent=2, default=str)
    
    logger.info(f"Results saved to {results_path}")
    logger.info("LLM experiment completed!")


if __name__ == "__main__":
    main()

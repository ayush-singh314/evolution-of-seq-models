"""
Evaluation utilities for sequence models.
"""

import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from typing import Dict, List, Any, Optional
import numpy as np
from sklearn.metrics import accuracy_score, precision_recall_fscore_support
import logging

from utils import setup_logging


class Evaluator:
    """Evaluator class for sequence models."""
    
    def __init__(self, model: nn.Module, device: Optional[torch.device] = None):
        """
        Initialize evaluator.
        
        Args:
            model: Model to evaluate
            device: Device to run evaluation on
        """
        self.model = model
        self.device = device or torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.model.to(self.device)
        self.logger = setup_logging()
    
    def evaluate(self, data_loader: DataLoader) -> Dict[str, float]:
        """
        Evaluate model on dataset.
        
        Args:
            data_loader: Data loader for evaluation
            
        Returns:
            Dictionary of evaluation metrics
        """
        self.model.eval()
        total_loss = 0.0
        all_predictions = []
        all_labels = []
        
        criterion = nn.CrossEntropyLoss(ignore_index=0)
        
        with torch.no_grad():
            for batch in data_loader:
                input_ids = batch['input_ids'].to(self.device)
                attention_mask = batch['attention_mask'].to(self.device)
                labels = batch['labels'].to(self.device)
                
                # Forward pass
                outputs = self.model(input_ids=input_ids, attention_mask=attention_mask)
                
                if hasattr(outputs, 'logits'):
                    logits = outputs.logits
                else:
                    logits = outputs
                
                # Calculate loss
                loss = criterion(logits.view(-1, logits.size(-1)), labels.view(-1))
                total_loss += loss.item()
                
                # Get predictions
                predictions = torch.argmax(logits, dim=-1)
                
                # Filter out padding tokens
                mask = labels != 0
                predictions = predictions[mask]
                labels_filtered = labels[mask]
                
                all_predictions.extend(predictions.cpu().numpy())
                all_labels.extend(labels_filtered.cpu().numpy())
        
        # Calculate metrics
        avg_loss = total_loss / len(data_loader)
        accuracy = accuracy_score(all_labels, all_predictions)
        precision, recall, f1, _ = precision_recall_fscore_support(
            all_labels, all_predictions, average='weighted', zero_division=0
        )
        
        metrics = {
            'loss': avg_loss,
            'accuracy': accuracy,
            'precision': precision,
            'recall': recall,
            'f1_score': f1
        }
        
        self.logger.info(f"Evaluation Results: {metrics}")
        return metrics
    
    def generate_text(self, tokenizer, prompt: str, max_length: int = 100, 
                     temperature: float = 1.0, top_k: int = 50) -> str:
        """
        Generate text using the model.
        
        Args:
            tokenizer: Tokenizer instance
            prompt: Input prompt
            max_length: Maximum generation length
            temperature: Sampling temperature
            top_k: Top-k sampling parameter
            
        Returns:
            Generated text
        """
        self.model.eval()
        
        # Encode prompt
        input_ids = tokenizer.encode(prompt, add_special_tokens=True)
        input_ids = torch.tensor(input_ids, dtype=torch.long).unsqueeze(0).to(self.device)
        
        generated_ids = input_ids.clone()
        
        with torch.no_grad():
            for _ in range(max_length):
                # Get model predictions
                outputs = self.model(input_ids=generated_ids)
                if hasattr(outputs, 'logits'):
                    logits = outputs.logits
                else:
                    logits = outputs
                
                # Get next token logits
                next_token_logits = logits[:, -1, :] / temperature
                
                # Apply top-k filtering
                if top_k > 0:
                    top_k_logits, top_k_indices = torch.topk(next_token_logits, top_k)
                    next_token_logits = torch.full_like(next_token_logits, float('-inf'))
                    next_token_logits.scatter_(1, top_k_indices, top_k_logits)
                
                # Sample next token
                probs = torch.nn.functional.softmax(next_token_logits, dim=-1)
                next_token = torch.multinomial(probs, num_samples=1)
                
                # Append to generated sequence
                generated_ids = torch.cat([generated_ids, next_token], dim=1)
                
                # Stop if end token is generated
                if next_token.item() == tokenizer.token_to_id.get('<EOS>', -1):
                    break
        
        # Decode generated text
        generated_text = tokenizer.decode(generated_ids[0].cpu().numpy(), skip_special_tokens=True)
        return generated_text
    
    def compute_perplexity(self, data_loader: DataLoader) -> float:
        """
        Compute perplexity on dataset.
        
        Args:
            data_loader: Data loader for evaluation
            
        Returns:
            Perplexity score
        """
        self.model.eval()
        total_loss = 0.0
        total_tokens = 0
        
        criterion = nn.CrossEntropyLoss(ignore_index=0, reduction='sum')
        
        with torch.no_grad():
            for batch in data_loader:
                input_ids = batch['input_ids'].to(self.device)
                attention_mask = batch['attention_mask'].to(self.device)
                labels = batch['labels'].to(self.device)
                
                outputs = self.model(input_ids=input_ids, attention_mask=attention_mask)
                
                if hasattr(outputs, 'logits'):
                    logits = outputs.logits
                else:
                    logits = outputs
                
                loss = criterion(logits.view(-1, logits.size(-1)), labels.view(-1))
                total_loss += loss.item()
                
                # Count non-padding tokens
                total_tokens += (labels != 0).sum().item()
        
        avg_loss = total_loss / total_tokens
        perplexity = torch.exp(torch.tensor(avg_loss)).item()
        
        self.logger.info(f"Perplexity: {perplexity:.4f}")
        return perplexity

"""
Experiment script for Attention-based RNN model.
"""

import torch
import torch.nn as nn
from torch.utils.data import random_split
from pathlib import Path
import sys
import os

# Add parent directory to path for imports
sys.path.append(str(Path(__file__).parent.parent))

from models.rnn_encoder_decoder import RNNEncoder
from models.attention import AttentionRNNDecoder, BahdanauAttention
from dataset import SequenceDataset, create_dataloaders
from tokenizer import SimpleTokenizer
from train import Trainer
from evaluate import Evaluator
from utils import load_config, set_seed, setup_logging


class AttentionRNNEncoderDecoder(nn.Module):
    """RNN Encoder-Decoder with Attention mechanism."""
    
    def __init__(self, vocab_size: int, embed_dim: int = 256, hidden_size: int = 256,
                 num_layers: int = 2, dropout: float = 0.1, bidirectional: bool = False,
                 attention_type: str = 'bahdanau'):
        super(AttentionRNNEncoderDecoder, self).__init__()
        self.encoder = RNNEncoder(
            vocab_size, embed_dim, hidden_size, num_layers, dropout, bidirectional
        )
        
        # Adjust decoder hidden size if encoder is bidirectional
        decoder_hidden_size = hidden_size * 2 if bidirectional else hidden_size
        self.decoder = AttentionRNNDecoder(
            vocab_size, embed_dim, decoder_hidden_size, num_layers, dropout, attention_type
        )
        
        self.vocab_size = vocab_size
        self.hidden_size = hidden_size
        self.bidirectional = bidirectional
        
    def forward(self, input_ids: torch.Tensor, target_ids: Optional[torch.Tensor] = None,
                attention_mask: Optional[torch.Tensor] = None,
                teacher_forcing_ratio: float = 0.5) -> torch.Tensor:
        """
        Forward pass through attention encoder-decoder.
        
        Args:
            input_ids: Input token IDs [batch_size, src_seq_len]
            target_ids: Target token IDs [batch_size, tgt_seq_len]
            attention_mask: Attention mask [batch_size, src_seq_len]
            teacher_forcing_ratio: Probability of using teacher forcing
            
        Returns:
            logits: Output logits [batch_size, tgt_seq_len, vocab_size]
        """
        batch_size, src_len = input_ids.shape
        
        # Encode input sequence
        encoder_outputs, encoder_hidden = self.encoder(input_ids, attention_mask)
        
        # Initialize decoder input with SOS token
        if target_ids is not None:
            tgt_len = target_ids.shape[1]
            # Use target_ids for training
            decoder_input = target_ids[:, :-1]  # Remove last token
            logits, _ = self.decoder(decoder_input, encoder_hidden, encoder_outputs, attention_mask)
        else:
            # Generate sequence during inference
            decoder_input = torch.full((batch_size, 1), 1, dtype=torch.long, device=input_ids.device)  # SOS token
            generated_logits = []
            
            for _ in range(src_len * 2):  # Generate up to 2x source length
                logits, encoder_hidden = self.decoder(decoder_input, encoder_hidden, encoder_outputs, attention_mask)
                generated_logits.append(logits)
                
                # Get next token
                next_token = torch.argmax(logits[:, -1:, :], dim=-1)
                decoder_input = next_token
                
                # Stop if EOS token is generated
                if (next_token == 2).all():  # EOS token
                    break
            
            logits = torch.cat(generated_logits, dim=1)
        
        return logits


def prepare_dummy_data(vocab_size: int = 1000, num_samples: int = 1000, seq_length: int = 50):
    """Prepare dummy data for testing."""
    import random
    
    # Generate random sequences
    sequences = []
    targets = []
    
    for _ in range(num_samples):
        # Random input sequence
        seq = [random.randint(3, vocab_size - 1) for _ in range(seq_length)]
        # Target sequence (shifted version of input)
        target = seq[1:] + [2]  # Add EOS token
        
        sequences.append(seq)
        targets.append(target)
    
    return sequences, targets


def main():
    """Main experiment function."""
    # Load configuration
    config_path = Path(__file__).parent.parent.parent / "configs" / "attention.yaml"
    config = load_config(str(config_path))
    
    # Set random seed
    set_seed(42)
    
    # Setup logging
    logger = setup_logging()
    logger.info("Starting Attention RNN experiment")
    
    # Prepare data
    logger.info("Preparing data...")
    sequences, targets = prepare_dummy_data(
        vocab_size=config['data']['vocab_size'],
        num_samples=2000,
        seq_length=config['data']['seq_length']
    )
    
    # Create tokenizer (for demonstration, using simple tokenizer)
    tokenizer = SimpleTokenizer(vocab_size=config['data']['vocab_size'])
    
    # Create datasets
    full_dataset = SequenceDataset(sequences, targets, max_length=config['data']['seq_length'])
    
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
    model = AttentionRNNEncoderDecoder(
        vocab_size=config['data']['vocab_size'],
        embed_dim=config['model']['hidden_size'],
        hidden_size=config['model']['hidden_size'],
        num_layers=config['model']['num_layers'],
        dropout=config['model']['dropout'],
        bidirectional=False  # Can be configured
    )
    
    logger.info(f"Model parameters: {sum(p.numel() for p in model.parameters() if p.requires_grad)}")
    
    # Create trainer
    trainer = Trainer(model, config['training'], train_loader, val_loader)
    
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
    
    # Save results
    results = {
        'config': config,
        'training_history': training_history,
        'test_metrics': metrics,
        'model_parameters': sum(p.numel() for p in model.parameters() if p.requires_grad)
    }
    
    import json
    results_path = Path(__file__).parent.parent.parent / "outputs" / "attention_results.json"
    with open(results_path, 'w') as f:
        json.dump(results, f, indent=2, default=str)
    
    logger.info(f"Results saved to {results_path}")
    logger.info("Attention RNN experiment completed!")


if __name__ == "__main__":
    main()

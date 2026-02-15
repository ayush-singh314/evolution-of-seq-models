"""
Decoder-only Language Model implementation.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional, Tuple
import math

from .attention import MultiHeadAttention


class PositionalEncoding(nn.Module):
    """Positional encoding for Transformer."""
    
    def __init__(self, d_model: int, max_length: int = 2048, dropout: float = 0.1):
        super(PositionalEncoding, self).__init__()
        self.dropout = nn.Dropout(dropout)
        
        # Create positional encoding matrix
        pe = torch.zeros(max_length, d_model)
        position = torch.arange(0, max_length, dtype=torch.float).unsqueeze(1)
        
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * 
                           (-math.log(10000.0) / d_model))
        
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0).transpose(0, 1)
        
        self.register_buffer('pe', pe)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Add positional encoding to input.
        
        Args:
            x: Input tensor [seq_len, batch_size, d_model]
            
        Returns:
            x: Input with positional encoding added
        """
        x = x + self.pe[:x.size(0), :]
        return self.dropout(x)


class TransformerBlock(nn.Module):
    """Transformer block for decoder-only model."""
    
    def __init__(self, d_model: int, num_heads: int, d_ff: int, dropout: float = 0.1):
        super(TransformerBlock, self).__init__()
        self.attention = MultiHeadAttention(d_model, num_heads, dropout)
        self.feed_forward = nn.Sequential(
            nn.Linear(d_model, d_ff),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(d_ff, d_model)
        )
        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        self.dropout = nn.Dropout(dropout)
        
    def forward(self, x: torch.Tensor, mask: Optional[torch.Tensor] = None) -> torch.Tensor:
        """
        Forward pass through transformer block.
        
        Args:
            x: Input tensor [batch_size, seq_len, d_model]
            mask: Attention mask [batch_size, 1, seq_len, seq_len]
            
        Returns:
            x: Output tensor [batch_size, seq_len, d_model]
        """
        # Self-attention with residual connection
        attn_output = self.attention(x, x, x, mask)
        x = self.norm1(x + self.dropout(attn_output))
        
        # Feed-forward with residual connection
        ff_output = self.feed_forward(x)
        x = self.norm2(x + self.dropout(ff_output))
        
        return x


class DecoderOnlyLLM(nn.Module):
    """Decoder-only Language Model (GPT-style)."""
    
    def __init__(self, vocab_size: int, d_model: int = 768, num_heads: int = 12,
                 num_layers: int = 12, d_ff: int = 3072, max_length: int = 2048,
                 dropout: float = 0.1, use_rotary_embeddings: bool = False):
        super(DecoderOnlyLLM, self).__init__()
        self.vocab_size = vocab_size
        self.d_model = d_model
        self.max_length = max_length
        
        # Token embedding
        self.token_embedding = nn.Embedding(vocab_size, d_model, padding_idx=0)
        
        # Positional encoding
        if use_rotary_embeddings:
            # For simplicity, we'll use standard positional encoding here
            # In practice, you might want to implement rotary embeddings
            self.pos_encoding = PositionalEncoding(d_model, max_length, dropout)
        else:
            self.pos_encoding = PositionalEncoding(d_model, max_length, dropout)
        
        # Transformer blocks
        self.blocks = nn.ModuleList([
            TransformerBlock(d_model, num_heads, d_ff, dropout)
            for _ in range(num_layers)
        ])
        
        # Final layer norm
        self.ln_f = nn.LayerNorm(d_model)
        
        # Output projection
        self.output_proj = nn.Linear(d_model, vocab_size, bias=False)
        
        # Tie weights (optional)
        self.output_proj.weight = self.token_embedding.weight
        
        self.dropout = nn.Dropout(dropout)
        
    def forward(self, input_ids: torch.Tensor, attention_mask: Optional[torch.Tensor] = None) -> torch.Tensor:
        """
        Forward pass through the model.
        
        Args:
            input_ids: Input token IDs [batch_size, seq_len]
            attention_mask: Attention mask [batch_size, seq_len]
            
        Returns:
            logits: Output logits [batch_size, seq_len, vocab_size]
        """
        batch_size, seq_len = input_ids.shape
        
        # Token embeddings
        token_embeds = self.token_embedding(input_ids)  # [batch_size, seq_len, d_model]
        
        # Scale embeddings
        token_embeds = token_embeds * math.sqrt(self.d_model)
        
        # Add positional encoding
        x = self.pos_encoding(token_embeds.transpose(0, 1)).transpose(0, 1)
        x = self.dropout(x)
        
        # Create causal mask
        causal_mask = self._create_causal_mask(seq_len, input_ids.device)
        
        # Combine with attention mask if provided
        if attention_mask is not None:
            # Convert attention mask to float and expand
            attention_mask = attention_mask.unsqueeze(1).unsqueeze(2)  # [batch_size, 1, 1, seq_len]
            attention_mask = attention_mask.expand(-1, 1, seq_len, -1)  # [batch_size, 1, seq_len, seq_len]
            mask = causal_mask & attention_mask
        else:
            mask = causal_mask
        
        # Pass through transformer blocks
        for block in self.blocks:
            x = block(x, mask)
        
        # Final layer norm
        x = self.ln_f(x)
        
        # Output projection
        logits = self.output_proj(x)
        
        return logits
    
    def _create_causal_mask(self, seq_len: int, device: torch.device) -> torch.Tensor:
        """Create causal mask for self-attention."""
        mask = torch.triu(torch.ones(seq_len, seq_len, device=device), diagonal=1)
        mask = mask.masked_fill(mask == 1, 0)
        return mask.unsqueeze(0)  # [1, seq_len, seq_len]
    
    def generate(self, input_ids: torch.Tensor, max_length: int = 100,
                temperature: float = 1.0, top_k: int = 50, top_p: float = 0.95,
                do_sample: bool = True, attention_mask: Optional[torch.Tensor] = None) -> torch.Tensor:
        """
        Generate text using various sampling strategies.
        
        Args:
            input_ids: Input token IDs [batch_size, seq_len]
            max_length: Maximum generation length
            temperature: Sampling temperature
            top_k: Top-k sampling parameter
            top_p: Top-p (nucleus) sampling parameter
            do_sample: Whether to use sampling or greedy decoding
            attention_mask: Attention mask [batch_size, seq_len]
            
        Returns:
            generated_ids: Generated token IDs [batch_size, gen_len]
        """
        self.eval()
        with torch.no_grad():
            batch_size = input_ids.shape[0]
            
            # Start with input_ids
            generated_ids = input_ids.clone()
            
            for _ in range(max_length):
                # Get model predictions
                logits = self.forward(generated_ids, attention_mask)
                
                # Get next token logits
                next_token_logits = logits[:, -1, :] / temperature
                
                # Apply top-k filtering
                if top_k > 0:
                    top_k_logits, top_k_indices = torch.topk(next_token_logits, top_k)
                    next_token_logits = torch.full_like(next_token_logits, float('-inf'))
                    next_token_logits.scatter_(1, top_k_indices, top_k_logits)
                
                # Apply top-p (nucleus) filtering
                if top_p < 1.0:
                    sorted_logits, sorted_indices = torch.sort(next_token_logits, descending=True)
                    cumulative_probs = torch.cumsum(F.softmax(sorted_logits, dim=-1), dim=-1)
                    
                    # Remove tokens with cumulative probability above the threshold
                    sorted_indices_to_remove = cumulative_probs > top_p
                    sorted_indices_to_remove[..., 1:] = sorted_indices_to_remove[..., :-1].clone()
                    sorted_indices_to_remove[..., 0] = 0
                    
                    indices_to_remove = sorted_indices_to_remove.scatter(1, sorted_indices, sorted_indices_to_remove)
                    next_token_logits = next_token_logits.masked_fill(indices_to_remove, float('-inf'))
                
                # Sample next token
                if do_sample:
                    probs = F.softmax(next_token_logits, dim=-1)
                    next_token = torch.multinomial(probs, num_samples=1)
                else:
                    next_token = torch.argmax(next_token_logits, dim=-1, keepdim=True)
                
                # Append to generated sequence
                generated_ids = torch.cat([generated_ids, next_token], dim=1)
                
                # Update attention mask if provided
                if attention_mask is not None:
                    attention_mask = torch.cat([
                        attention_mask,
                        torch.ones((batch_size, 1), device=attention_mask.device)
                    ], dim=1)
                
                # Stop if EOS token is generated for all sequences
                if (next_token == 2).all():  # EOS token
                    break
                
                # Stop if max sequence length is reached
                if generated_ids.shape[1] >= self.max_length:
                    break
            
            return generated_ids
    
    def compute_loss(self, input_ids: torch.Tensor, attention_mask: Optional[torch.Tensor] = None,
                    labels: Optional[torch.Tensor] = None) -> torch.Tensor:
        """
        Compute language modeling loss.
        
        Args:
            input_ids: Input token IDs [batch_size, seq_len]
            attention_mask: Attention mask [batch_size, seq_len]
            labels: Target labels [batch_size, seq_len]
            
        Returns:
            loss: Cross-entropy loss
        """
        if labels is None:
            labels = input_ids
        
        # Get model predictions
        logits = self.forward(input_ids, attention_mask)
        
        # Shift labels for next token prediction
        shift_logits = logits[..., :-1, :].contiguous()
        shift_labels = labels[..., 1:].contiguous()
        
        # Flatten for loss computation
        shift_logits = shift_logits.view(-1, shift_logits.size(-1))
        shift_labels = shift_labels.view(-1)
        
        # Compute cross-entropy loss
        loss_fct = nn.CrossEntropyLoss(ignore_index=0)  # Ignore padding token
        loss = loss_fct(shift_logits, shift_labels)
        
        return loss
    
    def get_num_parameters(self) -> int:
        """Get the number of parameters in the model."""
        return sum(p.numel() for p in self.parameters() if p.requires_grad)

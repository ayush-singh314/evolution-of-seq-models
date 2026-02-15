"""
Tokenizer utilities for sequence models.
"""

from typing import List, Dict, Tuple, Optional
import re
from collections import Counter
import json


class SimpleTokenizer:
    """Simple word-level tokenizer."""
    
    def __init__(self, vocab_size: int = 10000, min_freq: int = 2):
        """
        Initialize tokenizer.
        
        Args:
            vocab_size: Maximum vocabulary size
            min_freq: Minimum frequency for a token to be included
        """
        self.vocab_size = vocab_size
        self.min_freq = min_freq
        self.token_to_id = {}
        self.id_to_token = {}
        self.pad_token = "<PAD>"
        self.unk_token = "<UNK>"
        self.sos_token = "<SOS>"
        self.eos_token = "<EOS>"
    
    def build_vocab(self, texts: List[str]) -> None:
        """Build vocabulary from list of texts."""
        # Tokenize all texts
        all_tokens = []
        for text in texts:
            tokens = self._tokenize_text(text)
            all_tokens.extend(tokens)
        
        # Count token frequencies
        token_counts = Counter(all_tokens)
        
        # Create vocabulary with special tokens
        vocab = [self.pad_token, self.unk_token, self.sos_token, self.eos_token]
        
        # Add most frequent tokens
        for token, count in token_counts.most_common():
            if count >= self.min_freq and len(vocab) < self.vocab_size:
                vocab.append(token)
            else:
                break
        
        # Create mappings
        self.token_to_id = {token: i for i, token in enumerate(vocab)}
        self.id_to_token = {i: token for i, token in enumerate(vocab)}
    
    def _tokenize_text(self, text: str) -> List[str]:
        """Simple tokenization by splitting on whitespace and punctuation."""
        # Basic preprocessing
        text = text.lower()
        text = re.sub(r'[^\w\s]', ' ', text)
        
        # Split on whitespace
        tokens = text.split()
        return tokens
    
    def encode(self, text: str, add_special_tokens: bool = True) -> List[int]:
        """Encode text to token IDs."""
        tokens = self._tokenize_text(text)
        token_ids = [self.token_to_id.get(token, self.token_to_id[self.unk_token]) 
                    for token in tokens]
        
        if add_special_tokens:
            token_ids = [self.token_to_id[self.sos_token]] + token_ids + [self.token_to_id[self.eos_token]]
        
        return token_ids
    
    def decode(self, token_ids: List[int], skip_special_tokens: bool = True) -> str:
        """Decode token IDs to text."""
        tokens = [self.id_to_token.get(token_id, self.unk_token) for token_id in token_ids]
        
        if skip_special_tokens:
            tokens = [token for token in tokens if token not in 
                     [self.pad_token, self.unk_token, self.sos_token, self.eos_token]]
        
        return ' '.join(tokens)
    
    def save_vocab(self, path: str) -> None:
        """Save vocabulary to file."""
        vocab_data = {
            'token_to_id': self.token_to_id,
            'id_to_token': self.id_to_token,
            'vocab_size': self.vocab_size,
            'min_freq': self.min_freq
        }
        with open(path, 'w') as f:
            json.dump(vocab_data, f, indent=2)
    
    def load_vocab(self, path: str) -> None:
        """Load vocabulary from file."""
        with open(path, 'r') as f:
            vocab_data = json.load(f)
        
        self.token_to_id = vocab_data['token_to_id']
        self.id_to_token = {int(k): v for k, v in vocab_data['id_to_token'].items()}
        self.vocab_size = vocab_data['vocab_size']
        self.min_freq = vocab_data['min_freq']

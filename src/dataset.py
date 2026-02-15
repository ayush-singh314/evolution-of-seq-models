"""
dataset.py

This module handles:
1. Loading the Tiny Shakespeare dataset
2. Building a character-level vocabulary
3. Encoding and decoding text
4. Creating train/validation splits
5. Generating training batches for language modeling

We use character-level modeling because:
- Small dataset
- No tokenizer complexity
- Good for architectural comparison (RNN vs Transformer)
"""

import torch
import os


class TinyShakespeareDataset:
    def __init__(self, file_path, train_split=0.9, device="cpu"):
        """
        Args:
            file_path (str): Path to tiny_shakespeare.txt
            train_split (float): Fraction of data used for training
            device (str): 'cpu' or 'cuda'
        """

        self.file_path = file_path
        self.train_split = train_split
        self.device = device

        # Step 1: Load raw text
        self.text = self._load_text()

        # Step 2: Build vocabulary (character-level)
        self._build_vocab()

        # Step 3: Encode entire dataset into integers
        self.data = self._encode_text(self.text)

        # Step 4: Split into train and validation
        self._split_data()

    # -----------------------------------------------------------
    # Step 1: Load dataset
    # -----------------------------------------------------------
    def _load_text(self):
        """
        Reads the entire text file into memory.
        Language modeling needs full sequence continuity.
        """
        if not os.path.exists(self.file_path):
            raise FileNotFoundError(f"{self.file_path} not found.")

        with open(self.file_path, "r", encoding="utf-8") as f:
            text = f.read()

        return text

    # -----------------------------------------------------------
    # Step 2: Build character vocabulary
    # -----------------------------------------------------------
    def _build_vocab(self):
        """
        Build character-level vocabulary.

        Why character-level?
        - Simpler than word-level
        - Small vocab (~65 chars)
        - No OOV tokens
        """
        ## why not word level encoding?=> no oov  problem
        # Get sorted unique characters => sorting is done to ensure reproducablity => same char to index mapping for each text file
        chars = sorted(list(set(self.text)))

        self.vocab_size = len(chars)#65 for this case

        # String-to-index mapping (character → integer)
        self.stoi = {ch: i for i, ch in enumerate(chars)}

        # Index-to-string mapping (integer → character)
        self.itos = {i: ch for i, ch in enumerate(chars)}#for decoding model back to text

    # -----------------------------------------------------------
    # Step 3: Encode text
    # -----------------------------------------------------------
    def _encode_text(self, text):
        """
        Convert full text into tensor of integers.

        Each character is replaced by its vocabulary index.
        """

        encoded = [self.stoi[c] for c in text]

        # Convert to PyTorch tensor
        return torch.tensor(encoded, dtype=torch.long)

    # -----------------------------------------------------------
    # Decode helper (for generating readable text later)
    # -----------------------------------------------------------
    def decode(self, indices):
        """
        Convert list/tensor of indices back into string.
        Useful during generation phase.
        """

        if isinstance(indices, torch.Tensor):
            indices = indices.tolist()

        return "".join([self.itos[i] for i in indices])

    # -----------------------------------------------------------
    # Step 4: Train/Validation split
    # -----------------------------------------------------------
    def _split_data(self):
        """
        Split dataset into train and validation.

        We keep sequential order intact because:
        - Language modeling depends on continuity.
        """

        n = int(self.train_split * len(self.data))

        self.train_data = self.data[:n]
        self.val_data = self.data[n:]

    # -----------------------------------------------------------
    # Step 5: Batch sampling
    # -----------------------------------------------------------
    def get_batch(self, split, block_size, batch_size):
        """
        Generate a batch of input-target pairs.

        Args:
            split (str): 'train' or 'val'
            block_size (int): context length (sequence length)
            batch_size (int): number of sequences per batch

        Returns:
            x: input tensor (batch_size, block_size)
            y: target tensor (batch_size, block_size)

        Language modeling objective:
            Predict next token.
        So:
            y is x shifted by one position.
        """

        data = self.train_data if split == "train" else self.val_data

        # Random starting indices
        # Ensure we don't go out of bounds
        ix = torch.randint(
            low=0,
            high=len(data) - block_size - 1,
            size=(batch_size,)
        )

        # Input sequences
        x = torch.stack([data[i:i + block_size] for i in ix])

        # Target sequences (shifted by 1)
        y = torch.stack([data[i + 1:i + block_size + 1] for i in ix])

        # Move to device (GPU if available)
        x = x.to(self.device)
        y = y.to(self.device)

        return x, y

    # -----------------------------------------------------------
    # Utility: Print dataset statistics
    # -----------------------------------------------------------
    def summary(self):
        """
        Print useful dataset stats.
        Helpful for debugging and reporting.
        """

        print("---- Dataset Summary ----")
        print(f"Total characters: {len(self.text)}")
        print(f"Vocabulary size: {self.vocab_size}")
        print(f"Train size: {len(self.train_data)}")
        print(f"Validation size: {len(self.val_data)}")
        print("-------------------------")



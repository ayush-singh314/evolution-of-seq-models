import torch
import torch.nn as nn
import torch.optim as optim
import os
import math
import json

from models.rnn_language_model import RNNLanguageModel
from dataset import TinyShakespeareDataset


# ---------------------------------------------------------
# Directories
# ---------------------------------------------------------
CHECKPOINT_DIR = "outputs/checkpoints"
os.makedirs(CHECKPOINT_DIR, exist_ok=True)
os.makedirs("../outputs", exist_ok=True)


# ---------------------------------------------------------
# Hyperparameters
# ---------------------------------------------------------
embedding_dim = 128
hidden_dim = 256
block_size = 128
batch_size = 32
epochs = 5
learning_rate = 3e-4
steps_per_epoch = 500
val_steps = 200

device = "cuda" if torch.cuda.is_available() else "cpu"


# ---------------------------------------------------------
# Dataset
# ---------------------------------------------------------
dataset = TinyShakespeareDataset(
    file_path="../data/raw/tiny_shakespeare.txt",
    device=device
)

dataset.summary()


# ---------------------------------------------------------
# Model
# ---------------------------------------------------------
model = RNNLanguageModel(
    vocab_size=dataset.vocab_size,
    embedding_dim=embedding_dim,
    hidden_dim=hidden_dim
).to(device)

criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=learning_rate)


# ---------------------------------------------------------
# Metric storage (for comparative study later)
# ---------------------------------------------------------
train_losses = []
val_losses = []
perplexities = []


# ---------------------------------------------------------
# Training Loop
# ---------------------------------------------------------
for epoch in range(epochs):

    # -------------------
    # TRAINING PHASE
    # -------------------
    model.train()
    total_train_loss = 0

    for step in range(steps_per_epoch):

        xb, yb = dataset.get_batch("train", block_size, batch_size)

        logits, _ = model(xb)

        # Reshape for CrossEntropy
        B, T, C = logits.shape
        logits = logits.view(B * T, C)
        yb = yb.view(B * T)

        loss = criterion(logits, yb)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        total_train_loss += loss.item()

    avg_train_loss = total_train_loss / steps_per_epoch
    train_losses.append(avg_train_loss)

    # -------------------
    # VALIDATION PHASE
    # -------------------
    model.eval()
    total_val_loss = 0

    with torch.no_grad():
        for _ in range(val_steps):

            xb, yb = dataset.get_batch("val", block_size, batch_size)

            logits, _ = model(xb)

            B, T, C = logits.shape
            logits = logits.view(B * T, C)
            yb = yb.view(B * T)

            loss = criterion(logits, yb)
            total_val_loss += loss.item()

    avg_val_loss = total_val_loss / val_steps
    val_losses.append(avg_val_loss)

    # -------------------
    # Perplexity
    # -------------------
    perplexity = math.exp(avg_val_loss)
    perplexities.append(perplexity)

    print(
        f"Epoch {epoch+1} | "
        f"Train Loss: {avg_train_loss:.4f} | "
        f"Val Loss: {avg_val_loss:.4f} | "
        f"Perplexity: {perplexity:.2f}"
    )

    # -------------------
    # Save Checkpoint
    # -------------------
    checkpoint = {
        "epoch": epoch + 1,
        "model_state_dict": model.state_dict(),
        "optimizer_state_dict": optimizer.state_dict(),
        "train_loss": avg_train_loss,
        "val_loss": avg_val_loss,
        "perplexity": perplexity,
        "vocab_size": dataset.vocab_size,
        "embedding_dim": embedding_dim,
        "hidden_dim": hidden_dim,
        "block_size": block_size
    }

    torch.save(
        checkpoint,
        os.path.join(CHECKPOINT_DIR, f"rnn_epoch_{epoch+1}.pt")
    )


# ---------------------------------------------------------
# Save metrics for plotting and comparison
# ---------------------------------------------------------
metrics = {
    "train_loss": train_losses,
    "val_loss": val_losses,
    "perplexity": perplexities
}

with open("outputs/rnn_metrics.json", "w") as f:
    json.dump(metrics, f, indent=4)

print("Training complete. Metrics saved.")

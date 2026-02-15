import json
import matplotlib.pyplot as plt
import os


# ---------------------------------------------------------
# Load Attention Metrics
# ---------------------------------------------------------
attention_path = "../outputs/attention_metrics.json"

with open(attention_path, "r") as f:
    attention_metrics = json.load(f)

epochs = range(1, len(attention_metrics["train_loss"]) + 1)


# ---------------------------------------------------------
# Plot 1: Training vs Validation Loss
# ---------------------------------------------------------
plt.figure(figsize=(10, 5))

plt.plot(epochs, attention_metrics["train_loss"], marker='o', label="Train Loss")
plt.plot(epochs, attention_metrics["val_loss"], marker='s', label="Validation Loss")

plt.xlabel("Epoch")
plt.ylabel("Loss")
plt.title("RNN + Attention: Training vs Validation Loss")
plt.legend()
plt.grid(True)

plt.savefig("../outputs/attention_loss_curve.png")
plt.show()


# ---------------------------------------------------------
# Plot 2: Perplexity
# ---------------------------------------------------------
plt.figure(figsize=(10, 5))

plt.plot(epochs, attention_metrics["perplexity"], marker='o')

plt.xlabel("Epoch")
plt.ylabel("Perplexity")
plt.title("RNN + Attention: Perplexity")
plt.grid(True)

plt.savefig("../outputs/attention_perplexity_curve.png")
plt.show()

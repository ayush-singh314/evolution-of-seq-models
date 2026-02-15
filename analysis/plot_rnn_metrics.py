import json
import matplotlib.pyplot as plt

with open("../outputs/rnn_metrics.json", "r") as f:
    metrics = json.load(f)

epochs = range(1, len(metrics["train_loss"]) + 1)

plt.figure(figsize=(10,5))
plt.plot(epochs, metrics["train_loss"], label="Train Loss")
plt.plot(epochs, metrics["val_loss"], label="Validation Loss")
plt.xlabel("Epoch")
plt.ylabel("Loss")
plt.legend()
plt.title("RNN Training vs Validation Loss")
plt.savefig("outputs/rnn_loss_curve.png")
plt.show()

plt.figure(figsize=(10,5))
plt.plot(epochs, metrics["perplexity"], label="Perplexity")
plt.xlabel("Epoch")
plt.ylabel("Perplexity")
plt.title("RNN Perplexity")
plt.savefig("outputs/rnn_perplexity.png")
plt.show()

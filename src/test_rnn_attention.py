import torch
from models.rnn_attention_language_model import RNNAttentionLanguageModel
from dataset import TinyShakespeareDataset


device = "cuda" if torch.cuda.is_available() else "cpu"

# ---------------------------------------------------------
# Load dataset
# ---------------------------------------------------------
dataset = TinyShakespeareDataset(
    file_path="../data/raw/tiny_shakespeare.txt",
    device=device
)

# ---------------------------------------------------------
# Load checkpoint
# ---------------------------------------------------------
checkpoint = torch.load(
    "../outputs/checkpoints_attention/attention_epoch_5.pt",
    map_location=device
)

model = RNNAttentionLanguageModel(
    vocab_size=checkpoint["vocab_size"],
    embedding_dim=checkpoint["embedding_dim"],
    hidden_dim=checkpoint["hidden_dim"]
).to(device)

model.load_state_dict(checkpoint["model_state_dict"])
model.eval()


# ---------------------------------------------------------
# Generation function
# ---------------------------------------------------------
def generate(model, start_token, max_new_tokens):

    x = start_token

    for _ in range(max_new_tokens):

        logits = model(x)

        logits = logits[:, -1, :]
        probs = torch.softmax(logits, dim=-1)

        next_token = torch.multinomial(probs, num_samples=1)

        x = torch.cat((x, next_token), dim=1)

    return x


# ---------------------------------------------------------
# Generate text
# ---------------------------------------------------------
context = torch.zeros((1, 1), dtype=torch.long).to(device)

generated = generate(model, context, max_new_tokens=300)

print(dataset.decode(generated[0]))

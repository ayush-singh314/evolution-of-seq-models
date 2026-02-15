import torch
from models.rnn_language_model import RNNLanguageModel
from dataset import TinyShakespeareDataset

device = "cuda" if torch.cuda.is_available() else "cpu"

dataset = TinyShakespeareDataset(
    file_path="../data/raw/tiny_shakespeare.txt",
    device=device
)

checkpoint = torch.load("outputs/checkpoints/rnn_epoch_5.pt", map_location=device)

model = RNNLanguageModel(
    vocab_size=checkpoint["vocab_size"],
    embedding_dim=checkpoint["embedding_dim"],
    hidden_dim=checkpoint["hidden_dim"]
).to(device)

model.load_state_dict(checkpoint["model_state_dict"])
model.eval()
context = torch.zeros((1, 1), dtype=torch.long).to(device)
generated = model.generate(context, max_new_tokens=300)

print(dataset.decode(generated[0]))


# “The LSTM model achieves low perplexity, indicating strong local modeling capability. However, qualitative generation reveals limited long-term coherence due to sequential memory compression.”
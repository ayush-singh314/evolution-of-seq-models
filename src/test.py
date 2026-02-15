from dataset import TinyShakespeareDataset

dataset = TinyShakespeareDataset(
    file_path="../data/raw/tiny_shakespeare.txt",
    device="cpu"
)

dataset.summary()

xb, yb = dataset.get_batch("train", block_size=8, batch_size=4)

print("Input:\n", xb)
print("Target:\n", yb)
#since it is autoregressive model, target is input shifted by one
print("Decoded input:\n", dataset.decode(xb[0]))
print("Decoded target:\n", dataset.decode(yb[0]))

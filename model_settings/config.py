import torch

device = "cuda" if torch.cuda.is_available() else "cpu"
batch_size = 128
epochs = 15
learning_rate = 1e-3

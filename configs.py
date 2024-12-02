import torch

device = "cuda" if torch.cuda.is_available() else "cpu"
batch_size = 128
num_epochs = 10
learning_rate = 1e-3
train_dir = 'dataset/train_v2/train'
valid_dir = 'dataset/validation_v2/validation'
test_dir = 'dataset/test_v2/test'
train_csv = 'dataset/written_name_train_v2.csv'
valid_csv = 'dataset/written_name_validation_v2.csv'
test_csv = 'dataset/written_name_test_v2.csv'
max_length = 20
vocab_size = 30
image_size = [300, 100]
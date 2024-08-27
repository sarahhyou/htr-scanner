import torch

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
BATCH_SIZE = 128
NUM_EPOCHS = 15
LEARNING_RATE = 1e-3
TRAIN_IMAGES_DIR = 'datasets/train_v2/train'
VALID_IMAGES_DIR = 'datasets/validation_v2/validation'
TEST_IMAGES_DIR = 'datasets/test_v2/test'
TRAIN_CSV = 'datasets/written_name_train_v2.csv'
VALID_CSV = 'datasets/written_name_validation_v2.csv'
TEST_CSV = 'datasets/written_name_test_v2.csv'
MAX_LENGTH = 20
VOCAB_SIZE = 32
IMAGE_SIZE = [300, 100]
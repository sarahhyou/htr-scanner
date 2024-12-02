import torch
import configs
import torch.nn as nn

class CNNBlock(nn.Module):
    def __init__(self, input_channel, output_channel, kernel_sizes, strides, paddings, batch_norm: bool = False):
        super(CNNBlock, self).__init__()
        self.do_batch_norm = batch_norm
        self.conv = nn.Conv2d(input_channel, output_channel, kernel_sizes, strides, paddings)
        self.bn = nn.BatchNorm2d(output_channel)
        self.relu = nn.LeakyReLU(inplace=True)

    def forward(self, x):
        x = self.conv(x)
        if self.do_batch_norm:
            x = self.bn(x)
        x = self.relu(x)
        return x

class CNN(nn.Module):
    def __init__(self, img_channel, img_dim, collapse_layer_hidden=64, rnn_hidden=256):
        super(CNN, self).__init__()
        # CNN block
        self.cnn = nn.Sequential(
            CNNBlock(img_channel, 16, 3, 1, 1),
            nn.MaxPool2d(kernel_size=2, stride=2),
            CNNBlock(16, 32, 3, 1, 1),
            nn.MaxPool2d(kernel_size=2, stride=2),
            CNNBlock(32, 64, 3, 1, 1),
            CNNBlock(64, 128, 3, 1, 1),
            nn.MaxPool2d(kernel_size=(2, 2)),
            nn.Dropout(0.2),
            CNNBlock(128, 256, 3, 1, 1, batch_norm=True),
            nn.Dropout(0.2),
            CNNBlock(256, 512, 3, 1, 1, batch_norm=True),
            nn.MaxPool2d(kernel_size=(2, 2)),
            nn.Dropout(0.2),
            CNNBlock(512, 512, 2, 1, 0, batch_norm=True),
        )
        output_height = img_dim // 16 - 1
        self.collapse_features = nn.Linear(512 * output_height, collapse_layer_hidden)
    def forward(self, images):
        # Extract features
        conv = self.cnn(images)
        # Reformat array
        batch, channel, height, width = conv.size()
        conv = conv.view(batch, channel * height, width)
        conv = conv.permute(2, 0, 1)
        conv = self.collapse_features(conv)
        return conv


class LSTMClassifier(nn.Module):
    def __init__(self, input_dim, hidden_dim, layer_dim, classifier_dim, output_dim):
        super(LSTMClassifier, self).__init__()

        # Hidden dimensions
        self.hidden_dim = hidden_dim

        # Number of hidden layers
        self.layer_dim = layer_dim

        # Building your LSTM
        # batch_first=True causes input/output tensors to be of shape
        # (batch_dim, seq_dim, feature_dim)
        self.lstm = nn.LSTM(input_dim, hidden_dim, layer_dim, batch_first=True)

        # Classifier network
        self.classifier = nn.Sequential(
            nn.Linear(hidden_dim, classifier_dim),
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.Linear(classifier_dim, output_dim),
            #nn.Softmax(dim=1) # No need for softmax with logit loss.
       )

    # Forward method
    def forward(self, x):
        # Initialize hidden state with zeros
        h0 = torch.zeros(self.layer_dim, x.size(0), self.hidden_dim).to(configs.device)

        # Initialize cell state
        c0 = torch.zeros(self.layer_dim, x.size(0), self.hidden_dim).to(configs.device)

        # Forward pass
        out, (hn, cn) = self.lstm(x, (h0, c0))
        out = self.classifier(out[:, -1, :])
        return out


class CRNN(nn.Module):
    def __init__(self, input_size, hidden_size, output_size, num_layers):
        super(CRNN, self).__init__()
        self.RNN = LSTMClassifier(input_size, hidden_size,
                          output_size, num_layers)
        self.CNN = CNN()

    def forward(self, images):
        features = self.CNN(images)
        outputs = self.RNN(features)
        return outputs
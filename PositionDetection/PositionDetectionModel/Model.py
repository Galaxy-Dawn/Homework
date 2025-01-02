import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision.models import mobilenet_v3_small
from torch.nn import init


class Mobile_LSTMModel(nn.Module):
    def __init__(self, num_classes):
        super(Mobile_LSTMModel, self).__init__()
        # Define CNN layers
        self.conv1 = nn.Conv1d(in_channels=100, out_channels=100, kernel_size=3, padding=1)
        self.conv2 = nn.Conv1d(in_channels=100, out_channels=128, kernel_size=3, padding=1)
        self.conv3 = nn.Conv1d(in_channels=128, out_channels=256, kernel_size=3, padding=1)

        # Define BatchNorm layers
        self.bn1 = nn.BatchNorm1d(100)
        self.bn2 = nn.BatchNorm1d(128)
        self.bn3 = nn.BatchNorm1d(256)

        # Use MSRA initialization for convolutional layers
        for conv_layer in [self.conv1, self.conv2, self.conv3]:
            init.kaiming_normal_(conv_layer.weight, mode='fan_out', nonlinearity='relu')

        # Define LSTM layer
        self.lstm = nn.LSTM(input_size=256, hidden_size=512, num_layers=2, dropout=0.1, batch_first=True)

        # Define MobileNet V3 network
        self.mobilenet = mobilenet_v3_small(weights=None)

        self.conv4 = nn.Conv1d(in_channels=1000, out_channels=256, kernel_size=3, padding=1)
        self.conv5 = nn.Conv1d(in_channels=256 + 512, out_channels=512, kernel_size=3, padding=1)
        self.conv6 = nn.Conv1d(in_channels=512, out_channels=512, kernel_size=3, padding=1)

        # Define BatchNorm layers
        self.bn4 = nn.BatchNorm1d(256)
        self.bn5 = nn.BatchNorm1d(512)
        self.bn6 = nn.BatchNorm1d(512)

        # Define FC layer
        self.fc = nn.Linear(512, num_classes)  # 512 from LSTM + 1000 from MobileNet V3

    def forward(self, x, mobilenet_input):
        # CNNLSTMModel forward pass
        x = self.conv1(x)
        x = self.bn1(x)
        x = F.relu(x)

        x = self.conv2(x)
        x = self.bn2(x)
        x = F.relu(x)

        x = self.conv3(x)
        x = self.bn3(x)
        x = F.relu(x)

        x = x.permute(0, 2, 1)  # Reshape dimensions to [batch_size, 256, new_seq_len]
        lstm_output, _ = self.lstm(x)
        lstm_output = lstm_output[:, -1, :]  # Take the last time step's output

        # MobileNet V3 forward pass
        mobilenet_output = self.mobilenet(mobilenet_input)

        # Concatenate the outputs of CNN and MobileNet
        mobilenet_output = mobilenet_output.unsqueeze(2)
        mobilenet_output = self.conv4(mobilenet_output)
        mobilenet_output = self.bn4(mobilenet_output)
        mobilenet_output = F.relu(mobilenet_output)
        mobilenet_output = mobilenet_output.squeeze(2)

        combined_output = torch.cat((lstm_output, mobilenet_output), dim=1)

        # Additional convolutional layers
        x = combined_output.unsqueeze(2)  # Add a singleton dimension for conv1d
        x = self.conv5(x)
        x = self.bn5(x)
        x = F.relu(x)

        x = self.conv6(x)
        x = self.bn6(x)
        x = F.relu(x)

        # Fully connected layer forward pass with ReLU activation
        x = x.squeeze(2)  # Remove the singleton dimension before FC layer
        output = self.fc(x)

        return output
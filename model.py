import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset


class CNNLSTMModel(nn.Module):
    def __init__(self, img_height, img_width, channels, sequence_length):
        super(CNNLSTMModel, self).__init__()
        self.img_height = img_height
        self.img_width = img_width
        self.channels = channels
        self.sequence_length = sequence_length

        # 卷积层
        self.conv1 = nn.Conv2d(channels, 32, kernel_size=3)
        self.pool1 = nn.MaxPool2d(kernel_size=2)
        self.conv2 = nn.Conv2d(32, 64, kernel_size=3)
        self.pool2 = nn.MaxPool2d(kernel_size=2)
        self.conv3 = nn.Conv2d(64, 128, kernel_size=3)
        self.pool3 = nn.MaxPool2d(kernel_size=2)

        # 计算卷积层输出的特征数量
        self._to_linear = None
        x = torch.randn(self.sequence_length, self.channels, self.img_height, self.img_width)
        self.convs(x)

        # 考虑年份特征，增加一个输入维度
        self.lstm = nn.LSTM(self._to_linear + 1, 128, batch_first=True)

        # 全连接层
        self.fc1 = nn.Linear(128, 64)
        self.fc2 = nn.Linear(64, 1)

    def convs(self, x):
        x = torch.relu(self.conv1(x))
        x = self.pool1(x)
        x = torch.relu(self.conv2(x))
        x = self.pool2(x)
        x = torch.relu(self.conv3(x))
        x = self.pool3(x)

        if self._to_linear is None:
            self._to_linear = x[0].shape[0] * x[0].shape[1] * x[0].shape[2]
        return x

    def forward(self, x, years):
        batch_size = x.size(0)
        cnn_outputs = []
        for i in range(self.sequence_length):
            cnn_out = self.convs(x[:, i, :, :, :])
            cnn_out = cnn_out.view(batch_size, -1)
            cnn_outputs.append(cnn_out)

        lstm_input = torch.stack(cnn_outputs, dim=1)
        # 拼接年份特征
        years = years.unsqueeze(-1).unsqueeze(-1).repeat(1, self.sequence_length, 1)
        lstm_input = torch.cat((lstm_input, years), dim=-1)

        lstm_out, _ = self.lstm(lstm_input)
        lstm_out = lstm_out[:, -1, :]

        x = torch.relu(self.fc1(lstm_out))
        x = self.fc2(x)
        return x


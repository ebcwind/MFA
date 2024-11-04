import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn import Dropout
from torchvision.models import resnet34


class BasicConv(nn.Module):
    def __init__(self, n_vocabulary, embedding_size, channel):
        super(BasicConv, self).__init__()
        self.embed = torch.nn.Embedding(n_vocabulary, embedding_size)
        self.cbam = CBAMLayer(channel)
        self.cnn3 = nn.Conv2d(2, 2, (3, 4), padding=(1, 0))
        self.cnn5 = nn.Conv2d(2, 2, (5, 4), padding=(2, 0))
        self.cnn7 = nn.Conv2d(2, 2, (7, 4), padding=(3, 0))
        self.learn = nn.Sequential(
            nn.Conv2d(6, 2, (13608, 1), bias=True),
            nn.Sigmoid(),
        )
        for m in self.modules():
            if isinstance(m, (torch.nn.Linear, torch.nn.Conv2d)):
                nn.init.uniform_(m.weight, a=-0.1, b=0.1)

    def forward(self, x):
        x = self.embed(x)
        x3 = self.cnn3(x)
        x5 = self.cnn5(x)
        x7 = self.cnn7(x)
        x = torch.cat([x3, x5, x7], dim=1)
        x = self.cbam(x)
        x = self.learn(x)
        x = x.view(x.size(0), -1)
        return x


class NormalConv(nn.Module):
    def __init__(self, stride=1, padding=0, bias=False):
        super(NormalConv, self).__init__()
        self.conv3 = nn.Conv2d(1, 1, (3, 18), stride=stride, padding=padding, bias=bias)
        self.conv5 = nn.Conv2d(1, 1, (5, 18), stride=stride, padding=padding, bias=bias)
        self.conv7 = nn.Conv2d(1, 1, (7, 18), stride=stride, padding=padding, bias=bias)
        self.linear_relu_stack3 = nn.Sequential(
            nn.Linear(13606, 6048),
            nn.Sigmoid(),
            nn.Linear(6048, 1024),
            nn.Sigmoid(),
        )
        self.linear_relu_stack5 = nn.Sequential(
            nn.Linear(13604, 6048),
            nn.Sigmoid(),
            nn.Linear(6048, 1024),
            nn.Sigmoid(),
        )
        self.linear_relu_stack7 = nn.Sequential(
            nn.Linear(13602, 6048),
            nn.Sigmoid(),
            nn.Linear(6048, 1024),
            nn.Sigmoid(),
        )
        self.linear_relu_stack = nn.Sequential(
            nn.Linear(1024*3, 1024),
            nn.Sigmoid(),
            Dropout(p=0.5, inplace=False),
            nn.Linear(1024, 512),
            nn.Sigmoid(),
            Dropout(p=0.5, inplace=False),
            nn.Linear(512, 2)
        )
        for m in self.modules():
            if isinstance(m, (torch.nn.Linear, torch.nn.Conv2d)):
                nn.init.xavier_uniform_(m.weight)

        # for m in self.modules():
        #     if isinstance(m, torch.nn.Linear):
        #         torch.nn.init.xavier_uniform_(m.weight, gain=torch.nn.init.calculate_gain('relu'))
        #
        # for m in self.modules():
        #     if isinstance(m, torch.nn.Conv2d):
        #         torch.nn.kaiming_normal_(m.weight, mode='fan_in', nonlinearity='relu')

    def forward(self, x):
        x1 = self.conv3(x[:, :1, :, :])
        x2 = self.conv5(x[:, 1:2, :, :])
        x3 = self.conv7(x[:, 2:3, :, :])
        x1 = x1.view(x1.size(0), -1)
        x2 = x2.view(x2.size(0), -1)
        x3 = x3.view(x3.size(0), -1)

        x1 = self.linear_relu_stack3(x1)
        x2 = self.linear_relu_stack5(x2)
        x3 = self.linear_relu_stack7(x3)

        x = torch.cat([x1, x2, x3], dim=1)
        x = self.linear_relu_stack(x)

        return x


class Flatten(nn.Module):
    def forward(self, x):
        return x.view(x.size(0), -1)


class CBAMLayer(nn.Module):
    def __init__(self, channel, reduction=3, spatial_kernel=7):
        super(CBAMLayer, self).__init__()

        # channel attention 压缩H,W为1
        self.max_pool = nn.AdaptiveMaxPool2d(1)
        self.avg_pool = nn.AdaptiveAvgPool2d(1)

        # shared MLP
        self.mlp = nn.Sequential(
            # Conv2d比Linear方便操作
            # nn.Linear(channel, channel // reduction, bias=False)
            nn.Conv2d(channel, channel // reduction, 1, bias=False),
            # inplace=True直接替换，节省内存
            nn.ReLU(inplace=True),
            # nn.Linear(channel // reduction, channel,bias=False)
            nn.Conv2d(channel // reduction, channel, 1, bias=False)
        )

        # spatial attention
        # self.conv = nn.Conv2d(2, 1, kernel_size=spatial_kernel,
        #                       padding=spatial_kernel // 2, bias=False)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        max_out = self.mlp(self.max_pool(x))
        avg_out = self.mlp(self.avg_pool(x))
        channel_out = self.sigmoid(max_out + avg_out)
        x = channel_out * x

        # max_out, _ = torch.max(x, dim=1, keepdim=True)
        # avg_out = torch.mean(x, dim=1, keepdim=True)
        # spatial_out = self.sigmoid(self.conv(torch.cat([max_out, avg_out], dim=1)))
        # x = spatial_out * x
        return x


class CbamCnn(nn.Module):
    def __init__(self, channel):
        super(CbamCnn, self).__init__()
        self.cbam = CBAMLayer(channel)
        self.learn = nn.Sequential(
            nn.Conv2d(1, 2, (13608, 1), bias=True),
            nn.ReLU(),
            nn.Conv2d(2, 2, (1, 3), bias=True),
            # nn.Conv2d(1, 1, (1, 5)),
            # nn.ReLU(),
        )
        self.fc = nn.Sequential(
            nn.Linear(13608, 1024*6),
            nn.Sigmoid(),
            Dropout(p=0.5, inplace=False),
            nn.Linear(1024 * 6, 1024 * 3),
            nn.Sigmoid(),
            Dropout(p=0.5, inplace=False),
            nn.Linear(1024 * 3, 1024),
            nn.Sigmoid(),
            Dropout(p=0.5, inplace=False),
            nn.Linear(1024, 512),
            nn.Sigmoid(),
            nn.Linear(512, 2),
        )

        self.relu = nn.ReLU()

        self.conv = nn.Conv2d(6, 1, (1, 4), stride=1, bias=False)
        self.conv3 = nn.Conv2d(2, 1, (3, 4), stride=1, padding=(1, 0), bias=False)
        self.conv5 = nn.Conv2d(2, 1, (5, 4), stride=1, padding=(2, 0), bias=False)
        self.conv7 = nn.Conv2d(2, 1, (7, 4), stride=1, padding=(3, 0), bias=False)


        for m in self.modules():
            if isinstance(m, (torch.nn.Linear, torch.nn.Conv2d)):
                nn.init.xavier_uniform_(m.weight)
                # nn.init.kaiming_uniform_(m.weight)
                # nn.init.uniform_(m.weight, a=-1, b=1)

    def forward(self, x):
        # x = self.cbam(x)
        x3 = self.conv3(x[:, 0:2, :, :])
        x5 = self.conv5(x[:, 2:4, :, :])
        x7 = self.conv7(x[:, 4:6, :, :])

        x = torch.cat([x3, x5, x7], dim=3)

        x = self.learn(x)
        x = x.view(x.size(0), -1)
        return x


class myLSTM(nn.Module):
    def __init__(self, embedding_size, hidden_size, num_layers):
        super(myLSTM, self).__init__()  # 初始化
        self.lstm_pre = nn.LSTM(
            batch_first=True,
            input_size=embedding_size,  # 输入大小为转化后的词向量
            hidden_size=hidden_size,  # 隐藏层大小
            num_layers=num_layers,  # 堆叠层数，有几层隐藏层就有几层
            dropout=0.5,  # 遗忘门参数
            bidirectional=True  # 双向LSTM
        )
        self.lstm_in = nn.LSTM(
            batch_first=True,
            input_size=embedding_size,  # 输入大小为转化后的词向量
            hidden_size=hidden_size,  # 隐藏层大小
            num_layers=num_layers,  # 堆叠层数，有几层隐藏层就有几层
            dropout=0.5,  # 遗忘门参数
            bidirectional=True  # 双向LSTM
        )

        self.dropout = nn.Dropout(0.5)
        self.fc = nn.Sequential(
            nn.Linear(hidden_size * num_layers * 2 * 2, hidden_size * num_layers),  # 因为双向要*2
            nn.Sigmoid(),
            nn.Linear(hidden_size * num_layers, 2),
        )
        self.softmax = nn.Softmax(dim=-1)

        for name, param in self.lstm_pre.named_parameters():
            if 'bias' in name:
                nn.init.constant_(param, 0.0)
            elif 'weight' in name:
                nn.init.xavier_uniform_(param, gain=1)

        for name, param in self.lstm_in.named_parameters():
            if 'bias' in name:
                nn.init.constant_(param, 0.0)
            elif 'weight' in name:
                nn.init.xavier_uniform_(param, gain=1)

        for m in self.modules():
            if isinstance(m, torch.nn.Linear):
                nn.init.xavier_uniform_(m.weight)
                nn.init.constant_(m.bias, 0)

    def forward(self, x):

        lstm_out1, (h_n1, c_n1) = self.lstm_pre(x[:, 0, :, :])
        lstm_out2, (h_n2, c_n2) = self.lstm_in(x[:, 1, :, :])

        feature1 = self.dropout(h_n1)
        feature2 = self.dropout(h_n2)

        # 这里将所有隐藏层进行拼接来得出输出结果，没有使用模型的输出
        feature_map1 = torch.cat([feature1[i, :, :] for i in range(feature1.shape[0])], dim=-1)  # 2, 32, hidden_size
        feature_map2 = torch.cat([feature2[i, :, :] for i in range(feature2.shape[0])], dim=-1)
        feature_map = torch.cat([feature_map1, feature_map2], dim=-1)
        out = self.fc(feature_map)

        return self.softmax(out)


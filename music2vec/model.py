import torch as th
import torch.nn as nn


class BLSTM(nn.Module):

    def __init__(self, dim, layers=1):
        super().__init__()
        self.lstm = nn.LSTM(
            bidirectional=True, 
            num_layers=layers, hidden_size=dim, 
            input_size=dim
        )
        self.linear = nn.Linear(2 * dim, dim)

    def forward(self, x):
        x = x.permute(2, 0, 1)
        x = self.lstm(x)[0]
        x = self.linear(x)
        x = x.permute(1, 2, 0)
        return x


class Music2Vec(nn.Module):

    def __init__(
        self, feature_size=512,
        depth=4, kernel_size=8,
        stride=4, lstm_layers=2,
        output_size=10, audio_channel=1,
        channel=64
    ):
        super().__init__()

        self.feature_size = feature_size
        self.depth = depth
        self.kernel_size = kernel_size
        self.stride = stride
        self.lstm_layers = lstm_layers
        self.output_size = output_size
        self.audio_channel = audio_channel
        self.channel = channel

        self.lstm = BLSTM(self.feature_size, self.lstm_layers)

        self.feature_extructor = nn.ModuleList()
        self.fc = nn.Sequential(
            nn.Dropout(0.5),
            nn.Linear(self.feature_size, self.output_size)
        )

        in_channel = self.audio_channel
        conv_channel = self.channel

        for i in range(self.depth):

            if i == self.depth - 1:
                conv_channel = self.feature_size

            encode = [
                nn.Conv1d(in_channel, conv_channel, self.kernel_size, self.stride),
                nn.BatchNorm1d(conv_channel),
                nn.ReLU(),
                nn.Dropout(0.5)
            ]

            self.feature_extructor.append(
                nn.Sequential(*encode)
            )

            in_channel = conv_channel
            conv_channel = conv_channel * 2

        self.feature_extructor.append(
            nn.Sequential(
                self.lstm,
                nn.AdaptiveAvgPool1d((1,)),
                nn.Flatten()
            )
        )

    def features(self, x):
        for encode in self.feature_extructor:
            x = encode(x)
        return x

    def forward(self, x):
        x = self.features(x)
        return self.fc(x)


if __name__ == '__main__':

    print('Model test.')

    model = Music2Vec()
    print(model)

    dummy = th.randn(1, 1, 22050)
    print('input tensor size: [{}, {}, {}]'.format(*dummy.shape))

    features = model.features(dummy)
    print('output feature size: [{}, {}]'.format(*features.shape))

    output = model(dummy)
    print('output size: [{}, {}]'.format(*output.shape))
import torch as th
import torch.nn as nn
from demucs.pretrained import demucs



class Music2Vec(nn.Module):

    def __init__(
        self, feature_size=512,
        depth=4, kernel_size=8,
        stride=4, lstm_layers=2,
        output_size=10, audio_channel=1,
        channel=64
    ):
        super().__init__()

        basemodel = demucs()
        basemodel.encoder[0][0] = nn.Conv1d(1, 64, kernel_size=(8,), stride=(4,))

        self.encoder = nn.Sequential(
            *basemodel.encoder
        )
        self.lstm = basemodel.lstm
        self.feature = nn.Sequential(
            nn.AdaptiveAvgPool1d([1]),
            nn.Flatten()
        )
        self.fc = nn.Sequential(
            nn.Linear(2048, 10)
        )



    def features(self, x):
        x = self.encoder(x)
        x = self.lstm(x)
        return self.feature(x)


    def forward(self, x):
        x = self.features(x)
        return self.fc(x)



if __name__ == '__main__':

    print('Model test.')

    model = Music2Vec()
    print(model)

    dummy = th.randn(1, 1, 22050*2)
    print('input tensor size: [{}, {}, {}]'.format(*dummy.shape))

    features = model.features(dummy)
    print('output feature size: [{}, {}]'.format(*features.shape))

    output = model(dummy)
    print('output size: [{}, {}]'.format(*output.shape))
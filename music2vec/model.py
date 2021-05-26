import torch as th
import torch.nn as nn


class Music2Vec(nn.Module):

    def __init__(
        self, output_size=10
    ):
        super().__init__()

        self.basemodel = th.hub.load(
            'pytorch/vision:v0.9.0', 
            'densenet201', pretrained=True
        )
        self.basemodel.features.conv0 = nn.Conv2d(
            1, 64, kernel_size=25, 
            stride=(2, 2), padding=(3, 3), 
            bias=False
        )
        self.basemodel.classifier = nn.Sequential(
            nn.BatchNorm1d(1024),
            nn.ReLU(),
            nn.Linear(1024, output_size, bias=True)
        )

        self.features_ = nn.Sequential(
            self.basemodel.features,
            nn.Flatten(),
            nn.BatchNorm1d(17280),
            nn.ReLU(),
            nn.Linear(17280, 17280//2, bias=True),
            nn.BatchNorm1d(17280//2),
            nn.ReLU(),
            nn.Linear(17280//2, 17280//4, bias=True),
            nn.BatchNorm1d(17280//4),
            nn.ReLU(),
            nn.Linear(17280//4, 1024, bias=True)
        )
        self.classifier_ = self.basemodel.classifier
        
    
    def features(self, x):
        x = self.features_(x)
        return x
    

    def forward(self, x):
        x = self.features(x)
        x = self.classifier_(x)
        return nn.Sigmoid()(x)


if __name__ == '__main__':

    print('Model test.')

    model = Music2Vec()
    print(model)

    dummy = th.randn(1, 1, 128, 128)
    print('input tensor size: [{}, {}, {}, {}]'.format(*dummy.shape))

    features = model.features(dummy)
    print('output feature size: [{}, {}]'.format(*features.shape))

    output = model(dummy)
    print('output size: [{}, {}]'.format(*output.shape))
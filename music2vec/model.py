import torch as th
import torch.nn as nn


class Music2Vec(nn.Module):

    def __init__(
        self, output_size=10
    ):
        super().__init__()

        basemodel = th.hub.load(
            'pytorch/vision:v0.9.0', 
            'densenet201', pretrained=True
        )
        basemodel.features.conv0 = nn.Conv2d(
            1, 64, kernel_size=25, 
            stride=(2, 2), padding=(3, 3), 
            bias=False
        )

        self.features_ = basemodel.features
        self.classifier_ = nn.ModuleList(
          [
            nn.Sequential(
              nn.BatchNorm1d(1920),
              nn.ReLU(),
              nn.Linear(1920, 1920//2, bias=True),
              nn.BatchNorm1d(1920//2),
              nn.ReLU(),
              nn.Linear(1920//2, 1920, bias=True),
            ) for _ in range(50)
          ]
        )
        self.fc = nn.Sequential(
            nn.BatchNorm1d(1920),
            nn.ReLU(),
            nn.Linear(1920, output_size, bias=True),
        )
        
    
    def features(self, x):
        x = self.features_(x)
        x = nn.AdaptiveMaxPool2d((1, 1))(x)
        x = nn.Flatten()(x)
        return x
    

    def forward(self, x):
        x = self.features(x)
        for m in self.classifier_:
            skip = x
            x = m(x)
            x += skip
        return nn.Softmax(dim=-1)(self.fc(x))


if __name__ == '__main__':

    print('Model test.')

    model = Music2Vec()
    print(model)

    dummy = th.randn(2, 1, 128, 128)
    print('input tensor size: [{}, {}, {}, {}]'.format(*dummy.shape))

    features = model.features(dummy)
    print('output feature size: [{}, {}]'.format(*features.shape))

    output = model(dummy)
    print('output size: [{}, {}]'.format(*output.shape))
import torch as th
import torch.nn as nn


class Music2Vec(nn.Module):

    def __init__(
        self, output_size=10
    ):
        super().__init__()

        basemodel = th.hub.load(
            'pytorch/vision:v0.9.0', 
            'resnet18', pretrained=True
        )
        basemodel.conv1 = nn.Conv2d(
            5, 64, kernel_size=25, 
            stride=(2, 2), padding=(3, 3), 
            bias=False
        )
        self.fc = nn.Sequential(
            # nn.Dropout(0.5),
            nn.Linear(512, output_size, bias=True)
        )
        self.feature = nn.Sequential(*list(basemodel.children())[:-1])
        

    def features(self, x):
        return self.feature(x)
        

    def forward(self, x):
        x = self.features(x)
        x = nn.Flatten()(x)
        return self.fc(x)


if __name__ == '__main__':

    print('Model test.')

    model = Music2Vec()
    print(model)

    dummy = th.randn(1, 1, 224, 224)
    print('input tensor size: [{}, {}, {}]'.format(*dummy.shape))

    features = model.features(dummy)
    print('output feature size: [{}, {}]'.format(*features.shape))

    output = model(dummy)
    print('output size: [{}, {}]'.format(*output.shape))
import torch as th
import torch.nn as nn


class Music2Vec(nn.Module):

    def __init__(
        self, output_size=10
    ):
        super().__init__()

        basemodel = th.hub.load(
            'pytorch/vision:v0.9.0', 
            'densenet121', pretrained=True
        )
        basemodel.features.conv0 = nn.Conv2d(
            1, 64, kernel_size=25, 
            stride=(2, 2), padding=(3, 3), 
            bias=False
        )
        basemodel.classifier = nn.Sequential(
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.Linear(1024, output_size, bias=True)
        )

        self.features_ = basemodel.features
        self.classifier_ = basemodel.classifier
        
    
    def features(self, x):
        x = self.features_(x)
        x = nn.AdaptiveMaxPool2d((1, 1))(x)
        x = nn.Flatten()(x)
        return x
    

    def forward(self, x):
        x = self.features(x)
        return self.classifier_(x)


if __name__ == '__main__':

    print('Model test.')

    model = Music2Vec()
    print(model)

    dummy = th.randn(1, 5, 224, 224)
    print('input tensor size: [{}, {}, {}]'.format(*dummy.shape))

    features = model.features(dummy)
    print('output feature size: [{}, {}]'.format(*features.shape))

    output = model(dummy)
    print('output size: [{}, {}]'.format(*output.shape))
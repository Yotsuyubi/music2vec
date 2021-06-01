import torch as th
import torch.nn as nn
import os


class Swish(nn.Module):

    def __init__(self):
        super().__init__()
        self.beta = nn.Parameter(
            th.tensor(1.0)
        )


    def forward(self, x):
        return x*nn.Sigmoid()(x*self.beta)
        # return nn.ReLU()(x)


class ConvBlock(nn.Module):

    def __init__(self, filter, kernel):

        super().__init__()

        self.seq = nn.Sequential(
            nn.Conv2d(filter, filter, kernel, padding=(kernel-1)//2),
            nn.BatchNorm2d(filter),
            Swish()
        )

    def forward(self, x):
        return self.seq(x)


class MultiScaleBlock(nn.Module):

    def __init__(self, filter):

        super().__init__()

        self.inconv = nn.Conv2d(filter, filter, 1)

        self.conv1x1 = ConvBlock(filter, 1)
        self.conv3x3 = nn.Sequential(
            ConvBlock(filter, 1),
            ConvBlock(filter, 3)
        )
        self.conv5x5 = nn.Sequential(
            ConvBlock(filter, 1),
            ConvBlock(filter, 5)      
        )
        self.pool = nn.Sequential(
            nn.MaxPool2d(3, 1, 1),
            ConvBlock(filter, 1)
        )
        self.outconv = nn.Conv2d(filter*4, filter, 1)

    def forward(self, x):

        conv1x1 = self.conv1x1(x)
        conv3x3 = self.conv3x3(x)
        conv5x5 = self.conv5x5(x)
        pool = self.pool(x)

        out = th.cat([conv1x1, conv3x3, conv5x5, pool], dim=1)
        out = self.outconv(out)

        return out


class DenseBlock(nn.Module):

    def __init__(self, num_blocks, filter):

        super().__init__()
        self.dense = nn.ModuleList(
            [ MultiScaleBlock(filter) for i in range(num_blocks) ]
        )
        self.conv = nn.ModuleList([
            *[ nn.Conv2d(filter*2, filter, 1) for i in range(num_blocks) ]
        ])


    def forward(self, x):

        skip = x
        for i, dense in enumerate(self.dense):
            x = dense(x)
            x = th.cat([x, skip], dim=1)
            x = self.conv[i](x)
        
        return x

    
class TransitionBlock(nn.Module):

    def __init__(self, in_filter, out_filter):

        super().__init__()
        self.seq = nn.Sequential(
            nn.BatchNorm2d(in_filter),
            Swish(),
            nn.Conv2d(in_filter, out_filter, 1),
            nn.AvgPool2d(2, 2),
            nn.BatchNorm2d(out_filter),
            Swish(),
            nn.AdaptiveAvgPool2d((1, 1)),
            nn.Flatten()
        )


    def forward(self, x):      
        return self.seq(x)


class Music2Vec(nn.Module):

    def __init__(
        self, output_size=10, filter=64, num_blocks=3, features=1024
    ):
        super().__init__()

        self.in_conv = nn.Sequential(
            nn.Conv2d(1, filter, 3, 1),
            nn.BatchNorm2d(filter),
            Swish(),
            nn.MaxPool2d((1, 4))
        )
        self.dense_block = DenseBlock(num_blocks, filter)
        self.transition = TransitionBlock(filter, features)
        self.fc = nn.Sequential(
            # nn.Dropout(0.5),
            nn.Linear(features, output_size),
            nn.Softmax(dim=-1)
        )
        

    
    def features(self, x):
        x = self.in_conv(x)
        x = self.dense_block(x)
        x = self.transition(x)
        return x
    

    def forward(self, x):
        return self.fc( self.features(x) )



def music2vec(model_path=None, gpu=False):

    if model_path == None:
        model_path = os.path.join(
            os.path.dirname(os.path.abspath(__file__)), 'pretrained/model.pth'
        )

    model = Music2Vec()
    model.load_state_dict(
        th.load(
            model_path,
            map_location='cuda' if gpu else 'cpu'
        )
    )
    model.eval()

    return model



if __name__ == '__main__':

    print('Model test.')

    model = Music2Vec()
    print(model)

    dummy = th.randn(1, 8, 128, 128)
    print('input tensor size: [{}, {}, {}, {}]'.format(*dummy.shape))

    features = model.features(dummy)
    print('output feature size: [{}, {}]'.format(*features.shape))

    output = model(dummy)
    print('output size: [{}, {}]'.format(*output.shape))
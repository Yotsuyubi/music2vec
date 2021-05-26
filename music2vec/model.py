import torch as th
import torch.nn as nn
from self_attention_cv import ResNet50ViT
from einops import rearrange, repeat

def expand_to_batch(tensor, desired_size):
    tile = desired_size // tensor.shape[0]
    return repeat(tensor, 'b ... -> (b tile) ...', tile=tile)


class Music2Vec(nn.Module):

    def __init__(
        self, output_size=10
    ):
        super().__init__()

        self.basemodel = ResNet50ViT(img_dim=128, dim=1024, pretrained_resnet=True)
        self.basemodel.model[0][0] = nn.Conv2d(
            8, 64, kernel_size=(7, 7), 
            stride=(2, 2), padding=(3, 3), bias=False
        )
        self.resnet = self.basemodel.model[0]
        self.vit = self.basemodel.model[1]


    def features(self, img, mask=None):
        img = self.resnet(img)
        # Create patches
        # from [batch, channels, h, w] to [batch, tokens , N], N=p*p*c , tokens = h/p *w/p
        img_patches = rearrange(img,
                                'b c (patch_x x) (patch_y y) -> b (x y) (patch_x patch_y c)',
                                patch_x=self.vit.p, patch_y=self.vit.p)

        batch_size, tokens, _ = img_patches.shape

        # project patches with linear layer + add pos emb
        img_patches = self.vit.project_patches(img_patches)

        img_patches = th.cat((expand_to_batch(self.vit.cls_token, desired_size=batch_size), img_patches), dim=1)

        # add pos. embeddings. + dropout
        # indexing with the current batch's token length to support variable sequences
        img_patches = img_patches + self.vit.pos_emb1D[:tokens + 1, :]
        patch_embeddings = self.vit.emb_dropout(img_patches)

        # feed patch_embeddings and output of transformer. shape: [batch, tokens, dim]
        y = self.vit.transformer(patch_embeddings, mask)

        return y[:, 0, :]


    def forward(self, x):
        x = self.features(x)
        x = self.vit.mlp_head(x)
        return nn.Softmax(dim=-1)(x)


if __name__ == '__main__':

    print('Model test.')

    model = Music2Vec()
    print(model)

    dummy = th.randn(1, 8, 128, 128)
    print('input tensor size: [{}, {}, {}, {}]'.format(*dummy.shape))

    features = model.features(dummy)
    print('feature tensor size: [{}, {}]'.format(*features.shape))

    output = model(dummy)
    print('output size: [{}, {}]'.format(*output.shape))
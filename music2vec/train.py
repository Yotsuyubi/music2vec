import torch as th
import pytorch_lightning as pl
from torch.optim.adam import Adam
from torchvision import transforms
from .model import Music2Vec
from .dataset import GT
from torch.utils.data import DataLoader
from torchvision.datasets import ImageFolder
from torchvision.transforms import ToTensor, Compose, Grayscale
from pytorch_lightning.callbacks import Callback
import os
import shutil
import argparse
from torch_optimizer import AdaBelief




def accuracy(y_hat, y):

    total = 0
    correct = 0
    _, predicted = th.max(y_hat, 1)
    total += y.size(0)
    correct += (predicted == y).sum().item()
    return correct / total


class MyCallback(Callback):

    def __init__(self, path, num_each=10):

        super().__init__()

        self.path = path
        self.num_each = num_each

        self.epoch_saved_model_path = None


    def on_epoch_end(self, trainer, pl_module):

        current_epoch = trainer.current_epoch

        if (current_epoch+1) % self.num_each == 0:

            th.save(
                trainer.model.model.state_dict(), 
                self.path
            )



class Trainer(pl.LightningModule):

    def __init__(
        self, 
        # model params
        feature_size=512,
        depth=4, kernel_size=8,
        stride=4, lstm_layers=2,
        output_size=10, audio_channel=1,
        channel=64,
        # optimizer
        optimizer=Adam, lr=1e-3
    ):

        super().__init__()

        self.model = Music2Vec()

        self.lr = lr

        self.optimizer = optimizer(
			  self.model.parameters(), 
          	self.lr#, weight_decay=1e-6
        )
        self.scheduler = th.optim.lr_scheduler.StepLR(
            self.optimizer, 50, 0.5
        )


    def loss_func(self, y, y_true):
        return th.nn.NLLLoss()(th.log(y), y_true)

    def forward(self, x):
        return self.model(x)


    def configure_optimizers(self):
        return [self.optimizer], [self.scheduler]


    def training_step(self, train_batch, batch_idx):

        x, y = train_batch
        y_hat = self(x)
        loss = self.loss_func(y_hat, y)
        acc = accuracy(y_hat, y)
        self.log('train_loss', loss, prog_bar=True, on_epoch=True, on_step=False)
        self.log('train_acc', acc, prog_bar=True, on_epoch=True, on_step=False)
        return loss


    def validation_step(self, val_batch, batch_idx):

        x, y = val_batch
        y_hat = self(x)
        loss = self.loss_func(y_hat, y)
        acc = accuracy(y_hat, y)
        self.log('val_loss', loss, prog_bar=True, on_epoch=True)
        self.log('val_acc', acc, prog_bar=True, on_epoch=True)

    def test_step(self, test_batch, batch_idx):

        x, y = test_batch
        y_hat = self(x)
        loss = self.loss_func(y_hat, y)
        acc = accuracy(y_hat, y)
        self.log('test_loss', loss, prog_bar=True, on_epoch=True)
        self.log('test_acc', acc, prog_bar=True, on_epoch=True)


if __name__ == '__main__':

    parser = argparse.ArgumentParser(
        description='music2vec.train: train music2vec model.'
    )
    parser.add_argument(
        'model_path', metavar='<model_path>', 
        help='dir for load/save model.'
    )
    parser.add_argument(
        'processed_root', metavar='<processed_root>', 
        help='root for processed dataset.'
    )
    parser.add_argument(
        '-l', '--learning_rate', 
        type=float, help='value of learning rate. default is 1e-4.', 
        default=1e-4
    )
    parser.add_argument(
        '-L', '--logging', 
        action='store_true', help='enable logging for tensorboard.', 
    )
    parser.add_argument(
        '-b', '--batch_size', 
        type=int, help='value of batch size. default is 10.', 
        default=10
    )
    parser.add_argument(
        '-e', '--num_per_epoch', 
        type=int, help='number of save model per epoch. default is 10.', 
        default=10
    )
    parser.add_argument(
        '-g', '--num_gpus', 
        type=int, help='number of gpu use. to train using cpu, this must be 0. default is 0.', 
        default=0
    )
    parser.add_argument(
        '-d', '--depth', 
        type=int, help='number of network layers. default is 4.', 
        default=4
    )
    parser.add_argument(
        '-s', '--sample_length', 
        type=int, help='length of samples. default is 22050.', 
        default=22050
    )

    args = parser.parse_args()

    train_model = Trainer(
        lr=args.learning_rate,
        depth=args.depth
    )
    train_loader = DataLoader(
        GT(args.processed_root, download=True, subset='training'), 
        batch_size=args.batch_size,
        num_workers=4, shuffle=True
    )
    valid_loader = DataLoader(
        GT(args.processed_root, download=True, subset='validation'), 
        batch_size=args.batch_size,
        num_workers=4, shuffle=False
    )

    trainer = pl.Trainer(
        gpus=args.num_gpus, 
        callbacks=[MyCallback(args.model_path, args.num_per_epoch)],
        checkpoint_callback=False, logger=args.logging,
        auto_lr_find=False,
    )

    if os.path.exists(args.model_path):
        print('Load model from {}.'.format(args.model_path))
        train_model.model.load_state_dict(
            th.load(
                args.model_path, 
                map_location='cpu' if args.num_gpus == 0 else 'cuda'
            )
        )
    else:
        print('train new model.')

    trainer.fit(train_model, train_loader, valid_loader)
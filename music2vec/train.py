import torch as th
import pytorch_lightning as pl
from torch.optim.adam import Adam
from torchvision import transforms
from .model import Music2Vec
from .dataset import Remixer
from torch.utils.data import DataLoader
from torchvision.datasets import ImageFolder
from torchvision.transforms import ToTensor, Compose, Grayscale
from pytorch_lightning.callbacks import Callback, ModelCheckpoint
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
          	self.lr, weight_decay=1e-6
        )
        self.scheduler = th.optim.lr_scheduler.ReduceLROnPlateau(
            self.optimizer, patience=3, factor=0.5
        )

        self.save_hyperparameters()


    def loss_func(self, y, y_true):
        return th.nn.NLLLoss()(th.log(y), y_true)

    def forward(self, x):
        return self.model(x)


    def configure_optimizers(self):
        return {
            "optimizer": self.optimizer, 
            "lr_scheduler": self.scheduler, 
            "monitor": "train_loss"
        }


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
        help='dir to save model.'
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
        Remixer(os.path.join(args.processed_root, 'train')), 
        batch_size=args.batch_size,
        num_workers=4, shuffle=True
    )
    valid_loader = DataLoader(
        Remixer(os.path.join(args.processed_root, 'valid')), 
        batch_size=args.batch_size,
        num_workers=4, shuffle=False
    )

    checkpoint_callback = ModelCheckpoint(
        monitor='val_loss',
        dirpath='ckeckpoints',
        filename='{epoch:02d}-{val_loss:.2f}',
        save_top_k=3,
        mode='min',
    )

    trainer = pl.Trainer(
        gpus=args.num_gpus, 
        callbacks=[
            MyCallback(args.model_path, args.num_per_epoch), 
            checkpoint_callback
        ],
        logger=args.logging,
        auto_lr_find=False,
    )

    trainer.fit(train_model, train_loader, valid_loader)
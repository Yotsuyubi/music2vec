import torch as th
import pytorch_lightning as pl
from .model import Music2Vec
from .dataset import Remixer
from torch.utils.data import DataLoader
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
        optimizer=AdaBelief, lr=1e-3
    ):

        super().__init__()

        self.model = Music2Vec()

        self.lr = lr

        self.optimizer = optimizer(self.parameters(), self.lr)
        self.scheduler = th.optim.lr_scheduler.StepLR(
            self.optimizer, 30, 0.5
        )


    def forward(self, x):
        return self.model(x)


    def configure_optimizers(self):
        return [self.optimizer], [self.scheduler]


    def training_step(self, train_batch, batch_idx):

        x, y = train_batch
        y_hat = self(x)
        loss = th.nn.CrossEntropyLoss()(y_hat, y)
        acc = accuracy(y_hat, y)
        self.log('train_loss', loss, prog_bar=True, on_epoch=True, on_step=False)
        self.log('train_acc', acc, prog_bar=True, on_epoch=True, on_step=False)
        return loss


    def validation_step(self, val_batch, batch_idx):

        x, y = val_batch
        y_hat = self(x)
        loss = th.nn.CrossEntropyLoss()(y_hat, y)
        acc = accuracy(y_hat, y)
        self.log('val_loss', loss, prog_bar=True, on_epoch=True)
        self.log('val_acc', acc, prog_bar=True, on_epoch=True)


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
        type=float, help='value of learning rate. default is 1e-3.', 
        default=1e-3
    )
    parser.add_argument(
        '-L', '--logging', 
        action='store_true', help='enable logging for tensorboard.', 
    )
    parser.add_argument(
        '-b', '--batch_size', 
        type=int, help='value of batch size. default is 128.', 
        default=64
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

    args = parser.parse_args()

    train_model = Trainer(
        lr=args.learning_rate
    )
    train_loader = DataLoader(
        Remixer(os.path.join(args.processed_root, 'train'), sample_length=128), 
        batch_size=args.batch_size,
        num_workers=4
    )
    valid_loader = DataLoader(
        Remixer(os.path.join(args.processed_root, 'valid'), sample_length=128), 
        batch_size=args.batch_size,
        num_workers=4
    )

    trainer = pl.Trainer(
        gpus=args.num_gpus, 
        callbacks=[MyCallback(args.model_path, args.num_per_epoch)],
        checkpoint_callback=False, logger=args.logging,
        auto_lr_find=True
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
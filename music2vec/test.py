import glob
from .dataset import GT
from .model import Music2Vec
from .train import accuracy, Trainer
import torch as th
import scipy.stats as stats
import pytorch_lightning as pl
from torch.utils.data import DataLoader
import argparse



if __name__ == '__main__':

    parser = argparse.ArgumentParser(
        description='music2vec.test: test music2vec model.'
    )
    parser.add_argument(
        'model_path', metavar='<model_path>', 
        help='dir for saved model.'
    )
    parser.add_argument(
        'dataset_root', metavar='<dataset_root>', 
        help='root for dataset.'
    )
    parser.add_argument(
        '-g', '--num_gpus', 
        type=int, help='number of gpu use. to test using cpu, this must be 0. default is 0.', 
        default=0
    )

    args = parser.parse_args()

    model = Music2Vec()
    model.load_state_dict(
        th.load(
            args.model_path,
            map_location='cpu' if args.num_gpus == 0 else 'cuda'
        )
    )
    model.eval()

    test_loader = DataLoader(
        GT(args.dataset_root, download=True, subset='testing'), 
        batch_size=8,
        num_workers=4, shuffle=False
    )

    train = Trainer()
    train.model = model

    trainer = pl.Trainer()

    trainer.test(model=train, test_dataloaders=test_loader)

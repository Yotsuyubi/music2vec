import glob
from .dataset import GT
from .model import Music2Vec
from .train import accuracy, Trainer
import torch as th
import scipy.stats as stats
import pytorch_lightning as pl
from torch.utils.data import DataLoader


model = Music2Vec()
model.load_state_dict(
    th.load('save/model_bbnn8.pth', map_location='cpu')
)
model.eval()

test_loader = DataLoader(
    GT('download', download=True, subset='testing'), 
    batch_size=8,
    num_workers=4, shuffle=False
)

train = Trainer()
train.model = model

trainer = pl.Trainer()

trainer.test(model=train, test_dataloaders=test_loader)

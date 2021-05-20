import glob
from .dataset import read_wav_and_random_crop
from .model import Music2Vec
from .argument import ToConstantQ
from .train import accuracy
import torch as th
import scipy.stats as stats


GENRES = [
    'blues',
    'classical',
    'country',
    'disco',
    'hiphop',
    'jazz',
    'metal',
    'pop',
    'reggae',
    'rock'
]


model = Music2Vec()
model.load_state_dict(
    th.load('save/model_.pth', map_location='cpu')
)
model.eval()


for genre in GENRES:

    filenames = glob.glob('process/test/{}/*.wav'.format(genre))
    res = th.zeros(20, 10)
    true = th.zeros(20, 1)

    for n, filename in enumerate(filenames):

        data = th.zeros(10, 1, 224, 224)

        for i in range(10):

            wav = read_wav_and_random_crop(filename, 22050*2)
            data[i] += ToConstantQ()(wav)

        with th.no_grad():
            y_hat = model(data)
        _, predicted = th.max(y_hat, 1)
        mode = stats.mode(predicted.detach().numpy())[0]
        res[n] += th.eye(10)[mode][0]
        true[n] += GENRES.index(genre)

    acc = accuracy(res, true)
    print(acc)
    print(res)
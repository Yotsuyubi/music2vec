import glob
from .dataset import read_wav_and_random_crop
from .model import Music2Vec
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
# model.load_state_dict(
#     th.load('save/model.pth', map_location='cpu')
# )
model.eval()


for genre in GENRES:

    filenames = glob.glob('process/test/{}/*.wav'.format(genre))
    res = th.zeros(20, 10)
    true = th.zeros(20, 1)

    for n, filename in enumerate(filenames):

        data = th.zeros(10, 1, 22050)

        for i in range(10):

            wav = read_wav_and_random_crop(filename, 22050)
            data[i] += (wav - wav.min()) / (wav.max() - wav.min()) * 2.0 - 1.0

        data = th.rand(data.shape)

        with th.no_grad():
            y_hat = model(data)

        _, predicted = th.max(y_hat, 1)
        mode = stats.mode(predicted.detach().numpy())[0]
        res[n] += th.eye(10)[mode][0]
        true[n] += GENRES.index(genre)
        print(predicted)

    acc = accuracy(res, true)
    print(acc/20)
    print(res)
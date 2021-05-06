import torch as th
from torch.utils.data import Dataset
import glob
import os
import random
import torchaudio
from .argument import RandomCrop
from scipy.io import wavfile


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
TRACKS = ['bass', 'drums', 'other', 'vocals']


def read_wav(filename, offset=0, duration=None):

    _, data = wavfile.read(filename)
    data = data[::2]

    if duration:
        return data[offset:offset+duration, 0] / 32768
    else:
        return data[offset:, 0] / 32768


def get_subset(path):

    subset = {}

    for genre in GENRES:

        genre_set = {}

        for track in TRACKS:

            file_dir = os.path.join(path, genre, track, '*.wav')
            filenames = glob.glob(file_dir)

            genre_set.update({ track: filenames })

        subset.update({ genre: genre_set })

    return subset


class Remixer(Dataset):

    def __init__(
        self, root,
        length=1024, sample_length=22050
    ):
        super().__init__()

        self.root = root

        self.length = length
        self.sample_length = sample_length

        self.subset = get_subset(self.root)


    def __len__(self):
        return self.length


    def compose_set(self, genre):

        composed = {}

        for track in TRACKS:
            random_file = random.choice(self.subset[genre][track])
            composed.update({ track: random_file })

        return composed


    def load_set(self, compose_set):

        wavs = th.zeros(4, self.sample_length)

        for i, track in enumerate(TRACKS):
            filename = compose_set[track]
            start = random.randint(0, 22050*30-self.sample_length)
            wav = read_wav(filename, start, self.sample_length)
            wavs[i,:] = th.tensor(wav)

        return wavs

    def random_mixer(self, wavs):

        mixed = th.zeros(1, self.sample_length)

        volumes = th.softmax(th.rand(4), dim=0)

        for i, volume in enumerate(volumes):
            volume_randamized = wavs[i,:] * volume
            mixed[0] += volume_randamized

        mixed = (mixed - mixed.min()) / (mixed.max() - mixed.min()) * 2.0 - 1.0

        return mixed

    
    def __getitem__(self, _):

        genre = random.choice(GENRES)
        compose_set = self.compose_set(genre)

        wavs = self.load_set(compose_set)
        mix = self.random_mixer(wavs)

        return mix, GENRES.index(genre)


if __name__ == '__main__':
    dataset = Remixer('process/train')
    mix, genre = dataset.__getitem__(None)
    torchaudio.save('test.wav', mix, sample_rate=22050)
    print(mix, genre)
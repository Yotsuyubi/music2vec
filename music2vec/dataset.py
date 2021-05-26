import torch as th
from torch.utils.data import Dataset
import glob
import os
import random
import torchaudio
from .argument import TimeStreach, PitchShift, Mask, ToConstantQ
from scipy.io import wavfile
import numpy as np
from torchvision.utils import save_image


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


def read_wav_and_random_crop(filename, duration=None):

    info = torchaudio.info(filename)
    sample_length = info.num_frames
    offset = random.randint(0, sample_length-duration*2)
    data, _ = torchaudio.load(
        filename, frame_offset=offset,
        num_frames=duration*2 if duration else -1
    )

    return data.detach().numpy()[0,::2]


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
        length=100, sample_length=22050
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

        wavs = np.zeros((4, self.sample_length))

        for i, track in enumerate(TRACKS):
            filename = compose_set[track]
            wav = read_wav_and_random_crop(filename, self.sample_length)
            wavs[i,:] = wav

        return wavs


    def random_mixer(self, wavs):

        mixed = np.zeros(self.sample_length)

        volumes = np.ones(4)

        for i, volume in enumerate(volumes):
            volume_randamized = wavs[i,:] * volume
            mixed += volume_randamized

        mixed = (mixed - mixed.min()) / (mixed.max() - mixed.min()) * 2.0 - 1.0

        return mixed

    
    def __getitem__(self, _):

        genre = random.choice(GENRES)
        compose_set = self.compose_set(genre)

        wavs = self.load_set(compose_set)
        mix = self.random_mixer(wavs)
        mix = TimeStreach()(mix)
        mix = PitchShift()(mix)
        constant_q = ToConstantQ()(mix)

        return constant_q, th.eye(10)[GENRES.index(genre)]



if __name__ == '__main__':
    dataset = Remixer('process/train', sample_length=22050*3)
    mix, genre = dataset.__getitem__(None)
    print(mix.shape, genre)
    for i in range(mix.shape[0]):
        save_image(mix[i], 'test_{}.png'.format(i))
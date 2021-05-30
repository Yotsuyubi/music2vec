from torch.autograd.grad_mode import no_grad
from .model import music2vec
from .argument import ToConstantQ
from .dataset import GENRES
import torchaudio
import math
import torch as th
import argparse



def read_audio_and_split_by_frame(audio_path, duration=30):

    offset = 0

    info = torchaudio.info(audio_path)
    sr = info.sample_rate
    num_samples = info.num_frames
    num_frames = math.ceil( num_samples / (sr * duration) )

    audio = th.zeros((num_frames, 22050*30))

    for i in range(num_frames):

        data, _ = torchaudio.load(
            audio_path, frame_offset=offset,
            num_frames=sr*duration
        )

        if sr != 22050:
            data = torchaudio.transforms.Resample(sr, 22050)(data)

        data = data[0]

        if len(data) != 22050*30:
            data = th.nn.functional.pad(data, (0, 22050*30-len(data)))

        audio[i] = (data - data.min()) / (data.max() - data.min()) * 2.0 - 1.0

        offset += 22050*30

    return audio


def to_spectrum(audio_batch, size=(128, 644)):

    spectrum = th.zeros((audio_batch.shape[0], 4, *size))

    for i, audio in enumerate(audio_batch):

        audio = audio.detach().numpy()
        spectrum[i] = ToConstantQ(size)(audio)

    return spectrum



class Extractor():

    def __init__(self, model_path=None, gpu=False):

        self.model = music2vec(model_path, gpu)


    def __call__(self, audio_path):

        audio = read_audio_and_split_by_frame(audio_path)
        spectrum = to_spectrum(audio)
        features = th.zeros((spectrum.shape[0], 1024))

        for i in range(spectrum.shape[0]):
            batch = th.unsqueeze(spectrum[i], 0)

            with th.no_grad():
                features[i] += self.model.features(batch)[0]

        features_mean = th.mean(features, dim=0)
        features_mean = th.unsqueeze(features_mean, 0)
        genre = self.model.fc(features_mean)
        
        return genre.detach().numpy()[0], features_mean.detach().numpy()[0]



if __name__ == '__main__':

    parser = argparse.ArgumentParser(
        description='music2vec.extraction: music feature extraction.'
    )
    parser.add_argument(
        'audio_path', metavar='<audio_path>', 
        help='audio filename.'
    )

    args = parser.parse_args()
    ext = Extractor()

    genre, features = ext(args.audio_path)

    for i in range(len(GENRES)):
        print("{}: {:.4f}".format(GENRES[i], genre[i]))
    
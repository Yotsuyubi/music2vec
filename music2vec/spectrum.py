import os
import io
import argparse
import progressbar
import glob
import subprocess
import shutil
from multiprocessing import Pool
from time import sleep
from .dataset import Remixer
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
IMAGE_PER_GENRE = 10_000


class CB:
    def __init__(self, bar):
        self.bar = bar
        self.n = 0

    def cb(self, x):
        self.n += 1
        self.update()

    def err(self, x):
        print("error callback args={}".format(x))

    def update(self):
        self.bar.update(self.n)


def save_spectrum(dataset, index, path):

    spectrum, label = dataset.__getitem__(index)
    genre = GENRES[label]

    save_image(spectrum[0], os.path.join(path, genre, '{}.png'.format(index)))

        

def make_process_dir(path):

    for subset in ['train', 'valid']:

        os.mkdir(os.path.join(args.processed_path, subset))

        for genre in GENRES:

            os.mkdir(os.path.join(args.processed_path, subset, genre))


if __name__ == '__main__':

    parser = argparse.ArgumentParser(
        description='music2vec.spectrum: preprocess GTZAN dataset.'
    )
    parser.add_argument(
        'dataset_path', metavar='<dataset_path>', 
        help='dir for dataset.'
    )
    parser.add_argument(
        'processed_path', metavar='<processed_path>', 
        help='dir for save processed dataset.'
    )
    parser.add_argument(
        '-c', '--num_cpu', 
        type=int, help='number of cpus are used. default is 1.', 
        default=1
    )

    args = parser.parse_args()

    make_process_dir(args.processed_path)

    total = int(10 * IMAGE_PER_GENRE * (1+0.3))
    bar = progressbar.ProgressBar(max_value=total)
    callback = CB(bar)

    train_dataset = Remixer(
        os.path.join(args.dataset_path, 'train'),
        length=int(10*IMAGE_PER_GENRE),
        sample_length=22050*3
    )
    valid_dataset = Remixer(
        os.path.join(args.dataset_path, 'valid'),
        length=int(10*IMAGE_PER_GENRE*0.3),
        sample_length=22050*3
    )
    dataset = {
        'train': train_dataset,
        'valid': valid_dataset
    }

    for subset in ['train', 'valid']:

        d = dataset[subset]


        def wrapper(*args, **kwargs):
            return save_spectrum(*args)

        p = Pool(args.num_cpu)
        res = []
        arg = [ (d, i, os.path.join(args.processed_path, subset)) for i in range(len(d)) ]
        
        for a in arg:
            r = p.apply_async(wrapper, a, callback=callback.cb, error_callback=callback.err)
            res.append(r)

        while True:

            callback.update()

            exited_processes = [ r.ready() for r in res ].count(True)
            if exited_processes == len(res):
                break

            sleep(1)

        p.close()

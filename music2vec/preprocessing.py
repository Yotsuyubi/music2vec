import os
import io
import argparse
import progressbar
import glob
import subprocess
import shutil
from multiprocessing import Pool
from time import sleep


SONGS_PER_GENRE = 100
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

        
def separate(wav_path, output_path):

    args = [
        'python3',
        '-m',
        'demucs.separate',
        '-n', 'demucs_quantized',
        '-d', 'cpu',
        wav_path
    ]
    p = subprocess.Popen(args, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
    p.wait()

    if p.returncode == 0:

        base_name = ''.join(os.path.splitext(os.path.basename(wav_path))[:-1])
        separated_path = os.path.join('separated', 'demucs_quantized', base_name)
        
        for track in ['drums', 'bass', 'other', 'vocals']:
            track_path = os.path.join(separated_path, '{}.wav'.format(track))
            dest = os.path.join(output_path, '{}'.format(track), '{}.wav'.format(base_name))
            shutil.move(track_path, dest)

        os.rmdir(separated_path)

        return True

    else:
        return False


def make_subset(path, training=0.5, validation=0.3, test=0.2):

    assert(training+validation+test, 1.0)
    
    num_train = SONGS_PER_GENRE * training
    num_valid = SONGS_PER_GENRE * validation
    # num_test = SONGS_PER_GENRE * test

    train = {}
    valid = {}
    test = {}

    for genre in GENRES:

        genre_train = []
        genre_valid = []
        genre_test = []

        genre_path = os.path.join(path, 'gtzan/genres', genre, '*.wav')
        filenames = glob.glob(genre_path)

        for i in range(SONGS_PER_GENRE):

            filename = filenames[i]

            if i < num_train:
                genre_train.append(filename)
            elif num_train <= i < num_train+num_valid:
                genre_valid.append(filename)
            else:
                genre_test.append(filename)

        train.update({ genre: genre_train })
        valid.update({ genre: genre_valid })
        test.update({ genre: genre_test })

    return {
        'train': train,
        'valid': valid,
        'test': test
    }


def make_process_dir(path, subset_filename):

    for subset in ['train', 'valid', 'test']:

        os.mkdir(os.path.join(args.processed_path, subset))

        for genre in subset_filename[subset]:

            os.mkdir(os.path.join(args.processed_path, subset, genre))

            if subset in ['train', 'valid']:

                for track in ['drums', 'bass', 'other', 'vocals']:
                    os.mkdir(os.path.join(args.processed_path, subset, genre, track))


if __name__ == '__main__':

    parser = argparse.ArgumentParser(
        description='music2vec.preprocessing: preprocess GTZAN dataset.'
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

    subset_filename = make_subset(args.dataset_path)
    make_process_dir(args.processed_path, subset_filename)

    total = 10 * SONGS_PER_GENRE * (0.5+0.3)
    bar = progressbar.ProgressBar(max_value=total)
    callback = CB(bar)

    for subset in ['train', 'valid', 'test']:

        for genre in subset_filename[subset]:

            genre_filenames = subset_filename[subset][genre]

            if subset in ['train', 'valid']:

                def separate_wrapper(*args, **kwargs):
                    return separate(*args)

                p = Pool(args.num_cpu)
                res = []
                arg = [ (wav_path, os.path.join(args.processed_path, subset, genre)) for wav_path in genre_filenames ]
                
                for a in arg:
                    r = p.apply_async(separate_wrapper, a, callback=callback.cb, error_callback=callback.err)
                    res.append(r)

                while True:

                    callback.update()

                    exited_processes = [ r.ready() for r in res ].count(True)
                    if exited_processes == len(res):
                        break

                    sleep(1)

                p.close()

            else:
                for filename in genre_filenames:
                    shutil.copy(filename, os.path.join(args.processed_path, subset, genre))
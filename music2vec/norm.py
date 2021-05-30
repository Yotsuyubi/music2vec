import torch as th
import argparse
from .extraction import Extractor


if __name__ == '__main__':

    parser = argparse.ArgumentParser(
        description='music2vec.norm: measure the norm bitween 2 music.'
    )
    parser.add_argument(
        'audio1_path', metavar='<audio1_path>', 
        help='audio filename.'
    )
    parser.add_argument(
        'audio2_path', metavar='<audio2_path>', 
        help='audio filename.'
    )

    args = parser.parse_args()
    ext = Extractor()

    genre, features1 = ext(args.audio1_path)
    genre, features2 = ext(args.audio2_path)

    norm = th.nn.CosineSimilarity(dim=0)(th.tensor(features1), th.tensor(features2))
    print(norm.item())
    
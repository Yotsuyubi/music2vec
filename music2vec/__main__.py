import argparse


if __name__ == '__main__':

    parser = argparse.ArgumentParser(
        description='music2vec.train: train music2vec model.'
    )
    parser.add_argument(
        'model_path', metavar='<model_path>', 
        help='dir for load/save model.'
    )
    parser.add_argument(
        'processed_root', metavar='<processed_root>', 
        help='root for processed dataset.'
    )
    parser.add_argument(
        '-l', '--learning_rate', 
        type=float, help='value of learning rate. default is 1e-3.', 
        default=1e-3
    )
    parser.add_argument(
        '-L', '--logging', 
        action='store_true', help='enable logging for tensorboard.', 
    )
    parser.add_argument(
        '-b', '--batch_size', 
        type=int, help='value of batch size. default is 128.', 
        default=64
    )
    parser.add_argument(
        '-e', '--num_per_epoch', 
        type=int, help='number of save model per epoch. default is 10.', 
        default=10
    )
    parser.add_argument(
        '-g', '--num_gpus', 
        type=int, help='number of gpu use. to train using cpu, this must be 0. default is 0.', 
        default=0
    )
    args = parser.parse_args()
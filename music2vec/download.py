import urllib.request
import os
import tarfile
import io
import argparse
import progressbar



GTZAN_URL = 'http://opihi.cs.uvic.ca/sound/genres.tar.gz'


def download_and_extract(url, path):

    downloaded_file_path = os.path.join(path, 'gtzan.tar.gz')

    with urllib.request.urlopen(url) as res:
        
        length = int(res.getheader('content-length'))
        chunk_size = 4096

        bar = progressbar.ProgressBar(max_value=length)

        size = 0
        
        with open(downloaded_file_path, 'wb') as file:

            while True:
                data = res.read(chunk_size)
                if not data:
                    break
                file.write(data)
                size += len(data)
                bar.update(size)

    with tarfile.open(downloaded_file_path) as t:
        def is_within_directory(directory, target):
            
            abs_directory = os.path.abspath(directory)
            abs_target = os.path.abspath(target)
        
            prefix = os.path.commonprefix([abs_directory, abs_target])
            
            return prefix == abs_directory
        
        def safe_extract(tar, path=".", members=None, *, numeric_owner=False):
        
            for member in tar.getmembers():
                member_path = os.path.join(path, member.name)
                if not is_within_directory(path, member_path):
                    raise Exception("Attempted Path Traversal in Tar File")
        
            tar.extractall(path, members, numeric_owner=numeric_owner) 
            
        
        safe_extract(t, path)

    os.remove(downloaded_file_path)


if __name__ == '__main__':

    parser = argparse.ArgumentParser(description='music2vec.download: download GTZAN dataset.')
    parser.add_argument('download_path', metavar='<path>', help='dir for save dataset.')

    args = parser.parse_args()

    path = os.path.join(args.download_path, 'gtzan')

    if os.path.exists(path):

        print('{} already exists. Adout.'.format(path))

    else:
    
        os.mkdir(path)

        download_and_extract(GTZAN_URL, path)
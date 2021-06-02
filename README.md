# music2vec

Music genre estimation and embedding.

## Installation

~~~
$ pip3 install git+https://github.com/Yotsuyubi/music2vec#egg=music2vec
~~~

## Usage

~~~
$ python -m music2vec.extraction <audio_file>
~~~

### Example output

~~~
$ python -m music2vec.extraction 514.wav
blues: 0.1740
classical: 0.0066
country: 0.1381
disco: 0.2899
hiphop: 0.0013
jazz: 0.0297
metal: 0.0021
pop: 0.0112
reggae: 0.0859
rock: 0.2612
~~~

### Implementation

~~~python

from music2vec.extraction import Extractor

extractor = Extractor()

genres, features = extractor('path_to_wav.wav')

print(genres.shape, features.shape)

~~~


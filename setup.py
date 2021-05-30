from glob import glob
from os.path import basename
from os.path import splitext

from setuptools import setup
from setuptools import find_packages


def _requires_from_file(filename):
    return open(filename).read().splitlines()


setup(
    name="music2vec",
    version="0.1.0",
    license="MIT",
    description="music2vec",
    author="Yotsuyubi",
    url="https://github.com/Yotsuyubi/music2vec",
    packages=find_packages(exclude=["contrib", "docs", "tests"]),
    include_package_data=True,
    zip_safe=False,
    install_requires=_requires_from_file('requirements.txt')
)
import multiprocessing
import os
import platform
import re
import subprocess
import sys
from distutils.version import LooseVersion

from setuptools import setup

setup(
    name='PRIME',
    version='0.1',
    url='https://github.com/sbuschjaeger/PRIME',
    author='Sebastian Buschjäger',
    author_email='sebastian.buschjaeger@tu-dortmund.de',
    description='Ensemble Learning via (biased) proximal gradient descent implemented in Python and C++.',
    long_description='Ensemble Learning via (biased) proximal gradient descent implemented in Python and C++. ',
    zip_safe=False,
    license='MIT',
    packages=['PRIME'],
    install_requires = [
        "numpy",
        "scikit-learn",
        "pip",
        "setuptools",
        "tqdm",
        "cvxpy"
    ]
)

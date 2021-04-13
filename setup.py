
from setuptools import find_packages    
from setuptools import setup
from os.path import splitext
from glob import glob
from os.path import basename

setup(
    name='PRIME',
    version='0.1',
    url='https://github.com/sbuschjaeger/PRIME',
    author='Sebastian Buschjäger',
    author_email='sebastian.buschjaeger@tu-dortmund.de',
    description='Ensemble Learning via (biased) proximal gradient descent implemented in Python and C++.',
    long_description='Ensemble Learning via (biased) proximal gradient descent implemented in Python and C++. ',
    packages=find_packages('src'),
    package_dir={'': 'src'},
    py_modules=[splitext(basename(path))[0] for path in glob('src/*.py')],
    include_package_data=True,
    zip_safe=False
)

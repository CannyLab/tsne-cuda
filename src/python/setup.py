from distutils.core import setup

from distutils.core import setup

setup(
    name='PyCTSNE',
    version='0.1.0',
    author='David M. Chan',
    author_email='davidchan@berkeley.edu',
    packages=['pyctsne', 'pyctsne.test'],
    scripts=[],
    url='',
    license='LICENSE.txt',
    description='CUDA Implementation of T-SNE with Python bindings',
    long_description=open('README.txt').read(),
    install_requires=[
        "numpy >= 1.14.1",
    ],
)


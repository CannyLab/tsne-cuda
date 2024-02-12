from setuptools import setup

# Get version from version.txt
with open('VERSION.txt', 'r') as vf:
    version = vf.read().strip()

setup(
    name='tsnecuda',
    version=version,
    author='Chan, David M., Huang, Forrest., Rao, Roshan.',
    author_email='davidchan@berkeley.edu',
    packages=['tsnecuda', 'tsnecuda.test'],
    package_data={'tsnecuda': ['libtsnecuda.so', 'tsnecuda.dll']},
    scripts=[],
    url='https://github.com/CannyLab/tsne-cuda',
    license='LICENSE.txt',
    description='CUDA Implementation of T-SNE with Python bindings',
    long_description=open('README.txt').read(),
    install_requires=[
        'numpy>=1.14.1',
        # NOTE: This package also requires FAISS, however we don't explicitly check for it on install.
    ],
    classifiers=[
        'Programming Language :: Python :: 3.6',
        'Programming Language :: Python :: 3.7',
        'Programming Language :: Python :: 3.8',
        'Programming Language :: Python :: 3.9',
        'Programming Language :: Python :: 3.10',
        'Programming Language :: Python :: 3.11',
        'Operating System :: POSIX :: Linux',
        'Intended Audience :: Developers',
        'Intended Audience :: Science/Research',
        'Topic :: Scientific/Engineering :: Artificial Intelligence'
    ],
    keywords=[
        'TSNE',
        'CUDA',
        'Machine Learning',
        'AI'
    ]

)

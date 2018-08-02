from setuptools import setup

setup(
    name='tsnecuda',
    version='0.1.0',
    author='Chan, David M., Huang, Forrest., Rao, Roshan.',
    author_email='davidchan@berkeley.edu',
    packages=['tsnecuda', 'tsnecuda.test'],
    package_data={'tsnecuda': ['libtsnecuda.so']},
    scripts=[],
    url='https://github.com/CannyLab/tsne-cuda',
    license='LICENSE.txt',
    description='CUDA Implementation of T-SNE with Python bindings',
    long_description=open('README.txt').read(),
    install_requires=[
        'numpy >= 1.14.1',    
    ],
    classifiers=[
        'Programming Language :: Python :: 3.6',
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


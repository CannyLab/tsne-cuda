from distutils.core import setup

setup(
    name='tsnecuda',
    version='0.1.0',
    author='Chan, David M., Huang, Forrest., Rao, Roshan.',
    author_email='davidchan@berkeley.edu,forrest_huang@berkeley.edu,roshan_rao@berkeley.edu',
    packages=['tsnecuda', 'tsnecuda.test'],
    package_data={'tsnecuda': ['libtsnecuda.so']},
    scripts=[],
    url='',
    license='LICENSE.txt',
    description='CUDA Implementation of T-SNE with Python bindings',
    long_description=open('README.txt').read(),
    install_requires=[
        "numpy >= 1.14.1",
    ],
)


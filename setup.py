
from setuptools import setup, find_packages

setup(
    name='python_csdl_backend',
    packages=find_packages(),
    #packages=['python_csdl_backend'],
    install_requires=[
        'networkx>=2.8.4',
        'csdl<1',
        'scipy>=1.8.0',
        'numpy>=1.21',
    ],
    version_config=True,
    setup_requires=['setuptools-git-versioning'],
)

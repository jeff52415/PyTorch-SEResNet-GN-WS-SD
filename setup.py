#!/usr/bin/env python
from setuptools import setup, find_packages

NAME = 'PyTorch-SEResNet-GN-WS-SD'
DESCRIPTION = 'PyTorch implementation of SE-ResNet with Group Normalization, Weight Standardization and Stochastic Depth'
URL = 'https://github.com/jeff52415/PyTorch-SEResNet-GN-WS-SD'


def list_reqs(fname='requirements.txt'):
    with open(fname) as fd:
        return fd.read().splitlines()


setup(
    name=NAME,
    description=DESCRIPTION,
    url=URL,
    version='1.0.0',
    include_package_data=True,
    packages=find_packages(),
    install_requires=list_reqs(),
    python_requires=">=3.6",
)

#nsml: pytorch/pytorch
from distutils.core import setup

setup(
    name='ladder_networks',
    version='1.0',
    install_requires=[
        'scikit-learn',
        'tqdm',
        'numpy',
        'opencv-python-headless',
        'torch_optimizer',
        'pytorch-lightning-bolts',
    ]
)

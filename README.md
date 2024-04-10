# EIFNet-MADR
ACMMM2023: Event-based Motion Deblurring with Modality-Aware Decomposition and Recomposition
# Installation

The model is built in PyTorch 1.1.0 and tested on Ubuntu 16.04 environment (Python3.7, CUDA9.0, cuDNN7.5).

For installing, follow these instructions

    conda create -n pytorch1 python=3.7
    conda activate pytorch1
    conda install pytorch=1.1 torchvision=0.3 cudatoolkit=9.0 -c pytorch
    pip install matplotlib scikit-image opencv-python yacs joblib natsort h5py tqdm

Install warmup scheduler

    cd pytorch-gradual-warmup-lr; python setup.py install; cd ..


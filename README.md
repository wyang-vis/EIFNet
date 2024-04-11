# EIFNet-MADR
ACMMM2023: Event-based Motion Deblurring with Modality-Aware Decomposition and Recomposition
# Installation

The model is built in PyTorch 1.8.0 and tested on Ubuntu 16.04 environment (Python3.7, CUDA9.0, cuDNN7.5).

For installing, follow these instructions

    conda create -n pytorch1 python=3.7
    conda activate pytorch1
    conda install pytorch=1.1 torchvision=0.3 cudatoolkit=9.0 -c pytorch
    pip install matplotlib scikit-image opencv-python yacs joblib natsort h5py tqdm

Install warmup scheduler

    cd pytorch-gradual-warmup-lr; python setup.py install; cd ..

# Training and Evaluation
## Train
- Download the [GoPro events train dataset](https://example.com) and [GoPro events test dataset](https://pan.baidu.com/s/1UKV-sPGo9mRf7XJjZDoF7Q) (code: kmaz) to ./Datasets
- Train the model with default arguments by running

  python main_train.py

## Evaluation
- Download the [GoPro events test dataset](https://pan.baidu.com/s/1sM5Y6uWMA5NVp7tmrMXYkg) (code: xd71) to ./Datasets
- Download the  [pretrained model](https://pan.baidu.com/s/193vCnygNkXT_GOq6PhRrhg) (code: svbb) to ./checkpoints/models/EIFNet
- Test the model with default arguments by running

  python main_test.py

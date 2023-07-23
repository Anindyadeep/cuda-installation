# cuda-installation
A simple guide for installing cuda stuff if you have a local gpu for doing deep learning and transformers

Setting up cuda and nvidia utils in Amazon LinuxAmazon Linux based on fedora. In order to support GPU capabilities, we need to install nvidia drivers nvidia-smi and cuda. To get started make sure you have gcc installed. Otherwise we can download it by 

sudo dnf install gcc

After that download the driver from the official channel

wget https://developer.download.nvidia.com/compute/cuda/12.2.0/local_installers/cuda-repo-fedora37-12-2-local-12.2.0_535.54.03-1.x86_64.rpm

Start the installation process by this command

sudo rpm -i cuda-repo-fedora37-12-2-local-12.2.0_535.54.03-1.x86_64.rpm
sudo dnf clean all
sudo dnf -y module install nvidia-driver:latest-dkms

This should install the nvidia-driver and to check whether it is installed or not, we can check this by running nvidia-smi and it should show:

+---------------------------------------------------------------------------------------+
| NVIDIA-SMI 535.54.03              Driver Version: 535.54.03    CUDA Version: 12.2     |
|-----------------------------------------+----------------------+----------------------+
| GPU  Name                 Persistence-M | Bus-Id        Disp.A | Volatile Uncorr. ECC |
| Fan  Temp   Perf          Pwr:Usage/Cap |         Memory-Usage | GPU-Util  Compute M. |
|                                         |                      |               MIG M. |
|=========================================+======================+======================|
|   0  NVIDIA A10G                    Off | 00000000:00:1E.0 Off |                    0 |
|  0%   27C    P0              52W / 300W |      4MiB / 23028MiB |     15%      Default |
|                                         |                      |                  N/A |
+-----------------------------------------+----------------------+----------------------+
                                                                                         
+---------------------------------------------------------------------------------------+
| Processes:                                                                            |
|  GPU   GI   CI        PID   Type   Process name                            GPU Memory |
|        ID   ID                                                             Usage      |
|=======================================================================================|
|  No running processes found                                                           |
+---------------------------------------------------------------------------------------+

But though we might have nvidia drivers installed, now we are going to install CUDA  (Compute Unified Device Architecture) library. It acts as an general interface to support parallel processing capabilities.  

sudo dnf -y install cuda

Also add cuda into the path for CUDA_HOME and env variable PATH

export CUDA_HOME=/usr/local/cuda-<cuda_version>
export PATH="/usr/local/cuda-<cuda_version>/bin:$PATH"

To check if cuda is installed or not we can just type

cuda --version

This should output

nvcc: NVIDIA (R) Cuda compiler driver
Copyright (c) 2005-2023 NVIDIA Corporation
Built on Tue_Jun_13_19:16:58_PDT_2023
Cuda compilation tools, release 12.2, V12.2.91
Build cuda_12.2.r12.2/compiler.32965470_0

NOTE: Some times I saw that, terminal might not find where cuda is located in different terminal session. Hence it is better to add  env variable into the path in bashrc . So in the bashrc just place this line.

export PATH="/usr/local/cuda-<cuda_version>/bin:$PATH"

Installation of cuDNN Library

 cuDNN is a GPU-accelerated library developed by NVIDIA specifically for deep learning tasks. It provides highly optimized implementations of many common deep learning operations, such as convolution, pooling, normalization, and activation functions. 
To install cuDNN first install the cuDNN tar xz file by going to this website https://developer.nvidia.com/cudnn, There select the version of cuda we have here it is 12.x. Once downloaded extract the package. 

tar -xz <cudnn_filename>.xz

Now that you have cuda installed and also have cuDNN package extracted, we need to put some parts of the package to the correct path

sudo cp -P include/cudnn*.h /usr/local/cuda-<cuda_version>/include
sudo cp -P lib64/libcudnn* /usr/local/cuda-<cuda_version>/lib64

Here cuda-version is 12.2
Additionally provide the necessary permission

sudo chmod a+r /usr/local/cuda-<cuda_version>/include/cudnn*.h /usr/local/cuda-<cuda_version>/lib64/libcudnn*

Installing python dependencies 

We install torch and transformers and other libraries specific to transformers fine tuning and optimization

pip install torch
pip install trl
pip install einops
pip install -q datasets
pip install -q -U bitsandbytes
pip install -q -U git+https://github.com/huggingface/transformers.git
pip install -q -U git+https://github.com/huggingface/peft.git
pip install -q -U git+https://github.com/huggingface/accelerate.git
pip install sentencepiece==0.1.97
pip install gradio

Right now I deleted tensorflow because at least for, in the experimentation, we do not need tensorflow backend, rather pytorch is just fine. Also the reason for uninstallation is because some how tensorflow is not detecting cuda. 

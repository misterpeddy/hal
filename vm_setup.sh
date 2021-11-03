#!/bin/bash
# This bash script can be run on a clean debian VM image to install needed system dependencies. 
# There is a fair amount of intricate versioning needed to support the pre-compiled PyTorch ops released by NVIDIA libraries (powering the StyleGAN family of models).
# Unless you really know what you're doing, I suggest running it in its entirety, without changing anything.

set -x

# Install build toolchain and kernel headers
sudo apt-get install libxml2
sudo apt install build-essential
sudo apt-get install linux-headers-`uname -r`
sudo apt-get install wget

# Download NVIDIA CUDA + Driver + NVCC toolchain installer (Interactive installation)
wget https://developer.download.nvidia.com/compute/cuda/11.2.0/local_installers/cuda_11.2.0_460.27.04_linux.run
sudo sh cuda_11.2.0_460.27.04_linux.run

# Set appropriate env vars on shell startup
export PATH=$PATH:"/usr/local/cuda-11.2/bin"
export LD_LIBRARY_PATH=$LD_LIBRARY_PATH:"/usr/local/cuda-11.2/lib64"

# Set up pyenv and python 3.7
sudo apt-get update; sudo apt-get install -yq git make build-essential libssl-dev zlib1g-dev \
libbz2-dev libreadline-dev libsqlite3-dev wget curl llvm \
libncursesw5-dev xz-utils tk-dev libxml2-dev libxmlsec1-dev libffi-dev liblzma-dev

curl https://pyenv.run | bash
sed -i '1s|^|export PYENV_ROOT="$HOME/.pyenv"\n|' ~/.profile
sed -i '1s|^|export PATH="$PYENV_ROOT/bin:$PATH"\n|' ~/.profile
sed -i '1s|^|eval "$(pyenv init --path)"|' ~/.profile

pyenv install 3.7.0
pyenv global 3.7.0

# Install LLVM toolchain needed for numba and librosa
sudo apt install lsb-release wget software-properties-common
wget https://apt.llvm.org/llvm.sh
chmod +x llvm.sh
sudo ./llvm.sh 11
export LLVM_CONFIG=/usr/bin/llvm-config-11
sudo apt-get -y install libsndfile1
sudo apt-get -y install libjpeg-dev zlib1g-dev
sudo apt-get -y install ffmpeg

# Install rclone and mount Google Drive (Interactive installation)
curl https://rclone.org/install.sh | sudo bash
rclone config
sudo apt-get install fuse
nohup rclone mount gdrive: ~/drive &

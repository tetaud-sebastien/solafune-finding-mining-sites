# Solafune finding mining sites

### Download and Install Miniconda

    wget https://repo.anaconda.com/miniconda/Miniconda3-latest-Linux-x86_64.sh
    sh Miniconda3-latest-Linux-x86_64.sh

### Install Nvidia drivers

First we can check our hardware, architecture, and distro with

```Bash
lspci | grep -i nvidia
uname -m && cat /etc/*release
```

which should show several NVIDIA devices, x86_64, and then Ubuntu 22.04 or some such.

Install build dependencies:

```Bash
sudo apt install gcc
sudo apt install linux-headers-$(uname -r)
```

On these systems we additionally had to remove an outdated signing key:
```Bash
sudo apt-key del 7fa2af80
```

Install the latest nvidia keyring:
```Bash
wget https://developer.download.nvidia.com/compute/cuda/repos/ubuntu2204/x86_64/cuda-keyring_1.0-1_all.deb
sudo dpkg -i cuda-keyring_1.0-1_all.deb
```
Finally, update sources, install CUDA, and reboot:
```Bash
sudo apt update
sudo apt install cuda
sudo reboot
```

Install environment:
Install Torch and Cuda
```Bash
pip install -r requirements.txt 
```

## Train model

```Bash
python train.py
```



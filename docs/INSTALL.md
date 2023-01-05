# Installation

In our experiments, we used the following setup:
- Ubuntu 20.04.2 LTS
- Python 3.6.11
- CUDA 10.2
- Pytorch 1.5.1


In order to reimplement our development environment, pelase, follow the below-mentioned instructions.


<hr>


## I. Setup Code Environment

NOTE: In case you have multiple CUDA versions installed, please, make sure to initialise the appropriate system CUDA version before running any command.
```bash
# <xx.x> - CUDA version number
module load cuda-xx.x 
```

1) Setup a conda environment:
    - With Conda [Recommended]:
    ```bash
    # Create a conda environment with dependencies from the environment.yml file
    conda env create --name smvit -f environment.yml
    # Activate the environment
    conda activate smvit
    ```
    
    - With PIP:
    ```bash
    # Create a conda environment
    conda create -n smvit python=3.6.11
    # Activate the environment
    conda activate smvit
    # Install dependencies from the requirements.txt file
    pip install -r requirements.txt

    ```

2) Install the Apex library for mixed-precision training:

    - Via conda [Our choice]:
    ```bash
    ## Both commands are necessary
    # May throw warnings, but it is okay
    conda install -c conda-forge nvidia-apex
    # Answer 'Yes' when numpy package upgrade is inquired
    conda install -c conda-forge nvidia-apex=0.1 
    ```

    - From source [Recommended]:
    ```bash
    git clone https://github.com/NVIDIA/apex
    cd apex
    # May throw unexpected system-specific errors
    pip install -v --disable-pip-version-check --no-cache-dir --global-option="--cpp_ext" --global-option="--cuda_ext" ./
    ```

3) [Additional] In case of runtime errors related to numpy or scikit-learn packages, force downgrade numpy to the '1.15.4' version:
    ```bash
    pip install numpy==1.15.4
    ```


<hr>


## II. Download Pre-trained 3d-party Models

### U2-Net:

_NOTE: This model is required for every experiment (training from scratch, fine-tuning, inference)._

Download the pre-trained model from the U2-Net author's [Google Drive link](https://drive.google.com/file/d/1ao1ovG1Qtx4b7EoskHXmi2E9rp5CHLcZ/view?usp=sharing) ("u2net.pth" - 176.3 MB).
P.S. Original shared link was taken from [here](https://github.com/xuebinqin/U-2-Net#usage-for-salient-object-detection).

The model must be located in:
```
sm-vit/
|–– U2Net/
|   |–– saved_models/
|   |   |–– u2net/
|   |   |   |–– u2net.pth
```


### ViT-B/16:

_NOTE: This model is required for training from scratch only._

Download the pre-trained model from the ViT author's [Google Cloud link](https://console.cloud.google.com/storage/browser/_details/vit_models/imagenet21k/ViT-B_16.npz) ("ViT-B_16.npz" - 393.7 MB).
```bash
wget https://storage.googleapis.com/vit_models/imagenet21k/ViT-B_16.npz
```
P.S. Original shared link was taken from [here](https://github.com/jeonsworld/ViT-pytorch#1-download-pre-trained-model-googles-official-checkpoint).

The model must be located in:
```
sm-vit/
|–– checkpoint/
|   |–– ViT-B_16.npz
```

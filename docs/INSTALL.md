# Installation

In our experiments, we used the following setup:
- Ubuntu 20.04.2 LTS
- Python 3.6
- CUDA 10.2


In order to reimplement our development environment, pelase, follow the below-mentioned instructions.

NOTE: In case you have multiple CUDA versions installed, please, make sure to initialise the appropriate system CUDA version before running any command.
```
# <xx.x> - version number
module load cuda-xx.x 
```

1) Setup a conda environment:
```bash
# Create a conda environment from the environment.yml file:
conda env create --name smvit -f environment.yml
# Activate the environment
conda activate smvit
```

2) Install the Apex library for mixed-precision training:

    - Via conda [Our choice]:
    ```
    ## Both commands are necessary
    # May throw warnings, but it is okay
    conda install -c conda-forge nvidia-apex
    # Answer 'Yes' when numpy package upgrade is inquired
    conda install -c conda-forge nvidia-apex=0.1 
    ```

    - From source [Recommended]:
    ```
    git clone https://github.com/NVIDIA/apex
    cd apex
    # May throw unexpected system-specific errors
    pip install -v --disable-pip-version-check --no-cache-dir --global-option="--cpp_ext" --global-option="--cuda_ext" ./
    ```

3) [Additional] In case of runtime errors related to numpy or scikit-learn packages, force downgrade numpy to the '1.15.4' version:
```
pip install numpy==1.15.4
```

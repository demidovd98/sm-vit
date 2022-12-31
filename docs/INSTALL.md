# Installation

In our experiments, we used the following setup:
- Ubuntu 20.04.2 LTS
- Python 3.6
- CUDA 10.2


In order to reimplement our testing environment, pelase, follow the below-mentioned instructions.

NOTE: In case you have multiple CUDA verstions, please, make sure to initialise the appropriate system CUDA version before running any command.
```
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

    - Via conda [Recommended]:
    ```
    conda install -c conda-forge nvidia-apex=0.1
    ```

    - From source:
    ```
    git clone https://github.com/NVIDIA/apex
    cd apex
    pip install -v --disable-pip-version-check --no-cache-dir --global-option="--cpp_ext" --global-option="--cuda_ext" ./
    ```

# Setup
We use various dependencies that need to be installed. In general, this should happen automatically, but since we're 
limited in our test environments, we cannot guarantee that it will work in all cases. Should you have trouble with these
steps, feel free to issue a bug report.

In addition to the dependencies, we will need to set up a couple of cache directories. The caches may become quite large
since we cache the covariance descriptors for every image (we do not delete the caches automatically in order to 
reuse them, especially for the epochs of SPDNet). The images themselves are not cached.
Note that the evaluation and training procedures are time-consuming.

To properly work, we need write access to the `config.json` file in the project's root directory if you are on a new computer.
The following directories are needed:
* __Dataset directory__: <br/> The root directory of the datasets. We look for a directory `medmnist` in that directory. If a 
  medmnist dataset is not available yet, we'll download it as soon as needed.
* __Result directory__: <br/> This is the directory in which the results will be placed.
* __Model directory__: <br/> This is the directory in which our SPDNets will be stored.
* __Cache directory__: <br/> Here, the large cache files will be stored.


The setup involves the following steps:

1. Install Python 3.11 or later ([Python Downloads](https://www.python.org/downloads/))
2. Create a new pip-environment (conda should also work if pip is installed)
3. Run `main.py` and follow the instructions. It should automatically install all requirements. In case it fails, 
   please use the manual setup below

## Manual setup
You only need to follow these steps in case the automatic setup should fail:

1. Install PyTorch (torch at least v2.0 and torchvision). We used
   [torch 2.2.1 and torchvision 0.17.1 for cuda 12.1](https://pytorch.org/get-started/previous-versions/#v221).<br/> 
``pip install torch==2.2.1 torchvision==0.17.1 torchaudio==2.2.1 --index-url https://download.pytorch.org/whl/cu121``

2. Install the `requirements.txt` file<br/>
``pip install -r requirements.txt``

3. Install medsam from its [github-repository](https://github.com/bowang-lab/MedSAM.git)<br/>
``pip install git+https://github.com/bowang-lab/MedSAM.git``


# Overview
The `main.py` performs all the necessary training steps, the `main_check.py` is intended to give a short test run 
through all steps to check whether everything is set up correctly.

The `handle_*` methods in the [main.py](src/main.py) perform the respective operations for the traditional methods or 
SPDNet and the handcrafted or general vision encoder features, respectively.
The models themselves are implemented in the `experiments` directory.

We use PyTorch lightning to wrap our model training. You can add `wandb` and initialize it (just initialize it in the 
[main.py](src/main.py), or the respective spdnet-based methods in it). If wandb is not initialized, the training 
progress gets tracked via tensorboard.

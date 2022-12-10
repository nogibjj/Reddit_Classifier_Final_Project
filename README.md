[![CI](https://github.com/nogibjj/mlops-template/actions/workflows/cicd.yml/badge.svg?branch=GPU)](https://github.com/nogibjj/mlops-template/actions/workflows/cicd.yml)
[![Codespaces Prebuilds](https://github.com/nogibjj/mlops-template/actions/workflows/codespaces/create_codespaces_prebuilds/badge.svg?branch=GPU)](https://github.com/nogibjj/mlops-template/actions/workflows/codespaces/create_codespaces_prebuilds)

## Template for MLOPs projects with GPU

1. First thing to do on launch is to open a new shell and verify virtualenv is sourced.

Things included are:

* `Makefile`

* `Pytest`

* `pandas`

* `Pylint`

* `Dockerfile`

* `GitHub copilot`

* `jupyter` and `ipython` 

* Most common Python libraries for ML/DL and Hugging Face

* `githubactions` 

## Two fun tools to explore:

* Zero-shot classification:  ./hugging-face/zero_shot_classification.py classify
* Yake for candidate label creation: ./utils/kw_extract.py


### Verify GPU works

The following examples test out the GPU

* run pytorch training test: `python utils/quickstart_pytorch.py`
* run pytorch CUDA test: `python utils/verify_cuda_pytorch.py`
* run tensorflow training test: `python utils/quickstart_tf2.py`
* run nvidia monitoring test: `nvidia-smi -l 1` it should show a GPU

Additionally, this workspace is setup to fine-tune Hugging Face

![fine-tune](https://user-images.githubusercontent.com/58792/195709866-121f994e-3531-493b-99af-c3266c4e28ea.jpg)


`python hf_fine_tune_hello_world.py` 

### Login to Hugging Face 
* In terminal, run `huggingface-cli login`
* Go to Hugging Face settings to get access token and type it in
* In Hugging Face, create a new model repo 
* In terminal, run `git clone https://huggingface.co/michellejieli/reddit
_classifier` 
* cd into repo `cd reddit_classifier/1`
* add model `git add.`






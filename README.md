[![CI](https://github.com/nogibjj/mlops-template/actions/workflows/cicd.yml/badge.svg?branch=GPU)](https://github.com/nogibjj/mlops-template/actions/workflows/cicd.yml)
[![Codespaces Prebuilds](https://github.com/nogibjj/mlops-template/actions/workflows/codespaces/create_codespaces_prebuilds/badge.svg?branch=GPU)](https://github.com/nogibjj/mlops-template/actions/workflows/codespaces/create_codespaces_prebuilds)


### Get Reddit data
* Data pulled in notebook `reddit_data/reddit_new.ipynb`
### Verify GPU works
* Run pytorch training test: `python utils/quickstart_pytorch.py`
* Run pytorch CUDA test: `python utils/verify_cuda_pytorch.py`
* Run tensorflow training test: `python utils/quickstart_tf2.py`
* Run nvidia monitoring test: `nvidia-smi -l 1`

### Finetune text classifier model and upload to Hugging Face 
* In terminal, run `huggingface-cli login`
* Run `python fine_tune_berft.py` to finetune the model on Reddit data 
* Run `rename_labels.py` to change the output labels of the classifier
* Check out the fine-tuned model [here](https://huggingface.co/michellejieli/inappropriate_text_classifier)









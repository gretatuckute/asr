# ASR models from HuggingFace

Repo for extracting model activations of HuggingFace models. Currently supports:

- wav2vec (`get_wav2vec_activations.py`)
- S2T (`get_S2T_activations.py`)

To extract permuted model activations, set the rand_netw flag = True in the beginning of each respective script.

Note that the environment file for wav2vec is named environment_wav2vecUPDATED_20220830.yml (which runs fine for the activations) -- however, the activations used for the first submission of the paper were extracted using a slightly different version of this environment (specifically, using transformers=4.5.0, torch=1.8.1 in Python 3.8.8). 

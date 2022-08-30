# ASR models from HuggingFace

Repo for extracting model activations of HuggingFace models. Currently supports:

- wav2vec (`get_wav2vec_activations.py`)
- S2T (`get_S2T_activations.py`)

To extract permuted model activations, set the rand_netw flag = True in the beginning of each respective script.

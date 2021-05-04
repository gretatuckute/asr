from transformers import Wav2Vec2Tokenizer, Wav2Vec2ForCTC, Wav2Vec2CTCTokenizer
import soundfile as sf
import torch
import matplotlib.pyplot as plt
from scipy import signal
import numpy as np
from scipy.io import wavfile
import librosa
import pickle
from os import listdir
from os.path import isfile, join
from pathlib import Path
import os
import random

random.seed(0)
torch.manual_seed(0)
torch.cuda.manual_seed(0)
np.random.seed(0)

DATADIR = '/Users/gt/Documents/GitHub/aud-dnn/data/stimuli/165_natural_sounds_16kHz/'
RESULTDIR = '/Users/gt/Documents/GitHub/aud-dnn/aud_dnn/model-actv/wav2vec/'
RESULTDIR = (Path(RESULTDIR))

rand_netw = True

if not (Path(RESULTDIR)).exists():
	os.makedirs((Path(RESULTDIR)))

if __name__ == '__main__':
	# WITH LM HEAD
	model = Wav2Vec2ForCTC.from_pretrained("facebook/wav2vec2-base-960h", output_hidden_states=True, return_dict=True)
	tokenizer = Wav2Vec2Tokenizer.from_pretrained("facebook/wav2vec2-base-960h")
	
	if rand_netw:
		state_dict = model.state_dict()
		state_dict_rand = {}
		print('OBS! RANDOM NETWORK!')
		
		## The following code was used to generate indices for random permutation ##
		# d_rand_idx = {}  # create dict for storing the indices for random permutation
		# for k, v in state_dict.items():
		#     w = state_dict[k]
		#     idx = torch.randperm(w.nelement())  # create random indices across all dimensions
		#     d_rand_idx[k] = idx
		#
		# with open(os.path.join(os.getcwd(), 'wav2vec_randnetw_indices.pkl'), 'wb') as f:
		#     pickle.dump(d_rand_idx, f)
		d_rand_idx = pickle.load(open(os.path.join(os.getcwd(), 'wav2vec_randnetw_indices.pkl'), 'rb'))
		
		for k, v in state_dict.items():
			w = state_dict[k]
			# Load random indices
			print(f'________ Loading random indices from permuted architecture for {k} ________')
			idx = d_rand_idx[k]
			rand_w = w.view(-1)[idx].view(w.size())  # permute, and reshape back to original shape
			state_dict_rand[k] = rand_w
	
		# force to use this other state dict
		model = Wav2Vec2ForCTC.from_pretrained("facebook/wav2vec2-base-960h", output_hidden_states=True,
											   return_dict=True, state_dict=state_dict_rand)
	
	files = [f for f in listdir(DATADIR) if isfile(join(DATADIR, f))]
	wav_files = [f for f in files if f.endswith('wav')]
	
	for file in wav_files:
		model.eval()
		# get identifier (sound file name)
		identifier = file[:-4]
		
		audio_input, _ = librosa.load(join(DATADIR, file), sr=16000)
		input_values = tokenizer(audio_input, return_tensors="pt", padding=True).input_values
		# input_values = tokenizer(join(DATADIR, file), return_tensors="pt", padding=True).input_ids
		
		hidden = model(input_values)['hidden_states']
		logits = model(input_values).logits
	
		detached_activations = {}
	
		for i, l in enumerate(hidden):
			if i == 0:
				print(f'Layer {i}, shape: {np.shape(l)}')
			
			# squeeze batch
			layer = l.squeeze()
			# avg over time
			layer_avg = layer.mean(axis=0).detach().numpy()
			if i == 0: # The embedding
				detached_activations[f'Embedding'] = layer_avg
			else:
				detached_activations[f'Encoder_{i}'] = layer_avg
		
		# deal with logits
		logits_layer = logits.squeeze()
		detached_activations['Logits'] = logits_layer.mean(axis=0).detach().numpy()
		
		# save
		if rand_netw:
			filename = os.path.join(RESULTDIR, f'{identifier}_activations_randnetw.pkl')
		else:
			filename = os.path.join(RESULTDIR, f'{identifier}_activations.pkl')
		
		with open(filename, 'wb') as f:
			pickle.dump(detached_activations, f)
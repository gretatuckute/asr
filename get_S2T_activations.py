
import matplotlib.pyplot as plt
import numpy as np
import librosa
from os import listdir
from os.path import isfile, join
from pathlib import Path
import os
import pandas as pd
import pickle
from transformers import Speech2TextProcessor, Speech2TextForConditionalGeneration
import random
import torch

DATADIR = '/Users/gt/Documents/GitHub/aud-dnn/data/stimuli/165_natural_sounds_16kHz/'
RESULTDIR = '/Users/gt/Documents/GitHub/aud-dnn/aud_dnn/model-actv/S2T/'
RESULTDIR = (Path(RESULTDIR))
randnetw = False

random.seed(0)
torch.manual_seed(0)
torch.cuda.manual_seed(0)
np.random.seed(0)

if not (Path(RESULTDIR)).exists():
	os.makedirs((Path(RESULTDIR)))

if __name__ == '__main__':
	
	model = Speech2TextForConditionalGeneration.from_pretrained("facebook/s2t-large-librispeech-asr",
																output_hidden_states=True, return_dict=True)
	processor = Speech2TextProcessor.from_pretrained("facebook/s2t-large-librispeech-asr")
	
	if randnetw:
		state_dict = model.state_dict()
		state_dict_rand = {}
		print('OBS! RANDOM NETWORK!')
		
		## The following code was used to generate indices for random permutation ##
		if not Path(os.path.join(os.getcwd(), 'S2T_randnetw_indices.pkl')).exists():  # random network indices not generated
			d_rand_idx = {}  # create dict for storing the indices for random permutation
			for k, v in state_dict.items():
				w = state_dict[k]
				idx = torch.randperm(w.nelement())  # create random indices across all dimensions
				d_rand_idx[k] = idx

			with open(os.path.join(os.getcwd(), 'S2T_randnetw_indices.pkl'), 'wb') as f:
				pickle.dump(d_rand_idx, f)
				
		else:
			d_rand_idx = pickle.load(open(os.path.join(os.getcwd(), 'S2T_randnetw_indices.pkl'), 'rb'))
		
		for k, w in state_dict.items():
			# Load random indices
			print(f'________ Loading random indices from permuted architecture for {k} ________')
			idx = d_rand_idx[k]
			rand_w = w.view(-1)[idx].view(
				w.size())  # permute using the stored indices, and reshape back to original shape
			state_dict_rand[k] = rand_w
		
		# force to use this other state dict
		model = Speech2TextForConditionalGeneration.from_pretrained("facebook/s2t-large-librispeech-asr",
																output_hidden_states=True, return_dict=True, state_dict=state_dict_rand)
	
	## LOOP OVER FILES ##
	files = [f for f in listdir(DATADIR) if isfile(join(DATADIR, f))]
	wav_files = [f for f in files if f.endswith('wav')]
	
	for file in wav_files:
		model.eval()
		# get identifier (sound file name)
		identifier = file[:-4]
		
		audio_input, _ = librosa.load(join(DATADIR, file), sr=16000)

		input_features = processor(
			audio_input,
			sampling_rate=16_000,
			return_tensors="pt"
		).input_features  # Batch size 1
		generated_ids = model.generate(input_ids=input_features, output_hidden_states=True, return_dict_in_generate=True)
		
		# transcription = processor.batch_decode(generated_ids)
		
		detached_activations = {}
		# spect = input_features.squeeze().detach().numpy()
		# detached_activations['Spect'] = np.mean(spect, axis=0)
		
		for i, l in enumerate(generated_ids.encoder_hidden_states):
			# squeeze batch
			layer = l.squeeze()
			
			# avg over time
			layer_avg = layer.mean(axis=0).detach().numpy()
			
			if i == 0:
				detached_activations['Embedding'] = layer_avg
			else:
				detached_activations[f'Encoder_{i}'] = layer_avg
		
		# save
		if randnetw:
			filename = os.path.join(RESULTDIR, f'{identifier}_activations_randnetw.pkl')
		else:
			filename = os.path.join(RESULTDIR, f'{identifier}_activations.pkl')
		
		with open(filename, 'wb') as f:
			pickle.dump(detached_activations, f)

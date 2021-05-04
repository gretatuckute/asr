from transformers import AutoTokenizer, AutoModel
from transformers import Wav2Vec2CTCTokenizer
from transformers import Wav2Vec2Processor
# from datasets import load_dataset
# import soundfile as sf
# import torch
import matplotlib.pyplot as plt
# from scipy import signal
import numpy as np
from scipy.io import wavfile
import librosa
import pickle
from os import listdir
from os.path import isfile, join
from pathlib import Path
import os

DATADIR = '/Users/gt/Documents/GitHub/control-neural/data/stimuli/165_natural_sounds_16kHz/'
RESULTDIR = '/Users/gt/Documents/GitHub/control-neural/control_neural/model-actv-control/XLSR/'

RESULTDIR = (Path(RESULTDIR))

if not (Path(RESULTDIR)).exists():
	os.makedirs((Path(RESULTDIR)))

if __name__ == '__main__':
	tokenizer = AutoTokenizer.from_pretrained("facebook/wav2vec2-large-xlsr-53")
	# tokenizer = Wav2Vec2CTCTokenizer("facebook/wav2vec2-large-xlsr-53")
	model = AutoModel.from_pretrained("facebook/wav2vec2-large-xlsr-53", output_hidden_states=True, return_dict=True)
	
	files = [f for f in listdir(DATADIR) if isfile(join(DATADIR, f))]
	wav_files = [f for f in files if f.endswith('wav')]
	
	for file in wav_files:
		# get identifier (sound file name)
		identifier = file[:-4]
		
		audio_input, _ = librosa.load(join(DATADIR, file), sr=16000)
		input_values = tokenizer(audio_input, return_tensors="pt", padding=True).input_values
		
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
			detached_activations[f'Layer_{i+1}'] = layer_avg
		
		# deal with logits
		logits_layer = logits.squeeze()
		detached_activations['Logits'] = logits_layer.mean(axis=0).detach().numpy()
		
		# save
		filename = os.path.join(RESULTDIR, f'{identifier}_activations.pkl')
		
		with open(filename, 'wb') as f:
			pickle.dump(detached_activations, f)
		
	
		# predicted_ids = torch.argmax(logits, dim=-1)
		# transcription = tokenizer.batch_decode(predicted_ids)[0]
	

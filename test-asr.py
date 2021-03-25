from transformers import Wav2Vec2Tokenizer, Wav2Vec2ForCTC
from datasets import load_dataset
import soundfile as sf
import torch
import matplotlib.pyplot as plt
from scipy import signal
import numpy as np
import json
from scipy.io import wavfile
import librosa
from transformers import Wav2Vec2Model, Wav2Vec2Config
import pickle

# define function to read in sound file
def map_to_array(batch):
	speech, _ = sf.read(batch["file"])
	batch["speech"] = speech
	return batch

if __name__ == '__main__':
	# load model and tokenizer
	filename = "test_audio2.wav"
	tokenizer = Wav2Vec2Tokenizer.from_pretrained("facebook/wav2vec2-base-960h")
	
	# look into vocab:
	# tokenizer.save_vocabulary('/Users/gt/Documents/GitHub/asr/')
	# with open('/Users/gt/Documents/GitHub/asr/vocab.json') as f:
	# 	data = json.load(f)
	#
	# np.sort(list(data))
	
	######### RESAMPLING ###########
	# load audio
	# audio_input, _ = sf.read(filename)
	
	# data = wavfile.read(filename)
	# framerate = data[0]
	# sounddata = data[1]
	# time = np.arange(0, len(sounddata)) / framerate
	# print('Sample rate:', framerate, 'Hz')
	# print('Total time:', len(sounddata) / framerate, 's')
	#
	# # Resample to 16 kHz
	# audio_input, _ = librosa.load(filename, sr=16000)
	#
	# plt.plot(audio_input)
	# plt.show()
	#
	# Resampling for DS2
	# Resample to 16 kHz
	# import soundfile as sf
	# sf.write('/Users/gt/Documents/GitHub/deepspeech.pytorch/data/inference/test_audio_16khz.wav', audio_input, 16000)
	#
	# audio_input, _ = librosa.load('/Users/gt/Documents/GitHub/deepspeech.pytorch/data/inference/test_audio2.wav', sr=16000)
	
	# transcribe
	# input_values = tokenizer(audio_input, return_tensors="pt", padding="longest").input_values
	
	
	
	
	# load dummy dataset and read soundfiles
	ds = load_dataset("patrickvonplaten/librispeech_asr_dummy", "clean", split="validation")
	ds = ds.map(map_to_array)
	
	# tokenize -- two sentences
	input_values = tokenizer(ds["speech"][:2], return_tensors="pt", padding="longest").input_values  # Batch size 1
	
	# BASE MODEL
	# Initializing a model from the facebook/wav2vec2-base-960h style configuration
	# configuration = Wav2Vec2Config()
	# model = Wav2Vec2Model(configuration)
	model = Wav2Vec2Model.from_pretrained("facebook/wav2vec2-base-960h", output_hidden_states=True)
	hidden_states = model(input_values).last_hidden_state
	
	# WITH LM HEAD
	model = Wav2Vec2ForCTC.from_pretrained("facebook/wav2vec2-base-960h", output_hidden_states=True)
	
	
	logits = model(input_values).logits
	predicted_ids = torch.argmax(logits, dim=-1)
	transcription = tokenizer.batch_decode(predicted_ids)[0]
	
	# look into logits
	e = np.exp(logits[0][0, :].detach().numpy())
	torch.log(logits[1][200, :]).detach().numpy()
	np.sum(e)
	
	torch.exp(logits[0][0, :]).sum()
	
	# try one sentence
	# tokenize --
	model_name = "facebook/wav2vec2-large-xlsr-53-french"
	
	model = Wav2Vec2ForCTC.from_pretrained("facebook/wav2vec2-base-960h", output_hidden_states=True,
										   output_attentions=True, return_dict=True)
	
	## test different language
	from transformers import AutoTokenizer, Wav2Vec2ForCTC
	
	tokenizer = AutoTokenizer.from_pretrained("facebook/wav2vec2-large-xlsr-53-french")
	model = Wav2Vec2ForCTC.from_pretrained("facebook/wav2vec2-large-xlsr-53-french", output_hidden_states=True,
										   output_attentions=True, return_dict=True)
	
	model(input_values)['hidden_states'][0] # works :)
	
	# tokenize --
	input_values = tokenizer(ds["speech"][:1], return_tensors="pt", padding="longest").input_values  # Batch size 1
	
	# retrieve logits
	logits = model(input_values).logits
	
	logits_sq = logits.squeeze()
	e1 = torch.exp(logits_sq[0]).detach().numpy()
	e1.sum()
	# retrieve logits
	logits = model(input_values).logits
	
	# take argmax and decode
	predicted_ids = torch.argmax(logits, dim=-1)
	transcription = tokenizer.batch_decode(predicted_ids)

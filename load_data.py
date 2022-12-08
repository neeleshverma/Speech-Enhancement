import numpy as np
import os, csv
from scipy.io import wavfile
from tqdm import tqdm


# Domestic Audio Tagging Data
def loadDATData(DATFolder):
	sets = ['train', 'val']

	all_csv = {}
	all_csv[sets[0]] = DATFolder + "/development_chunks_refined.csv"
	all_csv[sets[1]] = DATFolder + "/evaluation_chunks_refined.csv"

	labels = {}
	names = {}
	datasets = {}

	print("\n", flush=True)
	print("Loading DAT data......", flush=True)
	# print("\n")

	for x in sets:
		labels[x] = []
		names[x] = []
		datasets[x] = []

		files = []
		l = []

		with open(all_csv[x]) as csvfile1:
			metareader1 = csv.reader(csvfile1, delimiter=',', quotechar='|')

			for row in metareader1:
				files.append(row[1] + ".wav")

				with open('%s/%s.csv' % (DATFolder, row[1])) as csvfile2:
					metareader2 = csv.reader(csvfile2, delimiter=',', quotechar='|')

					for row2 in metareader2:
						if row2[0] == 'majorityvote':
							l.append(row2[1])


		for i in tqdm(range(len(files))):
			file_name = files[i]
			fs, input_audio = wavfile.read(DATFolder + "/" + file_name)

			if not (fs == 16000):
				raise ValueError('Sample frequency is not 16kHz')

			audio_shape = np.shape(input_audio)

			if len(audio_shape) > 1 and audio_shape[1] > 1:
				for j in range(audio_shape[1]):
					input_data = np.reshape(input_audio[:, j], [1, 1, 1, audio_shape[0]])
					labels[x].append(l[i])
					names[x].append(file_name)
					datasets[x].append(input_data)

			else:
				input_data = np.reshape(input_audio, [1, 1, 1, audio_shape[0]])
				labels[x].append(l[i])
				names[x].append(file_name)
				datasets[x].append(input_data)


	labels_list = []

	for label in labels[sets[0]]:
		for ch in list(label):
			if not (label == 'S'):
				labels_list.append(ch)

	labels_list = list(set(labels_list))

	return labels, names, datasets, labels_list


def loadASCData(ASCFolder):
	sets = ['train', 'val']
	folders = {}

	for x in sets:
		folders[x] = ASCFolder + "/" + x + "set"

	labels = {}
	names = {}
	datasets = {}

	print("\n", flush=True)
	print("Loading ASC data.....", flush=True)
	# print("\n")

	for x in sets:
		folder_name = folders[x]
		labels[x] = []
		names[x] = []
		datasets[x] = []

		files = []
		l = []

		with open("%s/meta.txt" % folder_name) as csvfile:
			metareader = csv.reader(csvfile, delimiter='\t', quotechar='|')

			for row in metareader:
				files.append(row[0][6:])
				l.append(row[1])

		for i in tqdm(range(len(files))):
			file_name = files[i]
			fs, input_audio = wavfile.read(folder_name + "/" + file_name)

			if not (fs == 16000):
				raise ValueError('Sample frequency is not 16kHz')

			audio_shape = np.shape(input_audio)

			if len(audio_shape) > 1 and audio_shape[1] > 1:
				for j in range(audio_shape[1]):
					input_data = np.reshape(input_audio[:, j], [1, 1, 1, audio_shape[0]])
					labels[x].append(l[i])
					names[x].append(file_name)
					datasets[x].append(input_data)

			else:
				input_data = np.reshape(input_audio, [1, 1, 1, audio_shape[0]])
				labels[x].append(l[i])
				names[x].append(file_name)
				datasets[x].append(input_data)

		labels_list = list(set(labels[sets[0]]))

	return labels, names, datasets, labels_list


def loadEntireDataList(dataFolder = 'dataset'):

	sets = ['train', 'val']
	dataset = {}
	data_folder_path = {}

	for x in sets:
		dataset[x] = {}
		data_folder_path[x] = dataFolder + "/" + x + "set"

	# print("Loading Files ...", flush=True)
	print("\n", flush=True)
	print("Loading Entire Data Files Name .....", flush=True)

	for x in sets:
		folder_name = data_folder_path[x]

		dataset[x]['innames'] = []
		dataset[x]['outnames'] = []
		dataset[x]['shortnames'] = []

		files_list = os.listdir("%s_noisy" % (folder_name))
		files_list = [f for f in files_list if f.endswith(".wav")]

		for i in tqdm(files_list):
			dataset[x]['innames'].append("%s_noisy/%s" % (folder_name, i))
			dataset[x]['outnames'].append("%s_clean/%s" % (folder_name, i))
			dataset[x]['shortnames'].append("%s" % (i))


	return dataset['train'], dataset['val']


def loadEntireData(trainingSet, validationSet):

	print("\n", flush=True)
	print("Loading Entire Data Audio .....", flush=True)

	for dataset in [trainingSet, validationSet]:

		dataset['inaudio'] = [None]*len(dataset['innames'])
		dataset['outaudio'] = [None]*len(dataset['outnames'])

		for i in tqdm(range(len(dataset['innames']))):

			if dataset['inaudio'][i] is None:
				fs, input_data = wavfile.read(dataset['innames'][i])
				fs, output_data = wavfile.read(dataset['outnames'][i])

				input_data = np.reshape(input_data, [-1,1])
				output_data = np.reshape(output_data, [-1,1])

				audio_shape = np.shape(input_data)

				input_data = np.reshape(input_data, [1, audio_shape[1], 1, audio_shape[0]])
				output_data = np.reshape(output_data, [1, audio_shape[1], 1, audio_shape[0]])

				dataset['inaudio'][i] = np.float32(input_data)
				dataset['outaudio'][i] = np.float32(output_data)


	return trainingSet, validationSet


def loadNoisyDataList(validationFolder = ''):

	sets = ['val']
	dataset = {'val' : {}}
	datafolders = {'val' : validationFolder}

	# print("Loading Noisy data set for validation")
	print("\n", flush=True)
	print("Loading Noisy Data Files Name .....", flush=True)

	for x in sets:
		folder_name = datafolders[x]

		dataset[x]['innames'] = []
		dataset[x]['shortnames'] = []

		files_list = os.listdir("%s" % (folder_name))
		files_list = [f for f in files_list if f.endswith(".wav")]

		for i in tqdm(files_list):
			dataset[x]['innames'].append("%s/%s" % (folder_name, i))
			dataset[x]['shortnames'].append("%s" % (i))

	return dataset['val']


def loadNoisyData(validationSet):

	print("\n", flush=True)
	print("Loading Validation Data Audio .....", flush=True)

	for dataset in [validationSet]:

		dataset['inaudio'] = [None] * len(dataset['innames'])

		for i in tqdm(range(len(dataset['innames']))):

			if dataset['inaudio'][i] is None:
				fs, input_data = wavfile.read(dataset['innames'][i])

				input_data = np.reshape(input_data, [-1,1])
				audio_shape = np.shape(input_data)

				input_data = np.reshape(input_data, [1, audio_shape[1], 1, audio_shape[0]])

				dataset['inaudio'][i] = np.float32(input_data)

	return validationSet


# def loadNoisyDataList()


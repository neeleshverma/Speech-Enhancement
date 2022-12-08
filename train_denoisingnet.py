from models import *
from load_data import *

import sys, getopt

# Denoising Network Parameters
DN_LAYERS = 13
DN_CHANNELS = 64
DN_LOSS_LAYERS = 6
DN_NORM_TYPE = "ADN" # Adaptive Batch Norm
DN_LOSS_TYPE = "FL" # Feature Loss

# Feature Loss Network
LOSS_LAYERS = 14
LOSS_BASE_CHANNELS = 32
LOSS_BASE_CHANNELS_INCREMENT = 5
LOSS_NORM_TYPE = "SBN" # Stochastic Batch Norm

SET_WEIGHT_EPOCH = 10
SAVE_MODEL_EPOCH = 10

# logs_file = open("logfile.txt", 'w+')

# Command line options
data_folder = "dataset"
loss_model_path = "models/loss_model.pth"
output_folder = "."

try:
	opts, args = getopt.getopt(sys.argv[1:], "hd:l:o:", ["datafolder=,lossfolder=,outfolder="])
except getopt.GetoptError:
	print('Usage train_denoisingnet.py -d dataFolder -l lossModelPath -s outputFolder', flush=True)
	sys.exit(2)

for opt, arg in opts:
	if opt == '-h':
		print('Usage train_denoisingnet.py -d dataFolder -l lossModelPath -s outputFolder', flush=True)
	elif opt in ("-d", "--dataFolder"):
		data_folder = arg
	elif opt in ("-l", "--lossModelPath"):
		loss_model_path = arg
	elif opt in ("-s", "--outputFolder"):
		output_folder = arg

# print 'Dataset folder is ' + 
print('Dataset folder is "' + data_folder + '/"', flush=True)
print('Loss model path is "' + loss_model_path, flush=True)
print('Folder to save model is "' + output_folder + '/"', flush=True)

# Loading of data
training_set, validation_set = loadEntireDataList(dataFolder = data_folder)
training_set, validation_set = loadEntireData(training_set, validation_set)

if DN_LOSS_TYPE == "FL":
	loss_weights = np.ones(DN_LOSS_LAYERS)
else:
	loss_weights = []

if DN_LOSS_TYPE == "FL":
	training_loss = np.zeros((len(training_set["innames"]), DN_LOSS_LAYERS+1))
	validation_loss = np.zeros((len(validation_set["innames"]), DN_LOSS_LAYERS+1))
else:
	training_loss = np.zeros((len(training_set["innames"]), 1))
	validation_loss = np.zeros((len(validation_set["innames"]), 1))



#################################################### MODELS INITIALIZATION #################################################################
total_epochs = 320

denoising_model = DenoisingNet().cuda()

feature_loss_model = FeatureLossNet().cuda()
feature_loss_model.load_state_dict(torch.load(loss_model_path))

learning_rate = 1e-4
optimizer = torch.optim.Adam(denoising_model.parameters(), lr=learning_rate)

l1_loss = nn.L1Loss()
l2_loss = nn.MSELoss()


######################################################## LOSS CALCULATION ##############################################################
def featureLoss(actualOutput, modelOutput, lossWeights):
	width_1 = actualOutput.shape[3]
	width_2 = modelOutput.shape[3]

	if width_1 > width_2:
		modelOutput = F.pad(modelOutput, (0, width_1 - width_2, 0, 0, 0, 0, 0, 0))

	else:
		actualOutput = F.pad(actualOutput, (0, width_2 - width_1, 0, 0, 0, 0, 0, 0))

	_ , model_output_vectors = feature_loss_model(modelOutput, 0)
	_ , actual_output_vectors = feature_loss_model(actualOutput, 0)

	loss_vectors = [0]
	# total_loss = 0

	for i in range(DN_LOSS_LAYERS):
		loss_vectors.append(l1_loss(model_output_vectors[i], actual_output_vectors[i]) / lossWeights[i])

	for i in range(1,DN_LOSS_LAYERS+1):
		loss_vectors[0] += loss_vectors[i]

	return loss_vectors


for epoch in range(1, total_epochs+1):

	print("\n", flush=True)
	print("################################################## Epoch " + str(epoch) + " started ########################################################", flush=True)

	training_ids = np.random.permutation(len(training_set["innames"]))

	############################################### Training Epoch ########################################################

	for id in range(0, len(training_ids)):

		index = training_ids[id]
		input_data = training_set["inaudio"][index]
		output_data = training_set["outaudio"][index]

		input_data = torch.tensor(input_data).cuda()
		output_data = torch.tensor(output_data).cuda()

		enhanced_data = denoising_model(input_data)


		# print(output_data.shape)
		# print(enhanced_data.shape)
		# exit(0)

		loss = []

		if DN_LOSS_TYPE == "L1":
			loss = l1_loss(output_data, enhanced_data)

		elif DN_LOSS_TYPE == "L2":
			loss = l2_loss(output_data, enhanced_data)

		else:
			loss = featureLoss(output_data, enhanced_data, loss_weights)


		optimizer.zero_grad()
		loss[0].backward()
		optimizer.step()


		training_loss[id][0] = loss[0]

		if DN_LOSS_TYPE == "FL":
			for j in range(DN_LOSS_LAYERS):
				training_loss[id][j+1] = loss[j+1]



	####################################### Printing Training Errors ################################################

	to_print = "TRAINING ERRORS : \n"

	if DN_LOSS_TYPE == "FL":
		for j in range(DN_LOSS_LAYERS + 1):
			to_print += "\n%10.6e" % (np.mean(training_loss, axis=0)[j])

	else:
		to_print += "\n%10.6e" % (np.mean(training_loss, axis=0)[0])

	to_print += "\n"
	# logs_file.write(to_print + "\n")
	# logs_file.flush()
	print(to_print, flush=True)


	###################################### Change loss weights #############################################

	if DN_LOSS_TYPE == "FL" and epoch == SET_WEIGHT_EPOCH:
		print("\nSetting loss weights for the loss calculation ....\n", flush=True)
		loss_weights = np.mean(training_loss, axis=0)[1:]
		print("Weights has been set\n")


	if epoch % SAVE_MODEL_EPOCH != 0:
		print("************************************************ Epoch " + str(epoch) + " ended *********************************************", flush=True)
		print("\n", flush=True)
		continue


	####################################### Model saving ##################################################

	
	print("Saving the model .....", flush=True)
	torch.save(denoising_model.state_dict(), output_folder + "/denoising_model.pth")
	print("Model saving done", flush=True)
	print("\n", flush=True)



	###################################### Validation Epoch ###############################################

	print("------------------------ Validation loop started ------------------------", flush=True)
	print("\n", flush=True)

	for id in range(0, len(validation_set['innames'])):

		index = id
		input_data = validation_set['inaudio'][index]
		output_data = validation_set['outaudio'][index]

		input_data = torch.tensor(input_data).cuda()
		output_data = torch.tensor(output_data).cuda()

		enhanced_data = denoising_model(input_data)

		loss = []

		if DN_LOSS_TYPE == "L1":
			loss = l1_loss(output_data, enhanced_data)

		elif DN_LOSS_TYPE == "L2":
			loss = l2_loss(output_data, enhanced_data)

		else:
			loss = featureLoss(output_data, enhanced_data, loss_weights)


		validation_loss[id][0] = loss[0]

		if DN_LOSS_TYPE == "FL":
			for j in range(DN_LOSS_LAYERS):
				validation_loss[id][j+1] = loss[j+1]


	####################################### Printing Validation Errors ################################################

	# to_print = "\n"
	to_print = "VALIDATION ERROS : \n"

	if DN_LOSS_TYPE == "FL":
		for j in range(DN_LOSS_LAYERS + 1):
			to_print += "\n%10.6e" % (np.mean(validation_loss, axis=0)[j] * 1e9)

	else:
		to_print += "\n%10.6e" % (np.mean(validation_loss, axis=0)[0] * 1e9)

	to_print += "\n"
	print(to_print, flush=True)

	print("************************************************ Epoch " + str(epoch) + " ended *********************************************", flush=True)
	print("\n", flush=True)
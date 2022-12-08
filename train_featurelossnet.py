from models import *
from load_data import *
import numpy as np
import sys, getopt

# Command line options
out_folder = "models"

try:
	opts, args = getopt.getopt(sys.argv[1:], "ho:", ["out_folder="])
except getopt.GetoptError:
	print('HERE : Commad to run : python train_featurelossnet.py -o <out_folder>', flush=True)
	sys.exit(2)

for opt, arg in opts:
	if opt == '-h':
		print('Commad to run : python train_featurelossnet.py -o <out_folder>', flush=True)
		sys.exit()

	elif opt in ("-o", "--out_folder"):
		out_folder = arg

print('Folder to save the model ---- ' + out_folder, flush=True)


# Feature Loss Network Parameters
FEATURE_LOSS_LAYERS = 14
BASE_CHANNELS = 32
BASE_CHANNELS_INCREMENT = 5

# Network Setup
no_of_tasks = 0
no_of_classes = []
error_type = []
task_labels = []
layers_task = []
pred_output_task = []
loss_layer_task = []
optimizer_task = []

# Data Loading
file_names = []
labels = []
datasets = []
labels_lists = []
sets = ['train', 'val']


# Acoustic Scene Classification Task
no_of_tasks += 1

ase_labels, ase_names, ase_datasets, ase_labels_lists = loadASCData("dataset/asc")
no_of_classes.append(len(ase_labels_lists))
error_type.append(1)

file_names.append({})
labels.append({})
datasets.append({})

# print(len(ase_names['train']))

for x in sets:
	file_names[no_of_tasks-1][x] = ase_names[x]
	labels[no_of_tasks-1][x] = ase_labels[x]
	datasets[no_of_tasks-1][x] = ase_datasets[x]

labels_lists.append(ase_labels_lists)

# Domestic Audio Tagging Task
no_of_tasks += 1

dat_labels, dat_names, dat_datasets, dat_labels_lists = loadDATData("dataset/dat")
no_of_classes.append(len(dat_labels_lists))
error_type.append(2)

file_names.append({})
labels.append({})
datasets.append({})

for x in sets:
	file_names[no_of_tasks-1][x] = dat_names[x]
	labels[no_of_tasks-1][x] = dat_labels[x]
	datasets[no_of_tasks-1][x] = dat_datasets[x]

labels_lists.append(dat_labels_lists)

print("Data loading completed !!! ", flush=True)
# Epoch Initialization

train_error = []
test_error = []
threshold_error = []

for task in range(no_of_tasks):
	train_error.append(np.zeros(len(file_names[task]['train'])))
	test_error.append(np.zeros(len(file_names[task]['val'])))
	threshold_error.append(0.5 * np.ones(len(labels_lists[task])))

MAX_NUM_FILE = 0

training_prediction_label = []
training_true_label = []
test_prediction_label = []
test_true_label = []

for task in range(no_of_tasks):
	# print(len(file_names[task]['train']))
	MAX_NUM_FILE = np.maximum(MAX_NUM_FILE, len(file_names[task]['train']))
	
	# Training Data
	training_prediction_label.append([])
	training_true_label.append([])
	
	for file in range(len(file_names[task]['train'])):
		training_prediction_label[task].append(np.zeros((0, len(labels_lists[task]))))
		training_true_label[task].append(np.zeros((0, len(labels_lists[task]))))
	
	# Testing Data
	test_prediction_label.append([])
	test_true_label.append([])
	
	for file in range(len(file_names[task]['val'])):
		test_prediction_label[task].append(np.zeros((0, len(labels_lists[task]))))
		test_true_label[task].append(np.zeros((0, len(labels_lists[task]))))



# Epoch Loop
Epochs = 2500
feature_loss_model = FeatureLossNet().cuda()
# feature_loss_model.zero_grad()
learning_rate = 1e-4
# optimizer = torch.optim.Adam(feature_loss_model.parameters(), lr=learning_rate)
optimizer = []
output_conv = []
loss = nn.CrossEntropyLoss()
prediction_layer_softmax = nn.Softmax().cuda()
prediction_layer_sigmoid = nn.Sigmoid().cuda()

# print("Task 1 classes : " + str(no_of_classes[0]))
# print("Task 2 classes : " + str(no_of_classes[1]))
# width = [400000,50000]

for task in range(no_of_tasks):
	optimizer.append(torch.optim.Adam(feature_loss_model.parameters(), lr=learning_rate))
	# output_conv.append(nn.Conv2d(128, no_of_classes[task], kernel_size=(1,1)).cuda())

for epoch in range(1, Epochs+1):
	print("\n", flush=True)
	print("################################################## Epoch " + str(epoch) + " started ########################################################", flush=True)
	# print("\n")
	ids = []

	for task in range(no_of_tasks):
		ids.append(np.random.permutation(len(file_names[task]['train'])))

	for id in range(MAX_NUM_FILE*no_of_tasks):
		task = id % no_of_tasks
		
		file_id = ids[task][id % len(ids[task])]
		
		input_data = datasets[task]['train'][file_id]
		# print(input_data.shape)
		data_size = np.size(input_data)

		input_data = torch.tensor(input_data).cuda()
		data_shape = np.shape(input_data)
		
		min_width = 2**(FEATURE_LOSS_LAYERS + 1) - 1
		top_width = 1.*data_size
		max_width = top_width
		
		exponent = np.random.uniform(np.log10(min_width - 0.5), np.log10(max_width + 0.5))
		width = int(np.round(10. ** exponent))
		# print(width)
		start_point = np.random.randint(0, data_size - width + 1)
		# print(input_data.shape)
		input_data = input_data[:, :, :, start_point:start_point + width]
		# print(input_data.shape)
		input_label = torch.tensor(np.reshape(1. * np.array([(l in labels[task]['train'][file_id]) for l in labels_lists[task]]), (1,-1)), dtype=torch.long).cuda()
		
		######  GRADIENT DESCENT ######
		# optimizer[task].zero_grad()

		# outputs = feature_loss_model(input_data, task)
		# print(features[-1].shape)
		# avg_features = torch.mean(features[-1], 3, True)
		# print(avg_features.shape)
		# output_conv = nn.Conv2d(avg_features.size()[1], no_of_classes[task], kernel_size=(1,1))
		# outputs = output_conv[task](avg_features)
		# print(outputs.shape)
		# print(input_data.size())
		# print(no_of_classes[task])
		# logits = torch.reshape(outputs[-1], [input_data.size()[0], no_of_classes[task]])
		# loss = nn.CrossEntropyLoss()
		prediction, outputs = feature_loss_model(input_data, task)

		cross_entropy_loss = loss(outputs[-1], torch.max(input_label,1)[1])
		
		optimizer[task].zero_grad()
		# cross_entropy_loss.backward()
		cross_entropy_loss.backward()
		optimizer[task].step()
		reduced_cross_entropy_loss = torch.mean(cross_entropy_loss)
		
		train_error[task][file_id] = reduced_cross_entropy_loss
		# print(reduced_cross_entropy_loss)
		prediction = prediction.cpu().detach()
		training_prediction_label[task][file_id] = np.reshape(prediction, [1,-1])
		# print(prediction)
		training_true_label[task][file_id] = input_label.cpu()

		# if error_type[task] == 1:
		# 	# print("Task %d: Softmax" % task)
		# 	# prediction_layer = nn.Softmax(dim=-1)
		# 	# prediction = prediction_layer_softmax(logits)
		# 	# print(prediction)
		# 	# print(input_label)
		# 	# print(prediction)

		# 	cross_entropy_loss = loss(prediction, torch.max(input_label,1)[1])
		# 	# print(torch.max(input_label,1)[1])
		# 	# print(cross_entropy_loss)
		# 	# Check if we can write it after if else
		# 	optimizer[task].zero_grad()
		# 	# cross_entropy_loss.backward()
		# 	cross_entropy_loss.backward()
		# 	optimizer[task].step()
		# 	reduced_cross_entropy_loss = torch.mean(cross_entropy_loss)
			
		# 	train_error[task][file_id] = reduced_cross_entropy_loss
		# 	# print(reduced_cross_entropy_loss)
		# 	prediction = prediction.detach()
		# 	training_prediction_label[task][file_id] = np.reshape(prediction, [1,-1])
		# 	# print(training_prediction_label[task][file_id])
		# 	training_true_label[task][file_id] = input_label
		# 	# print(input_label)

		# else:
		# 	# print("Task %d: Sigmoid" % task)
		# 	# prediction_layer = nn.Sigmoid()
		# 	# prediction = prediction_layer_sigmoid(logits)

		# 	# print(input_label)
		# 	# print(prediction)

		# 	cross_entropy_loss = loss(prediction, torch.max(input_label,1)[1])
			
		# 	optimizer[task].zero_grad()
		# 	# cross_entropy_loss.backward()
		# 	cross_entropy_loss.backward()
		# 	optimizer[task].step()
		# 	reduced_cross_entropy_loss = torch.mean(cross_entropy_loss)
			
		# 	train_error[task][file_id] = reduced_cross_entropy_loss
		# 	# print(reduced_cross_entropy_loss)
		# 	prediction = prediction.detach()
		# 	training_prediction_label[task][file_id] = np.reshape(prediction, [1,-1])
		# 	# print(prediction)
		# 	training_true_label[task][file_id] = input_label
	

	################################### ALL TRAINING ERROS COMPUTATIONS ##################################### 
	# print("\n")
	to_print = "TRAINING ERRORS : \n\n"

	for task in range(no_of_tasks):
		to_print += "Training Error for task " + str(task) + " = "
		to_print += "%.6f " % (np.mean(train_error[task][np.where(train_error[task])]))
		to_print += "\n"

		# to_print += "\n"

		if error_type[task] == 1:
			# Mean classification error
			to_print += "Mean Classification error for task " + str(task) + " = "
			to_print += "%.6f " % (np.mean(1.0 * (np.argmax(np.vstack(training_prediction_label[task]),axis=1) == np.argmax(np.vstack(training_true_label[task]),axis=1))))
			to_print += "\n"

		elif error_type[task] == 2:
			# Mean equal error
			# to_print += "0.6f " % ()
			to_print += "Mean Equal error for task " + str(task) + " = "
			eq_error_rate = 0.

			for n1, label in enumerate(labels_lists[task]):
				thres = np.array([0.,1.,0.])
				fp = 1
				fn = 0

				while abs(fp-fn) > 1e-4 and abs(np.diff(thres[:-1])) > 1e-10:
					thres[-1] = np.mean(thres[:-1])
					fp = (np.sum((np.vstack(training_prediction_label[task])[:,n1] > thres[-1]) * (1.-np.vstack(training_true_label[task])[:,n1]))) / (1e-15 + np.sum(1.-np.vstack(training_true_label[task])[:,n1]))
					fn = (np.sum((np.vstack(training_prediction_label[task])[:,n1] <= thres[-1]) * np.vstack(training_true_label[task])[:,n1])) / (1e-15 + np.sum(np.vstack(training_true_label[task])[:,n1]))

					if fp < fn:
						thres[1] = thres[-1]
					else:
						thres[0] = thres[-1]

				threshold_error[task][n1] = thres[-1]
				eq_error_rate += (fp+fn)/2

			eq_error_rate /= len(labels_lists[task])
			to_print += "%.6f " % eq_error_rate
			to_print += "\n"

		# to_print += "\n"


	print(to_print, flush=True)

	if epoch % 2 > 0:
		# print("--------------------------- Epoch " + str(epoch) + " ended --------------------------------")
		print("################################################## Epoch " + str(epoch) + " ended ########################################################", flush=True)
		print("\n", flush=True)
		continue



	#### SAVE MODEL HERE
	print("Saving the model .....", flush=True)
	torch.save(feature_loss_model.state_dict(), out_folder + "/loss_model.pth")
	print("Model saving done", flush=True)
	print("\n", flush=True)



	################################################ VALIDATION LOOP ######################################################
	print("------------------------ Validation loop started ------------------------", flush=True)
	print("\n", flush=True)

	for task in range(no_of_tasks):

		for id in tqdm(range(len(file_names[task]['val'])), file=sys.stdout):
			file_id = id
			input_data = torch.tensor(datasets[task]['val'][file_id]).cuda()
			input_label = torch.tensor(np.reshape(1. * np.array([(l in labels[task]['val'][file_id]) for l in labels_lists[task]]), (1,-1)), dtype=torch.long).cuda()

			# features = feature_loss_model.featureLoss(input_data)
			# outputs = feature_loss_model(input_data, task)
			# avg_features = torch.mean(features[-1], 3, True)
			
			# output_conv = nn.Conv2d(avg_features.size()[2], no_of_classes[task], kernel_size=(1,1))
			# outputs = output_conv[task](avg_features)
			prediction, outputs = feature_loss_model(input_data, task)
			
			# logits = torch.reshape(outputs[-1], [input_data.size()[0], no_of_classes[task]])

			cross_entropy_loss = loss(outputs[-1], torch.max(input_label,1)[1])
			reduced_cross_entropy_loss = torch.mean(cross_entropy_loss)
			# # Check if we can write it after if else
			# optimizer.zero_grad()
			# cross_entropy_loss.backward()
			# optimizer.step()
			prediction = prediction.cpu().detach()
			test_error[task][file_id] = reduced_cross_entropy_loss
			test_prediction_label[task][file_id] = prediction
			test_true_label[task][file_id] = input_label.cpu()
			
			# if error_type[task] == 1:
			# 	# print("Task %d: Softmax" % task)
			# 	# prediction_layer = nn.Softmax(dim=-1)
			# 	# prediction = prediction_layer_softmax(logits)
			# 	cross_entropy_loss = loss(prediction, torch.max(input_label,1)[1])
			# 	reduced_cross_entropy_loss = torch.mean(cross_entropy_loss)
			# 	# # Check if we can write it after if else
			# 	# optimizer.zero_grad()
			# 	# cross_entropy_loss.backward()
			# 	# optimizer.step()
			# 	prediction = prediction.detach()
			# 	test_error[task][file_id] = reduced_cross_entropy_loss
			# 	test_prediction_label[task][file_id] = prediction
			# 	test_true_label[task][file_id] = input_label
				
			# else:
			# 	# print("Task %d: Sigmoid" % task)
			# 	# prediction_layer = nn.Sigmoid()
			# 	# prediction = prediction_layer_sigmoid(logits)
			# 	cross_entropy_loss = loss(prediction, torch.max(input_label,1)[1])
			# 	reduced_cross_entropy_loss = torch.mean(cross_entropy_loss)
			# 	# optimizer.zero_grad()
			# 	# cross_entropy_loss.backward()
			# 	# optimizer.step()
			# 	prediction = prediction.detach()
			# 	test_error[task][file_id] = reduced_cross_entropy_loss
			# 	test_prediction_label[task][file_id] = prediction
			# 	test_true_label[task][file_id] = input_label


	################################### ALL VALIDATION ERROS COMPUTATIONS ##################################### 
	to_print = "\n"
	to_print += "VALIDATION ERROS : \n"

	for task in range(no_of_tasks):
		to_print += "Validation Error for task " + str(task) + " = "
		to_print += "%.6f " % (np.mean(test_error[task][np.where(test_error[task])]))
		to_print += "\n"

		# to_print += "\n"

		if error_type[task] == 1:
			to_print += "Mean Classification error for task " + str(task) + " = "
			to_print += "%.6f " % (np.mean(1.0 * (np.argmax(np.vstack(test_prediction_label[task]), axis=1) == np.argmax(np.vstack(test_true_label[task]), axis=1))))
			to_print += "\n"

		elif error_type[task] == 2:
			to_print += "Mean Equal error for task " + str(task) + " = "
			eq_error_rate = 0

			for n1, label in enumerate(labels_lists[task]):
				thres = np.array([0.,1.,.0])
				fp = 1
				fn = 0

				while abs(fp-fn) > 1e-4 and abs(np.diff(thres[:-1])) > 1e-10:
					thres[-1] = np.mean(thres[:-1])
					fp = (np.sum((np.vstack(test_prediction_label[task])[:,n1] > thres[-1]) * (1.-np.vstack(test_true_label[task])[:,n1]))) / (1e-15 + np.sum(1.-np.vstack(test_true_label[task])[:,n1]))
					fn = (np.sum((np.vstack(test_prediction_label[task])[:,n1] <= thres[-1]) * np.vstack(test_true_label[task])[:,n1])) / (1e-15 + np.sum(np.vstack(test_true_label[task])[:,n1]))

					if fp < fn:
						thres[1] = thres[-1]
					else:
						thres[0] = thres[-1]

				# threshold_error[task][n1] = thres[-1]
				eq_error_rate += (fp+fn)/2

			eq_error_rate /= len(labels_lists[task])
			to_print += "%.6f " % (eq_error_rate)
			to_print += "\n"

		# to_print += "\n"

	print(to_print, flush=True)

	print("################################################## Epoch " + str(epoch) + " ended ########################################################", flush=True)
	print("\n", flush=True)
	# print("\n")
	# print("--------------------------- Epoch " + str(epoch) + " ended --------------------------------")

# Training Cod
# for epoch in range(1, Epochs+1):

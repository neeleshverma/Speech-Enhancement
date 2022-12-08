from models import *
from load_data import *

import sys, getopt

validation_folder = "dataset/valset_noisy"
denoising_model_path = "models/denoising_model.pth"

try:
	opts, args = getopt.getopt(sys.argv[1:], "hd:m:", ["val_folder=,lossModelPath="])
except getopt.GetoptError:
	print('Usage: test_denoisingnet.py -d dataFolder -m denoisingModelPath', flush=True)
	sys.exit(2)

for opt, arg in opts:
	if opt == '-h':
		print('Usage: test_denoisingnet.py -d dataFolder -m denoisingModelPath', flush=True)
	elif opt in ("-d", "--dataFolder"):
		validation_folder = arg
	elif opt in ("-m", "--denoisingModelPath"):
		denoising_model_path = arg

print("\n", flush=True)
print('Input Data folder is : "' + validation_folder + '/"', flush=True)
print('Denoising model path is : "' + denoising_model_path + '/"', flush=True)
print("Denoised outputs will be in the folder : " + validation_folder + "_denoised", flush=True)

if validation_folder[-1] == '/':
	validation_folder = validation_folder[:-1]

if not os.path.exists(validation_folder + '_denoised_v2'):
	os.mkdir(validation_folder + '_denoised_v2')

# Denoising Network Parameters
DN_LAYERS = 13
DN_CHANNELS = 64
DN_LOSS_LAYERS = 6
DN_NORMAL_TYPE = "ADN" # Adaptive Batch Norm
# DN_LOSS_TYPE = "FL"

frequency = 16000

# Loading of Data
validation_set = loadNoisyDataList(validationFolder = validation_folder)
validation_set = loadNoisyData(validation_set)


# Loading the saved model
denoising_model = DenoisingNet().cuda()
denoising_model.load_state_dict(torch.load(denoising_model_path))
denoising_model.eval()

########### Running on Validation Set ####################
print("\n---------------- Evaluation of validation dataset started ----------------------------\n")
for i in tqdm(range(0, len(validation_set['innames']))):

	index = i
	input_data = validation_set['inaudio'][index]
	input_data = torch.tensor(input_data).cuda()
	enhanced_data = denoising_model(input_data)

	enhanced_data = enhanced_data.detach().cpu().numpy()
	enhanced_data = np.reshape(enhanced_data, -1)
	wavfile.write("%s_denoised_v2/%s" % (validation_folder,validation_set['shortnames'][i]), frequency, enhanced_data)

print("\n---------------- Evaluation of validation dataset ended ----------------------------\n")


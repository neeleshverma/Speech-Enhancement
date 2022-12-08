# Speech Denoising Using Deep Feature Losses

-------------------------
DATA DOWNLOADING FILES: |
-------------------------

preprocess_lossdata.sh - This script downloads all the (Acoustic Scene Dataset and Domestic Audio Tagging) data, resamples it to 16 kHz and the entire data is saved inside dataset/asc and dataset/dat.

preprocess_denoisingdata.sh - It downloads the Voice Bank Corpus Dataset and resamples all the files to 16 kHz. 4 folders are created - 
							  1) trainset_clean
							  2) trainset_noisy
							  3) valset_clean
							  4) valset_noisy

----------------------
FEATURELOSS NET FILES |
----------------------

train_featurelossnet.py - This trains the featureloss network (or decoder network) on both the tasks and also calculates the validation scores for both of these.

						  USAGE : python train_featurelossnet.py -o models

						  The model is saved inside the "models" folder with name "loss_model.pth"


--------------------
DENOISING NET FILES |
--------------------

train_denoisingnet.py - This trains the denoising network (or encoder network) on Voice Bank Corpus training dataset and also calculates the validation scores on the validaation dataset.
						
						USAGE : python train_denoisingnet.py -d dataset -l models/loss_model.pth -s models

						The model is saved inside the "models" folder with name "denoising_model.pth". Specify the loss model path in -l option.


test_denosingnet.py - This test the denoising network on any noisy audios. It takes as input, the input data folder that should contain all the audios that we wish to denoise.

					  USAGE : test_denoisingnet.py -d dataFolder -m denoisingModelPath

					  Here dataFolder is the folder containing all the noisy audios. denoisingModelPath is the path for our denoised network model (encoder model).

					  The denoised audios will get saved on the same location as the input data folder. (dataFolder_denoised folder will get created).


models.py - Contains the architecture of both the encoder and the decoder.

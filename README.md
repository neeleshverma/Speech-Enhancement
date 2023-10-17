# End-to-End Speech Enhancement With Perceptual Feature Losses

An end-to-end deep neural network for speech denoising using perceptual feature differences as a loss function (using PyTorch framework).

## DATA DOWNLOADING FILES: ##

**1.)** **preprocess_lossdata.sh** - This script downloads all the (Acoustic Scene Dataset and Domestic Audio Tagging) data, resamples it to 16 kHz and the entire data is saved inside dataset/asc and dataset/dat. This dataset is used for training of the loss network.

					USAGE : ./preprocess_lossdata.sh

**2.)** **preprocess_denoisingdata.sh** - It downloads the Voice Bank Corpus Dataset and resamples all the files to 16 kHz. It will create 4 new folders inside dataset folder (clean training set, noisy training set, clean validation set and noisy validation set) - 
1) trainset_clean
2) trainset_noisy
3) valset_clean
4) valset_noisy

					USAGE : ./preprocess_denoisingdata.sh

## TRAINING FEATURE LOSS NETWORK ##

The feature loss network is trained on 2 datasets - Acoustic Scene Classification and Domestic Audio Tagging. 
The network architecture is shown below - 

<a href="url"><img src="https://github.com/neeleshverma/Speech-Enhancement/blob/main/report_and_figs/feature_loss_net.png" align="center" height="280" width="700" class="center">
</a>  


### Feature Loss Network Files ###

* train_featurelossnet.py - This trains the featureloss network (or decoder network) on both the tasks and also calculates the validation scores for both of these.

					USAGE : python train_featurelossnet.py -o models

   The model is saved inside the "models" folder with name "loss_model.pth"



## TRAINING SPEECH DENOISING NETWORK ## 

The speech-denoising network is trained on the Voice Bank Corpus Dataset.  
The network architecture is shown below - 

<a href="url"><img src="https://github.com/neeleshverma/Speech-Enhancement/blob/main/report_and_figs/denoising_net.png" align="center" height="280" width="800" >
</a>  


### Speech Denoising Network Files ###

* train_denoisingnet.py - This trains the denoising network (or encoder network) on Voice Bank Corpus training dataset and also calculates the validation scores on the validation dataset. It takes the loss network trained earlier as an argument.
						
			USAGE : python train_denoisingnet.py -d dataset -l models/loss_model.pth -s models

   The model is saved inside the "models" folder with name "denoising_model.pth". Specify the loss model path in -l option.


* test_denosingnet.py - This test the denoising network on any noisy audios. It takes as input, the input data folder that should contain all the audios that we wish to denoise.

			USAGE : python test_denoisingnet.py -d data_folder -m denoising_model_path

   data_folder - folder containing all the noisy audios   
   denoising_model_path - path for our denoised network model (encoder model).  

   The denoised audios will get saved on the same location as the input data folder. ($(data_folder)_denoised folder will get created).


models.py - Contains the architecture of both the encoder and the decoder.



### Examples
#### Noisy Audio 1



https://user-images.githubusercontent.com/39479994/215962178-ac55e72c-e13f-4264-9fd2-9820c52bbc83.mp4


#### Clean Audio 1


https://user-images.githubusercontent.com/39479994/215962236-3e22642f-0963-4b12-b88b-86ab0b4fdc28.mp4


#### Noisy Audio 2



https://user-images.githubusercontent.com/39479994/215962270-7f055176-6e59-4b01-8e6c-a4d6314f4ae3.mp4


#### Clean Audio 2


https://user-images.githubusercontent.com/39479994/215962320-79e5c43d-b30b-40a4-a650-f6394ffc3063.mp4




Contact Info : neeleshverma13@gmail.com

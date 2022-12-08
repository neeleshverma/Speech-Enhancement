import torch 
import torch.nn as nn
import torch.nn.functional as F
import numpy as np

class FeatureLossNet(nn.Module):

	def __init__(self, numLayers=14, baseChannels=32, baseChannelsIncrement=5, inputChannels=1, kernelSize=3):

		super(FeatureLossNet, self).__init__()

		outputChannels = baseChannels

		self.conv1 = nn.Conv2d(inputChannels, outputChannels, kernel_size=(1,3), stride=(1,2), padding=(0,1))
		self.conv1_bn = nn.BatchNorm2d(outputChannels)
		self.conv2 = nn.Conv2d(outputChannels, outputChannels, kernel_size=(1,3), stride=(1,2), padding=(0,1))
		self.conv2_bn = nn.BatchNorm2d(outputChannels)
		self.conv3 = nn.Conv2d(outputChannels, outputChannels, kernel_size=(1,3), stride=(1,2), padding=(0,1))
		self.conv3_bn = nn.BatchNorm2d(outputChannels)
		self.conv4 = nn.Conv2d(outputChannels, outputChannels, kernel_size=(1,3), stride=(1,2), padding=(0,1))
		self.conv4_bn = nn.BatchNorm2d(outputChannels)
		self.conv5 = nn.Conv2d(outputChannels, outputChannels, kernel_size=(1,3), stride=(1,2), padding=(0,1))
		self.conv5_bn = nn.BatchNorm2d(outputChannels)

		inputChannels = outputChannels
		outputChannels = outputChannels * 2

		self.conv6 = nn.Conv2d(inputChannels, outputChannels, kernel_size=(1,3), stride=(1,2), padding=(0,1))
		self.conv6_bn = nn.BatchNorm2d(outputChannels)
		self.conv7 = nn.Conv2d(outputChannels, outputChannels, kernel_size=(1,3), stride=(1,2), padding=(0,1))
		self.conv7_bn = nn.BatchNorm2d(outputChannels)
		self.conv8 = nn.Conv2d(outputChannels, outputChannels, kernel_size=(1,3), stride=(1,2), padding=(0,1))
		self.conv8_bn = nn.BatchNorm2d(outputChannels)
		self.conv9 = nn.Conv2d(outputChannels, outputChannels, kernel_size=(1,3), stride=(1,2), padding=(0,1))
		self.conv9_bn = nn.BatchNorm2d(outputChannels)
		self.conv10 = nn.Conv2d(outputChannels, outputChannels, kernel_size=(1,3), stride=(1,2), padding=(0,1))
		self.conv10_bn = nn.BatchNorm2d(outputChannels)

		inputChannels = outputChannels
		outputChannels = outputChannels * 2

		self.conv11 = nn.Conv2d(inputChannels, outputChannels, kernel_size=(1,3), stride=(1,2), padding=(0,1))
		self.conv11_bn = nn.BatchNorm2d(outputChannels)
		self.conv12 = nn.Conv2d(outputChannels, outputChannels, kernel_size=(1,3), stride=(1,2), padding=(0,1))
		self.conv12_bn = nn.BatchNorm2d(outputChannels)
		self.conv13 = nn.Conv2d(outputChannels, outputChannels, kernel_size=(1,3), stride=(1,2), padding=(0,1))
		self.conv13_bn = nn.BatchNorm2d(outputChannels)
		self.conv14 = nn.Conv2d(outputChannels, outputChannels, kernel_size=(1,3), padding=(0,1))
		self.conv14_bn = nn.BatchNorm2d(outputChannels)		

		self.conv15 = nn.Conv2d(outputChannels, 15, kernel_size=(1,1), padding=0)

		self.conv16 = nn.Conv2d(outputChannels, 7, kernel_size=(1,1), padding=0)

		self.softmax = nn.Softmax()
		self.sigmoid = nn.Sigmoid()

		# self.conv14_bn = nn.BatchNorm2d(outputChannels)
		# self.conv15 = nn.Conv2d(outputChannels, outputChannels, kernel_size=(1,3), padding=0)
		# self.conv15_bn = nn.BatchNorm2d(outputChannels)
		# if featureNetwork:
		# 	self.features = FeatureLossNet(inputChannels)
		# elif denoisingNetwork:
		# 	self.features = DenoisingNetwork(inputChannels)
		# self.features = nn.Sequential(
			# nn.Conv2d(inputChannels, , kernel_size=11, stride=4, padding=2),
		# )

	def forward(self, myinput, tasknum):
		allOutputs = []
		# print(myinput.shape)
		output = F.leaky_relu(self.conv1_bn(self.conv1(myinput)))
		# print(output.shape)
		allOutputs.append(output)
		output = F.leaky_relu(self.conv2_bn(self.conv2(output)))
		allOutputs.append(output)
		output = F.leaky_relu(self.conv3_bn(self.conv3(output)))
		allOutputs.append(output)
		output = F.leaky_relu(self.conv4_bn(self.conv4(output)))
		allOutputs.append(output)
		output = F.leaky_relu(self.conv5_bn(self.conv5(output)))
		allOutputs.append(output)
		output = F.leaky_relu(self.conv6_bn(self.conv6(output)))
		allOutputs.append(output)
		output = F.leaky_relu(self.conv7_bn(self.conv7(output)))
		allOutputs.append(output)
		output = F.leaky_relu(self.conv8_bn(self.conv8(output)))
		allOutputs.append(output)
		output = F.leaky_relu(self.conv9_bn(self.conv9(output)))
		allOutputs.append(output)
		output = F.leaky_relu(self.conv10_bn(self.conv10(output)))
		allOutputs.append(output)
		output = F.leaky_relu(self.conv11_bn(self.conv11(output)))
		allOutputs.append(output)
		output = F.leaky_relu(self.conv12_bn(self.conv12(output)))
		allOutputs.append(output)
		output = F.leaky_relu(self.conv13_bn(self.conv13(output)))
		allOutputs.append(output)
		output = F.leaky_relu(self.conv14_bn(self.conv14(output)))
		allOutputs.append(output)

		# print(output.shape)
		avg_features = torch.mean(output, 3, True)
		# print(avg_features.shape)
		if tasknum == 0:
			output = self.conv15(avg_features)
			output = torch.reshape(output, [1,15])
			allOutputs.append(output)
			prediction = self.softmax(output)
			# print(output.shape)
			# allOutputs.append(output)
			return prediction, allOutputs

		elif tasknum == 1:
			output = self.conv16(avg_features)
			output = torch.reshape(output, [1,7])
			allOutputs.append(output)
			prediction = self.sigmoid(output)
			# print(output.shape)
			# allOutputs.append(output)
			return prediction, allOutputs
		# output = F.leaky_relu(self.conv15_bn(self.conv15(output)))
		# allOutputs.append(output)

		# return allOutputs


	# def featureLoss(self, networkResult, actualResult, lossWeights, lossLayers):
	# 	networkOutputs = self.forward(self, networkResult)
	# 	actualOutputs = self.forward(self, actualResult)

	# 	lossVectors = [0]

	# 	for i in range(lossLayers):
	# 		lossVectors.append(l1_loss(networkOutputs[i], actualOutputs[i]) / lossWeights[i])

	# 	for i in range(1, lossLayers + 1):
	# 		lossVectors[0] += lossVectors[i]

	# 	return lossVectors



class DenoisingNet(nn.Module):

	def __init__(self, numLayers=13, baseChannels=32, inputChannels=1, kernelSize=3):
		super(DenoisingNet, self).__init__()

		self.conv1 = nn.Conv2d(inputChannels, baseChannels, kernel_size=(1,3), padding=(0,1))
		self.conv1_bn = nn.BatchNorm2d(baseChannels)
		self.conv2 = nn.Conv2d(baseChannels, baseChannels, kernel_size=(1,3), padding=(0,1))
		self.conv2_bn = nn.BatchNorm2d(baseChannels)
		self.conv3 = nn.Conv2d(baseChannels, baseChannels, kernel_size=(1,3), padding=(0,1))
		self.conv3_bn = nn.BatchNorm2d(baseChannels)
		self.conv4 = nn.Conv2d(baseChannels, baseChannels, kernel_size=(1,3), padding=(0,1))
		self.conv4_bn = nn.BatchNorm2d(baseChannels)
		self.conv5 = nn.Conv2d(baseChannels, baseChannels, kernel_size=(1,3), padding=(0,1))
		self.conv5_bn = nn.BatchNorm2d(baseChannels)
		self.conv6 = nn.Conv2d(baseChannels, baseChannels, kernel_size=(1,3), padding=(0,1))
		self.conv6_bn = nn.BatchNorm2d(baseChannels)
		self.conv7 = nn.Conv2d(baseChannels, baseChannels, kernel_size=(1,3), padding=(0,1))
		self.conv7_bn = nn.BatchNorm2d(baseChannels)
		self.conv8 = nn.Conv2d(baseChannels, baseChannels, kernel_size=(1,3), padding=(0,1))
		self.conv8_bn = nn.BatchNorm2d(baseChannels)
		self.conv9 = nn.Conv2d(baseChannels, baseChannels, kernel_size=(1,3), padding=(0,1))
		self.conv9_bn = nn.BatchNorm2d(baseChannels)
		self.conv10 = nn.Conv2d(baseChannels, baseChannels, kernel_size=(1,3), padding=(0,1))
		self.conv10_bn = nn.BatchNorm2d(baseChannels)
		self.conv11 = nn.Conv2d(baseChannels, baseChannels, kernel_size=(1,3), padding=(0,1))
		self.conv11_bn = nn.BatchNorm2d(baseChannels)
		self.conv12 = nn.Conv2d(baseChannels, baseChannels, kernel_size=(1,3), padding=(0,1))
		self.conv12_bn = nn.BatchNorm2d(baseChannels)
		self.conv13 = nn.Conv2d(baseChannels, baseChannels, kernel_size=(1,3), padding=(0,1))
		self.conv13_bn = nn.BatchNorm2d(baseChannels)
		self.conv14 = nn.Conv2d(baseChannels, baseChannels, kernel_size=(1,3), padding=(0,1))
		self.conv14_bn = nn.BatchNorm2d(baseChannels)
		self.conv15 = nn.Conv2d(baseChannels, 1, kernel_size=(1,1), padding=(0,1))




	def signalDilation(self, signal, channels, dilation):
		signal_shape = signal.shape
		num_elements_to_pad = dilation - 1 - (signal_shape[3] + dilation - 1) % dilation
		dilated_signal = F.pad(signal, (0, num_elements_to_pad, 0, 0, 0, 0, 0, 0))
		dilated_signal = torch.reshape(dilated_signal, (signal_shape[0], channels, -1, dilation))
		return torch.transpose(dilated_signal, 2, 3), num_elements_to_pad



	def inverseSignalDilation(self, dilated_signal, channels, toPad):
		signal_shape = dilated_signal.shape
		dilated_signal = torch.transpose(dilated_signal, 2, 3)
		dilated_signal = torch.reshape(dilated_signal, (signal_shape[0], channels, 1, -1))
		return dilated_signal[:,:,:,:signal_shape[2] * signal_shape[3] - toPad]



	def forward(self, myinput):

		baseChannels = 32
		output = F.leaky_relu(self.conv1_bn(self.conv1(myinput)))

		dilation_depth = 2
		dilated_input, padding = self.signalDilation(output, channels=baseChannels, dilation=dilation_depth)
		dilated_output = F.leaky_relu(self.conv2_bn(self.conv2(dilated_input)))
		output = self.inverseSignalDilation(dilated_output, channels=baseChannels, toPad=padding)

		dilation_depth *= 2
		dilated_input, padding = self.signalDilation(output, channels=baseChannels, dilation=dilation_depth)
		dilated_output = F.leaky_relu(self.conv3_bn(self.conv3(dilated_input)))
		output = self.inverseSignalDilation(dilated_output, channels=baseChannels, toPad=padding)

		dilation_depth *= 2
		dilated_input, padding = self.signalDilation(output, channels=baseChannels, dilation=dilation_depth)
		dilated_output = F.leaky_relu(self.conv4_bn(self.conv4(dilated_input)))
		output = self.inverseSignalDilation(dilated_output, channels=baseChannels, toPad=padding)

		dilation_depth *= 2
		dilated_input, padding = self.signalDilation(output, channels=baseChannels, dilation=dilation_depth)
		dilated_output = F.leaky_relu(self.conv5_bn(self.conv5(dilated_input)))
		output = self.inverseSignalDilation(dilated_output, channels=baseChannels, toPad=padding)

		dilation_depth *= 2
		dilated_input, padding = self.signalDilation(output, channels=baseChannels, dilation=dilation_depth)
		dilated_output = F.leaky_relu(self.conv6_bn(self.conv6(dilated_input)))
		output = self.inverseSignalDilation(dilated_output, channels=baseChannels, toPad=padding)

		dilation_depth *= 2
		dilated_input, padding = self.signalDilation(output, channels=baseChannels, dilation=dilation_depth)
		dilated_output = F.leaky_relu(self.conv7_bn(self.conv7(dilated_input)))
		output = self.inverseSignalDilation(dilated_output, channels=baseChannels, toPad=padding)

		dilation_depth *= 2
		dilated_input, padding = self.signalDilation(output, channels=baseChannels, dilation=dilation_depth)
		dilated_output = F.leaky_relu(self.conv8_bn(self.conv8(dilated_input)))
		output = self.inverseSignalDilation(dilated_output, channels=baseChannels, toPad=padding)

		dilation_depth *= 2
		dilated_input, padding = self.signalDilation(output, channels=baseChannels, dilation=dilation_depth)
		dilated_output = F.leaky_relu(self.conv9_bn(self.conv9(dilated_input)))
		output = self.inverseSignalDilation(dilated_output, channels=baseChannels, toPad=padding)

		dilation_depth *= 2
		dilated_input, padding = self.signalDilation(output, channels=baseChannels, dilation=dilation_depth)
		dilated_output = F.leaky_relu(self.conv10_bn(self.conv10(dilated_input)))
		output = self.inverseSignalDilation(dilated_output, channels=baseChannels, toPad=padding)

		dilation_depth *= 2
		dilated_input, padding = self.signalDilation(output, channels=baseChannels, dilation=dilation_depth)
		dilated_output = F.leaky_relu(self.conv11_bn(self.conv11(dilated_input)))
		output = self.inverseSignalDilation(dilated_output, channels=baseChannels, toPad=padding)

		dilation_depth *= 2
		dilated_input, padding = self.signalDilation(output, channels=baseChannels, dilation=dilation_depth)
		dilated_output = F.leaky_relu(self.conv12_bn(self.conv12(dilated_input)))
		output = self.inverseSignalDilation(dilated_output, channels=baseChannels, toPad=padding)

		dilation_depth *= 2
		dilated_input, padding = self.signalDilation(output, channels=baseChannels, dilation=dilation_depth)
		dilated_output = F.leaky_relu(self.conv13_bn(self.conv13(dilated_input)))
		output = self.inverseSignalDilation(dilated_output, channels=baseChannels, toPad=padding)

		output = F.leaky_relu(self.conv14_bn(self.conv14(output)))
		output = self.conv15(output)

		return output



# def FeatureLossNet(numLayers=14, baseChannels=32, baseChannelsIncrement=5, inputChannels, kernelSize=3):
	
# 	layers = []

# 	for currentLayer in numLayers:
# 		outputChannels = baseChannels * (2 ** (currentLayer // baseChannelsIncrement))

# 		if currentLayer < numLayers - 1:
# 			layers += [nn.Conv2d(inputChannels, outputChannels, kernel_size=(1,3), stride=(1,2), padding=0)]
# 			layers += [nn.BatchNorm2d(inputChannels)]
# 			layers += [nn.LeakyReLU(inplace=True)]
# 			inputChannels = outputChannels

# 		else:
# 			layers += [nn.Conv2d(inputChannels, outputChannels, kernel_size=(1,3), padding=0)]
# 			layers += [nn.BatchNorm2d(outputChannels)]
# 			layers += [nn.LeakyReLU(inplace=True)]
# 			inputChannels = outputChannels

# 	return nn.Sequential(layers)

# def DenoisingNetwork(numLayers=14, baseChannels=32, inputChannels, kernelSize=3, numberOfChannels=64):

# 	layers = []

# 	for currentLayer in numLayers:


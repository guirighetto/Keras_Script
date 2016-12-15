#!-*- conding: utf8 -*-
import theano
from keras import backend as K
from keras.models import Sequential
from keras.layers.core import Dense, Dropout, Activation, Flatten
from keras.layers.convolutional import Convolution2D, MaxPooling2D
from keras.utils import np_utils

import matplotlib.pyplot as plt
from sklearn.metrics import accuracy_score as acc

import random
import addOns as add
import numpy as np
import sys

np.random.seed(1337)

# PARAMETERS
try:
	if(sys.argv[1] != None):
		path_images = sys.argv[1]
		path_weights = sys.argv[2]
		images_format = '.bmp'
		nb_img_class = sys.argv[3]
		train = 60
		test = 40
		classes = sys.argv[4]
		epoch = 30
		batch_size = 48
		nb_block = sys.argv[5]
		block_height = sys.argv[6]
		block_width = sys.argv[6]
		preName = "CF"
		sep="_"
		channels_img = 1
except:
	path_images = "../FolderImages"
	path_weights = "../file_weights.hdf5"
	images_format = '.bmp'
	nb_img_class = 3
	train = 60
	test = 40
	classes = 50
	epoch = 30
	batch_size = 48
	nb_block = 105
	block_height = 128
	block_width = 128
	preName = "CF"
	sep="_"
	channels_img = 1


def loadDatabase():
	numSamplesTrain = float(nb_img_class*(float(train)/100))
	numSamplesTrain = round(numSamplesTrain)

	dataTrain = []
	labelTrain = []
	dataTest = []
	labelTest = []
	samples_test = []
	filesCount = 0
	patchesCount = 0

	for c in range(1,classes+1):
		samples_test.append([])

		for s in range(1,nb_img_class+1):

			if(s < numSamplesTrain+1):
				folderTrainTest = 'Train/'
			else:
				folderTrainTest = 'Test/'

			for b in range(1,nb_block+1):
				nameImg = preName + str(c).zfill(5) + sep + str(s) + sep + str(b)
				folderClass = preName + str(c).zfill(5) + '/'
				fullPathImg = path_images + folderClass + folderTrainTest + nameImg + images_format
				image = plt.imread(fullPathImg)

				image = image[np.newaxis]

				if(folderTrainTest == 'Train/'):
					dataTrain.append(image)
					labelTrain.append(c-1)
				else:
					dataTest.append(image)
					labelTest.append(c-1)
					samples_test[filesCount].append(patchesCount)
					patchesCount += 1
		filesCount+=1

	#train
	dataTrain = np.array(dataTrain)
	labelTrain = np.array(labelTrain)
		
	LUT = np.arange(len(dataTrain), dtype=int)
	random.shuffle(LUT)
	randomDataTrain = dataTrain[LUT]
	randomLabelTrain = labelTrain[LUT]

	X_Train = randomDataTrain.astype("float32")/255
	Y_Train = np_utils.to_categorical(randomLabelTrain, classes)

	#test		
	dataTest = np.array(dataTest)
	labelTest = np.array(labelTest)
	samples_test = np.array(samples_test)

	X_Test = dataTest.astype("float32")/255
	Y_Test = np_utils.to_categorical(labelTest, classes)

	print("Number samples train: ",numSamplesTrain*nb_block*classes)
	print("Number samples test: ",(nb_img_class-numSamplesTrain)*nb_block*classes)		

	return X_Train, Y_Train, X_Test, Y_Test, samples_test


def buildModel():
	model = Sequential()

	model.add(Convolution2D(32, 3, 3, activation='relu', name='conv1_1',input_shape=(channels_img, block_height, block_width), border_mode='same'))

	model.add(MaxPooling2D(pool_size=(2,2), strides=(2,2)))

	model.add(Convolution2D(32, 3, 3, border_mode='same'))
	model.add(Activation('relu'))

	model.add(Convolution2D(64, 3, 3, border_mode='same'))
	model.add(Activation('relu'))

	model.add(MaxPooling2D(pool_size=(2,2), strides=(2,2)))

	model.add(Convolution2D(64, 3, 3, border_mode='same'))
	model.add(Activation('relu'))

	model.add(MaxPooling2D(pool_size=(2,2), strides=(2,2)))

	model.add(Convolution2D(128, 2, 2, border_mode='same'))
	model.add(Activation('relu'))

	model.add(MaxPooling2D(pool_size=(2,2), strides=(2,2)))

	model.add(Flatten())

	model.add(Dense(2048))
	model.add(Activation("relu"))

	model.add(Dropout(0.5))

	model.add(Dense(classes))
	model.add(Activation("softmax"))

	return model


def evaluateModel(model, X_Train, Y_Train, X_Test, Y_Test):
	fakeScoreSeg = model.evaluate(X_Train, Y_Train, verbose=1)
	scoreSeg = model.evaluate(X_Test, Y_Test, verbose=1)

	print()
	print('CLASSIFICATION OF EACH SEGMENT (block):')
	print('Train score:', fakeScoreSeg[0])
	print('Train accuracy:', fakeScoreSeg[1])
	print('Test score:', scoreSeg[0])
	print('Test accuracy:', scoreSeg[1])
	print()

	predict = model.predict(X_Test, verbose=1)

	return predict

def combinationPredict(predict, samples_test):
	labels_samples, merger_min, merger_max, merger_sum, merger_pro = add.fusoesDiego(predict, samples_test)

	classSeg = np_utils.categorical_probas_to_classes(predict)
	classMin = np_utils.categorical_probas_to_classes(merger_min)	
	classMax = np_utils.categorical_probas_to_classes(merger_max)
	classSom = np_utils.categorical_probas_to_classes(merger_sum)
	classPro = np_utils.categorical_probas_to_classes(merger_pro)

	print ()
	print ("Min: " + str(acc(labels_samples,classMin)))
	print ("Max: " + str(acc(labels_samples,classMax)))
	print ("Sum: " + str(acc(labels_samples,classSom)))
	print ("Product: " + str(acc(labels_samples,classPro)))
	print ()

if __name__ == "__main__":
	
	X_Train, Y_Train, X_Test, Y_Test, samples_test = loadDatabase()

	print("Database successfully loaded")
	print("Loading model")
	model = buildModel()
	model.summary()
	print("Running the model")
	model.compile(loss="categorical_crossentropy", optimizer="Adadelta",metrics=['accuracy'])
	model.fit(X_Train, Y_Train, nb_epoch=epoch, batch_size=batch_size, verbose=1, validation_data=(X_Test, Y_Test))
	print("Validation of the model")
	predict = evaluateModel(model, X_Train, Y_Train, X_Test, Y_Test)
	combinationPredict(predict, samples_test)
	print("Saving the weights")
	model.save_weights(path_weights)

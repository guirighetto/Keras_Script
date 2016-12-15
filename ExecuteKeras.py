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



np.random.seed(1337) # for reproducibility ## uat???

# PARAMETERS

pathImages = '/home/guilherme/Documentos/Bases/BFL_50_128_105/'
imagesFormat = '.bmp'
numImgClass = 3
train = 60
test = 40
classes = 50
epoch = 30
batchSize = 48
numblock = 105
blockHeight = 128
blockWidth = 128
preName = "CF"
sep="_"
channelsImg = 1


def loadDatabase():

	numSamplesTrain = float(numImgClass*(float(train)/100))
	numSamplesTrain = round(numSamplesTrain)

	dataTrain = []
	labelTrain = []
	dataTest = []
	labelTest = []
	filesTest = []
	filesCount = 0
	patchesCount = 0

	for c in range(1,classes+1):
		filesTest.append([])

		for s in range(1,numImgClass+1):

			if(s < numSamplesTrain+1):
				folderTrainTest = 'Treino/'
			else:
				folderTrainTest = 'Teste/'

			for b in range(1,numblock+1):
				nameImg = preName + str(c).zfill(5) + sep + str(s) + sep + str(b)
				folderClass = preName + str(c).zfill(5) + '/'
				fullPathImg = pathImages + folderClass + folderTrainTest + nameImg + imagesFormat
				image = plt.imread(fullPathImg)

				image = image[np.newaxis]

				if(folderTrainTest == 'Treino/'):
					dataTrain.append(image)
					labelTrain.append(c-1)
				else:
					dataTest.append(image)
					labelTest.append(c-1)
					filesTest[filesCount].append(patchesCount)
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
	filesTest = np.array(filesTest)

	X_Test = dataTest.astype("float32")/255
	Y_Test = np_utils.to_categorical(labelTest, classes)

	print("Number samples train: ",numSamplesTrain*numblock*classes)
	print("Number samples test: ",(numImgClass-numSamplesTrain)*numblock*classes)		

	return X_Train, Y_Train, X_Test, Y_Test, filesTest


def buildModel():
	model = Sequential()

	model.add(Convolution2D(32, 3, 3, activation='relu', name='conv1_1',input_shape=(channelsImg, blockHeight, blockWidth), border_mode='same'))

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

	model.add(Dense(500))
	model.add(Activation("relu"))

	model.add(Dropout(0.5))

	model.add(Dense(classes))
	model.add(Activation("softmax"))

	return model


def buildModel2():
	model = Sequential()

	model.add(Convolution2D(64, 5, 5, activation='relu', name='conv1_1',input_shape=(channelsImg, blockHeight, blockWidth), border_mode='same'))

	model.add(MaxPooling2D(pool_size=(2,2), strides=(2,2)))

	model.add(Convolution2D(64, 5, 5, border_mode='same'))
	model.add(Activation('relu'))

	model.add(MaxPooling2D(pool_size=(2,2), strides=(2,2)))

	model.add(Convolution2D(64, 3, 3, border_mode='same'))
	model.add(Activation('relu'))

	model.add(MaxPooling2D(pool_size=(2,2), strides=(2,2)))

	model.add(Flatten())

	model.add(Dense(500))
	model.add(Activation("relu"))

	model.add(Dropout(0.5))

	model.add(Dense(classes))
	model.add(Activation("softmax"))

	return model

def evaluateModel(model, X_Train, Y_Train, X_Test, Y_Test):
	fakeScoreSeg = model.evaluate(X_Train, Y_Train, verbose=1)
	scoreSeg = model.evaluate(X_Test, Y_Test, verbose=1)

	print()
	print('CLASSIFICACAO DE CADA SEGMENTO:')
	print('Train score:', fakeScoreSeg[0])
	print('Train accuracy:', fakeScoreSeg[1])
	print('Test score:', scoreSeg[0])
	print('Test accuracy:', scoreSeg[1])
	print()

	predict = model.predict(X_Test, verbose=1)

	return predict

def combinationPredict(predict, filesTest):
	labelsAmostras, fusaoMin, fusaoMax, fusaoSom, fusaoPro = add.fusoesDiego(predict, filesTest)

	classSeg = np_utils.categorical_probas_to_classes(predict)
	classMin = np_utils.categorical_probas_to_classes(fusaoMin)	
	classMax = np_utils.categorical_probas_to_classes(fusaoMax)
	classSom = np_utils.categorical_probas_to_classes(fusaoSom)
	classPro = np_utils.categorical_probas_to_classes(fusaoPro)

	print ()
	print ("minimo: " + str(acc(labelsAmostras,classMin)))
	print ("maximo: " + str(acc(labelsAmostras,classMax)))
	print ("soma: " + str(acc(labelsAmostras,classSom)))
	print ("produto: " + str(acc(labelsAmostras,classPro)))
	print ()


def predict2svm(predict):
	f = open('/home/guilherme/Documentos/CNN/Script/BFL_50_1.predict','w')
	f.write('labels ')
	for c in range(classes):
		f.write(str(c+1)+' ')
	f.write('\n')
	i=0
	for p in predict:
		f.write(str(np.argmax(p)+1)+' ')
		for pi in p:
			if(i == classes-1):
				f.write(str(pi))
			else:
				f.write(str(pi)+' ')
			i+=1
		i=0
		f.write('\n')
	f.close()	

if __name__ == "__main__":
	
	X_Train, Y_Train, X_Test, Y_Test, filesTest = loadDatabase()

	print("Database successfully loaded")
	print("Loading model")
	model = buildModel2()
	model.summary()
	#model.outputs()
	print("Running the model")
	model.compile(loss="categorical_crossentropy", optimizer="Adadelta",metrics=['accuracy'])
	model.fit(X_Train, Y_Train, nb_epoch=epoch, batch_size=batchSize, verbose=1, validation_data=(X_Test, Y_Test))
	predict = evaluateModel(model, X_Train, Y_Train, X_Test, Y_Test)
	predict2svm(predict)
	combinationPredict(predict, filesTest)
	#model.save_weights("/local/home/projeto/Exp_Righetto/Script/BFL_50_128_weight.hdf5")

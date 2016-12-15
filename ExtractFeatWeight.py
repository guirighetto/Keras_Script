from keras.models import Sequential
from keras.layers.core import Flatten, Dense, Dropout, Activation
from keras.layers.convolutional import Convolution2D, MaxPooling2D, ZeroPadding2D
from keras.optimizers import SGD
import cv2, numpy as np
import matplotlib.pyplot as plt
from keras.utils import np_utils
from keras import backend as K

import theano
import random


#pathImages = '/home/guilherme/Documentos/Bases/CVL_309_100_128/'
pathImages = '/media/guilherme/1e8ee6ef-1ae9-40f8-920c-939d24a7a626/guilherme/Bases/BFL_315_100_128/'
imagesFormat = '.bmp'
numImgClass = 3
train = 60
test = 40
classes = 230
epoch = 30
batchSize = 48
numblock = 9
blockHeight = 128
blockWidth = 128
preName = "CF"
sep="_"
channelsImg = 1

def buildModelCVL():
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

    model.add(Convolution2D(128, 3, 3, border_mode='same'))
    model.add(Activation('relu'))

    model.add(Convolution2D(128, 3, 3, border_mode='same'))
    model.add(Activation('relu'))

    model.add(MaxPooling2D(pool_size=(2,2), strides=(2,2)))

    model.add(Flatten())

    model.add(Dense(2048))
    model.add(Activation("relu"))

    model.add(Dropout(0.5))

    model.add(Dense(classes))
    model.add(Activation("softmax"))


    model.load_weights("/home/guilherme/Documentos/CNN/Script/BFL_115_100_128_2048.hdf5")
    #model.load_weights("/media/guilherme/1e8ee6ef-1ae9-40f8-920c-939d24a7a626/guilherme/Bases/Pesos/QUWI/QUWI_115_128_2048_Ingles1.hdf5")

    return model

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

    model.add(Convolution2D(128, 2, 2, border_mode='same'))
    model.add(Activation('relu'))

    model.add(MaxPooling2D(pool_size=(2,2), strides=(2,2)))

    model.add(Flatten())

    model.add(Dense(4096))
    model.add(Activation("relu"))

    model.add(Dropout(0.5))

    model.add(Dense(classes))
    model.add(Activation("softmax"))

    model.load_weights("/home/guilherme/Documentos/CNN/Script/CVL_115_100_128_4096.hdf5")

    return model


if __name__ == "__main__":

    diss = False
   
    numSamplesTrain = float(numImgClass*(float(train)/100))
    numSamplesTrain = round(numSamplesTrain)

    model = buildModelCVL()

    model.compile(loss="categorical_crossentropy", optimizer="Adadelta",metrics=['accuracy'])

    get_feature = K.function([model.layers[0].input],[model.layers[16].output])

    for c in range(116,classes+1):
        #if(c == 194):
        #    c+=1
        for s in range(1,numImgClass+1):

            if(s == 1 or s == 2):
                folderTrainTest = 'Treino/'
            elif(s == 3):
                folderTrainTest = 'Teste/'
        #    if(s == 1):
        #        folderTrainTest = 'Arabe1/'
        #    elif(s == 2):
        #        folderTrainTest = 'Arabe2/'
        #    elif(s == 3):
        #        folderTrainTest = 'Ingles1/'
        #    elif(s == 4):
        #        folderTrainTest = 'Ingles2/'
            #elif(s == 5):
            #    folderTrainTest = 'Alemao1/'
            else:
                folderTrainTest = None

            for b in range(1,numblock+1):
                #if(folderTrainTest == 'Arabe1/' or folderTrainTest == 'Arabe2/' or folderTrainTest == 'Ingles1/' or folderTrainTest == 'Ingles2/'): #or folderTrainTest == 'Alemao1/'):
                #    nameImg = preName + str(c).zfill(6) + sep +'0'+str(s) + sep + str(b)
                #    folderClass = preName + str(c).zfill(6) + '/'
                #    fullPathImg = pathImages + folderClass + folderTrainTest + nameImg + imagesFormat
                #    image = plt.imread(fullPathImg)
                #    image = image[np.newaxis]

                if(folderTrainTest == 'Treino/' or folderTrainTest == 'Teste/'):     
                    nameImg = preName + str(c).zfill(5) + sep + str(s) + sep + str(b)
                    folderClass = preName + str(c).zfill(5) + '/'
                    fullPathImg = pathImages + folderClass + folderTrainTest + nameImg + imagesFormat
                    image = plt.imread(fullPathImg)
                    image = image[np.newaxis]

                    dataTrain = []

                    dataTrain.append(image)

                    dataTrain = np.array(dataTrain)
                    dataTrain = dataTrain.astype("float32")/255

                #    print(dataTrain)


                if(diss == False):
                    if(folderTrainTest == 'Treino/'):
                    #if(folderTrainTest == 'Arabe1/' or folderTrainTest == 'Ingles2/' or folderTrainTest == 'Arabe2/'):# or folderTrainTest == 'Ingles4/'): #or folderTrainTest == 'Arabe1/'): #or folderTrainTest == 'Arabe2/'):
                        fi = open('/media/guilherme/1e8ee6ef-1ae9-40f8-920c-939d24a7a626/guilherme/Bases/115_Outro/BFL/BFL_115_128_2048_Outro.Train','a')
                        k=1
                        fi.write(str(c))
                        for i in get_feature([dataTrain])[0][0]:
                            fi.write(' '+str(k)+':')
                            fi.write(str(i))
                            k+=1
                        fi.write('\n')
                        k=1
                        fi.close()
                    elif(folderTrainTest == 'Teste/'):
                    #elif(folderTrainTest == 'Ingles1/'):
                        fj = open('/media/guilherme/1e8ee6ef-1ae9-40f8-920c-939d24a7a626/guilherme/Bases/115_Outro/BFL/BFL_115_128_2048_Outro.Test','a')
                        j=1
                        fj.write(str(c))
                        for i in get_feature([dataTrain])[0][0]:
                            fj.write(' '+str(j)+':')
                            fj.write(str(i))
                            j+=1
                        fj.write('\n')
                        j=1
                        fj.close()
                elif(diss == True):
                    if(folderTrainTest == 'Ingles1/' or folderTrainTest == 'Ingles2/' or folderTrainTest == 'Ingles4/' or folderTrainTest == 'Ingles3/'):
                    #if(folderTrainTest == 'Treino/'):
                        fi = open('/media/guilherme/1e8ee6ef-1ae9-40f8-920c-939d24a7a626/guilherme/Bases/Restante/CVL/4_1_Alemao1_Full/CVL_309_128_2048_Diss_Alemao1.Train','a')
                        for i in get_feature([dataTrain])[0][0]:
                            fi.write(str(i)+' ')
                        fi.write(str(c))                            
                        fi.write('\n')
                        fi.close()
                    elif(folderTrainTest == 'Alemao1/'):
                    #elif(folderTrainTest == 'Teste/'):
                        fj = open('/media/guilherme/1e8ee6ef-1ae9-40f8-920c-939d24a7a626/guilherme/Bases/Restante/CVL/4_1_Alemao1_Full/CVL_309_128_2048_Diss_Alemao1.Test','a')
                        for i in get_feature([dataTrain])[0][0]:
                            fj.write(str(i)+' ')
                        fj.write(str(c))
                        fj.write('\n')
                        j=1
                        fj.close()   

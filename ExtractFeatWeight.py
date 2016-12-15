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
import time
import sys

#OUTPUT FORMAT FILES:
#   Diss = True: feat_1 feat_2 feat_3 ... feat_n class_1
#                feat_1 feat_2 feat_3 ... feat_n class_2
#                                                      .
#                                                      .
#                                                      .                                         
#                feat_1 feat_2 feat_3 ... feat_n class_n

#   Diss = False (SVM FORMAT): class_1 1:feat_1 2:feat_2 3:feat_3 ... n:feat_n
#                              class_2 1:feat_1 2:feat_2 3:feat_3 ... n:feat_n
#                                                                            .
#                                                                            .
#                                                                            .                                        
#                              class_n 1:feat_1 2:feat_2 3:feat_3 ... n:feat_n 

try:
    if(sys.argv[1] != None):
        path_images = sys.argv[1]
        path_weights = sys.argv[2]
        path_output_file_train = sys.argv[7]
        path_output_file_test = sys.argv[8]
        images_format = '.bmp'
        nb_img_class = sys.argv[3]
        train = 60
        test = 40
        classes = sys.argv[4]
        nb_block = sys.argv[5]
        block_height = sys.argv[6]
        block_width = sys.argv[6]
        preName = "CF"
        sep="_"
        channels_img = 1
except:
    path_images = "../FolderImages"
    path_weights = "../file_weights.hdf5"
    path_output_file_train = "../file.train"
    path_output_file_test = "../file.test"
    images_format = '.bmp'
    nb_img_class = 3
    train = 60
    test = 40
    classes = 50
    nb_block = 105
    block_height = 128
    block_width = 128
    preName = "CF"
    sep="_"
    channels_img = 1


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

    model.add(Convolution2D(128, 2, 2, border_mode='same'))
    model.add(Activation('relu'))

    model.add(MaxPooling2D(pool_size=(2,2), strides=(2,2)))

    model.add(Flatten())

    model.add(Dense(2048))
    model.add(Activation("relu"))

    model.add(Dropout(0.5))

    model.add(Dense(classes))
    model.add(Activation("softmax"))

    model.load_weights(path_weights)

    return model


if __name__ == "__main__":

    #Flag to calculate the dissimilarity approach (github.com/guirighetto/Dissimilaridade)
    diss = False
   
    numSamplesTrain = float(nb_img_class*(float(train)/100))
    numSamplesTrain = round(numSamplesTrain)

    model = buildModel()

    model.compile(loss="categorical_crossentropy", optimizer="Adadelta",metrics=['accuracy'])

    start_time = time.time()

    get_feature = K.function([model.layers[0].input],[model.layers[16].output])

    for c in range(1,classes+1):
        for s in range(1,nb_img_class+1):
            if(s == 1 or s == 2):
                folderTrainTest = 'Train/'
            elif(s == 3):
                folderTrainTest = 'Test/'
            else:
                folderTrainTest = None

            for b in range(1,nb_block+1):
                if(folderTrainTest == 'Train/' or folderTrainTest == 'Test/'):     
                    nameImg = preName + str(c).zfill(5) + sep + str(s) + sep + str(b)
                    folderClass = preName + str(c).zfill(5) + '/'
                    fullPathImg = path_images + folderClass + folderTrainTest + nameImg + images_format
                    image = plt.imread(fullPathImg)
                    image = image[np.newaxis]

                    dataTrain = []

                    dataTrain.append(image)

                    dataTrain = np.array(dataTrain)
                    dataTrain = dataTrain.astype("float32")/255

                if(diss == False):
                    if(folderTrainTest == 'Train/'):
                        fi = open(path_output_file_train,'a')
                        k=1
                        fi.write(str(c))
                        for i in get_feature([dataTrain])[0][0]:
                            fi.write(' '+str(k)+':')
                            fi.write(str(i))
                            k+=1
                        fi.write('\n')
                        k=1
                        fi.close()
                    elif(folderTrainTest == 'Test/'):
                        fj = open(path_output_file_test,'a')
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
                    if(folderTrainTest == 'Train/'):
                        fi = open(path_output_file_train,'a')
                        for i in get_feature([dataTrain])[0][0]:
                            fi.write(str(i)+' ')
                        fi.write(str(c))                            
                        fi.write('\n')
                        fi.close()
                    elif(folderTrainTest == 'Test/'):
                        fj = open(path_output_file_test,'a')
                        for i in get_feature([dataTrain])[0][0]:
                            fj.write(str(i)+' ')
                        fj.write(str(c))
                        fj.write('\n')
                        j=1
                        fj.close()

    print("Done. Time: ",(time.time() - start_time))
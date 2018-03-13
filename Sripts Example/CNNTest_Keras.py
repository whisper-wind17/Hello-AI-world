'''
Created on 11-03-2017

@author: ChenPei
'''

from __future__ import print_function
import scipy.io as sio  
import matplotlib.pyplot as plt  
import numpy as np  
from sklearn.model_selection import train_test_split
import time 
from sklearn.metrics import log_loss

from keras.models import Sequential
from keras.layers import Dense, Dropout, Activation, Flatten, Input
from keras.layers import Conv2D, MaxPooling2D, AveragePooling2D, BatchNormalization, GlobalAveragePooling2D, GlobalMaxPooling2D
from keras.layers import add, concatenate
from keras.utils import np_utils
from keras.optimizers import SGD, Adam, RMSprop, Adagrad, Adadelta, Adamax, Nadam
from keras.callbacks import ModelCheckpoint
from keras.models import load_model, Model

import gc

### one-hot encode
def one_hot_encode_object_array(arr):
    uniques, ids = np.unique(arr, return_inverse=True)
    return np_utils.to_categorical(ids, len(uniques))

def LoadingBinaryData(_path, vecNum, rssNum):
    rssData = np.fromfile(_path, dtype='int8')
    binNum = len(rssData)
    sampleNum = np.int32(binNum/(vecNum*rssNum+2))
    rssData =  np.reshape(rssData[:sampleNum*(vecNum*rssNum+2)], (sampleNum, vecNum*rssNum+2))    

    label = rssData[:,0]
    label.shape = sampleNum, 1 

    rssData = np.delete(rssData, [0,1], axis=1)
    rssData = np.reshape(rssData, (sampleNum*vecNum, rssNum))

    tmpMean = np.zeros((sampleNum*vecNum, 1), dtype='int8')
    tmpMean[:,0] = np.mean(rssData, axis=1).astype('int8')
    rssData = rssData - tmpMean  

    rssData = np.reshape(rssData, (sampleNum, vecNum, rssNum, 1))
    
    return rssData, label    


def FirstConvBlock(input, nb_filters):
    out1 = Conv2D(filters=nb_filters, kernel_size=(1,7),strides=(1,1),padding='same',
                activation=None, use_bias=True, kernel_initializer='random_normal', bias_initializer='zeros')(input)
    out1 = BatchNormalization(axis=-1, momentum=0.9, epsilon=0.0001,
                beta_initializer='zeros', gamma_initializer='ones')(out1)
    out1 = Activation('relu')(out1)
    
    return out1

def IdentityBlock(input, nb_filters):   
            
    out = BatchNormalization(axis=-1, momentum=0.9, epsilon=0.0001,
                beta_initializer='zeros', gamma_initializer='ones')(input)
    out = Activation('relu')(out)
    out = Conv2D(filters=nb_filters, kernel_size=(1,3),strides=(1,1),padding='same',
                activation=None, use_bias=True, kernel_initializer='random_normal', bias_initializer='zeros')(out)      
    
    return out



##################################################################################
VECNUM = 10 
RSSNUM = 400 
train_split_size = 0.9 

dataDir = "/home/jidian/chen_dnn/DataSet_BinaryFormat/"
dataPosName = "V2_Pre80_DataSetCar_Factory_0109_08_10_12_15_16_17__10x400.dbin"

dataNegName = "V2_Pre80_DataSetNeg_Factory_0205_10x400_1_10.dbin"

print(dataPosName,dataNegName )

################   training data reading 
#####  human data reading : WuShi (training)
print('\n*** Training Data Human  Factory : ')
_path = dataDir + dataPosName
dataHumanW, labelHumanW = LoadingBinaryData(_path, VECNUM, RSSNUM)
dataHumanWTrain, dataHumanWVal, labelHumanTrain, labelHumanWVal = train_test_split(dataHumanW, labelHumanW, 
                                                              train_size = train_split_size, random_state = 0)
del dataHumanW, labelHumanW, labelHumanTrain
gc.collect()
print('xTrain and xVal(WuShi) : ', dataHumanWTrain.shape, dataHumanWVal.shape )

#### rain data reading: WuShi airport (training)
print('\n*** Training Data rain  WuShi airport : ')
_path = dataDir + dataNegName
dataRain, labelRain = LoadingBinaryData(_path, VECNUM, RSSNUM)
labelRain = 1 - labelRain 
dataRainTrain, dataRainVal, labelRainTrain, labelRainVal = train_test_split(dataRain, labelRain, 
                                                              train_size = train_split_size, random_state = 0)
del dataRain, labelRain, labelRainTrain
gc.collect()
print('xTrain and xVal(low) : ', dataRainTrain.shape, dataRainVal.shape )

####### validation set formning
dataVal = np.concatenate(( dataHumanWVal, dataRainVal))
labelVal = np.concatenate((labelHumanWVal, labelRainVal))
del dataHumanWVal, dataRainVal
del labelHumanWVal, labelRainVal
gc.collect()
labelValOneHot = one_hot_encode_object_array(labelVal)

print('\n validation data: ', dataVal.shape)

###############################################################################################################
###############################################################################################################
###############################################################################################################
#######  cnn model
##### Input layer
input = Input(shape=(10, 400, 1))
#### First layer
out = FirstConvBlock(input, nb_filters=32)
out = MaxPooling2D(pool_size=[1,2],strides=[1,2],padding='valid')(out)   # 10x200

#### ConvBlock and Identityblock: time layer
out = IdentityBlock(out, 32) 
out = MaxPooling2D(pool_size=[1,2],strides=[1,2],padding='valid')(out)    # 10x100

out = IdentityBlock(out, 32) 
#out = ConvBlock(out, 48)
out = MaxPooling2D(pool_size=[1,2],strides=[1,2],padding='valid')(out)    # 10x50


out = IdentityBlock(out, 64) 
out = MaxPooling2D(pool_size=[1,2],strides=[1,2],padding='valid')(out)    # 10x25

out = IdentityBlock(out, 64) 
out = MaxPooling2D(pool_size=[1,2],strides=[1,2],padding='valid')(out)    # 10x12

out = IdentityBlock(out, 64) 
out = MaxPooling2D(pool_size=[1,2],strides=[1,2],padding='valid')(out)    # 10x6

out = IdentityBlock(out, 64)
out = MaxPooling2D(pool_size=[1,2],strides=[1,2],padding='valid')(out)    # 10x3

out = Flatten()(out)
out = Dropout(0.7)(out)
out = Dense(60, activation='relu')(out)
out = Dense(2, activation = 'softmax')(out)

#################################################################################################################
#################################################################################################################
#################################################################################################################
model = Model(inputs=[input], outputs=out)

####  optimizer, loss
sgd = SGD(lr=0.0002, decay=5e-2, momentum=0.9, nesterov=True)  
adadelta = Adadelta(lr=1.0, rho=0.95, epsilon=1e-08, decay=0.0)   #### best

#rmsprop = RMSprop(lr=0.001, rho=0.9, epsilon=1e-08, decay=0.0)
#adagrad = Adagrad(lr=0.01, epsilon=1e-08, decay=0.0)
#adam = Adam(lr=0.001, beta_1=0.9, beta_2=0.999, epsilon=1e-08, decay=0.0)
#adamax = Adamax(lr=0.002, beta_1=0.9, beta_2=0.999, epsilon=1e-08, decay=0.0)
#nadam = Nadam(lr=0.002, beta_1=0.9, beta_2=0.999, epsilon=1e-08, schedule_decay=0.004)
model.compile(loss='categorical_crossentropy', optimizer=adadelta, metrics=['accuracy'])

for i, layer in enumerate(model.layers):
   print(i, layer.name)

print('\n model.summary: \n')
model.summary()


print('\n training ....... \n')

#################################################################################################

######## 1st stage trainning 
dataHumanWNum, vecNum, rssNum, chanNum = dataHumanWTrain.shape
dataRainNum = len(dataRainTrain)

maxLen = np.max([dataHumanWNum, dataRainNum])
dataTrain = np.zeros(shape=(2*maxLen, vecNum, rssNum, chanNum))
labelTrain = np.zeros(shape=(2*maxLen, 1))
indexVec = np.arange(maxLen)
print('\n traing data: ', dataTrain.shape)

np.random.shuffle(indexVec)
for index in np.arange(maxLen):        
    humanWInd = np.mod(indexVec[index], dataHumanWNum)                
    rainInd = np.mod(indexVec[index], dataRainNum)        
    dataTrain[index*2, :, :, :] = dataHumanWTrain[humanWInd, :, :, :]        
    dataTrain[index*2+1, :, :, :] = dataRainTrain[rainInd, :, :, :]
    labelTrain[index*2, :] = 1
    labelTrain[index*2+1, :] = 0 
labelTrainOneHot = one_hot_encode_object_array(labelTrain)

history = model.fit(dataTrain, labelTrainOneHot, batch_size=128, epochs=10, verbose=1, 
              validation_split=0.0, shuffle = True,
              validation_data = (dataVal, labelValOneHot))

del dataTrain, labelTrainOneHot
gc.collect()

########################################################################

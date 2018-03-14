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
from keras.callbacks import ModelCheckpoint, EarlyStopping
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

def BinDataToSampleFormat(dataReadBytes, sampleLoadNum):
    vecNum = 10 
    rssNum = 400
    rssData = np.reshape(dataReadBytes, (sampleLoadNum, 4002)) 
    label = rssData[:,0]
    label.shape = sampleLoadNum, 1 

    rssData = np.delete(rssData, [0,1], axis=1)
    rssData = np.reshape(rssData, (sampleLoadNum*vecNum, rssNum))

    tmpMean = np.zeros((sampleLoadNum*vecNum, 1), dtype='int8')
    tmpMean[:,0] = np.mean(rssData, axis=1).astype('int8')
    rssData = rssData - tmpMean  

    rssData = np.reshape(rssData, (sampleLoadNum, vecNum, rssNum, 1))
    return rssData, label

   
def PredictionSummary(predLabel, trueLabel):
    lenLabel = len(trueLabel)
    tmpDiff = predLabel[:, 0] - predLabel[:, 1]
    tmpDiff = np.reshape(tmpDiff, [lenLabel, 1]) 
    idx_0 = np.where(tmpDiff>0)[0]
    idx_1 = np.where(tmpDiff<0)[0] 
    tmpDiff[idx_0] = 0  
    tmpDiff[idx_1] = 1
    tmpDiff2 = tmpDiff - trueLabel
    accuracy = len(tmpDiff2[tmpDiff2==0])/lenLabel
    missAlarm = len(tmpDiff2[tmpDiff2==-1])/len(trueLabel==1) 
    falseAlarm = len(tmpDiff2[tmpDiff2==1])/len(trueLabel==0)
    return accuracy, missAlarm, falseAlarm

def FirstConvBlock(input, nb_filters):
    k1, k2, k3 = nb_filters
    out1 = Conv2D(filters=k1, kernel_size=(1,15),strides=(1,1),padding='same',
                activation=None, use_bias=True, kernel_initializer='random_normal', bias_initializer='zeros')(input)
    out1 = BatchNormalization(axis=-1, momentum=0.9, epsilon=0.0001,
                beta_initializer='zeros', gamma_initializer='ones')(out1)
    out1 = Activation('relu')(out1)

    out2 = Conv2D(filters=k2, kernel_size=(1,11),strides=(1,1),padding='same',
                activation=None, use_bias=True, kernel_initializer='random_normal', bias_initializer='zeros')(input)
    out2 = BatchNormalization(axis=-1, momentum=0.9, epsilon=0.0001,
                beta_initializer='zeros', gamma_initializer='ones')(out2)
    out2 = Activation('relu')(out2) 

    out3 = Conv2D(filters=k3, kernel_size=(1,7),strides=(1,1),padding='same',
                activation=None, use_bias=True, kernel_initializer='random_normal', bias_initializer='zeros')(input)
    out3 = BatchNormalization(axis=-1, momentum=0.9, epsilon=0.0001,
                beta_initializer='zeros', gamma_initializer='ones')(out3)
    out3 = Activation('relu')(out3)
    '''
    out4 = Conv2D(filters=k4, kernel_size=(1,3),strides=(1,1),padding='same',
                activation=None, use_bias=True, kernel_initializer='random_normal', bias_initializer='zeros')(input)
    out4 = BatchNormalization(axis=-1, momentum=0.9, epsilon=0.0001,
                beta_initializer='zeros', gamma_initializer='ones')(out4)
    out4 = Activation('relu')(out4)
    '''
    out = concatenate([out1,out2,out3])
    return out

def IdentityBlock(input, nb_filters):
    k1, k2, k3 = nb_filters
    out = Conv2D(filters=k1, kernel_size=(1,1),strides=(1,1),padding='same',
                activation=None, use_bias=True, kernel_initializer='random_normal', bias_initializer='zeros')(input)
    out = BatchNormalization(axis=-1, momentum=0.9, epsilon=0.0001,
                beta_initializer='zeros', gamma_initializer='ones')(out)
    out = Activation('relu')(out)

    out = Conv2D(filters=k2, kernel_size=(1,3),strides=(1,1),padding='same',
                activation=None, use_bias=True, kernel_initializer='random_normal', bias_initializer='zeros')(out)
    out = BatchNormalization(axis=-1, momentum=0.9, epsilon=0.0001,
                beta_initializer='zeros', gamma_initializer='ones')(out)
    out = Activation('relu')(out)

    out = Conv2D(filters=k2, kernel_size=(1,3),strides=(1,1),padding='same',
                activation=None, use_bias=True, kernel_initializer='random_normal', bias_initializer='zeros')(out)
    out = BatchNormalization(axis=-1, momentum=0.9, epsilon=0.0001,
                beta_initializer='zeros', gamma_initializer='ones')(out)
    out = Activation('relu')(out)
    '''
    out = Conv2D(filters=k2, kernel_size=(1,3),strides=(1,1),padding='same',
                activation=None, use_bias=True, kernel_initializer='random_normal', bias_initializer='zeros')(out)
    out = BatchNormalization(axis=-1, momentum=0.9, epsilon=0.0001,
                beta_initializer='zeros', gamma_initializer='ones')(out)
    out = Activation('relu')(out)
    '''
    out = Conv2D(filters=k3, kernel_size=(1,1),strides=(1,1),padding='same',
                activation=None, use_bias=True, kernel_initializer='random_normal', bias_initializer='zeros')(out)
    out = BatchNormalization(axis=-1, momentum=0.9, epsilon=0.0001,
                beta_initializer='zeros', gamma_initializer='ones')(out)

    out = add([input, out])
    out = Activation('relu')(out)
    return out

def ConvBlock(input, nb_filters):
    k1, k2, k3 = nb_filters
    out = Conv2D(filters=k1, kernel_size=(1,1),strides=(1,1),padding='same',
                activation=None, use_bias=True, kernel_initializer='random_normal', bias_initializer='zeros')(input)
    out = BatchNormalization(axis=-1, momentum=0.9, epsilon=0.0001,
                beta_initializer='zeros', gamma_initializer='ones')(out)
    out = Activation('relu')(out)

    out = Conv2D(filters=k2, kernel_size=(1,3),strides=(1,1),padding='same',
                activation=None, use_bias=True, kernel_initializer='random_normal', bias_initializer='zeros')(out)
    out = BatchNormalization(axis=-1, momentum=0.9, epsilon=0.0001,
                beta_initializer='zeros', gamma_initializer='ones')(out)
    out = Activation('relu')(out)

    out = Conv2D(filters=k3, kernel_size=(1,1),strides=(1,1),padding='same',
                activation=None, use_bias=True, kernel_initializer='random_normal', bias_initializer='zeros')(out)
    out = BatchNormalization(axis=-1, momentum=0.9, epsilon=0.0001,
                beta_initializer='zeros', gamma_initializer='ones')(out)
    out = Activation('relu')(out)

    out2 = Conv2D(filters=k3, kernel_size=(1,1),strides=(1,1),padding='same',
                activation=None, use_bias=True, kernel_initializer='random_normal', bias_initializer='zeros')(input)
    out2 = BatchNormalization(axis=-1, momentum=0.9, epsilon=0.0001,
                beta_initializer='zeros', gamma_initializer='ones')(out2)

    out = add([out, out2])
    out = Activation('relu')(out)
    return out 

def ConvSpaceBlock(input, nb_filters):
    k1, k2, k3 = nb_filters
    out = Conv2D(filters=k1, kernel_size=(1,3),strides=(1,1),padding='same',
                activation=None, use_bias=True, kernel_initializer='random_normal', bias_initializer='zeros')(input)
    out = BatchNormalization(axis=-1, momentum=0.9, epsilon=0.0001,
                beta_initializer='zeros', gamma_initializer='ones')(out)
    out = Activation('relu')(out)

    out = Conv2D(filters=k2, kernel_size=(1,3),strides=(1,1),padding='same',
                activation=None, use_bias=True, kernel_initializer='random_normal', bias_initializer='zeros')(out)
    out = BatchNormalization(axis=-1, momentum=0.9, epsilon=0.0001,
                beta_initializer='zeros', gamma_initializer='ones')(out)
    out = Activation('relu')(out)

    out = Conv2D(filters=k3, kernel_size=(1,1),strides=(1,1),padding='same',
                activation=None, use_bias=True, kernel_initializer='random_normal', bias_initializer='zeros')(out)
    out = BatchNormalization(axis=-1, momentum=0.9, epsilon=0.0001,
                beta_initializer='zeros', gamma_initializer='ones')(out)
    out = Activation('relu')(out)

    return out 
    

###############################################################################################################
###############################################################################################################
###############################################################################################################
#######  cnn model
##### Input layer
input = Input(shape=(10, 400, 1))
#### First layer
out = FirstConvBlock(input, [16, 16, 16])

#### ConvBlock and Identityblock: time layer
out = IdentityBlock(out, [32,32,48]) 
#out = IdentityBlock(out, [64,64,96])  
#out = AveragePooling2D(pool_size=[1,2],strides=[1,2],padding='valid')(out)
out = MaxPooling2D(pool_size=[1,2],strides=[1,2],padding='valid')(out)

#out = ConvBlock(out, [96, 96, 128])
out = IdentityBlock(out, [32,32,48]) 
#out = IdentityBlock(out, [64,64,96])  
#out = AveragePooling2D(pool_size=[1,2],strides=[1,2],padding='valid')(out)
out = MaxPooling2D(pool_size=[1,2],strides=[1,2],padding='valid')(out)

#out = ConvBlock(out, [128, 128, 160])
out = IdentityBlock(out, [32,32,48]) 
#out = IdentityBlock(out, [64,64,96])  
#out = AveragePooling2D(pool_size=[1,2],strides=[1,2],padding='valid')(out)
out = MaxPooling2D(pool_size=[1,2],strides=[1,2],padding='valid')(out)

out = IdentityBlock(out, [32,32,48]) 
#out = IdentityBlock(out, [32,32,48]) 
#out = IdentityBlock(out, [64,64,96])  
#out = AveragePooling2D(pool_size=[1,2],strides=[1,2],padding='valid')(out)
out = MaxPooling2D(pool_size=[1,2],strides=[1,2],padding='valid')(out)

out = IdentityBlock(out, [32,32,48]) 
#out = IdentityBlock(out, [64,64,96])  
#out = AveragePooling2D(pool_size=[1,2],strides=[1,2],padding='valid')(out)
out = MaxPooling2D(pool_size=[1,2],strides=[1,2],padding='valid')(out)

out = IdentityBlock(out, [32,32,48])
out = MaxPooling2D(pool_size=[1,2],strides=[1,2],padding='valid')(out)

####### space-time layer
out = ConvSpaceBlock(out, [48, 48, 48])

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

checkpointer = ModelCheckpoint(filepath='/home/jidian/chen_dnn/Models/weights2_10x400_Factory_totalSet.hdf5', monitor='val_loss', verbose=1, 
                                save_best_only=False, save_weights_only=True, mode='auto', period=1)
earlyStopping = EarlyStopping(monitor='val_loss', min_delta=1e-3, patience=1, verbose=0, mode='auto')

print('\n training ....... \n')
print('loading the saved weights .......')
model.load_weights('/home/jidian/chen_dnn/Models/weights2_10x400_Factory_totalSet.hdf5')
#################################################################################################
##########  Data reading from directory 
VECNUM = 10 
RSSNUM = 400 
train_split_size = 0.8 

dataDir = "/home/jidian/chen_dnn/DataSet_BinaryFormat/"
dataNegFilePath = dataDir + "DataSetNeg_Factory_WuShi_Total_10x400.dbin"
dataPosFilePath = dataDir + "DataSetPositive_Factory_WuShi_QinShan_Total_10x400.dbin"
####### neg data file 
frid = open(dataNegFilePath, "rb")
frid.seek(0, 2)
dataNegBytesNum = frid.tell()
frid.close()
sampNegNum = int(dataNegBytesNum / 4002 )
print("Neg data : %d, %d"%(dataNegBytesNum, sampNegNum))
###### pos data file
frid = open(dataPosFilePath, "rb")
frid.seek(0, 2)
dataPosBytesNum = frid.tell()
frid.close()
sampPosNum = int(dataPosBytesNum / 4002 )
print("Pos data : %d, %d"%(dataPosBytesNum, sampPosNum))

######
sampleLoadNum = 50000 ; 
dataLoadByteNum = sampleLoadNum * 4002 
dataLoadByteSlew = 40000 * 4002
offsetNeg = 0 
offsetPos = 0
indexNeg = 0 
indexPos = 0 
for loop_ind in range(1000):
    print("\n\nloop_ind:%d"%(loop_ind))

    ###### negative samples reading
    endOffset = offsetNeg + dataLoadByteNum 
    if ( endOffset > dataNegBytesNum ):
        dataLoadNumTmp1 = dataNegBytesNum-offsetNeg        
        dataReadBytes1 = np.memmap(dataNegFilePath, dtype="int8", mode="r", offset=offsetNeg, shape=(1,dataLoadNumTmp1))
        offsetNeg = (offsetNeg+dataLoadNumTmp1)%dataNegBytesNum        
        dataLoadNumTmp1 = dataLoadByteNum - dataLoadNumTmp1
        dataReadBytes2 = np.memmap(dataNegFilePath, dtype="int8", mode="r", offset=offsetNeg, shape=(1,dataLoadNumTmp1))
        offsetNeg = (offsetNeg+dataLoadNumTmp1)%dataNegBytesNum        

        dataReadBytes = np.append(dataReadBytes1,dataReadBytes2)
    else:
        dataReadBytes = np.memmap(dataNegFilePath, dtype="int8", mode="r", offset=offsetNeg, shape=(1,dataLoadByteNum))
        offsetNeg = (offsetNeg+dataLoadByteSlew)%dataNegBytesNum 

    dataRain, label = BinDataToSampleFormat(dataReadBytes, sampleLoadNum)       
    labelRain = 1 - label 
    print("negative sample: sum of labels:%d"%(np.sum(labelRain))) 

    ###### positive samples reading
    endOffset = offsetPos + dataLoadByteNum 
    if ( endOffset > dataPosBytesNum ):
        dataLoadNumTmp1 = dataPosBytesNum-offsetPos        
        dataReadBytes1 = np.memmap(dataPosFilePath, dtype="int8", mode="r", offset=offsetPos, shape=(1,dataLoadNumTmp1))
        offsetPos = (offsetPos+dataLoadNumTmp1)%dataPosBytesNum        
        dataLoadNumTmp1 = dataLoadByteNum - dataLoadNumTmp1
        dataReadBytes2 = np.memmap(dataPosFilePath, dtype="int8", mode="r", offset=offsetPos, shape=(1,dataLoadNumTmp1))
        offsetPos = (offsetPos+dataLoadNumTmp1)%dataPosBytesNum        

        dataReadBytes = np.append(dataReadBytes1,dataReadBytes2) 
    else:
        dataReadBytes = np.memmap(dataPosFilePath, dtype="int8", mode="r", offset=offsetPos, shape=(1,dataLoadByteNum))
        offsetPos = (offsetPos+dataLoadByteSlew)%dataPosBytesNum    

    dataHuman, labelHuman = BinDataToSampleFormat(dataReadBytes, sampleLoadNum)
    print("positive samples : sum of labels:%d"%(np.sum(labelHuman))) 

    indexNeg = ( indexNeg + sampleLoadNum ) % sampNegNum
    indexPos = ( indexPos + sampleLoadNum ) % sampPosNum
    print("indexNeg : %d / %d(total) "%(indexNeg, sampNegNum ))
    print("indexPos : %d / %d(total) "%(indexPos, sampPosNum ))

    ##### spliting 
    dataRainTrain, dataRainVal, labelRainTrain, labelRainVal = train_test_split(dataRain, labelRain, 
                                                              train_size = train_split_size, random_state = 0)
    dataHumanTrain, dataHumanVal, labelHumanTrain, labelHumanVal = train_test_split(dataHuman, labelHuman, 
                                                              train_size = train_split_size, random_state = 0)
    #### validation set
    dataVal = np.concatenate(( dataHumanVal, dataRainVal))
    labelVal = np.concatenate((labelHumanVal, labelRainVal))
    labelValOneHot = one_hot_encode_object_array(labelVal)

    #### training set 
    dataTrain = np.concatenate(( dataRainTrain, dataHumanTrain))
    labelTrain = np.concatenate((labelRainTrain, labelHumanTrain))
    labelTrainOneHot = one_hot_encode_object_array(labelTrain)

    #model.compile(loss='categorical_crossentropy', optimizer=adadelta, metrics=['accuracy'])
    #checkpointer = ModelCheckpoint(filepath='/home/jidian/chen_dnn/Models/weights2_10x400_Factory_totalSet.hdf5', monitor='val_loss', verbose=1, 
    #                            save_best_only=True, save_weights_only=True, mode='auto', period=1)
    #earlyStopping = EarlyStopping(monitor='val_loss', min_delta=1e-3, patience=1, verbose=0, mode='auto')
    #print('loading the saved weights .......')
    #model.load_weights('/home/jidian/chen_dnn/Models/weights2_10x400_Factory_totalSet.hdf5')

    history = model.fit(dataTrain, labelTrainOneHot, batch_size=128, epochs=1, verbose=1, 
                  validation_split=0.0, shuffle = True,
                  validation_data = (dataVal, labelValOneHot), callbacks=[checkpointer, earlyStopping])


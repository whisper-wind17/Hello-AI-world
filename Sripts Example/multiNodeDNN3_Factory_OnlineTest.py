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
    wrongInd = np.where(tmpDiff2 != 0)[0]
    return accuracy, missAlarm, falseAlarm, wrongInd

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

    out = concatenate([out1,out2,out3])
    return out

def IdentityBlock(input, nb_filters):
    k1, k2, k3 = nb_filters    
        
    out = BatchNormalization(axis=-1, momentum=0.9, epsilon=0.0001,
                beta_initializer='zeros', gamma_initializer='ones')(input)
    out = Activation('relu')(out)
    out = Conv2D(filters=k1, kernel_size=(1,1),strides=(1,1),padding='same',
                activation=None, use_bias=True, kernel_initializer='random_normal', bias_initializer='zeros')(out)
        
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
    out = Conv2D(filters=k3, kernel_size=(1,1),strides=(1,1),padding='same',
                activation=None, use_bias=True, kernel_initializer='random_normal', bias_initializer='zeros')(out)

    out = add([input, out])    
    return out

def TransitionBlock(input, nb_filters, filter_size=(1,1)):        
    out = BatchNormalization(axis=-1, momentum=0.9, epsilon=0.0001,
                beta_initializer='zeros', gamma_initializer='ones')(input)
    out = Activation('relu')(out)
    out = Conv2D(filters=nb_filters, kernel_size=filter_size,strides=(1,1),padding='valid',
                activation=None, use_bias=True, kernel_initializer='random_normal', bias_initializer='zeros')(out)
    return out

##################################################################################
VECNUM = 10 
RSSNUM = 400 
train_split_size = 0.95 

dataDir = "/home/jidian/chen_dnn/DataSet_BinaryFormat/"
#dataPosName = "Pre120_DataSetPositive_Factory_Total_10x400_0115.dbin"    ### 正常入侵
#dataPosName = "Pre120_DataSetPositive_Factory_Total_10x400_0118.dbin"    ### 正常入侵+入侵停靠数据
#dataPosName = "Pre80_DataSetPositive_Factory_Total_10x400_0119.dbin"
#dataPosName = "Pre120_DataSetPositive_Factory_0116_17__10x400.dbin"
#dataPosName = "Pre80_DataSetPositive_Factory_0112_15_16_17__10x400.dbin"
#dataPosName = "V2_Pre80_DataSetCar_Factory_0109__10x400.dbin"
#dataPosName = "V2_Pre80_DataSetCar_Factory_0109_08__10x400.dbin"
#dataPosName = "V2_Pre80_DataSetCar_Factory_0110__10x400.dbin"
#dataPosName = "V2_Pre80_DataSetCar_Factory_0109_08_10__10x400.dbin"
#dataPosName = "V2_Pre80_DataSetCar_Factory_0112__10x400.dbin"
#dataPosName = "V2_Pre80_DataSetCar_Factory_0109_08_10_12__10x400.dbin"
#dataPosName = "V2_Pre80_DataSetCar_Factory_0117__10x400.dbin"
dataPosName = "V2_Pre80_DataSetCar_Factory_0109_08_10_12_15_16_17__10x400.dbin"

#dataNegName = "Pre120_DataSetNeg_Factory_Total_10x400_0118.dbin"
#dataNegName = "Pre80_DataSetNeg_Factory_0119_10x400_2.dbin"
#dataNegName = "Pre80_DataSetNeg_Factory_Total_10x400_0123_1.dbin"
#dataNegName = "Pre80_DataSetCar_Factory_0126__10x400.dbin"
#dataNegName = "Pre80_DataSetNeg_Car_Factory_Total_10x400_0123_1.dbin"

#dataNegName = "Pre80_DataSetNeg_Car_Factory_Total_10x400_0129_5.dbin"
#dataNegName = "Pre80_DataSetNeg_Factory_Total_10x400_0129_5.dbin"
#dataNegName = "Pre80_DataSetNeg_Factory_0205_10x400_4.dbin"
#dataNegName = "Pre80_DataSetNeg_Factory_Total_10x400_0205_4.dbin"
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
out = FirstConvBlock(input, nb_filters=[24, 24, 16])
out = MaxPooling2D(pool_size=[1,2],strides=[1,2],padding='valid')(out)   # 10x200

#out = TransitionBlock(out,nb_filters=64, filter_size=(1,1))

#### ConvBlock and Identityblock: time layer
out = IdentityBlock(out, [48,48,64]) 
out = MaxPooling2D(pool_size=[1,2],strides=[1,2],padding='valid')(out)    # 10x100

out = IdentityBlock(out, [48,48,64]) 
#out = ConvBlock(out, 48)
out = MaxPooling2D(pool_size=[1,2],strides=[1,2],padding='valid')(out)    # 10x50

out = TransitionBlock(out,nb_filters=96, filter_size=(1,1))

out = IdentityBlock(out, [64,64,96]) 
out = MaxPooling2D(pool_size=[1,2],strides=[1,2],padding='valid')(out)    # 10x25

out = IdentityBlock(out, [64,64,96]) 
out = MaxPooling2D(pool_size=[1,2],strides=[1,2],padding='valid')(out)    # 10x12

#out = TransitionBlock(out,nb_filters=128, filter_size=(1,1))

out = IdentityBlock(out, [64,64,96]) 
out = MaxPooling2D(pool_size=[1,2],strides=[1,2],padding='valid')(out)    # 10x6

out = IdentityBlock(out, [64,64,96])
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

checkpointer = ModelCheckpoint(filepath='/home/jidian/chen_dnn/Models/weights6_10x400_Factory.hdf5', monitor='val_loss', verbose=1, 
                                save_best_only=True, save_weights_only=True, mode='auto', period=1)
print('\n training ....... \n')
print('loading the saved weights .......')
model.load_weights('/home/jidian/chen_dnn/Models/weights6_10x400_Factory.hdf5')

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
              validation_data = (dataVal, labelValOneHot), callbacks=[checkpointer])

del dataTrain, labelTrainOneHot
gc.collect()

########################################################################
print('loading the saved weights .......')
model.load_weights('/home/jidian/chen_dnn/Models/weights6_10x400_Factory.hdf5')
model.save('/home/jidian/chen_dnn/Models/KerasModels4b_Factory_10x400_0223.h5')
print('from weights to models finished!')
############ Testing dataset  : factory  human

############ Testing dataset (positive) : WuShi human 
print('\n*** Testing Data Positive  Factory : ')
print(dataPosName)
_path = dataDir + dataPosName
dataHumanW, labelHumanW = LoadingBinaryData(_path, VECNUM, RSSNUM)
print('data shape: ', dataHumanW.shape)

startTime = time.time()
predLabel = model.predict(dataHumanW, batch_size=256)
endTime = time.time()
accuracy, misAlarm, falseAlarm, wrongInd = PredictionSummary(predLabel, labelHumanW)
print('\n *** WuShi data prediction (human): ')
print('cost time(total): %.2f(s), average time: %.2f(ms)'%(endTime-startTime, (endTime-startTime)/len(dataHumanW)*1000))
print('accuracy: %.6f, misAlarm: %.6f, falseAlarm: %.6f\n'%(accuracy, misAlarm, falseAlarm))

idx_reserved = np.where(predLabel[:,1] < 0.99 )[0]
np.savetxt("pos_idx_reserved_0117.txt", idx_reserved, fmt="%d", delimiter="\n")

del dataHumanW, labelHumanW, predLabel
gc.collect()

############ Testing dataset (negative) : Factory rain 
print('\n*** Testing Data Neg  Factory : ')
print(dataNegName)
_path = dataDir + dataNegName
dataRain, labelRain = LoadingBinaryData(_path, VECNUM, RSSNUM)
print('data shape: ', dataRain.shape)
labelRain = 1 - labelRain 

startTime = time.time()
predLabel = model.predict(dataRain, batch_size=256)
endTime = time.time()
accuracy, misAlarm, falseAlarm, wrongInd= PredictionSummary(predLabel, labelRain)
print('\n *** WuShi data prediction (rain): ')
print('cost time(total): %.2f(s), average time: %.2f(ms)'%(endTime-startTime, (endTime-startTime)/len(dataRain)*1000))
print('accuracy: %.6f, misAlarm: %.6f, falseAlarm: %.6f\n'%(accuracy, misAlarm, falseAlarm))

idx_reserved = np.where(predLabel[:,0] < 0.99 )[0]
np.savetxt("neg_idx_reserved_0223_10.txt", idx_reserved, fmt="%d", delimiter="\n")

del dataRain, labelRain, predLabel
gc.collect()

########### Testing dataset: QinShan Airport

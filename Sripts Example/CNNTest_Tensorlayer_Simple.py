from __future__ import print_function
import tensorflow as tf 
import tensorlayer as tl
import numpy as np 
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OneHotEncoder

import time
import gc

##########################################################################################
###############  Macros 
VEC_NUM = 10 
RSS_NUM = 400 
CLASS_NUM = 2 
NUM_EPOCH = 10 
BATCH_SIZE = 256
PRINT_FREQ = 1

#############################################################################################
############################  Data preparation 
### one-hot encode
def one_hot_encode_object_array(arr):
    enc = OneHotEncoder()
    enc.fit(arr)    
    return enc.transform(arr).toarray()    

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

print(" loading data to memory ...." )
train_split_size = 0.9 
dataDir = "/home/jidian/chen_dnn/DataSet_BinaryFormat/"
dataPosName = "V2_Pre80_DataSetCar_Factory_0109_08_10_12_15_16_17__10x400.dbin"
dataNegName = "V2_Pre80_DataSetNeg_Factory_0205_10x400_1_10.dbin"

print(dataPosName,dataNegName)

################   training data reading 
#####  human data reading : WuShi (training)
print('\n*** Training Data Human  Factory : ')
_path = dataDir + dataPosName
dataHumanW, labelHumanW = LoadingBinaryData(_path, VEC_NUM, RSS_NUM)
dataHumanWTrain, dataHumanWVal, labelHumanTrain, labelHumanWVal = train_test_split(dataHumanW, labelHumanW, 
                                                              train_size = train_split_size, random_state = 0)
del dataHumanW, labelHumanW, labelHumanTrain
gc.collect()
print('xTrain and xVal(WuShi) : ', dataHumanWTrain.shape, dataHumanWVal.shape )

#### rain data reading: WuShi airport (training)
print('\n*** Training Data rain  WuShi airport : ')
_path = dataDir + dataNegName
dataRain, labelRain = LoadingBinaryData(_path, VEC_NUM, RSS_NUM)
labelRain = 1 - labelRain 
dataRainTrain, dataRainVal, labelRainTrain, labelRainVal = train_test_split(dataRain, labelRain, 
                                                              train_size = train_split_size, random_state = 0)
del dataRain, labelRain, labelRainTrain
gc.collect()
print('xTrain and xVal(low) : ', dataRainTrain.shape, dataRainVal.shape )

######## trainning reader
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
    labelTrain[index*2] = 1
    labelTrain[index*2+1] = 0 
labelTrainOneHot = one_hot_encode_object_array(labelTrain)
del dataHumanWTrain, dataRainTrain, labelTrain
gc.collect()

######### evaluation reader
####### validation set forming
dataVal = np.concatenate(( dataHumanWVal, dataRainVal))
labelVal = np.concatenate((labelHumanWVal, labelRainVal))
labelValOneHot = one_hot_encode_object_array(labelVal)
del dataHumanWVal, dataRainVal
del labelHumanWVal, labelRainVal
gc.collect()
print('\n validation data: ', dataVal.shape)


############################################################################################
############################ Graph
with tf.device('/cpu:0'):
    sess = tf.InteractiveSession(config=tf.ConfigProto(allow_soft_placement=True))    

    w_init = tf.truncated_normal_initializer(stddev=5e-2)
    #W_init2 = tf.truncated_normal_initializer(stddev=0.04)
    b_init = tf.constant_initializer(value=0.01)

    def conv_layer_first(net_in, filter_num=32, kernel_size=(1,15), strides=(1,1), padding='SAME', layer_name="conv",  training_flag=True):
        ##### conv-bn-relu 
        network = tl.layers.Conv2d(net_in, n_filter=filter_num, filter_size=kernel_size, strides=strides, act=None, W_init=w_init,b_init=b_init,name=layer_name+"_conv" )
        network = tl.layers.BatchNormLayer(network, act=tf.nn.relu, is_train=training_flag, beta_init=tf.zeros_initializer, gamma_init=tf.ones_initializer, name =layer_name+"_bn")
        #network = tf.nn.relu(network)
        return network 

    def conv_layer_stack(net_in, filter_num=32, kernel_size=(1,15), strides=(1,1), padding='SAME', layer_name="conv",  training_flag=True):
        #### bn-relu-conv
        network = tl.layers.BatchNormLayer(net_in, act=tf.nn.relu, is_train=training_flag, beta_init=tf.zeros_initializer, gamma_init=tf.ones_initializer, name =layer_name+"_bn")
        #network = tf.nn.relu(network)
        network = tl.layers.Conv2d(network, n_filter=filter_num, filter_size=kernel_size, strides=strides, act=None, W_init=w_init,b_init=b_init,name=layer_name+"_conv" )
        return network

    def NetworkCNN_Model(x, y, reuse, is_training):        
        with tf.variable_scope("StackCNN", reuse=reuse):
            tl.layers.set_name_reuse(reuse)

            net = tl.layers.InputLayer(x, name='input_layer')
            ####### first conv_block
            net = conv_layer_first(net, filter_num=32, kernel_size=(1,7), strides=(1,1), padding='SAME', layer_name="1st_conv_block", training_flag=is_training)
            net = tl.layers.MaxPool2d(net, filter_size=(1,2), padding='VALID', name='maxpool_0')

            ####### stacked conv_block
            net = conv_layer_stack(net, filter_num=32, kernel_size=(1,3), strides=(1,1), padding='SAME', layer_name="stack_conv_1", training_flag=is_training)
            net = tl.layers.MaxPool2d(net, filter_size=(1,2), padding='VALID', name='maxpool_1')

            net = conv_layer_stack(net, filter_num=32, kernel_size=(1,3), strides=(1,1), padding='SAME', layer_name="stack_conv_2", training_flag=is_training)
            net = tl.layers.MaxPool2d(net, filter_size=(1,2), padding='VALID', name='maxpool_2')

            net = conv_layer_stack(net, filter_num=64, kernel_size=(1,3), strides=(1,1), padding='SAME', layer_name="stack_conv_3", training_flag=is_training)
            net = tl.layers.MaxPool2d(net, filter_size=(1,2), padding='VALID', name='maxpool_3')

            net = conv_layer_stack(net, filter_num=64, kernel_size=(1,3), strides=(1,1), padding='SAME', layer_name="stack_conv_4", training_flag=is_training)
            net = tl.layers.MaxPool2d(net, filter_size=(1,2), padding='VALID', name='maxpool_4')

            net = conv_layer_stack(net, filter_num=64, kernel_size=(1,3), strides=(1,1), padding='SAME', layer_name="stack_conv_5", training_flag=is_training)
            net = tl.layers.MaxPool2d(net, filter_size=(1,2), padding='VALID', name='maxpool_5')

            net = conv_layer_stack(net, filter_num=64, kernel_size=(1,3), strides=(1,1), padding='SAME', layer_name="stack_conv_6", training_flag=is_training)
            net = tl.layers.MaxPool2d(net, filter_size=(1,2), padding='VALID', name='maxpool_6')

            net = tl.layers.FlattenLayer(net, name='flatten_layer')
            net = tl.layers.DropoutLayer(net, keep=0.7, is_fix=True, is_train=is_training, name='drop1')
            net = tl.layers.DenseLayer(net, n_units=100, act=tf.nn.relu, W_init=w_init,b_init=tf.zeros_initializer, name='dense1')
            net = tl.layers.DenseLayer(net, n_units=2, act=tf.identity, W_init=w_init,b_init=tf.zeros_initializer, name='dense2')

            cnn_output = net.outputs
            ################ loss 
            cost = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(labels=y, logits=cnn_output))  ### loss fun
            #cost = tl.cost.cross_entropy(cnn_output, y, name='xentropy')

            correct_prediction = tf.equal(tf.argmax(cnn_output, 1), tf.argmax(y, 1))
            accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))

            return net, cost, accuracy

    # define placeholder
    x = tf.placeholder(tf.float32, shape=[None, VEC_NUM, RSS_NUM, 1])
    y_ = tf.placeholder(tf.int64, shape=[None,2])

    with tf.device('/gpu:0'):
        network, cost, accuracy, = NetworkCNN_Model(x, y_, reuse=None, is_training=True)  ### training 
        #_, cost_test, acc_test = NetworkCNN_Model(x_test_batch, y_test_batch, reuse=True, is_training=False)  ### eval/testing 
        train_op = tf.train.AdadeltaOptimizer(learning_rate=1.0, rho=0.95, epsilon=1e-08, 
            use_locking=False, name='Adadelta').minimize(cost)

    ######### training
    tl.layers.initialize_global_variables(sess)
    network.print_params(False)
    network.print_layers()        

    # train the network
    tl.utils.fit(
        sess, network, train_op, cost, dataTrain, labelTrainOneHot, x, y_, acc=accuracy, 
        batch_size=BATCH_SIZE, n_epoch=NUM_EPOCH, print_freq=PRINT_FREQ, X_val=dataVal, y_val=labelValOneHot, eval_train=True)

    
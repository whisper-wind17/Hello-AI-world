from __future__ import print_function
import tensorflow as tf 
import tensorlayer as tl
import numpy as np 
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OneHotEncoder

import time
import gc
import os
import io


##########################################################################################
###############  Macros
VEC_NUM = 10 
RSS_NUM = 400 
CLASS_NUM = 2 
NUM_EPOCH = 20 
BATCH_SIZE = 128
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
train_split_size = 0.95 
dataDir = "/home/jidian/chen_dnn/DataSet_BinaryFormat/"
dataPosName = "V2_Pre80_DataSetCar_Factory_0109_08_10_12_15_16_17__10x400.dbin"
dataNegName = "V2_Pre80_DataSetNeg_Factory_0205_10x400_1_10.dbin"

print(dataPosName,dataNegName)

################   training data reading 
#####  human data reading : WuShi (training)
print('\n*** Training Data Human  Factory : ')
_path = dataDir + dataPosName
dataHumanW, labelHumanW = LoadingBinaryData(_path, VEC_NUM, RSS_NUM)
dataHumanTotalNum = len(dataHumanW)
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
dataRainTotalNum = len(dataRain)
split_size = dataHumanTotalNum * (1-train_split_size)/dataRainTotalNum
dataRainTrain, dataRainVal, labelRainTrain, labelRainVal = train_test_split(dataRain, labelRain, 
                                                              train_size = 1-split_size, random_state = 0)
del dataRain, labelRain, labelRainTrain
gc.collect()
print('xTrain and xVal(low) : ', dataRainTrain.shape, dataRainVal.shape )

######## training data generator
dataHumanWNum, vecNum, rssNum, chanNum = dataHumanWTrain.shape
dataRainNum = len(dataRainTrain)
maxLen = np.max([dataHumanWNum, dataRainNum])
dataTrain = np.zeros(shape=(2*maxLen, vecNum, rssNum, chanNum),dtype=np.int8)
labelTrain = np.zeros(shape=(2*maxLen, 1),dtype=np.int8)
indexVec = np.arange(maxLen)
print('\n traing data size: ', 2*maxLen)

np.random.shuffle(indexVec)
for index in np.arange(maxLen):        
    humanWInd = np.mod(indexVec[index], dataHumanWNum)                
    rainInd = np.mod(indexVec[index], dataRainNum)        
    dataTrain[index*2, :, :, :] = dataHumanWTrain[humanWInd, :, :, :]        
    dataTrain[index*2+1, :, :, :] = dataRainTrain[rainInd, :, :, :]
    labelTrain[index*2, :] = 1
    labelTrain[index*2+1, :] = 0 

del dataHumanWTrain, dataRainTrain
gc.collect()

######### evaluation data generator
maxLen2 = np.max([len(dataHumanWVal), len(dataRainVal)])
dataVal = np.zeros(shape=(2*maxLen2, vecNum, rssNum, chanNum), dtype=np.int8)
labelVal = np.zeros(shape=(2*maxLen2, 1),dtype=np.int8)
print('\n testing data size: ', 2*maxLen2)

for index in range(maxLen2):
    humanWInd = np.mod(index, len(dataHumanWVal))
    rainInd = np.mod(index, len(dataRainVal))
    dataVal[index*2, :, :, :] = dataHumanWVal[humanWInd, :, :, :]
    dataVal[index*2+1, :, :, :] = dataRainVal[rainInd, :, :, :]
    labelVal[index*2, :] = 1 
    labelVal[index*2+1, :] = 0

del dataHumanWVal, dataRainVal, labelHumanWVal, labelRainVal
gc.collect()

def data_to_tfrecord(data, labels, filename):
    """ Save data into TFRecord """
    if os.path.isfile(filename):
        print("%s exists" % filename)
        return
    print("Converting data into %s ..." % filename)
    cwd = os.getcwd()
    writer = tf.python_io.TFRecordWriter(filename)
    for index, data in enumerate(data):
        img_raw = data.tobytes()
        ## Visualize a image
        # tl.visualize.frame(np.asarray(img, dtype=np.uint8), second=1, saveable=False, name='frame', fig_idx=1236)
        label = int(labels[index])
        # print(label)
        ## Convert the bytes back to image as follow:
            # image = Image.frombytes('RGB', (32, 32), img_raw)
        # image = np.fromstring(img_raw, np.float32)
        # image = image.reshape([32, 32, 3])
        # tl.visualize.frame(np.asarray(image, dtype=np.uint8), second=1, saveable=False, name='frame', fig_idx=1236)
        example = tf.train.Example(features=tf.train.Features(feature={
            "label": tf.train.Feature(int64_list=tf.train.Int64List(value=[label])),
            'img_raw': tf.train.Feature(bytes_list=tf.train.BytesList(value=[img_raw])),
        }))
        writer.write(example.SerializeToString())  # Serialize To String
    writer.close()

## Save data into TFRecord files
data_to_tfrecord(data=dataTrain, labels=labelTrain, filename="train.factory")
data_to_tfrecord(data=dataVal, labels=labelVal, filename="test.factory")


## Read Data Method 2: Queue and Thread =======================================
# use sess.run to get a batch of data
def read_and_decode(filename, vecNum, rssNum):
    # generate a queue with a given file name
    cwd = os.getcwd()
    filename_queue = tf.train.string_input_producer([filename])
    reader = tf.TFRecordReader()
    _, serialized_example = reader.read(filename_queue)     # return the file and the name of file
    features = tf.parse_single_example(serialized_example,  # see parse_single_sequence_example for sequence example
                                       features={
                                           'label': tf.FixedLenFeature([], tf.int64),
                                           'img_raw' : tf.FixedLenFeature([], tf.string),
                                       })
    # You can do more image distortion here for training data
    img = tf.decode_raw(features['img_raw'], tf.int8)
    img = tf.reshape(img, [vecNum, rssNum, 1])
    img = tf.cast(img, tf.float32)
    label = tf.cast(features['label'], tf.int32)
    return img, label

############################################################################################
############################ Graph
with tf.device('/cpu:0'):
    sess = tf.Session(config=tf.ConfigProto(allow_soft_placement=True,inter_op_parallelism_threads=1,intra_op_parallelism_threads=8))

    x_train_list, y_train_list = read_and_decode(filename="train.factory", vecNum=VEC_NUM, rssNum=RSS_NUM)
    x_train_batch, y_train_batch = tf.train.shuffle_batch([x_train_list, y_train_list],
        batch_size=BATCH_SIZE, capacity=2000, min_after_dequeue=1000, num_threads=32) 

    # for testing, use batch instead of shuffle_batch
    x_test_list, y_test_list = read_and_decode(filename="test.factory", vecNum=VEC_NUM, rssNum=RSS_NUM)
    x_test_batch, y_test_batch = tf.train.batch([x_test_list, y_test_list],
        batch_size=BATCH_SIZE, capacity=2000, num_threads=32) 

    w_init = tf.truncated_normal_initializer(stddev=5e-2)
    #W_init2 = tf.truncated_normal_initializer(stddev=0.04)
    b_init = tf.constant_initializer(value=0.01)

    def incep_layer_first(net_in, filter_num, layer_name="inception",  training_flag=True):
        ##### conv-bn-relu 
        k1, k2, k3 = filter_num
        out1 = tl.layers.Conv2d(net_in, n_filter=k1, filter_size=(1,15), strides=(1,1), padding="SAME", act=None, W_init=w_init,b_init=b_init,name=layer_name+"_conv15" )
        out1 = tl.layers.BatchNormLayer(out1, act=tf.nn.relu, is_train=training_flag, beta_init=tf.zeros_initializer, gamma_init=tf.ones_initializer, name =layer_name+"_bn15")
        
        out2 = tl.layers.Conv2d(net_in, n_filter=k2, filter_size=(1,11), strides=(1,1), padding="SAME", act=None, W_init=w_init,b_init=b_init,name=layer_name+"_conv11" )
        out2 = tl.layers.BatchNormLayer(out2, act=tf.nn.relu, is_train=training_flag, beta_init=tf.zeros_initializer, gamma_init=tf.ones_initializer, name =layer_name+"_bn11")
        
        out3 = tl.layers.Conv2d(net_in, n_filter=k3, filter_size=(1,7), strides=(1,1), padding="SAME", act=None, W_init=w_init,b_init=b_init,name=layer_name+"_conv7" )
        out3 = tl.layers.BatchNormLayer(out3, act=tf.nn.relu, is_train=training_flag, beta_init=tf.zeros_initializer, gamma_init=tf.ones_initializer, name =layer_name+"_bn7")
        
        out = tl.layers.ConcatLayer([out1, out2, out3], 3, name =layer_name+'concat_layer')
        return out 

    def res_layer_stack(net_in, filter_num, layer_name="stack_res",  training_flag=True):
        #### bn-relu-conv
        k1, k2, k3 = filter_num
        out = tl.layers.BatchNormLayer(net_in, act=tf.nn.relu, is_train=training_flag, beta_init=tf.zeros_initializer, gamma_init=tf.ones_initializer, name =layer_name+"_bn1x1")
        out = tl.layers.Conv2d(out, n_filter=k1, filter_size=(1,1), strides=(1,1), padding="SAME", act=None, W_init=w_init,b_init=b_init,name=layer_name+"_conv1x1" )
        
        out = tl.layers.BatchNormLayer(out, act=tf.nn.relu, is_train=training_flag, beta_init=tf.zeros_initializer, gamma_init=tf.ones_initializer, name =layer_name+"_bn1x3_1")
        out = tl.layers.Conv2d(out, n_filter=k2, filter_size=(1,3), strides=(1,1), padding="SAME", act=None, W_init=w_init,b_init=b_init,name=layer_name+"_conv1x3_1" )

        out = tl.layers.BatchNormLayer(out, act=tf.nn.relu, is_train=training_flag, beta_init=tf.zeros_initializer, gamma_init=tf.ones_initializer, name =layer_name+"_bn1x3_2")
        out = tl.layers.Conv2d(out, n_filter=k2, filter_size=(1,3), strides=(1,1), padding="SAME", act=None, W_init=w_init,b_init=b_init,name=layer_name+"_conv1x3_2" )

        out = tl.layers.BatchNormLayer(out, act=tf.nn.relu, is_train=training_flag, beta_init=tf.zeros_initializer, gamma_init=tf.ones_initializer, name =layer_name+"_bn1x1_2")
        out = tl.layers.Conv2d(out, n_filter=k3, filter_size=(1,1), strides=(1,1), padding="SAME", act=None, W_init=w_init,b_init=b_init,name=layer_name+"_conv1x1_2")

        network = tl.layers.ElementwiseLayer(layer=[net_in, out], combine_fn=tf.add, name=layer_name+'_add')
        return network

    def transition_layer(net_in, filter_num, layer_name="trans_layer", training_flag=True):
        #### bn-relu-conv
        out = tl.layers.BatchNormLayer(net_in, act=tf.nn.relu, is_train=training_flag, beta_init=tf.zeros_initializer, gamma_init=tf.ones_initializer, name =layer_name+"_bn1x1")
        out = tl.layers.Conv2d(out, n_filter=filter_num, filter_size=(1,1), strides=(1,1), padding="SAME", act=None, W_init=w_init,b_init=b_init,name=layer_name+"_conv1x1" )
        return out

    def NetworkCNN_Model(x, y, reuse, is_training):        
        with tf.variable_scope("StackCNN", reuse=reuse):
            tl.layers.set_name_reuse(reuse)

            net = tl.layers.InputLayer(x, name='input_layer')  ### 10x400
            ####### first inception_block
            net = incep_layer_first(net, filter_num=(24, 24, 16), layer_name="inception", training_flag=is_training )
            net = tl.layers.MaxPool2d(net, filter_size=(1,2), padding='VALID', name='maxpool_0')  ## 10x200

            ####### stacked conv_block
            net = res_layer_stack(net, filter_num=(48,48,64), layer_name="stack_res1",  training_flag=is_training)
            net = tl.layers.MaxPool2d(net, filter_size=(1,2), padding='VALID', name='maxpool_1')  ## 10x100

            net = res_layer_stack(net, filter_num=(48,48,64), layer_name="stack_res2",  training_flag=is_training)
            net = tl.layers.MaxPool2d(net, filter_size=(1,2), padding='VALID', name='maxpool_2')  ## 10x50

            net = transition_layer(net, filter_num=96, layer_name="trans_layer1", training_flag=is_training)

            net = res_layer_stack(net, filter_num=(64,64,96), layer_name="stack_res3",  training_flag=is_training)
            net = tl.layers.MaxPool2d(net, filter_size=(1,2), padding='VALID', name='maxpool_3')  ## 10x25

            net = res_layer_stack(net, filter_num=(64,64,96), layer_name="stack_res4",  training_flag=is_training)
            net = tl.layers.MaxPool2d(net, filter_size=(1,2), padding='VALID', name='maxpool_4')  ## 10x12

            net = res_layer_stack(net, filter_num=(64,64,96), layer_name="stack_res5",  training_flag=is_training)
            net = tl.layers.MaxPool2d(net, filter_size=(1,2), padding='VALID', name='maxpool_5')  ## 10x6

            net = res_layer_stack(net, filter_num=(64,64,96), layer_name="stack_res6",  training_flag=is_training)
            net = tl.layers.MaxPool2d(net, filter_size=(1,2), padding='VALID', name='maxpool_6')  ## 10x3            

            net = tl.layers.FlattenLayer(net, name='flatten_layer')
            net = tl.layers.DropoutLayer(net, keep=0.3, is_fix=True, is_train=is_training, name='drop1')
            net = tl.layers.DenseLayer(net, n_units=100, act=tf.nn.relu, W_init=w_init,b_init=tf.zeros_initializer, name='dense1')
            net = tl.layers.DenseLayer(net, n_units=2, act=tf.identity, W_init=w_init,b_init=tf.zeros_initializer, name='dense2')

            cnn_output = net.outputs
            ################ loss 
            #cost = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(labels=y, logits=cnn_output))  ### loss fun
            cost = tl.cost.cross_entropy(cnn_output, y, name='cost')
            correct_prediction = tf.equal(tf.cast(tf.argmax(cnn_output, 1),tf.int32), y)
            accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))

            return net, cost, accuracy

    with tf.device('/gpu:0'):
        network, cost, accuracy, = NetworkCNN_Model(x_train_batch, y_train_batch, reuse=None, is_training=True)  ### training 
        _, cost_test, acc_test = NetworkCNN_Model(x_test_batch, y_test_batch, reuse=True, is_training=False)  ### eval/testing 
        train_op = tf.train.AdadeltaOptimizer(learning_rate=1.0, rho=0.95, epsilon=1e-08, 
            use_locking=False, name='Adadelta').minimize(cost)


    ######### training
    model_file_name = "model_cifar10_advanced.ckpt"
    n_step_epoch = int(2*maxLen/BATCH_SIZE) 

    tl.layers.initialize_global_variables(sess)
    print("Load existing model " + "!"*10)
    saver = tf.train.Saver()
    saver.restore(sess, model_file_name)

    network.print_params(False)
    network.print_layers()

    coord = tf.train.Coordinator()
    threads = tf.train.start_queue_runners(sess=sess, coord=coord)
    step = 0
    ###### iterative train loop 
    for epoch in range(NUM_EPOCH):
        start_time = time.time()
        train_loss, train_acc, n_batch = 0, 0, 0
        for s in range(n_step_epoch):            
            err, ac, _ , x_train_2 = sess.run([cost, accuracy, train_op, x_train_list])
            step += 1; train_loss += err; train_acc += ac; n_batch += 1 
            print("epoch: %d/%d, step:%d/%d, train_los: %.8f, train_acc: %.4f"%(epoch, NUM_EPOCH, s, n_step_epoch, train_loss/n_batch, train_acc/n_batch), end="\r")            

        train_loss /= n_batch
        train_acc /= n_batch        

        line = "TimeCost: %.1f sec, epoch: %d/%d, train_loss: %.8f, train_acc: %.8f" % (time.time()-start_time, epoch, NUM_EPOCH, train_loss, train_acc)
        print(line)

        ######### validation
        start_time = time.time()
        test_loss, test_acc, n_batch = 0, 0, 0
        for _ in range(int(2*maxLen2/BATCH_SIZE)):
            err, ac = sess.run([cost_test, acc_test])
            test_loss += err; test_acc += ac; n_batch += 1
        print("   test loss: %.8f, test acc: %.8f" % (test_loss/ n_batch, test_acc/ n_batch))
        

        print("Save model " + "!"*10 +"\n")
        saver = tf.train.Saver()
        save_path = saver.save(sess, model_file_name)

    ### Terminate as usual.  It is innocuous to request stop twice.
    coord.request_stop()
    coord.join(threads)


from __future__ import division, print_function, absolute_import
import tensorflow as tf 
import tensorlayer as tl
import numpy as np 
from sklearn.model_selection import train_test_split

import time
import gc
import os

w_init = tf.truncated_normal_initializer(stddev=5e-2)
b_init = tf.constant_initializer(value=0.01)

def batch_norm_layer(input, is_train, scope ):     
    network = tf.layers.batch_normalization(inputs=input, training=is_train, name=scope+"_bn")
    print("  [TF Layers] : %s, shape: %s"%(scope+"_bn", str(input.shape)))
    return network

def relu_layer (input, scope):
    network = tf.nn.relu(input, name=scope+"_relu")
    print("  [TF Layers] : %s, shape: %s"%(scope+"_relu", str(input.shape)))
    return network 

def conv_layer(input, filters, kernel_size,  scope):
    network = tf.layers.conv2d(inputs=input, filters=filters, kernel_size=kernel_size, strides=(1,1), padding="same", 
        kernel_initializer= w_init, bias_initializer = b_init, activation=None, use_bias=True, name=scope+"_conv")
    print("  [TF Layers] : %s, shape: %s"%(scope+"_conv", str(network.shape)))
    return network

def sigmoid_layer(input, scope):
    network = tf.sigmoid(input, name=scope+"_sigmoid")
    print("  [TF Layers] : %s, shape: %s"%(scope+"_sigmoid", str(network.shape)))
    return network
    
def global_average_layer(input, scope):
    network = tf.reduce_mean(input, [1,2], name=scope+"_gap")
    print("  [TF Layers] : %s, shape: %s"%(scope+"_gap", str(network.shape)))
    return network

def dropout_layer(input, ratio, is_train):
    network = tf.layers.dropout(input,rate=ratio, training=is_train, name="dropout" )
    print("  [TF Layers] : %s, shape: %s"%("dropout", str(network.shape)))
    return network 

def dense_layer(input, units, activation, scope):
    network = tf.layers.dense(input,units=units, activation=activation, kernel_initializer=w_init,
            bias_initializer=b_init, name=scope+"_dense" )
    print("  [TF Layers] : %s, shape: %s"%(scope+"_dense", str(network.shape)))
    return network

def flatten_layer(input):
    network = tf.contrib.layers.flatten(input)
    print("  [TF Layers] : %s, shape: %s"%("flatten", str(network.shape)))
    return network

def scaling_layer(input, scale, scope):
    outdim = input.shape[-1]
    scale = tf.reshape(scale, [-1,1,1,outdim])
    network = input * scale 
    print("  [TF Layers] : %s, shape: %s"%(scope+"_scaling", str(network.shape)))
    return network

def concat_layer(input, scope):
    input_x = []
    for l in input:
        input_x.append(l)
    network = tf.concat(input_x, axis=-1, name=scope+"_concat")
    print("  [TF Layers] : %s, shape: %s"%(scope+"_concat", str(network.shape)))
    return network

def add_layer(input,scope):
    network = input[0]
    for l in input[1:]:
        network = tf.add(network, l, name=scope+"_add") 
    print("  [TF Layers] : %s, shape: %s"%(scope+"_add", str(network.shape)))
    return network

def maxpool_layer(input, scope):
    network = tf.layers.max_pooling2d(input, pool_size=(1,2),strides=(1,2),padding="valid", name=scope+"maxpool")
    print("  [TF Layers] : %s, shape: %s"%(scope+"_maxpool", str(network.shape))) 
    return network

def transition_layer(input, filters, is_train,  scope=None):
    network = batch_norm_layer(input, is_train=is_train, scope=scope)
    network = relu_layer(network, scope)
    network = conv_layer(network, filters, kernel_size=(1,1), scope=scope)
    return network

def inception_block (input, filters, is_train,  scope):
    k1, k2, k3 = filters
    with tf.variable_scope(scope ):
        out1 = conv_layer(input, k1, kernel_size=(1,15),  scope=scope+"_branch1")
        out1 = batch_norm_layer(out1, is_train=is_train,  scope=scope+"_branch1")
        out1 = relu_layer(out1, scope+"_branch1")

        out2 = conv_layer(input, k2, kernel_size=(1,11),  scope=scope+"_branch2")
        out2 = batch_norm_layer(out2, is_train=is_train,  scope=scope+"_branch2")
        out2 = relu_layer(out2, scope+"_branch2")

        out3 = conv_layer(input, k3, kernel_size=(1,7),  scope=scope+"_branch3")
        out3 = batch_norm_layer(out3, is_train=is_train, scope=scope+"_branch3")
        out3 = relu_layer(out3, scope+"_branch1")

        network = concat_layer([out1, out2, out3], scope)
        return network 

def res_block(input, filters, is_train,  scope):
    k1, k2, k3 = filters
    with tf.variable_scope(scope ):
        out = batch_norm_layer(input, is_train=is_train,  scope=scope+"_1x1")
        out = relu_layer(out, scope=scope+"_1x1")
        out = conv_layer(out, k1, kernel_size=(1,1), scope=scope+"_1x1")

        out = batch_norm_layer(out, is_train=is_train,  scope=scope+"_1x3")
        out = relu_layer(out, scope=scope+"_1x3")
        out = conv_layer(out, k2, kernel_size=(1,3), scope=scope+"_1x3")

        out = batch_norm_layer(out, is_train=is_train,  scope=scope+"_1x3_2")
        out = relu_layer(out, scope=scope+"_1x3_2")
        out = conv_layer(out, k2, kernel_size=(1,1), scope=scope+"_1x3_2")

        out = batch_norm_layer(out, is_train=is_train,  scope=scope+"_1x1_2")
        out = relu_layer(out, scope=scope+"_1x1_2")
        out = conv_layer(out, k3, kernel_size=(1,1), scope=scope+"_1x1_2")

        network = add_layer([input, out], scope=scope)
        return network

def SENetLikeModel(x, y_, reuse, is_train):
    with tf.variable_scope("SENetLikeModel", reuse=reuse):  
        network = inception_block(x, filters=[24,24,16], is_train=is_train, scope="inception_block")
        network = maxpool_layer(network, scope="inception_block") ### 10x200
        #### scope=res_block1
        network = res_block(network,filters=[48,48,64],is_train=is_train, scope="res_block1")
        network = maxpool_layer(network, scope="res_block1")  ### 10x100
        #### scope=res_block2
        network = res_block(network,filters=[48,48,64],is_train=is_train, scope="res_block2")
        network = maxpool_layer(network, scope="res_block2")  ### 10x50
        #### scope=transition_block
        network = transition_layer(network,filters=96, is_train=is_train, scope="transition_block")
        #### scope=res_block3
        network = res_block(network,filters=[64,64,96],is_train=is_train, scope="res_block3")
        network = maxpool_layer(network, scope="res_block3")  ### 10x25
        #### scope=res_block4
        network = res_block(network,filters=[64,64,96],is_train=is_train, scope="res_block4")
        network = maxpool_layer(network, scope="res_block4")  ### 10x12
        #### scope=res_block5
        network = res_block(network,filters=[64,64,96],is_train=is_train, scope="res_block5")
        network = maxpool_layer(network, scope="res_block5")  ### 10x6
        #### scope=res_block6
        network = res_block(network,filters=[64,64,96],is_train=is_train, scope="res_block6")
        network = maxpool_layer(network, scope="res_block6")  ### 10x3

        network = flatten_layer(network)
        network = dropout_layer(network, ratio=0.7, is_train=is_train)
        network = dense_layer(network, units=100, activation=tf.nn.relu,  scope="fc1")
        output = dense_layer(network, units=2, activation=tf.identity,  scope="fc2")

        prediction = tf.nn.softmax(output)
        #cost = tl.cost.cross_entropy(output, y_, name='cost')
        cost = tf.reduce_mean(tf.nn.sparse_softmax_cross_entropy_with_logits(labels=y_, logits=output))
        correct_prediction = tf.equal(tf.cast(tf.argmax(prediction, 1),tf.int32), y_)
        accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))

        return output, cost, accuracy
        
##########################################################################################
###############  Macros
VEC_NUM = 10 
RSS_NUM = 400 
CLASS_NUM = 2 
NUM_EPOCH = 20 
BATCH_SIZE = 128
BATCH_SIZE_Eval = 128
PRINT_FREQ = 1

#############################################################################################
############################  Data preparation 
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
_rootPath = os.getcwd() + "/TFRecords"
data_to_tfrecord(data=dataTrain, labels=labelTrain, filename=_rootPath+"/train.factory")
data_to_tfrecord(data=dataVal, labels=labelVal, filename=_rootPath+"/test.factory")


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

with tf.device('/cpu:0'):
    sess = tf.Session(config=tf.ConfigProto(allow_soft_placement=True,
                    inter_op_parallelism_threads=1,
                    intra_op_parallelism_threads=8))

    x_train_list, y_train_list = read_and_decode(filename=_rootPath+"/train.factory", 
                                            vecNum=VEC_NUM, 
                                            rssNum=RSS_NUM)
    x_train_batch, y_train_batch = tf.train.shuffle_batch([x_train_list, y_train_list],
        batch_size=BATCH_SIZE, capacity=2000, min_after_dequeue=1000, num_threads=32) 

    # for testing, use batch instead of shuffle_batch
    x_test_list, y_test_list = read_and_decode(filename=_rootPath+"/test.factory", 
                                            vecNum=VEC_NUM, 
                                            rssNum=RSS_NUM)
    x_test_batch, y_test_batch = tf.train.batch([x_test_list, y_test_list],
        batch_size=BATCH_SIZE_Eval, capacity=2000, num_threads=32) 

    with tf.device("/gpu:0"):
        network, cost, accuracy = SENetLikeModel(x_train_batch, y_train_batch, reuse=None, is_train=True)  ### training 
        _, cost_test, acc_test = SENetLikeModel(x_test_batch, y_test_batch, reuse=True, is_train=False)  ### test/eval 
        update_ops = tf.get_collection(tf.GraphKeys.UPDATE_OPS)
        with tf.control_dependencies(update_ops):
            train_op = tf.train.AdadeltaOptimizer(learning_rate=1.0, rho=0.95, epsilon=1e-08, 
                                                use_locking=False, name='Adadelta').minimize(cost)

    ######### training
    model_file_name = os.getcwd() + "/Model_SENet_like/model_senet_like.ckpt"
    n_step_epoch = int(2*maxLen/BATCH_SIZE) 

    sess.run(tf.global_variables_initializer()) 
    
    print("Load existing model " + "!"*10)
    saver = tf.train.Saver()
    saver.restore(sess, model_file_name)    
     

    coord = tf.train.Coordinator()
    threads = tf.train.start_queue_runners(sess=sess, coord=coord)
    step = 0
    ###### iterative train loop 
    for epoch in range(NUM_EPOCH):
        start_time = time.time()
        train_loss, train_acc, n_batch = 0, 0, 0
        for s in range(int(n_step_epoch/1)):            
            err, ac, _  = sess.run([cost, accuracy, train_op])
            step += 1; train_loss += err; train_acc += ac; n_batch += 1 
            print("epoch: %d/%d, step:%d/%d, train_los: %.8f, train_acc: %.4f"%(epoch, NUM_EPOCH, s, n_step_epoch, train_loss/n_batch, train_acc/n_batch), end="\r")            

        train_loss /= n_batch
        train_acc /= n_batch        

        line = "TimeCost: %.1f sec, epoch: %d/%d, train_loss: %.8f, train_acc: %.8f" % (time.time()-start_time, epoch, NUM_EPOCH, train_loss, train_acc)
        print(line)

        ######### validation
        start_time = time.time()
        test_loss, test_acc, n_batch = 0, 0, 0
        for _ in range(int(2*maxLen2/BATCH_SIZE_Eval)):
            err, ac = sess.run([cost_test, acc_test])
            test_loss += err; test_acc += ac; n_batch += 1
        print("   test loss: %.8f, test acc: %.8f" % (test_loss/ n_batch, test_acc/ n_batch))
        

        print("Save model " + "!"*10 +"\n")
        saver = tf.train.Saver()
        save_path = saver.save(sess, model_file_name)

    ### Terminate as usual.  It is innocuous to request stop twice.
    coord.request_stop()
    coord.join(threads)
    

from __future__ import print_function
import numpy as np  

from keras.models import Sequential
from keras.layers import Dense, Dropout, Activation, Flatten, Input
from keras.layers import Conv2D, MaxPooling2D, AveragePooling2D, BatchNormalization, GlobalAveragePooling2D
from keras.utils import np_utils
from keras.optimizers import SGD, Adam, RMSprop, Adagrad, Adadelta, Adamax, Nadam
from keras.callbacks import ModelCheckpoint
from keras.models import load_model, Model

###### pre-trained models 
from keras.applications.vgg16 import VGG16
from keras.applications.vgg19 import VGG19
from keras.applications.resnet50 import ResNet50 
from keras.applications.inception_v3 import InceptionV3
from keras.applications.xception import Xception

from keras.preprocessing.image import ImageDataGenerator


##### read dataset for training and validation 
batch_size = 32 
train_datagen = ImageDataGenerator()
train_generator = train_datagen.flow_from_directory('/home/jidian/chen_dnn/DataSet_10x400_PNG/train',
                                                    target_size = (224,224),
                                                    batch_size = batch_size,
                                                    class_mode='categorical', 
                                                    shuffle = 'True',
                                                    save_format='png')
train_samp_numbers = train_generator.samples 

validation_datagen = ImageDataGenerator() 
validation_generator = validation_datagen.flow_from_directory('/home/jidian/chen_dnn/DataSet_10x400_PNG/validation',
                                                    target_size =(224,224),
                                                    batch_size=batch_size,
                                                    class_mode = 'categorical',
                                                    save_format = 'png')
validation_samp_numbers = validation_generator.samples 

test_datagen = ImageDataGenerator()
test_generator = test_datagen.flow_from_directory('/home/jidian/chen_dnn/DataSet_10x400_PNG/test',
                                                    target_size = (224,224),
                                                    batch_size = batch_size,
                                                    class_mode='categorical', 
                                                    save_format='png')
test_samp_numbers = test_generator.samples
##############################################################################################
#### pre-trained model : VGG 16 
'''
base_model = VGG16(include_top=False, weights='imagenet')
x = base_mode.output

for i, layer in enumerate(base_model.layers):
   print(i, layer.name)

#### fine-tune layers
x = Flatten()(x)
x = Dropout(0.5)(x)
x = Dense(128, activation='relu')(x)

x = Dropout(0.5)(x)
x = Dense(128, activation='relu')(x)

x = Dropout(0.5)(x)
y_pred = Dense(2, activation='softmax')(x)

model = Model(inputs=base_model.input, outputs = y_pred)

# first: train only the top layers (which were randomly initialized)
# i.e. freeze all convolutional layers
for layer in base_model.layers:
    layer.trainable = False
'''
##############################################################################################
#### pre-trained model : InceptionV3
base_model = InceptionV3(weights=None, include_top=False)
for i, layer in enumerate(base_model.layers):
   print(i, layer.name)
base_model.load_weights('/home/jidian/chen_dnn/Models/inception_v3_weights_tf_dim_ordering_tf_kernels_notop.h5')

# add a global spatial average pooling layer
x = base_model.output
x = GlobalAveragePooling2D()(x)
x = Dense(1024, activation='relu')(x)
x = Dropout(0.5)(x)
y_pred = Dense(2, activation='softmax')(x)

# this is the model we will train
model = Model(inputs = base_model.input, outputs = y_pred)


# first: train only the top layers (which were randomly initialized)
# i.e. freeze all convolutional InceptionV3 layers
for layer in base_model.layers:
    layer.trainable = False

'''
for layer in model.layers[:249]:
   layer.trainable = False
for layer in model.layers[249:]:
   layer.trainable = True
'''

##############################################################################################
#####  CNN model


##############################################################################################
#### optimizer
adadelta = Adadelta(lr=1.0, rho=0.95, epsilon=1e-08, decay=0.0)
#rmsprop = RMSprop(lr=0.001, rho=0.9, epsilon=1e-08, decay=0.0)
#adagrad = Adagrad(lr=0.01, epsilon=1e-08, decay=0.0)
#adam = Adam(lr=0.001, beta_1=0.9, beta_2=0.999, epsilon=1e-08, decay=0.0)
#adamax = Adamax(lr=0.002, beta_1=0.9, beta_2=0.999, epsilon=1e-08, decay=0.0)
#nadam = Nadam(lr=0.002, beta_1=0.9, beta_2=0.999, epsilon=1e-08, schedule_decay=0.004)

model.compile(loss='categorical_crossentropy', optimizer=adadelta, metrics=['accuracy'])

print('\n model.summary: \n')
model.summary()


checkpointer = ModelCheckpoint(filepath='/home/jidian/chen_dnn/Models/weights.hdf5', monitor='val_loss', verbose=1, 
                                save_best_only=True, save_weights_only=True, mode='auto', period=1)
model.fit_generator(train_generator, steps_per_epoch = int(train_samp_numbers/batch_size),
                     epochs=10, verbose=1, callbacks=[checkpointer], 
                     validation_data=test_generator, validation_steps=int(test_samp_numbers/batch_size),
                      class_weight=None, max_q_size=10, workers=1, pickle_safe=False, initial_epoch=0)


############################################################################################
###### Testing dataset: QinShan-Human 
'''
print('loading the saved weights .......')
model.load_weights('/home/jidian/chen_dnn/Models/weights.hdf5') 

loss, score = model.evaluate_generator(test_generator, steps = int(test_samp_numbers/batch_size))
print('loss = %.4f, score = %.4f'%(loss, score))

predictionLabel = model.predict_generator(test_generator, steps = int(test_samp_numbers/batch_size),verbose=1)
print(predictionLabel.shape)
'''
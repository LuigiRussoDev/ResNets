import keras
from keras.datasets import cifar10
from keras.layers import Dense,Conv2D,MaxPooling2D,Flatten,AveragePooling2D,Dropout,BatchNormalization,Activation
from keras.models import Model,Input
from keras.optimizers import Adam
from keras.callbacks import LearningRateScheduler
from keras.callbacks import ModelCheckpoint
from math import ceil
import os
from keras.preprocessing.image import ImageDataGenerator
from keras import regularizers
import matplotlib.pyplot as plt
from sklearn.model_selection import StratifiedKFold
from keras import optimizers
import keras.backend as K
from tensorflow.keras.callbacks import TensorBoard
import tensorflow as tf
import datetime
from keras.regularizers import l1
from keras.constraints import max_norm
from keras.regularizers import l2
from keras.utils import multi_gpu_model
import pickle
from memory_profiler import profile
import json



def lr_schedule(epochs):
    """Learning Rate Schedule

    Learning rate is scheduled to be reduced after 80, 120, 160, 180 epochs.
    Called automatically every epoch as part of callbacks during training.

    # Arguments
        epoch (int): The number of epochs

    # Returns
        lr (float32): learning rate
    """
    lr = 1e-3
    if epochs > 180:
        lr *= 0.5e-3
    elif epochs > 160:
        lr *= 1e-3
    elif epochs > 120:
        lr *= 1e-2

    elif epochs > 80:
        lr *= 1e-1

    print('Learning rate: ', lr)
    return lr



def Unit(x,filters,pool=False):
    res = x
    if pool:
        x = MaxPooling2D(pool_size=(2, 2))(x)
        res = Conv2D(filters=filters,kernel_size=[3,3],strides=(2,2),padding="same"
                     )(res)
    out = BatchNormalization()(x)
    out = Activation("relu")(out)
    out = Conv2D(filters=filters, kernel_size=[3, 3], strides=[1, 1], padding="same"
                 )(out)

    out = BatchNormalization()(out)
    out = Activation("relu")(out)
    out = Conv2D(filters=filters, kernel_size=[3, 3], strides=[1, 1], padding="same")(out)

    print("shape out ",out.shape)
    print("shape res ",res.shape)

    out = keras.layers.add([res,out])

    return out

#Define the model

'''net = Conv2D(filters=32, kernel_size=[3, 3],
                 strides=[1, 1], padding="same")(images)


    net = Unit(net,32)
    net = Unit(net,32)
    #net = Unit(net,16)
    #4

    net = Unit(net,64,pool=True)
    net = Unit(net,64)
    #net = Unit(net,32)
    #5


    net = Unit(net,128,pool=True)
    net = Unit(net,128)
    #net = Unit(net,64)
    #5

    net = Unit(net, 256,pool=True)
    net = Unit(net, 256)
    #net = Unit(net, 128)
    #5
    '''


def MiniModel(input_shape):
    images = Input(input_shape)

    net = Conv2D(filters=16, kernel_size=[7, 7],
                 strides=[2, 2], padding="same")(images)

    net = Unit(net, 16)
    net = Unit(net, 16)
    net = Unit(net, 16)
    net = Unit(net, 16)
    net = Unit(net, 16)
    net = Unit(net, 16)

    net = Unit(net, 16)
    net = Unit(net, 16)
    net = Unit(net, 16)
    net = Unit(net, 16)
    net = Unit(net, 16)
    net = Unit(net, 16)

    net = Unit(net, 16)
    net = Unit(net, 16)
    net = Unit(net, 16)
    net = Unit(net, 16)
    net = Unit(net, 16)
    net = Unit(net, 16)

    net = Unit(net, 16)
    net = Unit(net, 16)
    net = Unit(net, 16)
    net = Unit(net, 16)
    net = Unit(net, 16)
    net = Unit(net, 16)

    net = Unit(net, 16)
    net = Unit(net, 16)
    net = Unit(net, 16)
    net = Unit(net, 16)
    net = Unit(net, 16)
    net = Unit(net, 16)


    # 7

    net = Unit(net, 32, pool=True)

    net = Unit(net, 32)
    net = Unit(net, 32)
    net = Unit(net, 32)
    net = Unit(net, 32)
    net = Unit(net, 32)

    net = Unit(net, 32)
    net = Unit(net, 32)
    net = Unit(net, 32)
    net = Unit(net, 32)
    net = Unit(net, 32)

    net = Unit(net, 32)
    net = Unit(net, 32)
    net = Unit(net, 32)
    net = Unit(net, 32)
    net = Unit(net, 32)

    net = Unit(net, 32)
    net = Unit(net, 32)
    net = Unit(net, 32)
    net = Unit(net, 32)
    net = Unit(net, 32)

    net = Unit(net, 32)
    net = Unit(net, 32)
    net = Unit(net, 32)
    net = Unit(net, 32)
    net = Unit(net, 32)


    # 7

    net = Unit(net, 64, pool=True)

    net = Unit(net, 64)
    net = Unit(net, 64)
    net = Unit(net, 64)
    net = Unit(net, 64)
    net = Unit(net, 64)

    net = Unit(net, 64)
    net = Unit(net, 64)
    net = Unit(net, 64)
    net = Unit(net, 64)
    net = Unit(net, 64)

    net = Unit(net, 64)
    net = Unit(net, 64)
    net = Unit(net, 64)
    net = Unit(net, 64)
    net = Unit(net, 64)

    net = Unit(net, 64)
    net = Unit(net, 64)
    net = Unit(net, 64)
    net = Unit(net, 64)
    net = Unit(net, 64)

    net = Unit(net, 64)
    net = Unit(net, 64)
    net = Unit(net, 64)
    net = Unit(net, 64)

    # 4

    net = BatchNormalization()(net)
    net = Activation("relu")(net)
    net = Conv2D(filters=32, kernel_size=[3, 3], strides=[1, 1], padding="same"
                 )(net)
    #5


    net = BatchNormalization()(net)
    net = Activation("relu")(net)

    net = Dropout(0.25)(net)

    net = AveragePooling2D(pool_size=(4,4))(net)
    net = Flatten()(net)


    net = Dense(units=10,activation="softmax")(net)

    model = Model(inputs=images,outputs=net)

    return model



#load the cifar10 dataset
(train_x, train_y) , (test_x, test_y) = cifar10.load_data()

#normalize the data
train_x = train_x.astype('float32') / 255
test_x = test_x.astype('float32') / 255

#Subtract the mean image from both train and test set
train_x = train_x - train_x.mean()
test_x = test_x - test_x.mean()

#Divide by the standard deviation
train_x = train_x / train_x.std(axis=0)
test_x = test_x / test_x.std(axis=0)




datagen = ImageDataGenerator(rotation_range=10,
                             width_shift_range=5. / 32,
                             height_shift_range=5. / 32,
                             horizontal_flip=True)

# Compute quantities required for featurewise normalization
# (std, mean, and principal components if ZCA whitening is applied).
datagen.fit(train_x)


#Encode the labels to vectors
train_y = keras.utils.to_categorical(train_y,10)
test_y = keras.utils.to_categorical(test_y,10)

#define a common unit
input_shape = (32,32,3)
model = MiniModel(input_shape)

#Print a Summary of the model

model.summary()
#Specify the training components
sgd = optimizers.SGD(lr=0.0001, momentum=0.0, decay=0.0, nesterov=False)

epochs = 150
steps_per_epoch = ceil(50000/128)




model.compile(optimizer=Adam(learning_rate=lr_schedule(0)),loss="categorical_crossentropy",metrics=["accuracy","top_k_categorical_accuracy"])



# Fit the model on the batches generated by datagen.flow().
history = model.fit_generator(datagen.flow(train_x, train_y, batch_size=128),
                    validation_data=[test_x,test_y],
                    epochs=epochs,steps_per_epoch=steps_per_epoch, verbose=1, workers=4)

with open('Resnet152.pickle', 'wb') as file_pi:
    pickle.dump(history.history, file_pi)

plt.plot(history.history['accuracy'])
plt.plot(history.history['val_accuracy'])
plt.title('model accuracy')
plt.ylabel('accuracy')
plt.xlabel('epoch')
plt.legend(['train', 'test'], loc='upper left')
    #plt.show()
plt.savefig('grafico_accuracy_val_accuracy_Resnet152.png')

plt.figure(2)
    # Plot training & validation loss values
plt.plot(history.history['loss'])
plt.plot(history.history['val_loss'])
plt.title('Model loss')
plt.ylabel('Loss')
plt.xlabel('Epoch')
plt.legend(['Train', 'Test'], loc='upper left')
plt.savefig('grafico_loss_e_val_loss_Resnet152.png')




#Evaluate the accuracy of the test dataset
accuracy = model.evaluate(x=test_x,y=test_y,batch_size=128)
print('Test loss:', accuracy[0])
print('Test accuracy:', accuracy[1])

model.save("cifar10model.h5")
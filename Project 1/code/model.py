import os
import keras
import numpy as np
import cv2
from keras.models import Sequential
from keras.layers import Dense,Dropout,Flatten
from keras.layers import Conv2D,MaxPooling2D
from keras import backend as K
from keras.optimizers import SGD
from keras import backend as K
import tensorflow as tf
import matplotlib.pyplot as plt
from keras.utils import to_categorical
from dataGenerator import DataGenerator
from tensorflow.compat.v1 import InteractiveSession
config = tf.compat.v1.ConfigProto()
config.gpu_options.allow_growth = True
session = InteractiveSession(config=config)
tf.config.experimental.list_physical_devices(device_type=None)

input_shape = (96,96,3)
class_map = {0 : 'helmet',1 : 'legs',2 : 'chest',3 : 'arms'}
num_classes = 4
epochs = 1000
batch_size = 32
class Model:
    def get_model(self):
        model = Sequential()
        model.add(Conv2D(32,kernel_size = (3, 3), activation = 'relu', input_shape = input_shape))

        model.add(Conv2D(32, (3, 3), activation = 'relu'))
        model.add(MaxPooling2D(pool_size = (2, 2)))
        model.add(Dropout(0.25))

        model.add(Conv2D(64, (3, 3), activation = 'relu', padding = 'same'))
        model.add(MaxPooling2D(pool_size = (2, 2)))
        model.add(Dropout(0.25))

        model.add(Conv2D(32, (3, 3), activation = 'relu'))
        model.add(MaxPooling2D(pool_size = (2, 2)))
        model.add(Dropout(0.25))

        model.add(Conv2D(32, (3, 3), activation = 'relu'))
        model.add(MaxPooling2D(pool_size = (2, 2)))
        model.add(Dropout(0.25))

        model.add(Flatten())
        model.add(Dense(1024, activation = 'relu'))
        model.add(Dropout(0.5))
        model.add(Dense(num_classes, activation = 'softmax'))

        return model
    def train_model(self):
        model = self.get_model()
        model.compile(loss = 'categorical_crossentropy', optimizer = SGD(0.01), metrics = ['accuracy'])
        model.summary()
        gen = DataGenerator("destiny2_dataset")
        (x_train, y_train, x_test , y_test) = gen.get_data()
        y_train = to_categorical(y_train)
        y_test = to_categorical(y_test)
        my_callbacks = [tf.keras.callbacks.EarlyStopping(patience=5)]
        history = model.fit(x_train, y_train, batch_size = batch_size, epochs = epochs, verbose = 1,callbacks = my_callbacks, validation_data = (x_test, y_test))
        self.plot_loss(history)
        model.save_weights("model.h5")
        score = model.evaluate(x_test,y_test, verbose = 0)
    def predict(self,model_path):
        gen = DataGenerator("destiny2_dataset")
        (x_train, y_train, x_test , y_test) = gen.get_data()
        model = self.get_model()
        model.compile(loss = 'categorical_crossentropy', optimizer = SGD(0.01), metrics = ['accuracy'])
        model.load_weights(model_path)
        rows = 5
        cols = 5
        axes = []
        fig=plt.figure()
        for a in range(rows*cols):
            rand = np.random.randint(0,len(x_test))
            input_img = x_test[rand]
            imageL = cv2.resize(input_img,None,fx = 1, fy = 1,interpolation = cv2.INTER_CUBIC)
            axes.append(fig.add_subplot(rows,cols,a+1))
            input_img = input_img.reshape(1,96,96,3)
            res = int(model.predict_classes(input_img,1,verbose = 0)[0])
            subplot_title=("Pred : "+str(class_map[res]))
            axes[-1].set_title(subplot_title)
            plt.imshow(imageL)
        fig.tight_layout()
        plt.show()
    def plot_loss(self,history):
        history_dict = history.history
        loss_values = history_dict['loss']
        val_loss_values = history_dict['val_loss']
        epochs_val = range(1,len(loss_values) + 1)
        line1 = plt.plot(epochs_val, val_loss_values, label = 'Validation / Test loss')
        line2 = plt.plot(epochs_val, loss_values, label = 'Training loss')
        plt.setp(line1, linewidth = 2.0, marker = '+', markersize = 10.0)
        plt.setp(line2, linewidth = 2.0, marker = '4', markersize = 10.0)
        plt.xlabel('Epoches')
        plt.ylabel('Loss')
        plt.grid(True)
        plt.legend()
        plt.show()
m = Model()
m.predict("Prueba1_0.9578(acc)/model.h5")
#m.train_model()

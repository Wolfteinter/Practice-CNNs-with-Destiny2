import os
import cv2
import numpy as np
from os.path import join
class DataGenerator:
    def __init__(self,dataset_path):
        self.dataset_path = dataset_path
    def get_data(self):
        class_map = {'helmet' : 0,'legs' : 1,'chest' : 2,'arms' : 3}
        x_train = []
        y_train = []
        x_test = []
        y_test = []
        classes = os.listdir(self.dataset_path)
        print(classes)
        for class_i in classes:
            images = os.listdir(join(self.dataset_path,class_i))
            num_split = int(len(images) * 70 / 100)
            imgs = [cv2.imread(join(self.dataset_path,class_i,img)) for img in images[:num_split]]
            x_train.extend(imgs)
            y_train.extend([class_map[class_i]]*num_split)
            imgs = [cv2.imread(join(self.dataset_path,class_i,img)) for img in images[num_split:]]
            x_test.extend(imgs)
            y_test.extend([class_map[class_i]]* (len(images) - num_split))
        x_train = np.array(x_train)
        x_test = np.array(x_test)

        x_train = x_train.astype('float32')
        x_test = x_test.astype('float32')

        x_train /= 255
        x_test /= 255

        y_train = np.array(y_train)
        y_test = np.array(y_test)
        print(np.unique(y_train),np.unique(y_test))
        print(x_train.shape,y_train.shape,x_test.shape,y_test.shape)
        return x_train,y_train,x_test,y_test

path_root = '/content/drive/My Drive/Colab Notebooks/DU/data/'
path_log = path_root + 'log/'
path_model = path_root + 'model'
path_zip = path_root + 'Coronahack-Chest-XRay-Dataset.zip'
path_unzip = path_root + 'Coronahack-Chest-XRay-Dataset/'
path_csv = path_unzip + 'Chest_xray_Corona_Metadata.csv'
path_train = path_unzip + 'Coronahack-Chest-XRay-Dataset/Coronahack-Chest-XRay-Dataset/train/'
path_test = path_unzip + 'Coronahack-Chest-XRay-Dataset/Coronahack-Chest-XRay-Dataset/test/'

import time
import cv2
import numpy as np
import pandas as pd
import tensorflow as tf
import matplotlib.pyplot as plt
from keras.models import Sequential
from keras.layers import Dense, Dropout, Flatten
from keras.layers import Conv2D, MaxPool2D
from keras.preprocessing.image import ImageDataGenerator
from sklearn.metrics import confusion_matrix
from sklearn.metrics import classification_report

pd.options.mode.chained_assignment = None

class Dataset():
    def __init__(self, path_csv, path_train, path_test, classes):
        self.path_csv = path_csv
        self.path_train = path_train
        self.path_test = path_test
        self.shape = (224, 224)
        self.classes = classes
        self.class_trans = {}
        for i, c in enumerate(self.classes):
            self.class_trans[c], self.class_trans[i] = i, c


    def load(self):
        df = pd.read_csv(self.path_csv).tail(700).head(100)
        df = df.drop(columns=['Unnamed: 0', 'Label_1_Virus_category', 'Label_2_Virus_category'])
        df = df.rename(columns={'X_ray_image_name':'x', 'Label':'y'})

        train_mask = df['Dataset_type'] == 'TRAIN'
        self.xy_train, self.xy_test = df[train_mask], df[~train_mask]

        self.xy_train['x'] = self.path_train + self.xy_train['x']
        self.xy_test['x'] = self.path_test + self.xy_test['x']

        self.y_test = self.xy_test['y'].apply(lambda x: self.class_trans[x])


dataset = Dataset(path_csv, path_train, path_test, ['Normal', 'Pnemonia'])
dataset.load()


class Model():
    def __init__(self, data, path_model, epochs=2, batch_size=128, learning_rate=0.01, workers=4):
        self.data = data
        self.path_model = path_model
        self.epochs = epochs
        self.batch_size = batch_size
        self.learning_rate = learning_rate
        self.workers = workers


    def build(self):
        self.model = Sequential()
        self.model.add(Conv2D(filters=96, kernel_size=11, strides=4, activation='relu'))
        self.model.add(MaxPool2D(pool_size=3, strides=2))
        self.model.add(Conv2D(filters=256, kernel_size=5, padding='same', activation='relu'))
        self.model.add(MaxPool2D(pool_size=3, strides=2))
        self.model.add(Conv2D(filters=384, kernel_size=3, padding='same', activation='relu'))
        self.model.add(Conv2D(filters=384, kernel_size=3, padding='same', activation='relu'))
        self.model.add(Conv2D(filters=256, kernel_size=3, padding='same', activation='relu'))
        self.model.add(MaxPool2D(pool_size=3, strides=2))
        self.model.add(Flatten())
        self.model.add(Dense(4096, activation='relu'))
        self.model.add(Dropout(0.5))
        self.model.add(Dense(4096, activation='relu'))
        self.model.add(Dropout(0.5))
        self.model.add(Dense(len(self.data.classes) // 2))
        self.model.compile(loss='binary_crossentropy', optimizer='Adam', metrics=['accuracy'])


    def fit(self):
        tensorboard_callback = tf.keras.callbacks.TensorBoard(path_log, histogram_freq=1)

        train_flow, test_flow = self._flow(self.data.xy_train), self._flow(self.data.xy_test)

        self.model.fit(train_flow,
                       batch_size=self.batch_size,
                       epochs=self.epochs,
                       callbacks=[tensorboard_callback],
                       validation_data=test_flow,
                       workers=self.workers)
        

    def predict(self):
        test_flow = self._flow(self.data.xy_test)
        y_pred = self.model.predict_classes(test_flow)
        
        cr = classification_report(self.data.y_test, y_pred, target_names=self.data.classes)
        print(cr)

        cm = confusion_matrix(self.data.y_test, y_pred)
        plt.matshow(cm)
        plt.colorbar()
        plt.xlabel('Predicted')
        plt.ylabel('Actual')
        plt.show()


    def save(self):
        self.model.save(self.path_model)


    def _flow(self, df):
        gen = ImageDataGenerator()
        return gen.flow_from_dataframe(
            dataframe=df,
            x_col='x',
            y_col='y',
            target_size=self.data.shape,
            color_mode='grayscale',
            classes=self.data.classes,
            class_mode='binary',
            batch_size=self.batch_size)


model = Model(dataset, path_model)
model.build()
model.fit()
model.predict()
model.save()
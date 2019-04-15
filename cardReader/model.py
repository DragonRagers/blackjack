import os
import numpy as np
import cv2
import pickle
import time
import tensorflow as tf
from random import shuffle
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout, Activation, Flatten
from tensorflow.keras.layers import Conv2D, MaxPooling2D

def getValue(name):
    return {
        "ace" : 0,
        "two" : 1,
        "three" : 2,
        "four" : 3,
        "five" : 4,
        "six" : 5,
        "seven" : 6,
        "eight" : 7,
        "nine" : 8,
        "ten" : 9,
        "jack" : 10,
        "queen" : 11,
        "king" : 12
    }[name]


def makeData(save=True):
    card_data = []
    value_data = []

    imgdir = "images\\{}"
    fileNames = []
    for filename in os.listdir(imgdir.format("")):
        fileNames.append(filename)
    shuffle(fileNames)
    for filename in fileNames:
        img = cv2.imread(imgdir.format(filename))
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        value = getValue(filename.split("_")[0])
        card_data.append(gray)
        value_data.append(value)

    data = [np.array(card_data), np.array(value_data)]

    if save:
        save = open("data{}.pkl".format(time.time()), 'wb')
        pickle.dump(data, save)
        save.close()
    return data


def loadData(filename):
    file = open(filename, 'rb')
    input = pickle.load(file)
    file.close()
    return input


def makeModel():
    data = loadData("data.pkl")
    x_train, y_train = data
    #x_train,y_train = makeData(False)
    #x_train = tf.keras.utils.normalize(x_train, axis=1)
    x_train = x_train / 255

    model = tf.keras.models.Sequential()
    model.add(tf.keras.layers.Flatten())
    model.add(tf.keras.layers.Dense(16 activation=tf.nn.relu))
    model.add(tf.keras.layers.Dense(16, activation=tf.nn.relu))
    #model.add(tf.keras.layers.Dense(32, activation=tf.nn.relu))
    model.add(tf.keras.layers.Dense(13, activation=tf.nn.softmax))

    model.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])
    model.fit(x_train, y_train, epochs=300)
    model.save("cardID{}.model".format(time.time()))


    """
    model = tf.keras.models.Sequential()
    model.add(Conv2D(256, (5, 5), input_shape=(1,125,175)))
    model.add(Activation('relu'))
    model.add(MaxPooling2D(pool_size=(3, 3)))

    model.add(Conv2D(256, (5, 5)))
    model.add(Activation('relu'))
    model.add(MaxPooling2D(pool_size=(3, 3)))

    model.add(Flatten())
    model.add(Dense(64))

    model.add(tf.keras.layers.Dense(13, activation=tf.nn.softmax))

    model.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])
    model.fit(x_train, y_train, epochs=10)
    model.save("cardReader\\cardID{}.model".format(time.time()))
    """

def main():
    #makeData()
    makeModel()



if __name__ == "__main__":
    main()

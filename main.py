import os
import time
import cv2
import pandas as pd
from numpy import mean, savetxt
from numpy import std
from matplotlib import pyplot as plt
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import KFold, train_test_split, cross_val_score
from sklearn.neural_network import MLPClassifier
from tensorflow.keras.datasets import mnist
from tensorflow.keras.utils import to_categorical
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D
from tensorflow.keras.layers import MaxPooling2D
from tensorflow.keras.layers import Dense
from tensorflow.keras.layers import Flatten
from tensorflow.keras.optimizers import SGD
from numpy import argmax
from keras.models import load_model
import numpy as np
import pickle


# load train and test dataset
def dataset():
    # leemos dataset
    imagenes_dataset = "dataset"
    # carpeta
    files_name = os.listdir(imagenes_dataset)
    y = []
    X = []
    # por cada imagen en el dataset la guardamos y ademas extraemos q numero es del nombre, ya q posee q nro es en
    # el primer elemento del string
    for file_name in files_name:
        imagen_dir = imagenes_dataset + "/" + file_name
        imagen_base = cv2.imread(imagen_dir, 0)  # la leemos en escala de grices para tenerlo en un solo canal
        if imagen_base is None:
            continue
        X.append(imagen_base)
        y.append(int(file_name[0]))
    # usamos esto para mezclar el dataset, 70 de entrenamiento y 30 de testeo
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)
    # valor esperado codificados one hot
    trainY = to_categorical(y_train)
    testY = to_categorical(y_test)
    # lo pasamos a array para evitar errores futuros
    X_train = np.array(X_train, dtype=np.float32)
    trainY = np.array(trainY, dtype=np.float32)
    X_test = np.array(X_test, dtype=np.float32)
    testY = np.array(testY, dtype=np.float32)

    return X_train, trainY, X_test, testY


# scale pixels
def preprocesar_pixeles(train, test):
    # convert from integers to floats
    # train_norm = train.astype('float32')
    # test_norm = test.astype('float32')
    # normalizamos entre 0 y 1
    train_norm = train / 255.0
    test_norm = test / 255.0
    return train_norm, test_norm


# define cnn modelo
def armar_modelo():
    modelo = Sequential()
    modelo.add(Conv2D(32, (3, 3), activation='relu', kernel_initializer='he_uniform', input_shape=(56, 56, 1)))
    modelo.add(MaxPooling2D((2, 2)))
    modelo.add(Conv2D(64, (3, 3), activation='relu', kernel_initializer='he_uniform'))
    modelo.add(Conv2D(64, (3, 3), activation='relu', kernel_initializer='he_uniform'))
    modelo.add(MaxPooling2D((2, 2)))
    modelo.add(Flatten())
    modelo.add(Dense(100, activation='relu', kernel_initializer='he_uniform'))
    modelo.add(Dense(10, activation='softmax'))
    # compilamos modelo
    opt = SGD(learning_rate=0.01, momentum=0.9)
    modelo.compile(optimizer=opt, loss='categorical_crossentropy', metrics=['accuracy'])
    return modelo


# evaluamos haciendo k-fold
def evaluar_modelo(dataX, dataY):
    scores, histories = list(), list()
    # prepare cross validation
    kfold = KFold(5, shuffle=True, random_state=1)
    # enumerate splits
    for train_ix, test_ix in kfold.split(dataX):
        # define modelo
        modelo = armar_modelo()
        # seleccionamos filas para entrenamiento y para test
        trainX, trainY, testX, testY = dataX[train_ix], dataY[train_ix], dataX[test_ix], dataY[test_ix]
        # entrenamos modelo
        history = modelo.fit(trainX, trainY, epochs=10, batch_size=32, validation_data=(testX, testY), verbose=0)
        # evaluamos modelo
        _, acc = modelo.evaluate(testX, testY, verbose=0)
        print('> %.3f' % (acc * 100.0))


# run the test harness for evaluating a modelo
def clasificar_digitos():
    # cargamos dataset
    trainX, trainY, testX, testY = dataset()
    # procesamos imagenes
    trainX, testX = preprocesar_pixeles(trainX, testX)
    # evaluamos con KFOLD al modelo
    evaluar_modelo(trainX, trainY)
    # entrenamos modelo
    modelo = armar_modelo()
    modelo.fit(trainX, trainY, epochs=10, batch_size=32, verbose=0)
    # guardamos el modelo
    modelo.save('CNN_clasif_digitos.h5')


# ejecutamos
# clasificar_digitos()
# una vez que entrenamos el modelo y lo tenemos guardado lo leemos
# cargamos la imagen y el modelo
modelo = load_model('CNN_clasif_digitos.h5')
img = cv2.imread("3.jpeg", 0)
# hacemos el preproceso para que el clasificador lo etiquete
img_resized = cv2.resize(img, (56, 56))
img_resized = cv2.bitwise_not(img_resized)
# reshape a una sola muestra
img_resized = img_resized.reshape(1, 56, 56)
# escalamos
img_resized = img_resized.astype('float32')
img_resized = img_resized / 255.0
# predecimos
digito = modelo.predict(img_resized)
digito = argmax(digito)
print(digito)
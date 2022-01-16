from enum import unique
import os, sys
import re
from webbrowser import get
import tensorflow as tf
import numpy as np
import pandas as pd
from PIL import Image
import matplotlib.pyplot as plt
import glob
import sklearn

IMAGE_SIZE = (100, 100)
IMAGE_SHAPE = (100, 100, 3)
MAX_RGB = 255.0

def getPic(imgPath):
    try:
        with Image.open(imgPath).convert("RGB").resize(IMAGE_SIZE, Image.ANTIALIAS) as image:
            return np.array(image, dtype=np.float32) / MAX_RGB
    except Exception as e:
        print(f"Error message on opening {imgPath}: {e}")
        return [[[None]]]

def getLabel(path):
    """Extract label for an image contained in folder 'path' as the first word of the folder name"""
    parentFolder = os.path.basename(path)
    split = re.split(r"(\S+)", parentFolder)
    label = split[1]
    return label

def getOneHot(labelList, label):
    """Get one-hot code identifying the index of label in labelList
    
    eg. For labelList = ["Apple", "Banana", "Orange"], label = "Orange", returns [0, 0, 1]
    """
    oneHot = [0 for _ in labelList]
    for i, l in enumerate(labelList):
        if l == label:
            oneHot[i] = 1
            break
    return oneHot

def printListToFile(listToPrint, filePath):
    file = open(filePath, "w")
    for item in listToPrint:
        file.write(f"{item}\n")
    file.close()

def getDataset(dataPath):
    fruitImgs = []
    labels = []
    uniqueLabels = []
    imgsLoaded = 0
    totalImgs = len(glob.glob(f"{dataPath}/*/*.jpg"))
    for filePath in glob.glob(f"{dataPath}/*"):
        label = getLabel(filePath)
        if label not in uniqueLabels:
            uniqueLabels.append(label)
        for img in glob.glob(f"{filePath}/*.jpg"):
            imgsLoaded += 1
            print(f"Loaded {imgsLoaded}/{totalImgs} images", end='\r')
            if getPic(img)[0][0][0] == None:
                print('skipped picture')
                continue
            fruitImgs.append(getPic(img))
            labels.append(label)
    print('\n')
    imgLabelPair = []
    for i, l in enumerate(labels):
        imgLabelPair.append([fruitImgs[i], getOneHot(uniqueLabels, l)])
        print(f"Converted {i}/{len(labels)} to One-Hot form", end='\r')
    print('\n\n')
    return (imgLabelPair, uniqueLabels)

    
    
if __name__ == "__main__":
    keras = tf.keras

    print(tf.__version__)
    config = tf.compat.v1.ConfigProto()
    config.gpu_options.allow_growth = True
    session = tf.compat.v1.InteractiveSession(config=config)

    trainDataPath = os.path.abspath(os.path.dirname(sys.argv[0]) + "/fruits-360/Training")
    train, uniqueLabelsTrain = getDataset(trainDataPath)

    testDataPath = os.path.abspath(os.path.dirname(sys.argv[0]) + "/fruits-360/Test")
    test, uniqueLabelsTest = getDataset(testDataPath)

    if uniqueLabelsTrain != uniqueLabelsTest:
        raise Exception("train and test datasets have different unique labels")
    
    uniqueLabels = uniqueLabelsTrain
    del(uniqueLabelsTest)

    labelFile = os.path.join(os.path.dirname(sys.argv[0]), "labels.txt")
    printListToFile(uniqueLabels, labelFile)

    baseModel = tf.keras.applications.MobileNetV2(input_shape=IMAGE_SHAPE, include_top=False, weights='imagenet')
    baseModel.summary()
    baseModel.trainable = False
    globalAvgLayer = tf.keras.layers.GlobalAveragePooling2D()
    predictionLayer = keras.layers.Dense(len(uniqueLabels), activation="sigmoid")
    
    model = tf.keras.Sequential([
        baseModel, 
        globalAvgLayer,
        predictionLayer
    ])
    
    baseLearningRate = 0.0001
    model.compile(optimizer=tf.keras.optimizers.RMSprop(learning_rate=baseLearningRate),
        loss=tf.keras.losses.BinaryCrossentropy(from_logits=True),
        metrics=['accuracy'])
    
    print('configuring data')
    
    np.random.shuffle(train)
    np.random.shuffle(test)
    #shuffledTrain, shuffledTest = sklearn.utils.shuffle(train, test, random_state=52)

    trainX = [row[0] for row in train]
    trainY = [row[1] for row in train]
    testX = [row[0] for row in test]
    testY = [row[1] for row in test]

    history = model.fit(x=np.array(trainX), y=np.array(trainY), batch_size=128, epochs=5)
    loss0, accuracy0 = model.evaluate(x=np.array(testX), y=np.array(testY), batch_size=128, steps=20)
    print('Saving model')

    modelFilePath = os.path.abspath(os.path.dirname(sys.argv[0]) + "/carrotVision.h5")
    model.save(modelFilePath)
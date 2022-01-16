import os, sys
import tensorflow as tf
import numpy as np
from PIL import Image
import matplotlib.pyplot as plt
import glob

config = tf.compat.v1.ConfigProto()
config.gpu_options.allow_growth = True
session = tf.compat.v1.InteractiveSession(config=config)

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

def getFruit(predictions, labels):
    maxVal = max(predictions)
    maxIndex = predictions.index(maxVal)
    return (labels[maxIndex], maxVal)

def getLabelsFromFile(filePath):
    file = open(filePath, "r")
    labels = []
    for line in file.readlines():
        label = line.replace("\n", "")
        labels.append(label)
    return labels

if __name__ == "__main__":
    model = tf.keras.models.load_model(os.path.abspath(os.path.dirname(sys.argv[0]) + "/carrotVision.h5"))

    imgPaths = glob.glob(os.path.abspath(os.path.dirname(sys.argv[0]) + "/fruits-360/Test/*/*.jpg"))

    imgArr = []
    imgToPrint = []
    for path in imgPaths:
        img = getPic(path)
        imgToPrint.append((img * 255.0).astype(np.uint8))
        imgArr.append(img)

    labelFile = os.path.join(os.path.dirname(sys.argv[0]), "labels.txt")
    uniqueLabels = getLabelsFromFile(labelFile)

    predictions = model.predict(np.array(imgArr))

    for i, p in enumerate(predictions):
        fruit, prob = getFruit(list(p), uniqueLabels)
        plt.figure()
        plt.title(f"I am {prob*100:0.2f}% sure this is a {fruit}")
        plt.imshow(imgToPrint[i])
        plt.show()
    
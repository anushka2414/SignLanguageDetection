from keras.models import Sequential 
from keras.layers import Dense, Conv2D, MaxPool2D, Flatten, Dropout, BatchNormalization 
from keras.preprocessing.image import ImageDataGenerator
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, confusion_matrix
import matplotlib.pyplot as plt
import pandas as pd 
from sklearn.preprocessing import LabelBinarizer

trainDF = pd.read_csv("./dataset/train.csv")
testDF = pd.read_csv("./dataset/test.csv")

yTrain = trainDF["label"]
yTest = testDF["label"]


del trainDF["label"]
del testDF["label"]

labelBinarizer = LabelBinarizer()

yTrain = labelBinarizer.fit_transform(yTrain) 
yTest = labelBinarizer.fit_transform(yTest)

xTrain = trainDF.values
xTest = testDF.values

#Normalization
xTest = xTest / 255 
xTrain = xTrain / 255 

xTrain = xTrain.reshape(-1, 28, 28, 1)
xTest = xTest.reshape(-1, 28, 28, 1) 

dataGenerator = ImageDataGenerator(
    featurewise_center= False, 
    samplewise_center=  False, 
    featurewise_std_normalization=False , 
    samplewise_std_normalization= False, 
    zca_whitening=False, 
    rotation_range=10, 
    zoom_range= 0.1, 
    width_shift_range=0.1, 
    height_shift_range=0.1, 
    horizontal_flip=False, 
    vertical_flip=False
)

dataGenerator.fit(xTrain) 

modelCNN = Sequential()

modelCNN.add(Conv2D(75, (3,3), strides= 1, padding="same", activation="relu", input_shape=(28,28,1)))

modelCNN.add(BatchNormalization())

modelCNN.add(MaxPool2D((2,2), strides= 2, padding="same"))

modelCNN.add(Conv2D(50, (3,3), strides = 1, padding= "same", activation = "relu"))

modelCNN.add(Dropout(0.2))

modelCNN.add(BatchNormalization())

modelCNN.add(MaxPool2D((2,2), strides = 2, padding= "same"))
modelCNN.add(Conv2D(25, (3,3), strides = 1, padding="same", activation="relu"))

modelCNN.add(BatchNormalization())

modelCNN.add(MaxPool2D((2,2), strides = 2, padding="same"))

modelCNN.add(Flatten())
modelCNN.add(Dense(units = 512, activation="relu"))
modelCNN.add(Dropout(0.3))
modelCNN.add(Dense(units = 24, activation = "softmax"))

modelCNN.compile(optimizer = 'adam', loss= "categorical_crossentropy", metrics = ["accuracy"])
modelCNN.summary()

history = modelCNN.fit(dataGenerator.flow(xTrain, yTrain, batch_size = 128), epochs = 20, validation_data = (xTest, yTest)) 

modelCNN.save('signLanguage.h5')


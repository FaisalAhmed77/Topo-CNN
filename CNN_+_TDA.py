
# Loading image Libraries

from matplotlib import image
from matplotlib import pyplot
import matplotlib.pyplot as plt
from PIL import Image
import pandas as pd
import numpy as np
import seaborn as sns
import glob

# Read image from folders and convert each image to array of pixels. 

file = "..../APTOS/train_images/" 

img_file = list(glob.glob1(file, "*.png"))
img = []
for i in img_file:
    img.append(i)

data = []
for i in img:
    image_path = file + i
    gray=Image.open(image_path)
    gray_resiz = gray.resize((256, 256))
    arr_gray = np.array(gray_resiz)
    data.append(arr_gray)
    
d = np.array(data)

file1 = ".../APTOS/val_images/" 

img_file1 = list(glob.glob1(file1, "*.png"))
img1 = []
for i in img_file1:
    img1.append(i)

data1 = []
for i in img1:
    image_path1 = file1 + i
    gray1=Image.open(image_path1)
    gray_resiz1 = gray1.resize((256, 256))
    arr_gray1 = np.array(gray_resiz1)
    data1.append(arr_gray1)
d1 = np.array(data1)

file2 = ".../APTOS/test_images/" 

img_file2 = list(glob.glob1(file2, "*.png"))
img2 = []
for i in img_file2:
    img2.append(i)

data2 = []
for i in img2:
    image_path2 = file2 + i
    gray2=Image.open(image_path2)
    gray_resiz2 = gray2.resize((256, 256))
    arr_gray2 = np.array(gray_resiz2)
    data2.append(arr_gray2)
d2 = np.array(data2)

# Concatenate all images

df = np.concatenate((d, d1, d2), axis = 0)
df.shape

# Add Labels 
trb = pd.read_csv("train_1.csv")
vb = pd.read_csv("valid.csv")
tb = pd.read_csv("test.csv")

yy = list( np.concatenate((trb["diagnosis"], vb["diagnosis"], tb["diagnosis"]), axis = 0))
y = np.array(yy)

# Spliting the data in 85:15 training to testing
from sklearn.model_selection import train_test_split
x_train, x_test, y_train, y_test = train_test_split(df, y, test_size = 0.15, random_state = 0)

import keras
y_train = keras.utils.np_utils.to_categorical(y_train,5)
y_test = keras.utils.np_utils.to_categorical(y_test,5)


# Adding TDA  features

df_tda = pd.read_excel("df_800.xlsx")
df_tda = df_tda.drop(df_tda.columns[[0]], axis=1)

# Spliting the TDA data in 85:15 training to testing
from sklearn.model_selection import train_test_split
x_train_tda, x_test_tda, y_train_tda, y_test_tda = train_test_split(df_tda.iloc[:,:800], y, test_size = 0.15, random_state = 0)

import keras
y_train_tda = keras.utils.np_utils.to_categorical(y_train_tda,5)
y_test_tda = keras.utils.np_utils.to_categorical(y_test_tda,5)

#importing CNN libraries
import matplotlib.pyplot as plt
from keras import Model
from keras.models import Sequential
from keras.layers import Conv2D, MaxPooling2D, Dropout, Flatten, Dense, Activation, BatchNormalization, concatenate
from tensorflow.keras import Input

# Neural Network features
def create_MLP(dim, regress=False):
    # define our MLP network
    model = Sequential()
    model.add(Dense(800, input_dim=dim, activation="relu"))
    model.add(Dense(256, activation="relu"))
    model.add(Dense(128, activation="relu"))
    #model.add(Dropout(0.2))
    # return our model
    return model
# Add CNN pre-trained Models

# Models 
import tensorflow as tf
from tensorflow.keras.layers import Input
from tensorflow.keras.layers import Flatten, Dense, Dropout, Conv2D, MaxPooling2D

from tensorflow.keras.applications.resnet50 import ResNet50
model_cnn = tf.keras.models.Sequential([
    ResNet50(input_shape=(256,256,3), include_top=False),
])
for layer in model_cnn.layers:
    layer.trainable = False

model_cnn.add(Conv2D(64, (3,3), activation='relu'))
model_cnn.add(MaxPooling2D(2,2))
model_cnn.add(Flatten())
model_cnn.add(Dense(64, activation='relu'))
# model.add(Dropout(0.4))
# model.add(Dense(2, activation='sigmoid'))

#model.summary()

# create the MLP and CNN models
mlp = create_MLP(x_train_tda.shape[1], regress=False)
#cnn = create_cnn(224, 224, 3, regress=False)
# create the input to our final set of layers as the *output* of both
# Concatenate TDA and CNN features 
combinedInput = concatenate([mlp.output, model_cnn.output])
# our final FC layer head will have two dense layers, the final one
x = Dense(256, activation="relu")(combinedInput)
x = Dense(128, activation='relu')(x)
x = Dense(128, activation='relu')(x)
Dropout(0.2)
x = Dense(5, activation="softmax")(x)

# Plotting the Model Architecture 

from tensorflow.keras.utils import plot_model

model = Model(inputs=[mlp.input, model_cnn.input], outputs=x)

plot_model(model, "multi_input_and_output_model.png", show_shapes=True)

# Compiling the Model 

model = Model(inputs=[mlp.input, model_cnn.input], outputs=x)

loss = keras.losses.categorical_crossentropy
model.compile(loss= loss, optimizer= 'adam', metrics=['accuracy','Precision','Recall','AUC'])
# train the model
print("[INFO] training model...")

h = model.fit(x=[x_train_tda,x_train], y=y_train,validation_data=([x_test_tda, x_test], y_test), epochs=50, batch_size=32)

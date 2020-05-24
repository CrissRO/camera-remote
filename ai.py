import cv2
import pandas as pd
import numpy as np

import tensorflow as tf
from tensorflow.keras.utils import to_categorical
from tensorflow.keras import Sequential
from tensorflow.keras.layers import Dense
from tensorflow.keras.layers import Conv2D
from tensorflow.keras.layers import MaxPool2D
from tensorflow.keras.layers import Flatten

impath = f'./data/train/images/'
training_data = pd.read_csv('./data/train/data.csv')
X = []
for img in training_data["image"].loc[:]:
    X.append(cv2.imread(f'{impath}{img}.jpg',0))
y = training_data["label"]

X = np.array(X)
y = np.array(y)

X = X.reshape((X.shape[0],X.shape[1],X.shape[2],1))

in_shape = X.shape[1:]
n_classes = len(np.unique(y))

X = X.astype("float32")/255
y = to_categorical(y)

model = Sequential()
model.add(Conv2D(256, (3,3), activation='relu',kernel_initializer='he_uniform',input_shape=in_shape))
model.add(MaxPool2D((2, 2)))
model.add(Flatten())
model.add(Dense(100, activation='relu', kernel_initializer='he_uniform'))
model.add(Dense(n_classes, activation='softmax'))

model.compile(optimizer='sgd', loss='categorical_crossentropy', metrics=['accuracy'])

model.fit(X, y, epochs=10, batch_size=128)
'''
video = cv2.VideoCapture(0)
running = True

while running:
    check,frame = video.read()
    frame = cv2.resize(frame,(256,256))
    frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    cv2.imshow("Capture",frame)
    key = cv2.waitKey(1)
    
    if key == ord('q'):
        running = False
        
video.release()
'''

import cv2
import pandas as pd
'''
import tensorflow as tf
from tensorflow.keras import Sequential
from tensorflow.keras.layers import Dense
from tensorflow.keras.layers import Conv2D
from tensorflow.keras.layers import MaxPool2D
from tensorflow.keras.layers import Flatten
'''

training_images = pd.read_csv('./data/train/data.csv')

f = pd.DataFrame([['a','5']])

training_images.append(f)



video = cv2.VideoCapture(0)
running = True


while running:
    check,frame = video.read()
    frame = cv2.resize(frame,(256,256))
    #frame[:,:,(0,2)] = 0
    frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    cv2.imshow("Capture",frame)
    key = cv2.waitKey(1)
    if key == ord('5'):
        running = False

    if key == ord('5'):
        running = False

    if key == ord('5'):
        running = False
    if key == ord('q'):
        running = False

video.release()
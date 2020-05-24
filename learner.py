import cv2
import pandas as pd
import numpy as np
'''
import tensorflow as tf
from tensorflow.keras import Sequential
from tensorflow.keras.layers import Dense
from tensorflow.keras.layers import Conv2D
from tensorflow.keras.layers import MaxPool2D
from tensorflow.keras.layers import Flatten
'''

training_data = pd.read_csv('./data/train/data.csv')
last_image_index = training_data.tail(1).index.item() + 1 if len(training_data.tail(1)) > 0 else 0
new_frames = []
command_keys = ['0','1','2','3','4','5','0']
#f = pd.DataFrame([['a','5']],columns=list(training_data.columns))
#training_data = training_data.append(f,ignore_index=True)
#print(training_data.head())



video = cv2.VideoCapture(0)
running = True

def add_new_image(image_list,image,image_name,label):
    image_list.append([image,image_name,label])

def save_images(image_list,impath,csvpath,dataframe):
    df = pd.DataFrame(image_list[:,1:3],columns=list(training_data.columns))
    images = image_list[:,0:2]
    dataframe = dataframe.append(df,ignore_index=True)
    for img in images:
        cv2.imwrite(f"{impath}{img[1]}.jpg",img[0])
    dataframe.to_csv(f'{csvpath}',index=False)

while running:
    check,frame = video.read()
    frame = cv2.resize(frame,(256,256))
    frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    cv2.imshow("Capture",frame)
    key = cv2.waitKey(1)

    for ckey in command_keys:
        if key == ord(ckey):
            add_new_image(new_frames,frame,f"image{last_image_index}",ckey)
            last_image_index += 1

    if key == ord('q'):
        running = False

video.release()
save_images(np.array(new_frames),"./data/train/images/","./data/train/data.csv",training_data)

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


def draw_rect(frame):
    rows, cols, _ = frame.shape
    global total_rectangle, hand_rect_one_x, hand_rect_one_y, hand_rect_two_x, hand_rect_two_y
    total_rectangle = 9
    hand_rect_one_x = np.array(
        [6 * rows / 20, 6 * rows / 20, 6 * rows / 20, 9 * rows / 20, 9 * rows / 20, 9 * rows / 20, 12 * rows / 20,
         12 * rows / 20, 12 * rows / 20], dtype=np.uint32)

    hand_rect_one_y = np.array(
        [9 * cols / 20, 10 * cols / 20, 11 * cols / 20, 9 * cols / 20, 10 * cols / 20, 11 * cols / 20, 9 * cols / 20,
         10 * cols / 20, 11 * cols / 20], dtype=np.uint32)

    hand_rect_two_x = hand_rect_one_x + 10
    hand_rect_two_y = hand_rect_one_y + 10

    for i in range(total_rectangle):
        cv2.rectangle(frame, (hand_rect_one_y[i], hand_rect_one_x[i]),
                      (hand_rect_two_y[i], hand_rect_two_x[i]),
                      (0, 255, 0), 1)

    return frame

def hand_histogram(frame):
    global hand_rect_one_x, hand_rect_one_y

    hsv_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
    roi = np.zeros([90, 10, 3], dtype=hsv_frame.dtype)

    for i in range(total_rectangle):
        roi[i * 10: i * 10 + 10, 0: 10] = hsv_frame[hand_rect_one_x[i]:hand_rect_one_x[i] + 10,
                                          hand_rect_one_y[i]:hand_rect_one_y[i] + 10]

    hand_hist = cv2.calcHist([roi], [0, 1], None, [180, 256], [0, 180, 0, 256])
    return cv2.normalize(hand_hist, hand_hist, 0, 255, cv2.NORM_MINMAX)

def hist_masking(frame, hist):
    hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
    dst = cv2.calcBackProject([hsv], [0, 1], hist, [0, 180, 0, 256], 1)

    disc = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (31, 31))
    cv2.filter2D(dst, -1, disc, dst)

    #ret, thresh = cv2.threshold(dst, 180, 255, cv2.THRESH_BINARY)
    
    ret, thresh = cv2.threshold(dst, 125, 125, cv2.THRESH_BINARY)
    
    th3 = cv2.adaptiveThreshold(dst,255,cv2.ADAPTIVE_THRESH_GAUSSIAN_C,cv2.THRESH_BINARY,5,2)

    thresh = cv2.merge((th3, th3, th3))

    return cv2.bitwise_and(frame, thresh)

hand_initialized = False
hist = None

while running:
    check,frame = video.read()
    frame = cv2.resize(frame,(512,512))
    #frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    if not hand_initialized:
        cv2.imshow("Capture",draw_rect(frame))
    else:
        cv2.imshow("Capture",hist_masking(frame,hist))
    key = cv2.waitKey(1)

    for ckey in command_keys:
        if key == ord(ckey):
            add_new_image(new_frames,frame,f"image{last_image_index}",ckey)
            last_image_index += 1

    if key == ord('q'):
        running = False

    if key == ord('i'):
        hand_initialized = True
        hist = hand_histogram(frame)

video.release()
save_images(np.array(new_frames),"./data/train/images/","./data/train/data.csv",training_data)

import cv2
import time
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
import os
import random
import numpy as np
TIMER = int(3)
  

cap = cv2.VideoCapture(0)
img_counter = 0

model = keras.models.load_model('dobrymodel.h5')

image_size = (180, 180)

choice = ['paper', 'rock', 'scissors']
score = [0, 0]

def playing(player_choice,bot_choice):
    if player_choice==bot_choice:
        return [1,1]
    elif player_choice==0:
        if bot_choice==1:
            return [1,0]
        else: 
            return [0,1]
    elif player_choice==1:
        if bot_choice==0:
            return [0,1]
        else: 
            return [1,0]
    elif player_choice==2:
        if bot_choice==0:
            return [1,0]
        else:
            return [0,1]




while True:
     

    ret, img = cap.read()
    cv2.imshow('a', img)

    k = cv2.waitKey(125)
 

    if k == ord('q'):
        prev = time.time()
 
        while TIMER >= 0:
            ret, img = cap.read()
 

            font = cv2.FONT_HERSHEY_SIMPLEX
            cv2.putText(img, str(TIMER),
                        (200, 250), font,
                        7, (0, 255, 255),
                        4, cv2.LINE_AA)
            cv2.imshow('a', img)
            cv2.waitKey(125)
 

            cur = time.time()
 
            if cur-prev >= 1:
                prev = cur
                TIMER = TIMER-1
 
        else:
            ret, img = cap.read()
 

            cv2.imshow('a', img)
 
            cv2.waitKey(2000)
 
            img_name = "datatest/{}.png".format(img_counter)
            cv2.imwrite(img_name, img)
            TIMER = int(3)
            


            cv2.putText(img, random.choice(choice),
                        (150, 250), font,
                        3, (0, 255, 255),
                        4, cv2.LINE_AA)
            cv2.imshow('a', img)
            cv2.waitKey(125)
            img = keras.preprocessing.image.load_img(
                img_name, target_size=image_size
            )
            img_array = keras.preprocessing.image.img_to_array(img)
            img_array = tf.expand_dims(img_array, 0)  # Create batch axis

            predictions = model.predict(img_array)
            max_prediction = max(max(predictions))
            result = np.where(predictions[0] == max_prediction)
            player_choice=result[0]

            bot_choice = choice.index(random.choice(choice))    
            outcome = playing(player_choice, bot_choice)

            score[0] += outcome[0]
            score[1] += outcome[1]
            img_counter += 1

 
    elif k == 27:
        break
    elif img_counter == 3:
        cv2.putText(img, str(score[0])+":"+str(score[1]),
                    (200, 250), font,
                    3, (0, 255, 255),
                    4, cv2.LINE_AA)
        cv2.imshow('a', img)
        cv2.waitKey(2000) 
        break

cap.release()
  

cv2.destroyAllWindows()
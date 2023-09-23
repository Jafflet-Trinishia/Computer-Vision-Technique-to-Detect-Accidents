import tensorflow as tf
from tensorflow import keras
import numpy as np
import cv2
import random

# Loading themodel
batch_size = 32
img_height = 64
img_width = 64
model_dl = keras.models.load_model("model.h5")  # look for local saved file

from keras.preprocessing import image

# Creating a dictionary to map each of the indexes to the corresponding number or letter

webcam_video_stream = cv2.VideoCapture('651RQwxB3WA.mp4')
dict = {0: "Accident", 1: "Non accident"}

import smtplib
from email.message import EmailMessage

def emailalert(subject,body,to):
    msg = EmailMessage()
    msg.set_content(body)
    msg['subject'] = subject
    msg['to'] = to

    user = 'jafflettrini@gmail.com'
    msg['from']=user
    passw = 'ammaappa123.'

    server = smtplib.SMTP("smtp.gmail.com",587)
    server.starttls()
    server.login(user,passw)
    server.send_message(msg)
    server.quit()
    print('successfull')

n = 0
while True:

    ret, current_frame = webcam_video_stream.read()
    if ret:

        img_to_detect = current_frame
        img_height = img_to_detect.shape[0]
        img_width = img_to_detect.shape[1]

        img = cv2.resize(img_to_detect, (250, 250))
        x = image.img_to_array(img)
        x = np.expand_dims(x, axis=0)

        imag = np.vstack([x])
        classes = model_dl.predict_classes(imag, batch_size=100)
        print(classes)
        num1 = random.randint(0, 19)
        # probabilities = model_dl.predict_proba(imag, batch_size=batch_size)
        # probabilities_formatted = list(map("{:.2f}%".format, probabilities[0]*100))
        text = str(dict[classes.item()])
        print(text)
        cv2.putText(img_to_detect, text, (45, 60), cv2.FONT_HERSHEY_SIMPLEX, 1.25, (255, 0, 0), 5)
        if text == 'Accident':
            n+=1
            if n==20:
                emailalert('accident', "Accident detected", 'Jafflettrini@gmail.com')
                n=0
    else:
        break

    cv2.imshow("Detection Output", img_to_detect)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

webcam_video_stream.release()
cv2.destroyAllWindows()
import time
import os
from keras.models import load_model
import cv2
from datetime import datetime
from gtts import gTTS
from pygame import mixer
import numpy as np
import playsound


mixer.init()
proj_path = "D:\\vs code files\\python files\\down Drowsiness detection\\Drowsiness detection"
sound = mixer.Sound(os.path.join(proj_path, "alarm1.wav"))

face_cascade = cv2.CascadeClassifier(os.path.join(
    proj_path, "haar cascade files\\haarcascade_frontalface_alt.xml"))
leye_cascade = cv2.CascadeClassifier(
    os.path.join(proj_path, "haar cascade files\\haarcascade_lefteye_2splits.xml"))
reye_cascade = cv2.CascadeClassifier(
    os.path.join(proj_path, "haar cascade files\\haarcascade_righteye_2splits.xml"))


label = ['Close', 'Open']

model = load_model(os.path.join(proj_path, "models\\cnnCat2.h5"))

cap = cv2.VideoCapture(0)
face_not_detect = 0
line_thickness = 2
font = cv2.FONT_HERSHEY_COMPLEX_SMALL
counter = 0
score = 0
right_eye_pred = [99]
left_eye_pred = [99]

while(True):
    ret, frame = cap.read()
    height, width = frame.shape[:2]

    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    faces = face_cascade.detectMultiScale(
        gray, minNeighbors=5, scaleFactor=1.1, minSize=(25, 25))
    left_eye = leye_cascade.detectMultiScale(gray)
    right_eye = reye_cascade.detectMultiScale(gray)

    cv2.rectangle(frame, (0, height-50), (200, height),
                  (0, 0, 0), thickness=cv2.FILLED)
    if len(faces) == 0:
        face_not_detect += 1
    else:
        face_not_detect = 0

    if(face_not_detect > 30):
        playsound.playsound(os.path.join(proj_path, "face_not_found.mp3"))

    for (x, y, w, h) in faces:
        cv2.rectangle(frame, (x, y), (x+w, y+h), (100, 100, 100), 1)

    for (x, y, w, h) in right_eye:
        right_eye = frame[y:y+h, x:x+w]
        counter = counter+1
        right_eye = cv2.cvtColor(right_eye, cv2.COLOR_BGR2GRAY)
        right_eye = cv2.resize(right_eye, (24, 24))
        right_eye = right_eye/255
        right_eye = right_eye.reshape(24, 24, -1)
        right_eye = np.expand_dims(right_eye, axis=0)
        right_eye_pred = model.predict_classes(right_eye)
        if(right_eye_pred[0] == 1):
            label = 'Open'
        if(right_eye_pred[0] == 0):
            label = 'Closed'
        break

    for (x, y, w, h) in left_eye:
        left_eye = frame[y:y+h, x:x+w]
        counter = counter+1
        left_eye = cv2.cvtColor(left_eye, cv2.COLOR_BGR2GRAY)
        left_eye = cv2.resize(left_eye, (24, 24))
        left_eye = left_eye/255
        left_eye = left_eye.reshape(24, 24, -1)
        left_eye = np.expand_dims(left_eye, axis=0)
        left_eye_pred = model.predict_classes(left_eye)
        if(left_eye_pred[0] == 1):
            label = 'Open'
        if(left_eye_pred[0] == 0):
            label = 'Closed'
        break

    if(right_eye_pred[0] == 0 and left_eye_pred[0] == 0):
        score = score+1
        cv2.putText(frame, "Closed", (10, height-20), font,
                    1, (255, 255, 255), 1, cv2.LINE_AA)

    else:
        score = score-2
        cv2.putText(frame, "Open", (10, height-20), font,
                    1, (255, 255, 255), 1, cv2.LINE_AA)

    if(score < 0):
        score = 0
    cv2.putText(frame, 'Score:'+str(score), (100, height-20),
                font, 1, (255, 255, 255), 1, cv2.LINE_AA)
    if(score > 15):
        path_to_images = os.path.join(proj_path, "drowsy_captures")
        now = datetime.now()

        current_time = now.strftime("%H:%M:%S")
        current_time = current_time.replace(":", "_")

        # index_of_person = len(os.listdir(path_to_images))+1
        filename = "drowsy_"+current_time+".jpg"
        if(right_eye_pred[0] == 0 and left_eye_pred[0] == 0):
            cv2.imwrite(os.path.join(path_to_images,
                                     filename), frame)
        try:
            sound.play()

        except:
            pass
        if(line_thickness < 16):
            line_thickness = line_thickness+2
        else:
            line_thickness = line_thickness-2
            if(line_thickness < 2):
                line_thickness = 2
        cv2.rectangle(frame, (0, 0), (width, height),
                      (0, 0, 255), line_thickness)
    cv2.imshow('frame', frame)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break
cap.release()
cv2.destroyAllWindows()

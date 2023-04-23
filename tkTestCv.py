from PIL import Image, ImageTk
import tkinter as tk
from mediapipe1 import *
import argparse
# from dbConnect import *

# import datetime
import cv2
import time
import os
from keras.models import load_model
import cv2
from datetime import datetime
from gtts import gTTS
from pygame import mixer
import numpy as np
import tensorflow as tf
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
# import os
height, width = 0, 0


def display(frame, gray):
    global face_not_detect, right_eye_pred, left_eye_pred, counter, line_thickness, font, score
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
    if(score > 15 or face_not_detect > 30):
        path_to_images = os.path.join(proj_path, "drowsy_captures")
        now = datetime.now()

        current_time = now.strftime("%H:%M:%S")
        current_time = current_time.replace(":", "_")

        # index_of_person = len(os.listdir(path_to_images))+1
        filename = "drowsy_"+current_time+".jpg"
        if((right_eye_pred[0] == 0 and left_eye_pred[0] == 0) or face_not_detect > 30):
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
    # if cv2.waitKey(1) & 0xFF == ord('q'):
    #     break


class Application:
    def __init__(self, output_path="./"):
        """ Initialize application which uses OpenCV + Tkinter. It displays
            a video stream in a Tkinter window and stores current snapshot on disk """
        self.vs = cv2.VideoCapture(
            0)  # capture video frames, 0 is your default video camera
        self.output_path = output_path  # store output path
        self.current_image = None  # current image from the camera

        self.root = tk.Tk()  # initialize root window
        self.root.title("PyImageSearch PhotoBooth")  # set window title
        # self.destructor function gets fired when the window is closed
        self.root.protocol('WM_DELETE_WINDOW', self.destructor)
        self.encoding = getEncoding()
        self.panel = tk.Label(self.root)  # initialize image panel
        self.panel.pack(padx=10, pady=10)

        # create a button, that when pressed, will take the current frame and save it to file
        # btn = tk.Button(self.root, text="Snapshot!", command=self.take_snapshot)
        # btn.pack(fill="both", expand=True, padx=10, pady=10)

        # start a self.video_loop that constantly pools the video sensor
        # for the most recently read frame
        self.video_loop()

    def video_loop(self):
        global height, width
        """ Get frame from the video stream and show it in Tkinter """
        ok, frame = self.vs.read()  # read frame from video stream
        if ok:  # frame captured without any errors
            frame = blurFace(myEncoding=self.encoding, frame=frame)
            height, width = frame.shape[:2]

            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            display(frame, gray)

            # convert colors from BGR to RGBA
            # cv2image = cv2.cvtColor(frame, cv2.COLOR_BGR2RGBA)

            # self.current_image = Image.fromarray(
            #     cv2image)  # convert image for PIL
            # # convert image for tkinter
            # imgtk = ImageTk.PhotoImage(image=self.current_image)
            # self.panel.imgtk = imgtk  # anchor imgtk so it does not be deleted by garbage-collector
            # self.panel.config(image=imgtk)  # show the image
        # call the same function after 30 milliseconds
        self.root.after(30, self.video_loop)

    # def take_snapshot(self):
    #     """ Take snapshot and save it to the file """
    #     ts = datetime.datetime.now() # grab the current timestamp
    #     filename = "{}.jpg".format(ts.strftime("%Y-%m-%d_%H-%M-%S"))  # construct filename
    #     p = os.path.join(self.output_path, filename)  # construct output path
    #     self.current_image.save(p, "JPEG")  # save image as jpeg file
    #     print("[INFO] saved {}".format(filename))

    def destructor(self):
        """ Destroy the root object and release all resources """
        print("[INFO] closing...")
        self.root.destroy()
        self.vs.release()  # release web camera
        cv2.destroyAllWindows()  # it is not mandatory in this application


# construct the argument parse and parse the arguments
ap = argparse.ArgumentParser()
ap.add_argument("-o", "--output", default="./",
                help="path to output directory to store snapshots (default: current folder")
args = vars(ap.parse_args())

# start the app
print("[INFO] starting...")
pba = Application(args["output"])
pba.root.mainloop()


def takemypic():
    cap = cv2.VideoCapture(0)
    while True:
        success, myimg = cap.read()
        if success == True:
            myimgCopy = myimg.copy()
            # myimg=cv2.resize(myimg,(200,200))
            cv2.rectangle(myimgCopy, (50, 50-30), (400, 50),
                          color=(255, 0, 0), thickness=-1)
            cv2.putText(myimgCopy, "Press 'q' to click..!!!", org=(
                50, 50), fontFace=cv2.FONT_HERSHEY_COMPLEX, fontScale=1, color=(255, 255, 255), thickness=1)
            cv2.imshow("WebCam", myimgCopy)

            if cv2.waitKey(1) & 0xFF == ord('q'):
                cv2.imwrite(
                    "D:\\vs code files\python files\Blur-Face-OpenCv\mypic.png", myimg)
                cv2.destroyAllWindows()
                break


# takemypic()

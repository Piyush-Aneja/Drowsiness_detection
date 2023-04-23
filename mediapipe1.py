import cv2
import mediapipe as mp
import face_recognition
import numpy as np


def blurFace(myEncoding, frame):
    # cap = cv2.VideoCapture(0)
    mpFaceDetection = mp. solutions.face_detection
    faceDetection = mpFaceDetection.FaceDetection(0.75)
    faces_encoding = [myEncoding]
    faces_names = ["Sudarsan", "Others"]
    faceLocations = []
    faceEncodings = []
    faceNames = []
    # while True:
    img = frame
    imgRGB = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    results = faceDetection.process(imgRGB)
    # print(results.detections)
    if results.detections:
        for id, detection in enumerate(results.detections):
            bboxC = detection.location_data.relative_bounding_box
            ih, iw, ic = img.shape

            bbox_x = int(bboxC.xmin * iw)
            bbox_y = int(bboxC.ymin * ih)
            old_width = int(bboxC.width * iw)
            old_height = int(bboxC.height * ih)

            # bbox = int(bboxC.xmin * iw), int(bboxC.ymin * ih), int(bboxC.width * iw), int(bboxC.height * ih)

            face = img[bbox_y:bbox_y+old_height, bbox_x:bbox_x+old_width]
            faceLocations = face_recognition.face_locations(face)
            faceEncodings = face_recognition.face_encodings(
                face, faceLocations)

            for faceEncoding in faceEncodings:
                matches = face_recognition.compare_faces(
                    faces_encoding, faceEncoding)
                name = ""
                face_distance = face_recognition.face_distance(
                    faces_encoding, faceEncoding)
                best_match_index = np.argmin(face_distance)
                if matches[best_match_index]:
                    name = faces_names[best_match_index]
                else:
                    name = "Others"
                if name == "Others":
                    img[bbox_y:bbox_y+old_height, bbox_x:bbox_x+old_width] = cv2.medianBlur(
                        img[bbox_y:bbox_y+old_height, bbox_x:bbox_x+old_width], 75)
    return img
    # cv2.imshow('Test 2', img)
    # if cv2.waitKey(1) & 0xFF == ord('q'):
    #     break

    # cap.release()
    # cv2.destroyAllWindows()


def getEncoding():
    myImage = face_recognition.load_image_file(
        "D:\\vs code files\python files\Blur-Face-OpenCv\mypic.png")
    # "D:\\vs code files\python files\down Drowsiness detection\Drowsiness detection\drowsy_captures\drowsy_11_25_17.jpg")
    myEncoding = face_recognition.face_encodings(myImage)[0]
    return myEncoding

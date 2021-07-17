#!/usr/bin/env python
# coding: utf-8

# In[ ]:


mport cv2
import os
import numpy as np
from datetime import datetime
import faceRecognition as fr

test_img = cv2.imread('TestImages/shivam.jpeg')  # test_img path
faces_detected, gray_img = fr.faceDetection(test_img)
print("faces_detected:", faces_detected)


faces, faceID = fr.labels_for_training_data('trainingImages')
face_recognizer = fr.train_classifier(faces, faceID)
face_recognizer.write('trainingData.yml')





name = {0: "Kamya", 1: "Sarthak", 2: "Ronak"}  # creating dictionary containing names for each label

for face in faces_detected:
    (x, y, w, h) = face
    roi_gray = gray_img[y:y + h, x:x + h]
    label, confidence = face_recognizer.predict(roi_gray)  # predicting the label of given image
    print("confidence:", confidence)
    print("label:", label)
    fr.draw_rect(test_img, face)
    predicted_name = name[label]
    if (confidence > 40): 
        continue
    fr.put_text(test_img, predicted_name, x, y)

def markAttendance(predicted_name):
    with open('Attendance.csv','r+') as f:
        myDataList = f.readlines()
        nameList = []
        for line in myDataList:
            entry = line.split(',')
            nameList.append(entry[0])
        if predicted_name not in nameList:
            now = datetime.now()
            dtString = now.strftime('%H:%M:%S')
            f.writelines(f'\n{predicted_name},{dtString}')
markAttendance(predicted_name)
resized_img = cv2.resize(test_img, (1000, 1000))
cv2.imshow("face detection tutorial", resized_img)
cv2.waitKey(0)  # Waits indefinitely until a key is pressed
cv2.destroyAllWindows


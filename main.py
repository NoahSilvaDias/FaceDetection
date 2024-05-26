import cv2
import streamlit as st
import pathlib

# INSTALAR haarcascade_frontalface_default ->
# face_cascade = pathlib.Path(cv2.__file__).parent.absolute() / "data/haarcascade_frontalface_default.xml"
# print(face_cascade)

face_cascade = cv2.CascadeClassifier('haarcascade_frontalface_default.xml')

# Capturar por Webcam. ->
cap = cv2.VideoCapture(0)

# Capturar por Video ->
# cap = cv2.VideoCapture('filename.mp4')

while True:
    _, img = cap.read()

    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    faces = face_cascade.detectMultiScale(gray, 1.1, 4)

    for (x, y, w, h) in faces:
        cv2.rectangle(img, (x, y), (x+w, y+h), (0, 255, 0), 2)

    cv2.imshow('img', img)

    k = cv2.waitKey(30) & 0xff
    if k==27:
        break
        
cap.release()
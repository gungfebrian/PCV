import cv2
import numpy as np

cap = cv2.VideoCapture(0)
while True:
    ret, frame = cap.read()
    if not ret:
        break
    
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    Blur = cv2.GaussianBlur(gray,(5,5), 0)
    edges = cv2.Canny(Blur, 200, 100)
    contours, hierarchy = cv2.findContours(edges, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    tepi = cv2.drawContours(frame.copy(), contours, -1, (0, 255, 0), 2)
    
    cv2.imshow('Edges', edges)
    cv2.imshow('tepi', tepi)
    
cap.realease(0)
cv2.destroyAllWindows()
    





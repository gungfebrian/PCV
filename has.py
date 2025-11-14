import cv2
import numpy as np
import os



def order_points(pts):  #order points dari kiri bawah ke kanan atas
    rect = np.zeros((4, 2), dtype = "float32") #array 4,2 isi 0
    s = pts.sum(axis = 1) # x+y
    rect[0] = pts[np.argmin(s)] # kiri bawah
    rect[2] = pts[np.argmax(s)] # kanan atas
    diff = np.diff(pts, axis = 1) # x-y
    rect[1] = pts[np.argmin(diff)] # kiri atas
    rect[3] = pts[np.argmax(diff)] # kanan bawah
    return rect

#fungsi klas kartu
    

cap = cv2.VideoCapture(0)
while True:
    ret, frame = cap.read()
    if not ret:
        break
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    blur = cv2.GaussianBlur(gray,(5,5), 0)
    edges = cv2.Canny(blur, 200, 100)
    contours, hierarchy = cv2.findContours(edges, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    tepi = cv2.drawContours(frame.copy(), contours, -1, (0, 255, 0), 2)





    cv2.imshow("Kamera", frame)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release(0)
cv2.destroyAllWindows()
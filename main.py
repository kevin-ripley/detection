import cv2 as cv
import numpy as np

firstFrame = None
cam = cv.VideoCapture(0)

while(True):
    (grabbed, frame) = cam.read()


    grayscale = cv.cvtColor(frame, cv.COLOR_BGR2GRAY)
    grayscale = cv.GaussianBlur(grayscale, (21,21), 0)

    if firstFrame is None:
        firstFrame = grayscale
        continue


    frameDiff = cv.absdiff(firstFrame, grayscale)
    threshold = cv.threshold(frameDiff, 25, 255, cv.THRESH_BINARY)[1]

    threshold = cv.dilate(threshold, None, iterations = 2)
    (count, _) = cv.findContours(threshold.copy(), cv.RETR_EXTERNAL, cv.CHAIN_APPROX_SIMPLE)

    for i in count:
        if cv.contourArea(i) < 4:
            continue

        (x,y,w,h) = cv.boundingRect(i)
        cv.rectangle(frame, (x,y), (x+w, y+h), (0,255,0), 2)

    cv.imshow("Frame", frame)
    cv.imshow("Threshold", threshold)
    cv.imshow("Diff", frameDiff)

cam.release()
cv.destroyAllWindows()
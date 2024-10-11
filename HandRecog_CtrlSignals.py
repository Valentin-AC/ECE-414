import numpy as np
import cv2
#if you plan to run this code run the following commands into your command prompt
#pip install numpy
#pip install cv2
#also, add directory for each to PATH (have to figure out how to do this myself)

webcam_capture = cv2.VideoCapture(0)

while True:
    webcam_vid_stream = webcam_capture.read()

    if cv2.waitKey(1) == ord('x'):
        break

webcam_capture.release()
cv2.destroyAllWindows()
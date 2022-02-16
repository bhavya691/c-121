import cv2
import numpy as np
import time

# fourcc = cv2.VideoWriter_fourcc(*'XVID')
# output_file = cv2.VideoWriter('output.avi', fourcc)
cap = cv2.VideoCapture(0)
time.sleep(2)
image = cv2.imread('image.jpg')
while(cap.isOpened()):
    ret, frame = cap.read()
    if not ret:
        break
    frame = cv2.resize(frame, (640,480))
    image = cv2.resize(image, (640,480))
    frame = np.flip(frame, axis=1)
    frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    u_black = np.array([104, 153, 70])
    l_black = np.array([30, 30, 0])
    mask = cv2.inRange(frame, l_black, u_black)
    res = cv2.bitwise_and(frame, frame, mask=mask)
    f = frame - res
    f = np.where(f == 0, image, f)
    cv2.imshow('Masked', f)
    cv2.imshow('Real', frame)
    if cv2.waitKey(1) == ord('q'):
        break
cap.release()
cv2.destroyAllWindows()
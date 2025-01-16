import cv2 as cv
import numpy as np

cap = cv.VideoCapture("../task_videos/task4_level1.mov")

if not cap.isOpened():
    print("无法打开摄像头")
    exit()

while True:
    ret, frame = cap.read()

    if not ret:
        break

    frame = cv.GaussianBlur(frame, (5, 5), 0)

    hsv = cv.cvtColor(frame, cv.COLOR_BGR2HSV)
    lower_blue = np.array([110, 50, 50])
    upper_blue = np.array([130, 255, 255])

    mask = cv.inRange(hsv, lower_blue, upper_blue)

    contours, _ = cv.findContours(mask, cv.RETR_EXTERNAL, cv.CHAIN_APPROX_SIMPLE)
    frame_draw = cv.drawContours(frame.copy(), contours, -1, (0, 0, 255), 2)
    for contour in contours:
        M = cv.moments(contour)
        if M['m00'] <= 0:
            continue
        cx = int(M['m10'] / M['m00'])
        cy = int(M['m01'] / M['m00'])
        cv.circle(frame_draw, (cx, cy), 3, (0, 0, 255), -1)

    cv.imshow("frame", frame)
    cv.imshow("frame_draw", frame_draw)
    # cv.imshow("mask", mask)

    if cv.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv.destroyAllWindows()

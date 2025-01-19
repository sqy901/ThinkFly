import cv2 as cv
import numpy as np

# cap = cv.VideoCapture(0)  # 640*480
cap = cv.VideoCapture("../task_videos/task4_level1.mov")

# 原始分辨率
original_width = cap.get(cv.CAP_PROP_FRAME_WIDTH)
original_height = cap.get(cv.CAP_PROP_FRAME_HEIGHT)
print(f"{original_width}*{original_height}")

# 设置摄像头分辨率
# cap.set(cv.CAP_PROP_FRAME_WIDTH, 800)
# cap.set(cv.CAP_PROP_FRAME_HEIGHT, 800)

if not cap.isOpened():
    print("无法打开摄像头")
    exit()

while True:
    ret, frame = cap.read()
    if not ret:
        break

    # 设置视频分辨率
    # frame = cv.cvtColor(frame, cv.COLOR_BGR2GRAY)
    # frame = cv.resize(frame, (640, 480))

    hsv = cv.cvtColor(frame, cv.COLOR_BGR2HSV)
    lower_blue = np.array([100, 43, 46])
    upper_blue = np.array([124, 255, 255])

    mask = cv.inRange(hsv, lower_blue, upper_blue)
    res = cv.bitwise_and(frame, frame, mask=mask)

    res = cv.cvtColor(res, cv.COLOR_BGR2GRAY)
    _, res = cv.threshold(res, 20, 255, cv.THRESH_BINARY_INV)

    cv.imshow("frame", frame)
    cv.imshow("mask", mask)
    cv.imshow("res", res)

    if cv.waitKey(10) & 0xFF == ord('q'):
        break

cap.release()
cv.destroyAllWindows()

import cv2 as cv
import numpy as np

cap = cv.VideoCapture("../task_videos/task4_level2.mp4")

template = cv.imread("../image/flag0.jpg")
template = cv.resize(template, (437, 640))
template_gray = cv.cvtColor(template, cv.COLOR_BGR2GRAY)
_, template_binary = cv.threshold(template_gray, 120, 255, cv.THRESH_BINARY_INV)
template_contour, _ = cv.findContours(template_binary, cv.RETR_EXTERNAL, cv.CHAIN_APPROX_SIMPLE)
# template_res = cv.drawContours(template.copy(), template_contour, -1, (0, 0, 255), 2)
# print(len(template_contour))
# cv.imshow("template_res", template_res)
# cv.waitKey(0)
# cv.destroyAllWindows()

if not cap.isOpened():
    print("打开失败")
    exit()

lower_blue = np.array([100, 43, 46])
upper_blue = np.array([124, 255, 255])
lower_white = np.array([0, 0, 211])
upper_white = np.array([180, 30, 255])

while True:
    ret, frame = cap.read()

    if not ret:
        print("视频结束")
        break

    frame_blur = cv.GaussianBlur(frame, (5, 5), 0)
    hsv = cv.cvtColor(frame_blur, cv.COLOR_BGR2HSV)

    mask = cv.inRange(hsv, lower_blue, upper_blue)

    contours, hierarchy = cv.findContours(mask, cv.RETR_CCOMP, cv.CHAIN_APPROX_SIMPLE)

    tar_contours = []
    roi_contours = []

    '''hierarchy类型?'''
    '''内轮廓判断不准确，外轮廓有很多非目标轮廓'''
    for i in range(len(contours)):
        if cv.contourArea(contours[i]) < 200:
            continue
        if cv.matchShapes(template_contour[0], contours[i], 1, 0.0) > 0.5:
            continue
        if hierarchy[0][i][3] != -1:
            continue
        tar_contours.append(contours[i])
        if hierarchy[0][i][2] != -1:
            roi_contours.append(contours[hierarchy[0][i][2]])

    res = cv.drawContours(frame.copy(), tar_contours, -1, (0, 0, 255), 2)
    res = cv.drawContours(res, roi_contours, -1, (0, 255, 255), 2)

    cv.imshow("frame", frame)
    cv.imshow("mask", mask)
    cv.imshow("res", res)

    if cv.waitKey(0) & 0xFF == ord('q'):
        break

cap.release()
cv.destroyAllWindows()

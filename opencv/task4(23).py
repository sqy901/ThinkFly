import cv2 as cv
import numpy as np

cap = cv.VideoCapture("../task_videos/task4_level2.mp4")

# 模板的内外轮廓
template = cv.imread("../image/flag0.jpg")
template = cv.resize(template, (36, 64))
template_gray = cv.cvtColor(template, cv.COLOR_BGR2GRAY)
_, template_binary = cv.threshold(template_gray, 120, 255, cv.THRESH_BINARY_INV)
t_contours, _ = cv.findContours(template_binary, cv.RETR_CCOMP, cv.CHAIN_APPROX_SIMPLE)
# 模板内轮廓
template_contours = [t_contours[2]]
# 模板外轮廓
num_contours = [t_contours[3]]

if not cap.isOpened():
    print("打开失败")
    exit()

lower_blue = np.array([100, 43, 46])
upper_blue = np.array([124, 255, 255])
# lower_white = np.array([0, 0, 211])
# upper_white = np.array([180, 30, 255])
cnt = 0

while True:
    ret, frame = cap.read()

    if not ret:
        print("视频结束")
        break

    # 拉对比度
    image = frame.copy()
    alpha = 0.6  # 对比度,第一个视频不适用
    beta = 10  # 亮度
    image = cv.convertScaleAbs(image, alpha=alpha, beta=beta)
    frame_blur = cv.GaussianBlur(image, (5, 5), 0)
    # kernel3 = cv.getStructuringElement(cv.MORPH_RECT, (3, 3))
    # frame_blur = cv.morphologyEx(frame_blur, cv.MORPH_CLOSE, kernel3, iterations=1)
    hsv = cv.cvtColor(frame_blur, cv.COLOR_BGR2HSV)

    mask = cv.inRange(hsv, lower_blue, upper_blue)

    contours, hierarchy = cv.findContours(mask, cv.RETR_TREE, cv.CHAIN_APPROX_SIMPLE)

    tar_contours = []
    roi_contours = []

    '''轮廓判断'''
    for i in range(len(contours)):
        # 面积小或太大，筛掉
        if cv.contourArea(contours[i]) < 300 or cv.contourArea(contours[i]) > 8000:
            continue
        # 匹配程度不高，筛掉
        if cv.matchShapes(template_contours[0], contours[i], 1, 0.0) > 0.4:
            continue
        # 有父轮廓，筛掉
        if hierarchy[0][i][3] != -1:
            continue
        # 无子轮廓，筛掉
        if hierarchy[0][i][2] == -1:
            continue
        # 有子轮廓时
        if hierarchy[0][i][2] > -1:
            # 顶级子轮廓，找面积最大的
            max_index = hierarchy[0][i][2]
            max_area = cv.contourArea(contours[max_index])
            now_index = hierarchy[0][max_index][0]
            while now_index > -1:
                now_area = cv.contourArea(contours[now_index])
                if now_area > max_area:
                    max_area = now_area
                    max_index = now_index
                now_index = hierarchy[0][now_index][0]
            # 去除噪点
            if max_area < 40:
                continue
            # 匹配程度不高，筛掉
            if cv.matchShapes(num_contours[0], contours[max_index], 1, 0.0) > 0.4:
                continue
            # 内外轮廓面积差距过大
            if cv.contourArea(contours[i]) / max_area > 10:
                continue
            epsilon = 0.005 * cv.arcLength(contours[i], True)
            approx_tar = cv.approxPolyDP(contours[i], epsilon, True)
            tar_contours.append(approx_tar)
            epsilon = 0.01 * cv.arcLength(contours[max_index], True)
            approx_roi = cv.approxPolyDP(contours[max_index], epsilon, True)
            roi_contours.append(approx_roi)

        for j in range(len(tar_contours)):
            contour = tar_contours[j]
            (cx, cy), (w, h), angle = cv.minAreaRect(roi_contours[j])
            (x, y), (Ma, ma), angle2 = cv.fitEllipse(contour)
            leftmost = tuple(contour[contour[:, :, 0].argmin()][0])
            rightmost = tuple(contour[contour[:, :, 0].argmax()][0])
            topmost = tuple(contour[contour[:, :, 1].argmin()][0])
            bottommost = tuple(contour[contour[:, :, 1].argmax()][0])
            distance = [(leftmost[0] - x) ** 2 + (leftmost[1] - y) ** 2,
                        (rightmost[0] - x) ** 2 + (rightmost[1] - y) ** 2,
                        (topmost[0] - x) ** 2 + (topmost[1] - y) ** 2,
                        (bottommost[0] - x) ** 2 + (bottommost[1] - y) ** 2]
            index_max = np.argmax(distance)

            if index_max == 0:
                angle += 180
            elif index_max == 2:
                if angle > 90:
                    angle += 180
            elif index_max == 3:
                if angle < 90:
                    angle += 180

            rows, cols = frame.shape[:2]
            M = cv.getRotationMatrix2D((cx, cy), angle, 1)
            # res = cv.drawContours(frame.copy(), [contour], -1, (0, 255, 0), 2)
            # res = cv.warpAffine(res, M, (rows, cols))
            # cv.imshow("res", res)
            # cv.waitKey(0)
            # cv.destroyAllWindows()
            # xuanzhuan = cv.drawContours(frame.copy(), [roi_contours[j]], -1, (0, 0, 255), 2)
            xuanzhuan = cv.warpAffine(frame.copy(), M, (rows, cols))
            roi = xuanzhuan[int(cy - h / 2):int(cy + h / 2), int(cx - w / 2):int(cx + w / 2)]
            if not roi.any():
                continue
            _ = cv.imwrite("generate_image/{}.png".format(cnt), roi)
            cnt += 1
            cv.imshow("roi", roi)
            cv.waitKey(0)
            cv.destroyAllWindows()
            # random_array = np.random.rand(1000,32,32,1).astype(np.float32)

    # res = cv.drawContours(frame.copy(), tar_contours, -1, (0, 0, 255), 2)
    # res = cv.drawContours(res, roi_contours, -1, (0, 255, 255), 2)

    # cv.imshow("frame", frame)
    # cv.imshow("image", image)
    # cv.imshow("res", res)

    # if cv.waitKey(0) & 0xFF == ord('q'):
    # break

cap.release()
cv.destroyAllWindows()

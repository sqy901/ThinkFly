import cv2 as cv
import numpy as np

# 原始图像
img = cv.imread("../image/flag1.jpg")
img = cv.resize(img, (480, 640))
cv.imshow("original", img)
cv.waitKey(0)
cv.destroyAllWindows()

img_blur = cv.GaussianBlur(img, (5, 5), 0)
cv.imshow("GaussianBlur", img_blur)
cv.waitKey(0)
cv.destroyAllWindows()

# kernel = np.ones((7, 7), np.uint8)
# img_blur = cv.morphologyEx(img_blur, cv.MORPH_CLOSE, kernel)
# cv.imshow("img_mor", img_blur)
# cv.waitKey(0)
# cv.destroyAllWindows()

# HSV蓝色范围
lower_blue = np.array([100, 43, 46])
upper_blue = np.array([124, 255, 255])

# 寻找图中蓝色区域
img_hsv = cv.cvtColor(img_blur, cv.COLOR_BGR2HSV)
img_mask = cv.inRange(img_hsv, lower_blue, upper_blue)
cv.imshow("img_mask", img_mask)
cv.waitKey(0)
cv.destroyAllWindows()

# 寻找轮廓
contours, _ = cv.findContours(img_mask, cv.RETR_EXTERNAL, cv.CHAIN_APPROX_SIMPLE)
img_draw = cv.drawContours(img.copy(), contours, -1, (0, 0, 255), 2)
cv.imshow("img_draw", img_draw)
cv.waitKey(0)
cv.destroyAllWindows()

# 选择目标轮廓
cnt = contours[0]
contours_max = cv.contourArea(contours[0])
for contour in contours:
    contours_tmp = cv.contourArea(contour)
    # print(contours_tmp)
    if contours_tmp > contours_max:
        contours_max = contours_tmp
        cnt = contour
# img_draw_tmp = cv.drawContours(img.copy(), cnt, -1, (0, 0, 255), 2)
# cv.imshow("img_draw_tmp", img_draw_tmp)
# cv.waitKey(0)
# cv.destroyAllWindows()

# 轮廓方向（最优拟合椭圆的方向）
(x, y), (Ma, ma), angle = cv.fitEllipse(cnt)
print(angle)
# ellipse = cv.fitEllipse(cnt)
# cv.ellipse(img, ellipse, (0, 255, 0), 2)
# cv.imshow("img", img)
# cv.waitKey(0)
# cv.destroyAllWindows()

# 根据angle进行图像旋转变换
rows, cols = img.shape[:2]
M = cv.getRotationMatrix2D((rows/2, cols/2), angle, 0.6)
res = cv.warpAffine(img.copy(), M, (rows, cols))
cv.imshow("res", res)
cv.waitKey(0)
cv.destroyAllWindows()

import cv2
import numpy as np

# 读取图像
img = cv2.imread("image.png")

gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
# 二值化灰度图
_, binary = cv2.threshold(gray, 127, 1, cv2.THRESH_BINARY)
# 调整图像大小为 n x n
n = 31
resized_binary = cv2.resize(binary, (n, n))
# 转换为矩阵
matrix = np.array(resized_binary)
# matrix 0,1 互换
matrix = 1 - matrix
print(matrix)

import numpy as np
import cv2
import os

from DetectService import resize, four_point_transform
from datetime import datetime
from paddleocr import PaddleOCR

# 读取原始图像
image = cv2.imread(r"E:\workspace\doc\img\1.PNG")
if image is None:
    raise ValueError("Image not loaded properly!")
orig = image.copy()
# 将图像转换为灰度
gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

# 使用阈值处理将图像转换为二值图像
ret, thresh = cv2.threshold(gray, 120, 255, cv2.THRESH_BINARY_INV | cv2.THRESH_OTSU)

# 使用自适应阈值或常规阈值
# 使用Otsu阈值并反转
#_, edged = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY | cv2.THRESH_OTSU)

# 兼容 OpenCV 版本差异
#cnts = cv2.findContours(edged.copy(), cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)
# 执行轮廓检测
contours, hierarchy = cv2.findContours(thresh, cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)

# 绘制轮廓
contour_image = cv2.drawContours(image.copy(), contours=contours, contourIdx=-1, color=(0, 255, 0), thickness=3)
screenCnt = sorted(contours, key=cv2.contourArea, reverse=True)[0]
peri = cv2.arcLength(screenCnt, True)
screenCnt = cv2.approxPolyDP(screenCnt, 0.02 * peri, True)

# 透视变换，裁剪区域
warped = four_point_transform(orig, screenCnt.reshape(4, 2))
ref_new = warped
# 显示带有轮廓的图像
cv2.imshow('Contours', contour_image)
cv2.waitKey(0)
cv2.destroyAllWindows()
#保存结果
output_path = os.path.join('output_images', f"cut_show_test1.png")
os.makedirs(os.path.dirname(output_path), exist_ok=True)  # 确保保存目录存在
cv2.imwrite(output_path, warped)
print(f"✅ 处理完成，结果已保存到：{output_path}")
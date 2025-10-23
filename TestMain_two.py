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

ratio = image.shape[0] / 500.0
orig = image.copy()
image = resize(orig, height=500)
# 四边各裁剪5px
image = image[5:-5, 5:-5]  # 上边裁5px，下边裁5px，左边裁5px，右边裁5px


# 转为灰度图
gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

# 进行高斯模糊去噪
gray = cv2.GaussianBlur(gray, (5, 5), 0)

# 使用自适应阈值或常规阈值
# 使用Otsu阈值并反转
_, edged = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY | cv2.THRESH_OTSU)

# 兼容 OpenCV 版本差异
cnts = cv2.findContours(edged.copy(), cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)
cnts = cnts[0] if len(cnts) == 2 else cnts[1]

# 确保找到轮廓
if not cnts:
    raise ValueError("No contours found in the image.")

# 在图像上绘制轮廓
output_image = image.copy()
cv2.drawContours(output_image, cnts, -1, (0, 255, 0), 2)  # 绿色，线宽2
#cv2.imshow("output_image", output_image)
#cv2.waitKey(0)
#cv2.destroyAllWindows()
# 找到最大轮廓并做近似多边形处理
screenCnt = sorted(cnts, key=cv2.contourArea, reverse=True)[0]
peri = cv2.arcLength(screenCnt, True)
screenCnt = cv2.approxPolyDP(screenCnt, 0.02 * peri, True)

# 透视变换，裁剪区域
warped = four_point_transform(orig, screenCnt.reshape(4, 2) * ratio)
ref_new = warped
# 显示效果
cv2.imshow("Original", image)
cv2.imshow("Edged", edged)
cv2.imshow("Warped Result", warped)
cv2.waitKey(0)
cv2.destroyAllWindows()

# 保存结果
output_path = os.path.join('output_images', f"cut_area_test1.png")
os.makedirs(os.path.dirname(output_path), exist_ok=True)  # 确保保存目录存在
cv2.imwrite(output_path, warped)
print(f"✅ 处理完成，结果已保存到：{output_path}")

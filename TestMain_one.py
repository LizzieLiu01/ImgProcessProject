import numpy as np
import cv2
import os
from DetectService import resize, four_point_transform
from datetime import datetime
from paddleocr import PaddleOCR

# 读取原始图像
image = cv2.imread(r"E:\WorkSpace\Doc\Image\test_quantized_pillow.png")
#image = cv2.imread(r"E:\WorkSpace\Doc\Image\test2.png")
ratio = image.shape[0] / 500.0
orig = image.copy()
image = resize(orig, height=500)

# 转为灰度图用于边缘检测
gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

# 使用阈值进行边缘检测
edged = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY | cv2.THRESH_OTSU)[1]
# 这里的 50 是黑色的阈值，黑色背景通常是低亮度，可以根据实际情况调整
#_, edged = cv2.threshold(gray, 50, 255, cv2.THRESH_BINARY_INV)
# 兼容 OpenCV 版本差异
cnts = cv2.findContours(edged.copy(), cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)
#cnts, _ = cv2.findContours(edged.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
cnts = cnts[0] if len(cnts) == 2 else cnts[1]

# 在图像上绘制轮廓
output_image = image.copy()
cv2.drawContours(output_image, cnts, -1, (0, 255, 0), 2)  # 绿色，线宽2

if not cnts:
    raise ValueError("No contours found in the image.")

# 找到最大轮廓并做近似多边形处理
screenCnt = sorted(cnts, key=cv2.contourArea, reverse=True)[0]
peri = cv2.arcLength(screenCnt, True)
screenCnt = cv2.approxPolyDP(screenCnt, 0.02 * peri, True)

# 透视变换，裁剪区域
warped = four_point_transform(orig, screenCnt.reshape(4, 2) * ratio)

# 透视变换后的图像直接使用原始背景，不做灰度化等处理
ref_new = warped  # 保持透视变换后的图像原样

# 显示效果
cv2.imshow("Original", image)
cv2.imshow("Edged", edged)
cv2.imshow("Warped Result", ref_new)
cv2.waitKey(0)
cv2.destroyAllWindows()

# 保存结果
output_path = os.path.join('output_images', f"cut_area_test.png")
os.makedirs(os.path.dirname(output_path), exist_ok=True)  # 确保保存目录存在
cv2.imwrite(output_path, ref_new)
print(f"✅ 处理完成，结果已保存到：{output_path}")

import numpy as np
import cv2
import os

from DetectService import  resize
from DetectService import  four_point_transform


image = cv2.imread(r"E:\WorkSpace\Doc\Image\test_quantized_pillow.png")
ratio = image.shape[0] / 500.0
orig = image.copy()
image = resize(orig, height=500) #缩放

gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
edged = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY | cv2.THRESH_OTSU)[1]
#cnts = cv2.findContours(edged.copy(), cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)[1]
# 兼容 OpenCV 版本差异
cnts = cv2.findContours(edged.copy(), cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)
cnts = cnts[0] if len(cnts) == 2 else cnts[1]
if not cnts:
    raise ValueError("No contours found in the image.")

screenCnt = sorted(cnts, key=cv2.contourArea, reverse=True)[0]
peri = cv2.arcLength(screenCnt, True)
screenCnt = cv2.approxPolyDP(screenCnt, 0.02 * peri, True)
warped = four_point_transform(orig, screenCnt.reshape(4, 2) * ratio) #拉平

warped = cv2.cvtColor(warped, cv2.COLOR_BGR2GRAY)
ref = cv2.threshold(warped, 0, 255, cv2.THRESH_BINARY | cv2.THRESH_OTSU)[1]
kernel = np.ones((2, 2), np.uint8)
ref_new = cv2.morphologyEx(ref, cv2.MORPH_CLOSE, kernel)

#rotated_image = cv2.rotate(ref_new, cv2.ROTATE_90_COUNTERCLOCKWISE)

# ==== 9. 显示效果 ====
cv2.imshow("Original", image)
cv2.imshow("Edged", edged)
cv2.imshow("Warped Result", ref_new)
cv2.waitKey(0)
cv2.destroyAllWindows()

# ==== 10. 保存结果 ====

cv2.imwrite( os.path.join('output_images', f"cut_area_test.png"), ref_new)
print(f"✅ 处理完成，结果已保存到：{os.path}")

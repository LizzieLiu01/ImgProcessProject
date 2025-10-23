import cv2
import os
import pytesseract
import numpy as np
from PIL import Image
from DetectService import  detect_ui_components
from DetectService import  detect_text_lightweight
from datetime import datetime
from paddleocr import PaddleOCR

# 手动指定 Tesseract 可执行文件的路径
pytesseract.pytesseract.tesseract_cmd = r'D:\install\tesseract\tesseract.exe'  # 根据实际路径修改

ocr = PaddleOCR(
    lang='ch',
    use_textline_orientation=False,  # 替代 use_angle_cls
    device='cpu'                     # 替代 use_gpu=False
)

# 图像加载
image_path = r"E:\WorkSpace\Doc\Image\test.png"  # 替换为实际图像路径
image = cv2.imread(image_path)

# 获取原尺寸
h, w = image.shape[:2]
print(f"原始尺寸: {w}x{h}")

# 使用Pillow将图像转换为16种颜色（量化）
pil_image = Image.open(image_path)
quantized = pil_image.convert("P", palette=Image.ADAPTIVE, colors=16)
quantized.save(r"E:\WorkSpace\Doc\Image\test_quantized_pillow.png")
#image1 = cv2.imread(r"E:\WorkSpace\Doc\Image\test_quantized_pillow.png")

#image1 = cv2.imread(r"C:\Users\Administrator\ImgProcessProject\output_images\cut_area_24.png")
image1 = cv2.imread(r"E:\WorkSpace\Doc\Image\test_byhand.png")

results1 = ocr.predict(image1)
# ✅ 打印结果
for line in results1:
    text = line['rec_texts']
    score = line['rec_scores']
    box = line['rec_polys']
    # 打印文本及其置信度
    for text, score, poly in zip(text, score, box):
        print(f"image1文字: {text} | 置信度: {score:.3f} ")
#   灰度识别切割图片
# # 转为灰度图
# gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
#
# # 进行高斯模糊去噪
# gray = cv2.GaussianBlur(gray, (5, 5), 0)
#
# # 使用自适应阈值处理，适应不同光照条件
# thresh = cv2.adaptiveThreshold(gray, 255, cv2.ADAPTIVE_THRESH_MEAN_C, cv2.THRESH_BINARY, 15, 2)

#
# # --- 使用不同的轮廓检索模式 ---
# # 这里使用 `cv2.RETR_LIST` 来查找图像中的所有轮廓（包括内部区域的轮廓）
# contours, _ = cv2.findContours(thresh, cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)
#
# # 输出文件夹（如果不存在则创建）
# output_folder = 'output_images'
# if not os.path.exists(output_folder):
#     os.makedirs(output_folder)
# image_copy = image.copy()  # 创建图像副本，防止修改原图
# # --- 筛选并切割UI区域 ---
# for i, contour in enumerate(contours):
#     # 获取每个轮廓的边界框
#     x, y, w, h = cv2.boundingRect(contour)
#
#     # --- 只切割出轮廓对应的区域 ---
#     roi = image[y:y+h, x:x+w]  # 切割出当前轮廓的区域
#
#     # # 创建副本图像，用于绘制当前轮廓
#     cv2.drawContours(image_copy, [contour], -1, (0, 255, 0), 2) # 绘制轮廓，绿色，线宽2
#
#          # 保存切割后的区域
#     output_image_path = os.path.join(output_folder, f"ui_area_{i + 1}.png")
#     cv2.imwrite(output_image_path, roi)
#     print(f"保存切割区域：{output_image_path}")
# # 显示原图上的切割区域（带矩形框）
# cv2.imshow("Detected UI Regions", image)
# cv2.waitKey(0)
# cv2.destroyAllWindows()


#背景色切割图片

# 转换到 HSV 色彩空间

# 转换到 HSV 色彩空间
hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)

# 定义背景颜色的范围（例如，假设背景是白色）
# 这里的范围可以根据具体情况调整
lower_bound = np.array([0, 0, 200])  # 背景颜色的下限（浅白色）
upper_bound = np.array([180, 30, 255])  # 背景颜色的上限（浅白色）

# 创建背景掩模
background_mask = cv2.inRange(hsv, lower_bound, upper_bound)

# 反转掩模得到前景掩模
foreground_mask = cv2.bitwise_not(background_mask)

# 形态学操作改善掩模质量
kernel = np.ones((5, 5), np.uint8)
#foreground_mask = cv2.morphologyEx(foreground_mask, cv2.MORPH_CLOSE, kernel)
#foreground_mask = cv2.morphologyEx(foreground_mask, cv2.MORPH_OPEN, kernel)

# foreground_mask = detect_by_edges(image)
# 查找所有轮廓
contours, _ = cv2.findContours(foreground_mask, cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)

#contours=detect_ui_components(image1)
# 创建输出文件夹
output_folder = 'output_images'
if not os.path.exists(output_folder):
    os.makedirs(output_folder)

# 按照原图背景切割 - 保持原有背景
for i, contour in enumerate(contours):
    # 获取每个轮廓的边界框
    x, y, w, h = cv2.boundingRect(contour)

    # 过滤掉太小的区域
    if w > 30 and h > 30:
        # 直接从原图切割对应区域，保持原有背景
        roi_with_original_bg = image[y:y + h, x:x + w].copy()
        # 使用 Tesseract 进行文字识别
       #   text = pytesseract.image_to_string(roi_with_original_bg)

        results = ocr.predict(roi_with_original_bg)
        # ✅ 打印结果
        for line in results:
            text = line['rec_texts']
            score = line['rec_scores']
            box = line['rec_polys']
            # 打印文本及其置信度
            for text, score, poly in zip(text, score, box):
                print(f"文字: {text} | 置信度: {score:.3f} ")

        # 保存切割后的区域（保持原背景）
        output_image_path = os.path.join(output_folder, f"cut_area_{i}.png")
        i = i + 1
        cv2.imwrite(output_image_path, roi_with_original_bg)
        print(f"保存切割区域：{output_image_path} (尺寸: {w}x{h})")

print("切割完成！共保存了 {} 个区域".format(len(contours)))

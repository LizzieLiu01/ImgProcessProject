from paddleocr import PaddleOCR
import cv2
import os
import paddle
# ===============================
# 配置部分
# ===============================
os.environ["CUDA_VISIBLE_DEVICES"] = "-1"  # 使用GPU（或改为 -1 强制使用CPU）
img_path = r"E:\workspace\doc\img\word.png"

print(paddle.__version__)


print(" 初始化 PaddleOCR 模型...")
# 只指定语言,use_angle_cls是可以将图片反转等，因为已经预处理过图片了，所以这里关掉，减少加载时间
# use_gpu 关掉使用gpu,加载模型时会检测环境适用gpu和cpu

ocr = PaddleOCR(
    lang='ch',
    use_textline_orientation=True,  # 替代 use_angle_cls
    device='cpu'  # 强制使用 CPU
)

#result = ocr.ocr(rotated_img, cls=True)
crop_results = ocr.predict(input=img_path)
#crop_results = ocr.ocr(img_path, cls=False)
print(crop_results)
for res in crop_results:
    rec_texts = res['rec_texts']  # 获取文本
    rec_scores = res['rec_scores']  # 获取置信度
    for text, score in zip(rec_texts, rec_scores):
     print(f"识别结果：{text} ({score:.2f})")

print("\n 完成！")

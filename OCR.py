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

# ocr = PaddleOCR(
#     lang='ch',
#     device='cpu',  # 强制使用 CPU
#    # enable_hpi=True #可执行高性能推理
#     use_doc_orientation_classify=False, # 通过 use_doc_orientation_classify 参数指定不使用文档方向分类模型
#     use_doc_unwarping=False, # 通过 use_doc_unwarping 参数指定不使用文本图像矫正模型
#     use_textline_orientation=False # 通过 use_textline_orientation 参数指定不使用文本行方向分类模型
#
# )
# 创建OCR实例，加载自定义模型
from paddleocr import PaddleOCR


# 创建OCR实例，使用新版API
ocr = PaddleOCR(
    # use_angle_cls=True,  # 启用文本方向分类器（新版参数名）
    lang='ch',           # 选择中文
    # 如果使用自定义模型，使用新版参数名
    text_recognition_model_dir='E:/workspace/code/git/ocr/PP-OCRv5_server_rec_infer',

    # use_gpu=False        # 根据你的环境设置
)
result = ocr.predict(img_path)  # 不使用方向分类器
print(f"result text: {result}")
# 检查结果结构
if result and len(result) > 0:
    first_result = result[0]
    texts = first_result.get('rec_texts', [])
    scores = first_result.get('rec_scores', [])

    # 遍历每一个识别到的文本和置信度
    for i in range(len(texts)):
        text = texts[i]
        score = scores[i]
        print(f"Detected11111111 text: {text} with22222222 confidence: {score}")
else:
    print("No result found.")
# 输出结果
for line in result:
    print(f"Detected text: {line[1][0]} with confidence: {line[1][1]}")

ocr.export_paddlex_config_to_yaml("ocr_config.yaml")
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

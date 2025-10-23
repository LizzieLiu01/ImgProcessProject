from paddleocr import PaddleOCR
import os

# 打印当前工作目录
print("当前工作目录：", os.getcwd())

# 初始化 OCR（中文，关闭文档校正功能）
ocr = PaddleOCR(
    use_doc_orientation_classify=False,
    use_doc_unwarping=False,
    use_textline_orientation=False,
    lang='ch'
)

# 本地图片路径
image_path = r"E:\WorkSpace\Doc\Image\test.png"

# 输出路径（自动创建）
save_dir = r"D:\output\ocr"
os.makedirs(save_dir, exist_ok=True)

# 运行 OCR
result = ocr.predict(input=image_path)

# 输出识别结果
for res in result:
    res.print()
    res.save_to_img(save_dir, draw_text=True)
    res.save_to_json(save_dir)

print(f"✅ 识别完成！结果保存在：{save_dir}")

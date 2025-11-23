import pytesseract
import cv2
import numpy as np
import os
from PIL import Image
from image_preprocessor import ImagePreprocessor

# 设置Tesseract路径
# 优先从环境变量获取 Tesseract 路径
tesseract_path = os.environ.get('TESSERACT_CMD')

if tesseract_path:
    pytesseract.pytesseract.tesseract_cmd = tesseract_path
else:
    # 如果环境变量未设置，抛出错误
    raise FileNotFoundError("Tesseract path not found. Please set the TESSERACT_CMD environment variable.")

# 读取图像
image = cv2.imread('test_image/4.png')

# 确保图像已成功加载，避免将 None 传入 cvtColor（IDE/静态检查和运行时错误）
if image is None:
    raise FileNotFoundError("Image 'test_image/1.png' not found or failed to load. Please check the path.")

# 转换为灰度图像
gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

# 二值化处理（Otsu的自适应二值化）
_, binary_image = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)

# 使用tesseract进行OCR识别
text = pytesseract.image_to_string(binary_image, lang='chi_sim')

# 打印识别到的文本
print("识别的文本是：")
print(text)

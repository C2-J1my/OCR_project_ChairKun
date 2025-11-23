import pytesseract
import cv2
import numpy as np
from PIL import Image
from image_preprocessor import ImagePreprocessor

# 设置Tesseract路径（如果路径不在默认位置）
pytesseract.pytesseract.tesseract_cmd = "/usr/bin/tesseract"

# 读取图像
image = cv2.imread('/home/jimi/图片/粘贴的图像 (3).png')

# 转换为灰度图像
gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

# 去噪（使用高斯模糊）
blurred = cv2.GaussianBlur(gray, (5, 5), 0)

# 二值化处理（Otsu的自适应二值化）
_, binary_image = cv2.threshold(blurred, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)

# 使用tesseract进行OCR识别
text = pytesseract.image_to_string(binary_image)

# 打印识别到的文本
print("识别的文本是：")
print(text)

# image_preprocessor.py
import cv2
import numpy as np
import os
from PIL import Image

def enhance_image(image_path, output_path):
    """
    专业级图像增强预处理
    """
    # 读取图像
    img = cv2.imread(image_path)
    if img is None:
        raise ValueError(f"无法读取图像: {image_path}")
    
    # 转换为灰度图
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    
    # 增强对比度
    gray = cv2.equalizeHist(gray)
    
    # 自适应阈值处理（关键步骤）
    thresh = cv2.adaptiveThreshold(
        gray, 255, 
        cv2.ADAPTIVE_THRESH_GAUSSIAN_C, 
        cv2.THRESH_BINARY, 11, 2
    )
    
    # 去噪处理
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (2, 2))
    denoised = cv2.morphologyEx(thresh, cv2.MORPH_OPEN, kernel)
    
    # 保存增强后的图像
    cv2.imwrite(output_path, denoised)
    return output_path

class ImagePreprocessor:
    def process(self, image_path: str, mode: str = 'auto') -> Image.Image:
        img = cv2.imread(image_path)
        if img is None:
            raise ValueError(f"无法读取图像: {image_path}")
        if mode == 'none':
            return Image.fromarray(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        if mode == 'light':
            eq = cv2.equalizeHist(gray)
            return Image.fromarray(eq)
        gray = cv2.equalizeHist(gray)
        thresh = cv2.adaptiveThreshold(gray, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, 11, 2)
        kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (2, 2))
        denoised = cv2.morphologyEx(thresh, cv2.MORPH_OPEN, kernel)
        return Image.fromarray(denoised)

if __name__ == "__main__":
    # 使用示例
    input_image = "test.jpg"  # 替换为您的图片路径
    output_image = "enhanced.jpg"
    
    try:
        enhanced_path = enhance_image(input_image, output_image)
        print(f"图像已增强并保存至: {enhanced_path}")
        print("请使用此增强后的图像进行OCR识别")
    except Exception as e:
        print(f"处理失败: {str(e)}")

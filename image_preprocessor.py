# image_preprocessor.py
import cv2
import numpy as np
import os
from PIL import Image
from typing import Optional, Tuple, Dict, Any


def enhance_image(image_path, output_path, preset: str = 'auto') -> str:
    """
    便捷函数：使用默认参数对单张图片进行增强并保存到 output_path
    """
    proc = ImagePreprocessor.from_preset(preset)
    img = proc.process(image_path)
    # 保存 PIL.Image 到 output_path
    img.save(output_path)
    return output_path


class ImagePreprocessor:
    """参数化图像预处理器，支持预设 'light' 和 'heavy'，也可通过构造参数自定义。

    process(...) 返回 PIL.Image 对象，方便直接传给 pytesseract。
    """
    def __init__(
        self,
        clip_limit: float = 2.0,
        tile_grid_size: Tuple[int, int] = (8, 8),
        denoise_method: str = 'bilateral',
        denoise_params: Optional[Tuple[Any, ...]] = None,
        thresh_blocksize: int = 11,
        thresh_C: int = 2,
        sharpen: bool = False,
        morph_kernel_size: Optional[Tuple[int, int]] = None,
    ):
        self.clip_limit = clip_limit
        self.tile_grid_size = tile_grid_size
        self.denoise_method = denoise_method
        self.denoise_params = denoise_params or (7, 50, 50)
        self.thresh_blocksize = thresh_blocksize
        self.thresh_C = thresh_C
        self.sharpen = sharpen
        self.morph_kernel_size = morph_kernel_size

    @classmethod
    def from_preset(cls, preset: str) -> 'ImagePreprocessor':
        preset = preset.lower()
        if preset == 'light':
            return cls(
                clip_limit=1.5,
                tile_grid_size=(8, 8),
                denoise_method='gassian',
                denoise_params=(5, 25, 25),
                thresh_blocksize=11,
                thresh_C=2,
                sharpen=False,
                morph_kernel_size=None,
            )
        elif preset == 'heavy':
            return cls(
                clip_limit=3.0,
                tile_grid_size=(8, 8),
                denoise_method='bilateral',
                denoise_params=(9, 75, 75),
                thresh_blocksize=15,
                thresh_C=4,
                sharpen=True,
                morph_kernel_size=(2, 2),
            )
        else:
            # 默认 'auto' 采用中等参数
            return cls()

    def _sharpen_image(self, img_gray: np.ndarray) -> np.ndarray:
        kernel = np.array([[0, -1, 0], [-1, 5, -1], [0, -1, 0]])
        return cv2.filter2D(img_gray, -1, kernel)

    def process(self, image_path: str, mode: Optional[str] = None) -> Image.Image:
        """读取 `image_path` 并按当前实例参数预处理，返回 PIL.Image。"""
        img = cv2.imread(image_path)
        if img is None:
            raise ValueError(f"无法读取图像: {image_path}")

        if mode == 'none':
            return Image.fromarray(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))

        # 转换为灰度并增强对比度
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        clahe = cv2.createCLAHE(clipLimit=self.clip_limit, tileGridSize=self.tile_grid_size)
        enhanced = clahe.apply(gray)

        # 去噪
        denoised = enhanced
        if self.denoise_method == 'bilateral':
            d, sigmaColor, sigmaSpace = self.denoise_params
            denoised = cv2.bilateralFilter(enhanced, d, sigmaColor, sigmaSpace)
        elif self.denoise_method == 'gaussian':
            ksize = self.denoise_params[0] if self.denoise_params else 5
            denoised = cv2.GaussianBlur(enhanced, (ksize, ksize), 0)

        # 可选锐化
        if self.sharpen:
            denoised = self._sharpen_image(denoised)

        # 自适应阈值
        block = max(3, self.thresh_blocksize | 1)  # ensure odd
        thresh = cv2.adaptiveThreshold(
            denoised, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, block, self.thresh_C
        )

        # 可选形态学去噪
        if self.morph_kernel_size:
            kernel = cv2.getStructuringElement(cv2.MORPH_RECT, self.morph_kernel_size)
            thresh = cv2.morphologyEx(thresh, cv2.MORPH_OPEN, kernel)

        return Image.fromarray(thresh)


if __name__ == "__main__":
    # 使用示例
    input_path = "test_image/1.png"
    output_path = "process/result_enhanced1.png"
    os.makedirs(os.path.dirname(output_path) or '.', exist_ok=True)
    try:
        enhanced_path = enhance_image(input_path, output_path, 'light')
        print(f"图像已增强并保存至: {enhanced_path}")
        print("请使用此增强后的图像进行OCR识别")
    except Exception as e:
        print(f"处理失败: {str(e)}")

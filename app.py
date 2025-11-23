import os
import time
from flask import Flask, request, render_template, jsonify
import pytesseract
from PIL import Image
import cv2
import numpy as np
from werkzeug.utils import secure_filename
import shutil
from image_preprocessor import ImagePreprocessor

app = Flask(__name__)
os.makedirs('uploads', exist_ok=True)

# 设置Tesseract路径（根据你安装的路径进行修改）
def _set_tesseract_cmd():
    # 优先从环境变量获取路径
    tesseract_path = os.environ.get('TESSERACT_CMD')
    
    if tesseract_path and os.path.exists(tesseract_path):
        pytesseract.pytesseract.tesseract_cmd = tesseract_path
        # 自动设置 tessdata 路径
        os.environ.setdefault('TESSDATA_PREFIX', os.path.join(os.path.dirname(tesseract_path), 'tessdata'))
        return True
    
    # 如果环境变量未设置，可以尝试一些常见路径作为备选
    candidates = [
        r'C:\Program Files\Tesseract-OCR\tesseract.exe',
        r'C:\Program Files (x86)\Tesseract-OCR\tesseract.exe',
        r'D:\Program Files\Tesseract-OCR\tesseract.exe',
        r'D:\Tesseract-OCR\tesseract.exe',
    ]
    
    for p in candidates:
        if os.path.exists(p):
            pytesseract.pytesseract.tesseract_cmd = p
            os.environ.setdefault('TESSDATA_PREFIX', os.path.join(os.path.dirname(p), 'tessdata'))
            return True
            
    return False

if not _set_tesseract_cmd():
    raise FileNotFoundError(
        "无法找到 Tesseract-OCR。\n"
        "请通过以下方式之一解决：\n"
        "1. 配置环境变量 TESSERACT_CMD 指向你的 tesseract.exe 文件。\n"
        "   例如: TESSERACT_CMD=D:\Tesseract-OCR\tesseract.exe\n"
        "2. 或者，在 _set_tesseract_cmd() 函数的 candidates 列表中添加你的安装路径。"
    )

# 首页路由，返回HTML表单
@app.route('/')
def index():
    return render_template('index.html')

# OCR路由，处理上传的图像
@app.route('/ocr', methods=['POST'])
def ocr():
    start = time.perf_counter()
    file = request.files.get('file') or request.files.get('image')
    if not file:
        return jsonify({"error": "No file part"})
    raw_filename = file.filename or ''
    if raw_filename == '':
        return jsonify({"error": "No selected file"})

    # 保存上传的文件（确保传入 secure_filename 的是 str）
    filename = secure_filename(raw_filename)
    filepath = os.path.join('uploads', filename)
    file.save(filepath)
    
    try:
        pre_start = time.perf_counter()
        # 仅使用 preset 参数（取代原有的 mode），preset 可为 'light'|'heavy'|'none' 或留空
        preset = request.args.get('preset')
        if preset in ('light', 'heavy'):
            preprocessor = ImagePreprocessor.from_preset(preset)
        else:
            preprocessor = ImagePreprocessor()

        # 如果用户选择 preset == 'none'，则跳过预处理并直接返回 RGB 图像
        process_mode = 'none' if preset == 'none' else None
        processed_image = preprocessor.process(filepath, mode=process_mode)


        # 保存处理后准备识别的图片到 process/result.png
        os.makedirs('process', exist_ok=True)
        result_path = os.path.join('process', 'result.png')
        try:
            processed_image.save(result_path)
        except Exception:
            # processed_image 可能不是 PIL.Image（保险起见），尝试转换
            try:
                Image.fromarray(processed_image).save(result_path)
            except Exception:
                pass


        pre_end = time.perf_counter()

        ocr_start = time.perf_counter()
        lang = request.args.get('lang', 'chi_sim')
        try:
            text = pytesseract.image_to_string(processed_image, lang=lang, config='--psm 6')
        except Exception:
            text = pytesseract.image_to_string(processed_image, lang='eng', config='--psm 6')
        ocr_end = time.perf_counter()

        total_end = time.perf_counter()
        timings = {
            'preprocess': round((pre_end - pre_start) * 1000, 1),
            'ocr': round((ocr_end - ocr_start) * 1000, 1),
            'total': round((total_end - start) * 1000, 1),
        }
        return render_template('index.html', text=text, timings=timings)
    except Exception as e:
        return render_template('index.html', text=f"识别失败: {str(e)}")

@app.route('/api/ocr', methods=['POST'])
def api_ocr():
    t0 = time.perf_counter()
    file = request.files.get('file') or request.files.get('image')
    if not file:
        return jsonify({"error": "No file uploaded"}), 400

    raw_filename = file.filename or ''
    if raw_filename == '':
        return jsonify({"error": "No file uploaded"}), 400

    filename = secure_filename(raw_filename)
    path = os.path.join('uploads', filename)
    file.save(path)
    # 仅使用 preset 参数（取代原有的 mode）
    preset = request.args.get('preset')
    lang = request.args.get('lang', 'chi_sim')
    if preset in ('light', 'heavy'):
        preproc = ImagePreprocessor.from_preset(preset)
    else:
        preproc = ImagePreprocessor()

    pre0 = time.perf_counter()
    process_mode = 'none' if preset == 'none' else None
    img = preproc.process(path, mode=process_mode)


    # 保存处理后准备识别的图片到 process/result.png
    os.makedirs('process', exist_ok=True)
    result_path = os.path.join('process', 'result.png')
    try:
        img.save(result_path)
    except Exception:
        try:
            Image.fromarray(img).save(result_path)
        except Exception:
            pass


    pre1 = time.perf_counter()
    ocr0 = time.perf_counter()
    try:
        text = pytesseract.image_to_string(img, lang=lang, config='--psm 6')
    except Exception:
        text = pytesseract.image_to_string(img, lang='eng', config='--psm 6')
    ocr1 = time.perf_counter()
    t1 = time.perf_counter()
    return jsonify({
        "text": text,
        "timings_ms": {
            "preprocess": round((pre1 - pre0) * 1000, 1),
            "ocr": round((ocr1 - ocr0) * 1000, 1),
            "total": round((t1 - t0) * 1000, 1)
        },
        "lang": lang,
        "preset": preset
    })

@app.route('/health', methods=['GET'])
def health():
    try:
        ver = str(pytesseract.get_tesseract_version())
    except Exception:
        ver = None
    langs = []
    try:
        langs = pytesseract.get_languages(config='')
    except Exception:
        langs = []
    return jsonify({
        "tesseract_version": ver,
        "languages": langs,
        "status": "ok" if ver else "error"
    })

if __name__ == '__main__':
    app.run(debug=True)

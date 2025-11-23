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
    candidates = [
        os.environ.get('TESSERACT_CMD'),
        r'C:\\Program Files\\Tesseract-OCR\\tesseract.exe',
        r'C:\\Program Files (x86)\\Tesseract-OCR\\tesseract.exe',
        r'D:\\Program Files\\Tesseract-OCR\\tesseract.exe',
        r'D:\\Tesseract-OCR\\tesseract.exe',
    ]
    auto = shutil.which('tesseract')
    if auto:
        candidates.insert(0, auto)
    for p in candidates:
        if p and os.path.exists(p):
            pytesseract.pytesseract.tesseract_cmd = p
            os.environ.setdefault('TESSDATA_PREFIX', os.path.join(os.path.dirname(p), 'tessdata'))
            return True
    return False

if not _set_tesseract_cmd():
    pytesseract.pytesseract.tesseract_cmd = r'C:\\Program Files\\Tesseract-OCR\\tesseract.exe'
    os.environ.setdefault('TESSDATA_PREFIX', r'C:\\Program Files\\Tesseract-OCR\\tessdata')

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
    if file.filename == '':
        return jsonify({"error": "No selected file"})
    
    # 保存上传的文件
    filename = secure_filename(file.filename)
    filepath = os.path.join('uploads', filename)
    file.save(filepath)
    
    try:
        pre_start = time.perf_counter()
        mode = request.args.get('mode', 'auto')
        preprocessor = ImagePreprocessor()
        processed_image = preprocessor.process(filepath, mode=mode)
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
    filename = secure_filename(file.filename)
    path = os.path.join('uploads', filename)
    file.save(path)
    mode = request.args.get('mode', 'auto')
    lang = request.args.get('lang', 'chi_sim')
    pre0 = time.perf_counter()
    img = ImagePreprocessor().process(path, mode=mode)
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
        "mode": mode
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

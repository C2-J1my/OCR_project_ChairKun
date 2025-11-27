import os
import io
import time
import base64
from flask import Flask, request, jsonify
from PIL import Image
import pytesseract
from pytesseract import Output
import numpy as np
import cv2

app = Flask(__name__)


def _set_tesseract_cmd():
    """确保Linux服务器上的tesseract路径正确"""
    candidates = [
        os.environ.get('TESSERACT_CMD'),
        '/usr/bin/tesseract',
        '/usr/local/bin/tesseract',
    ]
    for path in candidates:
        if path and os.path.exists(path):
            pytesseract.pytesseract.tesseract_cmd = path
            return True
    return False


if not _set_tesseract_cmd():
    raise FileNotFoundError("未找到tesseract可执行文件，请安装或设置TESSERACT_CMD环境变量。")


def _load_image(file_storage):
    data = file_storage.read()
    if not data:
        raise ValueError("上传图像为空")
    image = Image.open(io.BytesIO(data))
    # 预处理结果可能是灰度/二值，这里统一转为RGB以兼容pytesseract/image_to_data
    if image.mode != 'RGB':
        image = image.convert('RGB')
    return image


def _image_to_overlay(pil_image, boxes):
    cv_img = cv2.cvtColor(np.array(pil_image), cv2.COLOR_RGB2BGR)
    for box in boxes:
        try:
            conf = int(box.get('conf', -1))
        except Exception:
            conf = -1
        txt = box.get('text', '').strip()
        if conf > 30 and txt:
            x = box['left']
            y = box['top']
            w = box['width']
            h = box['height']
            cv2.rectangle(cv_img, (x, y), (x + w, y + h), (0, 255, 0), 2)
    overlay = cv2.cvtColor(cv_img, cv2.COLOR_BGR2RGB)
    buffer = io.BytesIO()
    Image.fromarray(overlay).save(buffer, format='PNG')
    buffer.seek(0)
    return base64.b64encode(buffer.getvalue()).decode('ascii')


@app.route('/api/remote_ocr', methods=['POST'])
def remote_ocr():
    start = time.perf_counter()
    uploaded = request.files.get('image') or request.files.get('file')
    if not uploaded:
        return jsonify({"error": "未收到图像文件"}), 400

    lang = request.form.get('lang', 'chi_sim')
    image = _load_image(uploaded)
    pre_end = time.perf_counter()

    ocr_start = time.perf_counter()
    try:
        text = pytesseract.image_to_string(image, lang=lang, config='--psm 11')
    except Exception:
        text = pytesseract.image_to_string(image, lang='eng', config='--psm 11')
    ocr_end = time.perf_counter()

    try:
        data = pytesseract.image_to_data(image, lang=lang, config='--psm 6', output_type=Output.DICT)
        boxes = []
        n = len(data.get('level', []))
        for i in range(n):
            boxes.append({
                "text": data.get('text', [''])[i],
                "left": int(data['left'][i]),
                "top": int(data['top'][i]),
                "width": int(data['width'][i]),
                "height": int(data['height'][i]),
                "conf": data.get('conf', [])[i],
                "level": data.get('level', [None])[i]
            })
    except Exception:
        boxes = []

    overlay_b64 = _image_to_overlay(image, boxes) if boxes else None
    end = time.perf_counter()
    width, height = image.size
    return jsonify({
        "text": text,
        "boxes": boxes,
        "image_size": {"width": width, "height": height},
        "overlay_image": overlay_b64,
        "timings_ms": {
            "preprocess": round((pre_end - start) * 1000, 1),
            "ocr": round((ocr_end - ocr_start) * 1000, 1),
            "total": round((end - start) * 1000, 1)
        },
        "lang": lang
    })


@app.route('/health', methods=['GET'])
def health():
    try:
        version = str(pytesseract.get_tesseract_version())
    except Exception:
        version = None
    langs = []
    try:
        langs = pytesseract.get_languages(config='')
    except Exception:
        langs = []
    status = "ok" if version else "error"
    return jsonify({
        "status": status,
        "tesseract_version": version,
        "languages": langs
    })


if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000, debug=False)


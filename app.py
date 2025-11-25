import os
import time
from flask import Flask, request, render_template, jsonify
import pytesseract
from PIL import Image
from pytesseract import Output
import cv2
import numpy as np
from werkzeug.utils import secure_filename
import shutil
from image_preprocessor import ImagePreprocessor
import uuid

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
        "   例如: TESSERACT_CMD=D:\\Tesseract-OCR\\tesseract.exe\n"
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
        # 支持前端已做部分预处理的标记：当前端先做了灰度+对比度增强，会提交 client_pre=1
        client_pre = request.args.get('client_pre') or request.args.get('client')
        client_pre = True if str(client_pre).lower() in ('1','true','yes') else False
        process_mode = 'none' if preset == 'none' else ('client' if client_pre else None)
        processed_image = preprocessor.process(filepath, mode=process_mode)


        # # 保存处理后准备识别的图片到 process/result.png
        # os.makedirs('process', exist_ok=True)
        # result_path = os.path.join('process', 'result.png')
        # try:
        #     processed_image.save(result_path)
        # except Exception:
        #     # processed_image 可能不是 PIL.Image（保险起见），尝试转换
        #     try:
        #         Image.fromarray(processed_image).save(result_path)
        #     except Exception:
                # pass


        pre_end = time.perf_counter()

        ocr_start = time.perf_counter()
        lang = request.args.get('lang', 'chi_sim')
        try:
            text = pytesseract.image_to_string(processed_image, lang=lang, config='--psm 11')
        except Exception:
            text = pytesseract.image_to_string(processed_image, lang='eng', config='--psm 6')
        ocr_end = time.perf_counter()

        # 使用 pytesseract 返回的位置信息绘制绿色矩形框并收集 boxes
        try:
            data = pytesseract.image_to_data(processed_image, lang=lang, config='--psm 11', output_type=Output.DICT)
            # 转为 OpenCV BGR 图像以便绘制
            cv_img = cv2.cvtColor(np.array(processed_image), cv2.COLOR_RGB2BGR)
            boxes = []
            n_boxes = len(data.get('level', []))
            for i in range(n_boxes):
                (x, y, w, h) = (data['left'][i], data['top'][i], data['width'][i], data['height'][i])
                conf = data.get('conf', [])[i] if 'conf' in data else '-1'
                txt = data.get('text', [''])[i]
                try:
                    c = int(conf)
                except Exception:
                    c = -1
                level = data.get('level', [None])[i]
                boxes.append({
                    'text': txt,
                    'left': int(x), 'top': int(y), 'width': int(w), 'height': int(h),
                    'conf': conf,
                    'level': level
                })
                # 仅对置信度有效且文本非空的位置画框
                if c > 30 and txt.strip():
                    cv2.rectangle(cv_img, (x, y), (x + w, y + h), (0, 255, 0), 2)
            static_dir = os.path.join(app.root_path, 'static')
            results_dir = os.path.join(static_dir, 'results')
            os.makedirs(results_dir, exist_ok=True)
            unique = f"result_{int(time.time()*1000)}_{uuid.uuid4().hex[:8]}.png"
            result_static_path = os.path.join(results_dir, unique)
            cv2.imwrite(result_static_path, cv_img)
            result_url = f'/static/results/{unique}'
        except Exception:
            boxes = []
            result_url = None

        total_end = time.perf_counter()
        timings = {
            'preprocess': round((pre_end - pre_start) * 1000, 1),
            'ocr': round((ocr_end - ocr_start) * 1000, 1),
            'total': round((total_end - start) * 1000, 1),
        }
        return render_template('index.html', text=text, timings=timings, result_img=result_url, boxes=boxes)
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
    client_pre = request.args.get('client_pre') or request.args.get('client')
    client_pre = True if str(client_pre).lower() in ('1','true','yes') else False
    process_mode = 'none' if preset == 'none' else ('client' if client_pre else None)
    img = preproc.process(path, mode=process_mode)


    # # 保存处理后准备识别的图片到 process/result.png
    # os.makedirs('process', exist_ok=True)
    # result_path = os.path.join('process', 'result.png')
    # try:
    #     img.save(result_path)
    # except Exception:
    #     try:
    #         Image.fromarray(img).save(result_path)
    #     except Exception:
    #         pass


    pre1 = time.perf_counter()
    ocr0 = time.perf_counter()
    try:
        text = pytesseract.image_to_string(img, lang=lang, config='--psm 6')
    except Exception:
        text = pytesseract.image_to_string(img, lang='eng', config='--psm 6')
    ocr1 = time.perf_counter()
    # 绘制识别框并保存到 static/results/<unique>.png，同时构建 boxes
    try:
        data = pytesseract.image_to_data(img, lang=lang, config='--psm 6', output_type=Output.DICT)
        cv_img = cv2.cvtColor(np.array(img), cv2.COLOR_RGB2BGR)
        boxes = []
        n_boxes = len(data.get('level', []))
        for i in range(n_boxes):
            (x, y, w, h) = (data['left'][i], data['top'][i], data['width'][i], data['height'][i])
            conf = data.get('conf', [])[i] if 'conf' in data else '-1'
            txt = data.get('text', [''])[i]
            try:
                c = int(conf)
            except Exception:
                c = -1
            level = data.get('level', [None])[i]
            boxes.append({
                'text': txt,
                'left': int(x), 'top': int(y), 'width': int(w), 'height': int(h),
                'conf': conf,
                'level': level
            })
            if c > 30 and txt.strip():
                cv2.rectangle(cv_img, (x, y), (x + w, y + h), (0, 255, 0), 2)
        static_dir = os.path.join(app.root_path, 'static')
        results_dir = os.path.join(static_dir, 'results')
        os.makedirs(results_dir, exist_ok=True)
        unique = f"result_{int(time.time()*1000)}_{uuid.uuid4().hex[:8]}.png"
        result_static_path = os.path.join(results_dir, unique)
        cv2.imwrite(result_static_path, cv_img)
        result_url = f'/static/results/{unique}'
    except Exception:
        boxes = []
        result_url = None
    if hasattr(img, 'size'):
        src_w, src_h = img.size
    else:
        shape = getattr(img, 'shape', None)
        if shape is not None and len(shape) >= 2:
            src_h, src_w = shape[:2]
        else:
            src_w = src_h = None
    t1 = time.perf_counter()
    return jsonify({
        "text": text,
        "boxes": boxes,
        "image_size": {"width": src_w, "height": src_h},
        "timings_ms": {
            "preprocess": round((pre1 - pre0) * 1000, 1),
            "ocr": round((ocr1 - ocr0) * 1000, 1),
            "total": round((t1 - t0) * 1000, 1)
        },
        "lang": lang,
        "preset": preset,
        "result_img": result_url
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


@app.route('/api/delete_results', methods=['POST'])
def delete_results():
    """安全地删除位于 static/results 目录下的文件。请求体 JSON: {"files": ["/static/results/xxx.png", ...]}"""
    try:
        data = request.get_json() or {}
        files = data.get('files', []) if isinstance(data, dict) else []
        if not isinstance(files, list):
            return jsonify({'error': 'files must be a list'}), 400

        static_dir = os.path.join(app.root_path, 'static')
        results_dir = os.path.join(static_dir, 'results')
        deleted = []
        failed = []
        for f in files:
            if not f:
                continue
            # accept either '/static/results/xxx' or 'static/results/xxx' or just the basename
            fname = os.path.basename(f)
            # simple safety: no path separators allowed in basename
            if '..' in fname or '/' in fname or '\\' in fname:
                failed.append({'file': f, 'reason': 'invalid name'})
                continue
            target = os.path.join(results_dir, fname)
            try:
                if os.path.exists(target) and os.path.commonpath([os.path.abspath(target), os.path.abspath(results_dir)]) == os.path.abspath(results_dir):
                    os.remove(target)
                    deleted.append(f)
                else:
                    failed.append({'file': f, 'reason': 'not_found'})
            except Exception as ex:
                failed.append({'file': f, 'reason': str(ex)})

        return jsonify({'deleted': deleted, 'failed': failed})
    except Exception as e:
        return jsonify({'error': str(e)}), 500

if __name__ == '__main__':
    app.run(debug=True)

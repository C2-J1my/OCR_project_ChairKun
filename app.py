import os
import time
import base64
import io
import uuid
import requests
from flask import Flask, request, render_template, jsonify
from PIL import Image
from werkzeug.utils import secure_filename
from image_preprocessor import ImagePreprocessor

app = Flask(__name__)
os.makedirs('uploads', exist_ok=True)
# 云端OCR服务地址（需要配置环境变量 CLOUD_OCR_URL）
CLOUD_OCR_URL = os.environ.get('CLOUD_OCR_URL', 'http://113.46.128.251:5000/api/remote_ocr')
# 云端OCR服务健康检查地址（需要配置环境变量 CLOUD_OCR_HEALTH_URL）
CLOUD_HEALTH_URL = os.environ.get('CLOUD_OCR_HEALTH_URL') or CLOUD_OCR_URL.replace('/api/remote_ocr', '/health')


def _ensure_pil_image(img_obj):
    if isinstance(img_obj, Image.Image):
        return img_obj
    return Image.fromarray(img_obj)


def _call_cloud_ocr(processed_image, lang='chi_sim', preset=None):
    """将预处理后的图片上传到云端OCR服务"""
    pil_image = _ensure_pil_image(processed_image)
    buffer = io.BytesIO()
    pil_image.save(buffer, format='PNG')
    buffer.seek(0)
    files = {'image': ('processed.png', buffer, 'image/png')}
    data = {'lang': lang}
    if preset:
        data['preset'] = preset
    response = requests.post(CLOUD_OCR_URL, files=files, data=data, timeout=60)
    response.raise_for_status()
    return response.json()


def _save_overlay_image(overlay_b64: str):
    if not overlay_b64:
        return None
    raw = base64.b64decode(overlay_b64)
    static_dir = os.path.join(app.root_path, 'static')
    results_dir = os.path.join(static_dir, 'results')
    os.makedirs(results_dir, exist_ok=True)
    unique = f"result_{int(time.time()*1000)}_{uuid.uuid4().hex[:8]}.png"
    target_path = os.path.join(results_dir, unique)
    with open(target_path, 'wb') as f:
        f.write(raw)
    return f'/static/results/{unique}'


def _check_cloud_health():
    try:
        resp = requests.get(CLOUD_HEALTH_URL, timeout=5)
        resp.raise_for_status()
        return resp.json()
    except Exception as exc:
        return {'status': 'error', 'error': str(exc)}

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
        pre_end = time.perf_counter()
        lang = request.args.get('lang', 'chi_sim')
        remote_result = _call_cloud_ocr(processed_image, lang=lang, preset=preset)
        text = remote_result.get('text', '')
        boxes = remote_result.get('boxes', [])
        result_url = _save_overlay_image(remote_result.get('overlay_image'))
        remote_timings = remote_result.get('timings_ms', {}) or {}
        cloud_total = remote_timings.get('total')
        cloud_ocr = remote_timings.get('ocr') or cloud_total

        total_end = time.perf_counter()
        timings = {
            'preprocess': round((pre_end - pre_start) * 1000, 1),
            'ocr': cloud_ocr,
            'cloud_total': cloud_total,
            'remote_breakdown': remote_timings,
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
    pre1 = time.perf_counter()
    remote_result = _call_cloud_ocr(img, lang=lang, preset=preset)
    text = remote_result.get('text', '')
    boxes = remote_result.get('boxes', [])
    result_url = _save_overlay_image(remote_result.get('overlay_image'))
    remote_timings = remote_result.get('timings_ms', {}) or {}
    cloud_total = remote_timings.get('total')
    cloud_ocr = remote_timings.get('ocr') or cloud_total
    image_size = remote_result.get('image_size')
    if not image_size and hasattr(img, 'size'):
        src_w, src_h = img.size
        image_size = {"width": src_w, "height": src_h}
    elif not image_size:
        image_size = {"width": None, "height": None}
    t1 = time.perf_counter()
    return jsonify({
        "text": text,
        "boxes": boxes,
        "image_size": image_size,
        "timings_ms": {
            "preprocess": round((pre1 - pre0) * 1000, 1),
            "ocr": cloud_ocr,
            "cloud_total": cloud_total,
            "remote_breakdown": remote_timings,
            "total": round((t1 - t0) * 1000, 1)
        },
        "lang": lang,
        "preset": preset,
        "result_img": result_url
    })

@app.route('/health', methods=['GET'])
def health():
    cloud_status = _check_cloud_health()
    return jsonify({
        "preprocess": "ok",
        "cloud": cloud_status
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

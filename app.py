from flask import Flask, request, jsonify, render_template
import numpy as np
import cv2
import tensorflow as tf
from mtcnn import MTCNN
from tensorflow.keras.applications.efficientnet import preprocess_input
import os
import uuid

# =============================
# APP CONFIG
# =============================
app = Flask(__name__, template_folder='frontend')
app.config['MAX_CONTENT_LENGTH'] = 2 * 1024 * 1024 * 1024  # 2 GB

model    = None
detector = None


# =============================
# CORS — applied to every response
# =============================
@app.after_request
def add_cors(response):
    response.headers['Access-Control-Allow-Origin']  = '*'
    response.headers['Access-Control-Allow-Methods'] = 'GET, POST, OPTIONS'
    response.headers['Access-Control-Allow-Headers'] = 'Content-Type'
    return response


# =============================
# LOAD MODEL
# =============================
def download_model_if_needed():
    """Download model from Google Drive if not present (for cloud deployment)."""
    model_path = 'model/unmask_b0_multidomain_boost.keras'
    if os.path.exists(model_path):
        return model_path

    gdrive_url = os.environ.get('MODEL_GDRIVE_URL')
    if not gdrive_url:
        raise FileNotFoundError(
            "Model not found locally and MODEL_GDRIVE_URL env var not set."
        )

    import urllib.request
    print("Downloading model from Google Drive...")
    os.makedirs('model', exist_ok=True)
    urllib.request.urlretrieve(gdrive_url, model_path)
    print("Model downloaded ✅")
    return model_path


def load_model():
    global model, detector

    model_path = download_model_if_needed()
    model = tf.keras.models.load_model(model_path)
    print("Model loaded ✅")

    detector = MTCNN()
    print("Params:", model.count_params())
    print("Model ready ✅")


# =============================
# PREPROCESS
# =============================
def preprocess_face(img):
    faces = detector.detect_faces(img)

    if faces:
        x, y, w, h = faces[0]['box']
        x, y = max(0, x - 20), max(0, y - 20)
        w    = min(w + 40, img.shape[1] - x)
        h    = min(h + 40, img.shape[0] - y)
        face = img[y:y + h, x:x + w]
    else:
        face = img

    face = cv2.resize(face, (224, 224))
    face = face.astype(np.float32)
    face = np.expand_dims(face, axis=0)
    face = preprocess_input(face)

    return face, len(faces)


# =============================
# SERVE FRONTEND
# =============================
@app.route('/')
def home():
    return render_template('index.html')


# =============================
# HANDLE OPTIONS (preflight)
# =============================
@app.route('/analyze/image', methods=['OPTIONS'])
@app.route('/analyze/video', methods=['OPTIONS'])
@app.route('/health',        methods=['OPTIONS'])
def handle_options():
    return '', 204


# =============================
# HEALTH CHECK
# =============================
@app.route('/health', methods=['GET'])
def health():
    return jsonify({
        'status':       'ok',
        'model_loaded': model is not None
    })


# =============================
# IMAGE ANALYSIS
# =============================
@app.route('/analyze/image', methods=['POST'])
def analyze_image():
    if model is None:
        return jsonify({'error': 'Model not loaded'}), 503

    if 'file' not in request.files:
        return jsonify({'error': 'No file provided'}), 400

    try:
        file  = request.files['file']
        npimg = np.frombuffer(file.read(), np.uint8)
        img   = cv2.imdecode(npimg, cv2.IMREAD_COLOR)

        if img is None:
            return jsonify({'error': 'Could not decode image. Check file format.'}), 400

        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

        face, face_count = preprocess_face(img)

        scores = [float(model.predict(face, verbose=0)[0][0]) for _ in range(3)]
        score  = float(np.mean(scores))

        verdict    = 'REAL' if score > 0.9 else 'FAKE'
        confidence = score if score > 0.9 else (1 - score)

        return jsonify({
            'verdict':     verdict,
            'confidence':  round(confidence * 100, 1),
            'raw_score':   round(score, 4),
            'faces_found': face_count,
            'model':       'EfficientNetB0'
        })

    except Exception as e:
        return jsonify({'error': str(e)}), 500


# =============================
# VIDEO ANALYSIS
# =============================
@app.route('/analyze/video', methods=['POST'])
def analyze_video():
    if model is None:
        return jsonify({'error': 'Model not loaded'}), 503

    if 'file' not in request.files:
        return jsonify({'error': 'No file provided'}), 400

    temp_path = f'temp_{uuid.uuid4().hex}.mp4'

    try:
        file = request.files['file']
        file.save(temp_path)

        cap       = cv2.VideoCapture(temp_path)
        fps       = cap.get(cv2.CAP_PROP_FPS)
        results   = []
        frame_num = 0

        while cap.isOpened():
            ret, frame = cap.read()
            if not ret:
                break

            if frame_num % 5 == 0:
                rgb     = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                face, _ = preprocess_face(rgb)
                score   = float(model.predict(face, verbose=0)[0][0])
                timestamp = round(frame_num / fps, 2) if fps > 0 else frame_num
                results.append({
                    'frame':     frame_num,
                    'timestamp': timestamp,
                    'score':     round(score, 4)
                })

            frame_num += 1

        cap.release()

        if not results:
            return jsonify({'error': 'No frames could be processed'}), 400

        scores    = [r['score'] for r in results]
        avg_score = float(np.mean(scores))

        verdict    = 'REAL' if avg_score > 0.9 else 'FAKE'
        confidence = avg_score if avg_score > 0.9 else (1 - avg_score)

        fake_frames = sum(1 for s in scores if s <= 0.9)
        real_frames = len(scores) - fake_frames

        return jsonify({
            'verdict':         verdict,
            'confidence':      round(confidence * 100, 1),
            'raw_score':       round(avg_score, 4),
            'frames_analyzed': len(results),
            'fake_frames':     fake_frames,
            'real_frames':     real_frames
        })

    except Exception as e:
        return jsonify({'error': str(e)}), 500

    finally:
        if os.path.exists(temp_path):
            os.remove(temp_path)


# =============================
# START SERVER
# =============================
if __name__ == '__main__':
    load_model()
    app.run(debug=False, host='0.0.0.0', port=5000, use_reloader=False)
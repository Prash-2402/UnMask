from flask import Flask, request, jsonify
from flask_cors import CORS
import numpy as np
import cv2
import tensorflow as tf
from mtcnn import MTCNN
import os

app = Flask(__name__)
CORS(app)

model = None
detector = None


def load_model():
    global model, detector
    if os.path.exists('model/unmask_b0_multidomain_boost.keras'):
        model = tf.keras.models.load_model(
            'model/unmask_b0_multidomain_boost.keras')
        print("Loaded EfficientNetB0 multidomain model ✅")
    elif os.path.exists('model/unmask_model.h5'):
        model = tf.keras.models.load_model('model/unmask_model.h5')
        print("Loaded base model ✅")
    else:
        raise FileNotFoundError("No model file found!")

    detector = MTCNN()
    print("Params:", model.count_params())
    print("Model ready ✅")


def preprocess_face(img):
    faces = detector.detect_faces(img)
    if faces:
        x, y, w, h = faces[0]['box']
        x, y = max(0, x-20), max(0, y-20)
        w = min(w+40, img.shape[1]-x)
        h = min(h+40, img.shape[0]-y)
        face = img[y:y+h, x:x+w]
    else:
        face = img

    face = cv2.resize(face, (224, 224))
    face = face.astype(np.float32) / 255.0
    face = np.expand_dims(face, axis=0)
    return face, len(faces)


@app.route('/')
def home():
    return jsonify({
        "status":  "UnMask API running ✅",
        "version": "2.0",
        "model":   "EfficientNetB0 Multidomain"
    })


@app.route('/analyze/image', methods=['POST'])
def analyze_image():
    try:
        file = request.files['file']
        npimg = np.frombuffer(file.read(), np.uint8)
        img = cv2.imdecode(npimg, cv2.IMREAD_COLOR)
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

        face, face_count = preprocess_face(img)

        # Run 3 times and average for stability
        scores = [float(model.predict(face, verbose=0)[0][0])
                  for _ in range(3)]
        score = float(np.mean(scores))

        # {'fake': 0, 'real': 1} → score > 0.5 = REAL
        verdict = "REAL" if score > 0.5 else "FAKE"
        confidence = score if score > 0.5 else (1 - score)

        return jsonify({
            "verdict":     verdict,
            "confidence":  round(confidence * 100, 1),
            "raw_score":   round(score, 4),
            "faces_found": face_count,
            "model":       "EfficientNetB0"
        })

    except Exception as e:
        return jsonify({"error": str(e)}), 500


@app.route('/analyze/video', methods=['POST'])
def analyze_video():
    try:
        file = request.files['file']
        temp_path = 'temp_video.mp4'
        file.save(temp_path)

        cap = cv2.VideoCapture(temp_path)
        fps = cap.get(cv2.CAP_PROP_FPS)
        results = []
        frame_num = 0

        while cap.isOpened():
            ret, frame = cap.read()
            if not ret:
                break
            if frame_num % 5 == 0:
                rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                face, _ = preprocess_face(rgb)
                score = float(model.predict(face, verbose=0)[0][0])
                timestamp = round(frame_num / fps, 2) if fps > 0 else frame_num
                results.append({
                    "frame":     frame_num,
                    "timestamp": timestamp,
                    "score":     round(score, 4)
                })
            frame_num += 1

        cap.release()
        os.remove(temp_path)

        if not results:
            return jsonify({"error": "No frames processed"}), 400

        scores = [r["score"] for r in results]
        avg_score = float(np.mean(scores))
        verdict = "REAL" if avg_score > 0.5 else "FAKE"
        confidence = avg_score if avg_score > 0.5 else (1 - avg_score)

        fake_frames = sum(1 for s in scores if s < 0.5)
        real_frames = len(scores) - fake_frames

        return jsonify({
            "verdict":         verdict,
            "confidence":      round(confidence * 100, 1),
            "raw_score":       round(avg_score, 4),
            "frames_analyzed": len(results),
            "fake_frames":     fake_frames,
            "real_frames":     real_frames,
            "timeline":        [round((1-r["score"])*100, 1) for r in results],
            "timestamps":      [r["timestamp"] for r in results]
        })

    except Exception as e:
        return jsonify({"error": str(e)}), 500


if __name__ == '__main__':
    load_model()
    app.run(debug=True, port=5000)

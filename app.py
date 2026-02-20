from flask import Flask, request, jsonify
from flask_cors import CORS
import numpy as np
import cv2
import tensorflow as tf
from mtcnn import MTCNN
import base64
import os

app = Flask(__name__)
CORS(app)

# Load model
model = None
detector = MTCNN()


def load_model():
    global model
    model = tf.keras.models.load_model('model/unmask_model.h5')
    print("Model loaded ✅")


def preprocess_face(img):
    # Detect face
    faces = detector.detect_faces(img)
    if faces:
        x, y, w, h = faces[0]['box']
        x, y = max(0, x-20), max(0, y-20)
        face = img[y:y+h+40, x:x+w+40]
    else:
        face = img
    # Resize and normalize
    face = cv2.resize(face, (224, 224))
    face = face.astype(np.float32) / 255.0
    face = np.expand_dims(face, axis=0)
    return face


@app.route('/')
def home():
    return jsonify({"status": "UnMask API running ✅"})


@app.route('/analyze/image', methods=['POST'])
def analyze_image():
    try:
        file = request.files['file']
        npimg = np.frombuffer(file.read(), np.uint8)
        img = cv2.imdecode(npimg, cv2.IMREAD_COLOR)
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

        face = preprocess_face(img)
        score = float(model.predict(face)[0][0])

        return jsonify({
            "verdict":    "FAKE" if score > 0.5 else "REAL",
            "confidence": round(score * 100, 1) if score > 0.5 else round((1-score)*100, 1),
            "raw_score":  round(score, 4)
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
        results = []
        frame_num = 0

        while cap.isOpened():
            ret, frame = cap.read()
            if not ret:
                break
            if frame_num % 5 == 0:
                rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                face = preprocess_face(rgb)
                score = float(model.predict(face)[0][0])
                results.append(score)
            frame_num += 1

        cap.release()
        os.remove(temp_path)

        avg_score = np.mean(results)
        return jsonify({
            "verdict":       "FAKE" if avg_score > 0.5 else "REAL",
            "confidence":    round(avg_score * 100, 1) if avg_score > 0.5 else round((1-avg_score)*100, 1),
            "frames_analyzed": len(results),
            "timeline":      [round(s*100, 1) for s in results]
        })

    except Exception as e:
        return jsonify({"error": str(e)}), 500


if __name__ == '__main__':
    load_model()
    app.run(debug=True, port=5000)

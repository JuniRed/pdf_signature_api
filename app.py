from flask import Flask, request, jsonify
from flask_cors import CORS
import base64
import numpy as np
import cv2
from pdf2image import convert_from_bytes
import os
import traceback

app = Flask(__name__)
CORS(app)

def preprocess_image(image):
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    blur = cv2.GaussianBlur(gray, (5, 5), 0)
    _, thresh = cv2.threshold(blur, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)
    kernel = np.ones((3, 3), np.uint8)
    cleaned = cv2.morphologyEx(thresh, cv2.MORPH_OPEN, kernel, iterations=1)
    return cleaned

def extract_signature(image):
    preprocessed = preprocess_image(image)
    contours, _ = cv2.findContours(preprocessed, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    mask = np.zeros(image.shape[:2], dtype=np.uint8)
    for cnt in contours:
        x, y, w, h = cv2.boundingRect(cnt)
        area = cv2.contourArea(cnt)
        aspect_ratio = w / float(h)
        if 1000 < area < 50000 and 1.5 < aspect_ratio < 10.0:
            cv2.drawContours(mask, [cnt], -1, 255, thickness=cv2.FILLED)

    if not np.any(mask):
        raise Exception("No valid signature found.")

    signature = cv2.bitwise_and(image, image, mask=mask)
    white_bg = np.full_like(image, 255)
    result = np.where(mask[:, :, None] == 255, signature, white_bg)
    return result

def read_image_from_base64(base64_string):
    try:
        base64_data = base64_string.split(',')[-1]
        file_data = base64.b64decode(base64_data)

        if file_data[:4] == b'%PDF':
            images = convert_from_bytes(file_data)
            image = np.array(images[0])
            return cv2.cvtColor(image, cv2.COLOR_RGB2BGR)

        np_array = np.frombuffer(file_data, np.uint8)
        return cv2.imdecode(np_array, cv2.IMREAD_COLOR)
    except Exception as e:
        print("Error decoding image:", e)
        return None

@app.route('/extract-signature', methods=['POST'])
def api_extract_signature():
    try:
        data = request.get_json()
        if 'file' not in data:
            return jsonify({'error': 'No file provided'}), 400

        image = read_image_from_base64(data['file'])
        if image is None:
            return jsonify({'error': 'Could not decode image'}), 400

        signature_img = extract_signature(image)
        _, buffer = cv2.imencode('.png', signature_img)
        encoded_signature = base64.b64encode(buffer).decode('utf-8')

        return jsonify({'signature': encoded_signature})
    except Exception as e:
        traceback.print_exc()
        return jsonify({'error': str(e)}), 500

if __name__ == "__main__":
    port = int(os.environ.get("PORT", 5000))
    app.run(host="0.0.0.0", port=port)

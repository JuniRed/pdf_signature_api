from flask import Flask, request, jsonify
from flask_cors import CORS
import base64
import numpy as np
import cv2
from PIL import Image
import io
import os
from pdf2image import convert_from_bytes

app = Flask(__name__)
CORS(app)

def preprocess_image(image):
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    _, binary = cv2.threshold(gray, 150, 255, cv2.THRESH_BINARY_INV)
    return binary

def extract_signature(image):
    preprocessed = preprocess_image(image)
    contours, _ = cv2.findContours(preprocessed, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    signature_contours = []
    for cnt in contours:
        x, y, w, h = cv2.boundingRect(cnt)
        aspect_ratio = w / float(h)
        area = cv2.contourArea(cnt)
        if 1000 < area < 50000 and 1.0 < aspect_ratio < 6.0:
            signature_contours.append(cnt)

    mask = np.zeros_like(preprocessed)
    cv2.drawContours(mask, signature_contours, -1, 255, -1)

    result = cv2.bitwise_and(image, image, mask=mask)
    white_bg = np.full_like(image, 255)
    final = np.where(mask[:, :, None] == 255, result, white_bg)

    return final

def read_image_from_base64(base64_string):
    header_removed = base64_string.split(',')[-1]
    img_data = base64.b64decode(header_removed)
    img_array = np.frombuffer(img_data, np.uint8)

    try:
        # Check if it's a PDF
        if base64_string.strip().startswith("JVBER"):  # %PDF
            images = convert_from_bytes(img_data)
            image = np.array(images[0])
            return cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
        else:
            return cv2.imdecode(img_array, cv2.IMREAD_COLOR)
    except Exception as e:
        print("Error decoding image:", e)
        return None

@app.route('/extract-signature', methods=['POST'])
def extract_signature_api():
    data = request.get_json()
    if 'file' not in data:
        return jsonify({'error': 'No file provided'}), 400

    image = read_image_from_base64(data['file'])
    if image is None:
        return jsonify({'error': 'Invalid image format'}), 400

    try:
        signature = extract_signature(image)
        _, buffer = cv2.imencode('.png', signature)
        encoded_signature = base64.b64encode(buffer).decode('utf-8')
        return jsonify({'signature': encoded_signature})
    except Exception as e:
        return jsonify({'error': str(e)}), 500

if __name__ == '__main__':
    app.run(debug=True)

from flask import Flask, request, jsonify
from flask_cors import CORS
import base64
import numpy as np
import cv2
from pdf2image import convert_from_bytes
import traceback

app = Flask(__name__)
CORS(app)

def preprocess_image(image):
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    blur = cv2.GaussianBlur(gray, (5, 5), 0)
    thresh = cv2.adaptiveThreshold(
        blur, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
        cv2.THRESH_BINARY_INV, 15, 8
    )
    kernel = np.ones((3, 3), np.uint8)
    return cv2.morphologyEx(thresh, cv2.MORPH_OPEN, kernel)

def extract_signature(image):
    preprocessed = preprocess_image(image)
    contours, _ = cv2.findContours(preprocessed, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    signature_contours = []
    for cnt in contours:
        x, y, w, h = cv2.boundingRect(cnt)
        aspect_ratio = w / float(h)
        area = cv2.contourArea(cnt)
        if 1000 < area < 50000 and 1.0 < aspect_ratio < 8.0:
            signature_contours.append(cnt)

    if not signature_contours:
        raise Exception("No signature detected")

    mask = np.zeros(image.shape[:2], dtype=np.uint8)
    cv2.drawContours(mask, signature_contours, -1, 255, -1)
    signature = cv2.bitwise_and(image, image, mask=mask)
    white_bg = np.full_like(image, 255)
    final = np.where(mask[:, :, None] == 255, signature, white_bg)
    return final

def read_image_from_base64(b64_string):
    try:
        base64_data = b64_string.split(',')[-1]
        file_data = base64.b64decode(base64_data + '===')
        if file_data[:4] == b'%PDF':
            image = convert_from_bytes(file_data, dpi=300)[0]
            return cv2.cvtColor(np.array(image), cv2.COLOR_RGB2BGR)
        np_array = np.frombuffer(file_data, np.uint8)
        return cv2.imdecode(np_array, cv2.IMREAD_COLOR)
    except Exception as e:
        print("Decoding failed:", e)
        return None

@app.route('/extract-signature', methods=['POST'])
def extract_signature_api():
    try:
        data = request.get_json()
        if not data or 'file' not in data:
            return jsonify({'error': 'Missing "file" in request'}), 400

        image = read_image_from_base64(data['file'])
        if image is None:
            return jsonify({'error': 'Failed to decode image'}), 400

        signature = extract_signature(image)
        _, buffer = cv2.imencode('.png', signature)
        encoded_signature = base64.b64encode(buffer).decode('utf-8')
        return jsonify({'signature': encoded_signature})

    except Exception as e:
        print(traceback.format_exc())
        return jsonify({'error': str(e)}), 500

if __name__ == '__main__':
    app.run(debug=True)

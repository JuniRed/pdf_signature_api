from flask import Flask, request, jsonify
from flask_cors import CORS
import cv2
import numpy as np
import base64
from PIL import Image
import io
from pdf2image import convert_from_bytes

app = Flask(__name__)
CORS(app)

def decode_base64_file(file_str):
    """Decode base64 string and return bytes."""
    if ',' in file_str:
        file_str = file_str.split(',')[1]
    return base64.b64decode(file_str)

def convert_pdf_to_image(pdf_bytes):
    """Convert first page of PDF to image (RGB)."""
    images = convert_from_bytes(pdf_bytes)
    img = np.array(images[0])
    img = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)
    return img

def extract_signature(img):
    """Extract the largest signature-like contour."""
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    _, thresh = cv2.threshold(gray, 180, 255, cv2.THRESH_BINARY_INV)

    # Denoise and find contours
    kernel = np.ones((3,3), np.uint8)
    morph = cv2.morphologyEx(thresh, cv2.MORPH_CLOSE, kernel, iterations=2)
    contours, _ = cv2.findContours(morph, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    # Find the largest contour by area
    max_contour = max(contours, key=cv2.contourArea, default=None)
    if max_contour is None or cv2.contourArea(max_contour) < 100:
        return None

    x, y, w, h = cv2.boundingRect(max_contour)
    signature = img[y:y+h, x:x+w]

    # Optional: clean white background
    white_bg = np.ones_like(signature, dtype=np.uint8) * 255
    gray_sig = cv2.cvtColor(signature, cv2.COLOR_BGR2GRAY)
    mask = gray_sig < 200
    white_bg[mask] = signature[mask]

    return white_bg

def encode_image_to_base64(img):
    """Encode BGR image to base64 PNG string."""
    _, buffer = cv2.imencode('.png', img)
    return base64.b64encode(buffer).decode()

@app.route('/extract_signature', methods=['POST'])
def extract_signature_api():
    try:
        data = request.get_json()
        file_data = data.get("file")

        if not file_data:
            return jsonify({"error": "No file data provided"}), 400

        file_bytes = decode_base64_file(file_data)

        if file_data.startswith("data:application/pdf") or file_bytes[:4] == b"%PDF":
            img = convert_pdf_to_image(file_bytes)
        else:
            nparr = np.frombuffer(file_bytes, np.uint8)
            img = cv2.imdecode(nparr, cv2.IMREAD_COLOR)

        signature = extract_signature(img)
        if signature is None:
            return jsonify({"error": "No signature found"}), 404

        encoded = encode_image_to_base64(signature)
        return jsonify({"signature_base64": encoded})

    except Exception as e:
        return jsonify({"error": str(e)}), 500

if __name__ == '__main__':
    app.run(debug=True)

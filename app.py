from flask import Flask, request, jsonify
from flask_cors import CORS
import base64
import numpy as np
import cv2
from pdf2image import convert_from_bytes
import traceback

app = Flask(__name__)
CORS(app)

def preprocess(image):
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    blurred = cv2.GaussianBlur(gray, (5, 5), 0)
    binary = cv2.adaptiveThreshold(
        blurred, 255, cv2.ADAPTIVE_THRESH_MEAN_C,
        cv2.THRESH_BINARY_INV, 15, 10
    )
    kernel = np.ones((2, 2), np.uint8)
    clean = cv2.morphologyEx(binary, cv2.MORPH_OPEN, kernel, iterations=1)
    return clean

def extract_signature_only(image):
    preprocessed = preprocess(image)
    contours, _ = cv2.findContours(preprocessed, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    mask = np.zeros(image.shape[:2], dtype=np.uint8)
    height = image.shape[0]

    for cnt in contours:
        x, y, w, h = cv2.boundingRect(cnt)
        area = cv2.contourArea(cnt)
        aspect_ratio = w / float(h)

        # Heuristic filters for signature
        if area > 800 and aspect_ratio > 2 and h < 150 and y > height // 2:
            cv2.drawContours(mask, [cnt], -1, 255, -1)

    if np.count_nonzero(mask) == 0:
        raise Exception("No signature found.")

    result = cv2.bitwise_and(image, image, mask=mask)
    white_bg = np.full_like(image, 255)
    signature = np.where(mask[:, :, None] == 255, result, white_bg)
    return signature

def decode_image(base64_str):
    try:
        base64_data = base64_str.split(",")[-1]
        raw = base64.b64decode(base64_data)

        if raw[:4] == b'%PDF':
            image = convert_from_bytes(raw, dpi=300)[0]
            return cv2.cvtColor(np.array(image), cv2.COLOR_RGB2BGR)

        np_arr = np.frombuffer(raw, np.uint8)
        return cv2.imdecode(np_arr, cv2.IMREAD_COLOR)

    except Exception as e:
        print("Decode error:", e)
        return None

@app.route("/extract-signature", methods=["POST"])
def extract_signature_route():
    try:
        data = request.get_json()
        if "file" not in data:
            return jsonify({"error": "Missing 'file' field"}), 400

        image = decode_image(data["file"])
        if image is None:
            return jsonify({"error": "Invalid image/PDF"}), 400

        signature_img = extract_signature_only(image)
        _, buffer = cv2.imencode(".png", signature_img)
        encoded = base64.b64encode(buffer).decode("utf-8")

        return jsonify({"signature": encoded})
    except Exception as e:
        print(traceback.format_exc())
        return jsonify({"error": str(e)}), 500

if __name__ == "__main__":
    app.run(host="0.0.0.0", port=10000)

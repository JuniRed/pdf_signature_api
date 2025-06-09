from flask import Flask, request, jsonify
from flask_cors import CORS
import base64
import numpy as np
import cv2
from pdf2image import convert_from_bytes
from skimage.metrics import structural_similarity as ssim
import traceback

app = Flask(__name__)
CORS(app)


# --- Decode base64 to image (PDF or image) ---
def decode_base64_to_image(base64_string):
    try:
        base64_data = base64_string.split(',')[-1]
        file_data = base64.b64decode(base64_data)

        # If PDF
        if file_data[:4] == b'%PDF':
            images = convert_from_bytes(file_data)
            img = np.array(images[0])
            return cv2.cvtColor(img, cv2.COLOR_RGB2BGR)

        # If image
        np_array = np.frombuffer(file_data, np.uint8)
        return cv2.imdecode(np_array, cv2.IMREAD_COLOR)

    except Exception as e:
        print(f"Error decoding base64: {e}")
        return None


# --- Preprocess to highlight signature ---
def preprocess_image(image):
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    blur = cv2.GaussianBlur(gray, (3, 3), 0)
    thresh = cv2.adaptiveThreshold(
        blur, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
        cv2.THRESH_BINARY_INV, 15, 8
    )
    kernel = np.ones((2, 2), np.uint8)
    cleaned = cv2.morphologyEx(thresh, cv2.MORPH_OPEN, kernel, iterations=1)
    return cleaned


# --- Extract signature region only ---
def extract_signature(image):
    preprocessed = preprocess_image(image)
    contours, _ = cv2.findContours(preprocessed, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    signature_contours = []
    for cnt in contours:
        x, y, w, h = cv2.boundingRect(cnt)
        area = cv2.contourArea(cnt)
        aspect_ratio = w / float(h)
        if 1500 < area < 50000 and 1.2 < aspect_ratio < 8.0:
            signature_contours.append(cnt)

    if not signature_contours:
        raise Exception("No signature-like contour detected.")

    mask = np.zeros(image.shape[:2], dtype=np.uint8)
    cv2.drawContours(mask, signature_contours, -1, 255, cv2.FILLED)

    signature_img = cv2.bitwise_and(image, image, mask=mask)
    white_bg = np.full_like(image, 255)
    result = np.where(mask[:, :, None] == 255, signature_img, white_bg)

    return result


# --- Compare two signature images using SSIM ---
def compare_signatures(img1, img2):
    img1_gray = cv2.cvtColor(img1, cv2.COLOR_BGR2GRAY)
    img2_gray = cv2.cvtColor(img2, cv2.COLOR_BGR2GRAY)

    # Resize to same shape
    h, w = min(img1_gray.shape[0], img2_gray.shape[0]), min(img1_gray.shape[1], img2_gray.shape[1])
    img1_gray = cv2.resize(img1_gray, (w, h))
    img2_gray = cv2.resize(img2_gray, (w, h))

    score, _ = ssim(img1_gray, img2_gray, full=True)
    return round(float(score), 4)


# --- API to extract + compare ---
@app.route('/signature_api', methods=['POST'])
def signature_api():
    try:
        data = request.get_json()
        if not data or 'original' not in data or 'input' not in data:
            return jsonify({'error': 'Missing "original" or "input" in request'}), 400

        original_img = decode_base64_to_image(data['original'])
        input_img = decode_base64_to_image(data['input'])

        if original_img is None or input_img is None:
            return jsonify({'error': 'Could not decode image'}), 400

        original_sig = extract_signature(original_img)
        input_sig = extract_signature(input_img)

        similarity = compare_signatures(original_sig, input_sig)

        return jsonify({
            'similarity': similarity,
            'match': similarity > 0.85
        })

    except Exception as e:
        print(traceback.format_exc())
        return jsonify({'error': f"Internal server error: {str(e)}"}), 500


# --- Run the server ---
if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000, debug=True)

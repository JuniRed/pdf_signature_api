from flask import Flask, request, jsonify, send_file
import cv2
import numpy as np
import io
import base64
import tempfile
from PIL import Image
from pdf2image import convert_from_bytes

app = Flask(__name__)

def extract_signature(image_data):
    # Convert image bytes to OpenCV format
    file_bytes = np.asarray(bytearray(image_data), dtype=np.uint8)
    open_cv_image = cv2.imdecode(file_bytes, cv2.IMREAD_COLOR)

    if open_cv_image is None:
        raise ValueError("Could not decode image")

    gray = cv2.cvtColor(open_cv_image, cv2.COLOR_BGR2GRAY)
    blur = cv2.GaussianBlur(gray, (3, 3), 0)
    _, thresh = cv2.threshold(blur, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)

    contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    potential_signatures_with_scores = []
    image_area = open_cv_image.shape[0] * open_cv_image.shape[1]

    for contour in contours:
        x, y, w, h = cv2.boundingRect(contour)
        aspect_ratio = w / float(h)
        area = cv2.contourArea(contour)

        if w < 20 or h < 10 or area < 150:
            continue
        if aspect_ratio < 1.5 or aspect_ratio > 10:
            continue
        if area / image_area > 0.25:
            continue

        score = area * aspect_ratio
        potential_signatures_with_scores.append({
            'score': score,
            'contour': contour,
            'bbox': (x, y, w, h)
        })

    if not potential_signatures_with_scores:
        raise ValueError("No signature-like contours found")

    # Sort by descending score
    potential_signatures_with_scores.sort(key=lambda x: x['score'], reverse=True)

    # Deduplicate and select best unique signature
    final_signature_contour = None
    unique_hashes = set()

    def hash_image(image_gray):
        resized = cv2.resize(image_gray, (64, 32), interpolation=cv2.INTER_AREA)
        return hash(resized.tobytes())

    similarity_threshold = 0.96
    compare_size = (180, 80)

    for candidate in potential_signatures_with_scores:
        x, y, w, h = candidate['bbox']

        if y < 0 or y + h > open_cv_image.shape[0] or x < 0 or x + w > open_cv_image.shape[1]:
            continue

        region = open_cv_image[y:y+h, x:x+w]
        if region.size == 0:
            continue

        try:
            resized = cv2.resize(region, compare_size, interpolation=cv2.INTER_AREA)
            gray = cv2.cvtColor(resized, cv2.COLOR_BGR2GRAY)
            hval = hash_image(gray)

            if hval in unique_hashes:
                continue  # skip duplicate
            unique_hashes.add(hval)

            final_signature_contour = candidate['contour']
            break  # keep only one

        except Exception as err:
            print(f"Error comparing or hashing signature: {err}")
            continue

    if final_signature_contour is None:
        raise ValueError("No unique signature contour found")

    x, y, w, h = cv2.boundingRect(final_signature_contour)
    extracted = open_cv_image[y:y+h, x:x+w]

    # Make white background
    mask = cv2.cvtColor(extracted, cv2.COLOR_BGR2GRAY)
    _, mask_bin = cv2.threshold(mask, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    mask_inv = cv2.bitwise_not(mask_bin)
    white_bg = np.ones_like(extracted, dtype=np.uint8) * 255
    signature_cleaned = cv2.bitwise_or(cv2.bitwise_and(extracted, extracted, mask=mask_inv), white_bg)

    # Convert to PNG in memory
    sig_image = Image.fromarray(cv2.cvtColor(signature_cleaned, cv2.COLOR_BGR2RGB))
    sig_io = io.BytesIO()
    sig_image.save(sig_io, format='PNG')
    return sig_io.getvalue()

@app.route('/extract_signature', methods=['POST'])
def extract_signature_api():
    try:
        data = request.json
        base64_input = data.get('file')

        if not base64_input:
            return jsonify({'error': 'Missing file in request'}), 400

        # Handle base64 header
        if "," in base64_input:
            base64_input = base64_input.split(",")[1]

        decoded = base64.b64decode(base64_input)

        # Convert PDF to image if needed
        if decoded[:4] == b'%PDF':
            with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as temp_pdf:
                temp_pdf.write(decoded)
                temp_pdf.flush()
                pages = convert_from_bytes(decoded, first_page=1, last_page=1)
                if not pages:
                    return jsonify({'error': 'No pages found in PDF'}), 400
                image = pages[0]
                img_byte_arr = io.BytesIO()
                image.save(img_byte_arr, format='PNG')
                decoded = img_byte_arr.getvalue()

        # Extract the signature
        signature_png = extract_signature(decoded)

        # Convert back to base64
        encoded_output = base64.b64encode(signature_png).decode('utf-8')
        return jsonify({'signature_base64': encoded_output}), 200

    except Exception as e:
        return jsonify({'error': str(e)}), 500

@app.route('/')
def index():
    return "Signature Extraction API is running."

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000)

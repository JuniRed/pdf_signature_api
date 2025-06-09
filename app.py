from flask import Flask, request, jsonify
from flask_cors import CORS
import base64, numpy as np, cv2
from pdf2image import convert_from_bytes
from skimage.metrics import structural_similarity as ssim
import os, traceback

app = Flask(__name__)
CORS(app)

# ... [preprocess_image, extract_signature, read_image_from_base64 go here] ...

@app.route('/signature-api', methods=['POST'])
def signature_api():
    try:
        data = request.get_json()
        if 'original' not in data or 'input' not in data:
            return jsonify({'error': 'Missing original or input fields'}), 400

        img1 = read_image_from_base64(data['original'])
        img2 = read_image_from_base64(data['input'])

        if img1 is None or img2 is None:
            return jsonify({'error': 'Invalid image(s)'}), 400

        sig1 = extract_signature(img1)
        sig2 = extract_signature(img2)

        score, _ = ssim(
            cv2.cvtColor(sig1, cv2.COLOR_BGR2GRAY),
            cv2.cvtColor(sig2, cv2.COLOR_BGR2GRAY),
            full=True
        )

        return jsonify({'similarity': round(score, 4)})

    except Exception as e:
        print("Traceback:\n", traceback.format_exc())
        return jsonify({'error': f"Internal error: {str(e)}"}), 500

if __name__ == '__main__':
    port = int(os.environ.get("PORT", 10000))
    app.run(host="0.0.0.0", port=port)

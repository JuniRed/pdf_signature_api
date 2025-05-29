from flask import Flask, request, jsonify
import base64
import fitz  # PyMuPDF
from extract_signature import extract_signature_from_image_bytes

app = Flask(__name__)

@app.route('/extract_signature_from_pdf', methods=['POST'])
def extract_signature_from_pdf():
    data = request.get_json()
    if not data or 'pdf_base64' not in data:
        return jsonify({"error": "Missing pdf_base64"}), 400

    try:
        pdf_data = base64.b64decode(data['pdf_base64'])

        # Convert PDF (1st page) to image
        pdf_doc = fitz.open(stream=pdf_data, filetype="pdf")
        page = pdf_doc.load_page(0)
        pix = page.get_pixmap(dpi=200)
        image_bytes = pix.tobytes("png")

        # Extract signature from image
        signature_bytes = extract_signature_from_image_bytes(image_bytes)
        signature_base64 = base64.b64encode(signature_bytes).decode('utf-8')

        return jsonify({"signature_base64": signature_base64})
    except Exception as e:
        return jsonify({"error": str(e)}), 500

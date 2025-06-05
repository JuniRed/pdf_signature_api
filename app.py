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
        # Ensure there is at least one page
        if pdf_doc.page_count == 0:
             return jsonify({"error": "PDF contains no pages"}), 400

        page = pdf_doc.load_page(0)
        # You might need to adjust the DPI based on your PDF quality
        pix = page.get_pixmap(dpi=300) # Increased DPI for potentially better detail
        image_bytes = pix.tobytes("png")
        pdf_doc.close() # Close the document after processing


        # Extract signature from image
        signature_bytes = extract_signature_from_image_bytes(image_bytes)

        # Check if signature_bytes is None or empty if your function might return that
        if not signature_bytes:
             return jsonify({"error": "Signature extraction returned empty result"}), 404 # Or 500 depending on expected behavior

        signature_base64 = base64.b64encode(signature_bytes).decode('utf-8')

        return jsonify({"signature_base64": signature_base64})

    except Exception as e:
        # Log the exception for debugging on the server side
        print(f"An error occurred: {e}")
        return jsonify({"error": str(e)}), 500

if __name__ == '__main__':
    # Consider running in debug=False for production
    # app.run(debug=True)
    # Specify host and port if needed
    app.run(host='0.0.0.0', port=5000, debug=True) # Example host/port

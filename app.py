from flask import Flask, request, jsonify
import base64
import fitz  # PyMuPDF
from extract_signature import extract_signature_from_image_bytes
import traceback # Import traceback to get detailed error info

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
             pdf_doc.close()
             return jsonify({"error": "PDF contains no pages"}), 400

        page = pdf_doc.load_page(0)
        # You might need to adjust the DPI based on your PDF quality (higher DPI can help with detail)
        # Increased DPI again for potentially better detail in the image conversion
        pix = page.get_pixmap(dpi=400)
        image_bytes = pix.tobytes("png")
        pdf_doc.close() # Close the document after processing


        # Extract signature from image
        signature_bytes = extract_signature_from_image_bytes(image_bytes)

        # Check if signature_bytes is None or empty if your function might return that
        if not signature_bytes:
             # This might indicate extraction failed to find a valid signature
             return jsonify({"error": "Signature extraction returned empty result or failed to find a unique signature that met the criteria."}), 404 # 404 Not Found might be appropriate here


        signature_base64 = base64.b64encode(signature_bytes).decode('utf-8')

        return jsonify({"signature_base64": signature_base64})

    except Exception as e:
        # Log the exception and traceback for debugging on the server side
        print(f"An error occurred during PDF processing or signature extraction: {e}")
        traceback.print_exc() # Print traceback for detailed error location
        # Return the error message in the response
        return jsonify({"error": str(e)}), 500

if __name__ == '__main__':
    # Consider running in debug=False for production
    # app.run(debug=True)
    # Specify host and port if needed
    app.run(host='0.0.0.0', port=5000, debug=True) # Example host/port

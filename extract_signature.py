import cv2
import numpy as np
from PIL import Image
import io

def extract_signature_from_image_bytes(image_bytes):
    image = Image.open(io.BytesIO(image_bytes)).convert("RGB")
    open_cv_image = np.array(image)
    open_cv_image = open_cv_image[:, :, ::-1].copy()  # RGB to BGR

    gray = cv2.cvtColor(open_cv_image, cv2.COLOR_BGR2GRAY)
    _, thresh = cv2.threshold(gray, 200, 255, cv2.THRESH_BINARY_INV)

    contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if not contours:
        raise Exception("No signature found")

    signature_contour = max(contours, key=cv2.contourArea)
    x, y, w, h = cv2.boundingRect(signature_contour)
    signature = open_cv_image[y:y+h, x:x+w]

    sig_image = Image.fromarray(cv2.cvtColor(signature, cv2.COLOR_BGR2RGB))
    sig_io = io.BytesIO()
    sig_image.save(sig_io, format='PNG')
    return sig_io.getvalue()

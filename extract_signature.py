import cv2
import numpy as np
from PIL import Image
import io

def extract_signature_from_image_bytes(image_bytes):
    image = Image.open(io.BytesIO(image_bytes)).convert("RGB")
    open_cv_image = np.array(image)
    open_cv_image = open_cv_image[:, :, ::-1].copy()  # RGB to BGR

    gray = cv2.cvtColor(open_cv_image, cv2.COLOR_BGR2GRAY)
    # Use Otsu's binarization
    _, thresh = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)

    contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if not contours:
        raise Exception("No signature found")

    # Filter contours by area and aspect ratio
    min_area = 100 # Minimum contour area to consider (adjust as needed)
    aspect_ratio_threshold = (0.2, 10) # Reasonable aspect ratio range (width/height, adjust as needed)

    potential_signatures = []
    for contour in contours:
        area = cv2.contourArea(contour)
        if area > min_area:
            x, y, w, h = cv2.boundingRect(contour)
            aspect_ratio = w / h if h > 0 else 0
            if aspect_ratio_threshold[0] < aspect_ratio < aspect_ratio_threshold[1]:
                potential_signatures.append(contour)

    if not potential_signatures:
        raise Exception("No potential signature contours found after filtering.")

    # Select the largest contour among potential signatures
    signature_contour = max(potential_signatures, key=cv2.contourArea)
    x, y, w, h = cv2.boundingRect(signature_contour)
    signature = open_cv_image[y:y+h, x:x+w]

    sig_image = Image.fromarray(cv2.cvtColor(signature, cv2.COLOR_BGR2RGB))
    sig_io = io.BytesIO()
    sig_image.save(sig_io, format='PNG')
    return sig_io.getvalue()

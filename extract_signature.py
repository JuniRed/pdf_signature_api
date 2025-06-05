import cv2
import numpy as np
from PIL import Image
import io

def extract_signature_from_image_bytes(image_bytes):
    image = Image.open(io.BytesIO(image_bytes)).convert("RGB")
    open_cv_image = np.array(image)
    open_cv_image = open_cv_image[:, :, ::-1].copy()  # RGB to BGR

    gray = cv2.cvtColor(open_cv_image, cv2.COLOR_BGR2GRAY)

    # Use Otsu's binarization for better thresholding
    _, thresh = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)

    # Apply morphological operations to clean up the image
    # Dilate to connect broken parts, then erode to remove small noise
    kernel = np.ones((3,3),np.uint8)
    thresh = cv2.dilate(thresh, kernel, iterations = 1)
    thresh = cv2.erode(thresh, kernel, iterations = 1)

    contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if not contours:
        raise Exception("No signature found")

    # Filter contours based on area and potentially other criteria
    # Increased minimum contour area to filter out smaller artifacts
    min_area = 200 # You might need to adjust this value
    potential_signatures = []
    for contour in contours:
        area = cv2.contourArea(contour)
        # Consider adding an upper bound for area if needed to exclude large non-signature elements
        if area > min_area:
             # Further filtering can be added here, e.g., based on aspect ratio if signatures have a typical shape
             # x, y, w, h = cv2.boundingRect(contour)
             # aspect_ratio = w / h if h > 0 else 0
             # if reasonable_aspect_ratio_condition:
             potential_signatures.append(contour)


    if not potential_signatures:
        raise Exception("No potential signature contours found after filtering.")

    # Select the largest contour among potential signatures
    # This is still the primary selection method, which might be improved with more specific filtering
    signature_contour = max(potential_signatures, key=cv2.contourArea)
    x, y, w, h = cv2.boundingRect(signature_contour)
    signature = open_cv_image[y:y+h, x:x+w]

    sig_image = Image.fromarray(cv2.cvtColor(signature, cv2.COLOR_BGR2RGB))
    sig_io = io.BytesIO()
    sig_image.save(sig_io, format='PNG')
    return sig_io.getvalue()

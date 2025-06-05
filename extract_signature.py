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
    thresh_cleaned = cv2.dilate(thresh, kernel, iterations = 1)
    thresh_cleaned = cv2.erode(thresh_cleaned, kernel, iterations = 1)

    contours, _ = cv2.findContours(thresh_cleaned, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if not contours:
        raise Exception("No signature found")

    # Filter contours based on area, pixel density, solidity, and extent
    min_area = 200 # Minimum contour area (adjust as needed)
    min_ink_density = 0.15 # Minimum ink density (adjust as needed)
    min_solidity = 0.3 # Minimum solidity (adjust as needed, signatures might be lower)
    max_solidity = 0.9 # Maximum solidity (adjust as needed, exclude solid blocks)
    min_extent = 0.2 # Minimum extent (adjust as needed)
    max_extent = 0.9 # Maximum extent (adjust as needed)

    potential_signatures = []

    for contour in contours:
        area = cv2.contourArea(contour)
        x, y, w, h = cv2.boundingRect(contour)

        if area > min_area:
            # Calculate ink density
            roi = thresh_cleaned[y:y+h, x:x+w]
            ink_density = cv2.countNonZero(roi) / (w * h) if (w * h) > 0 else 0

            # Calculate solidity
            hull = cv2.convexHull(contour)
            hull_area = cv2.contourArea(hull)
            solidity = float(area)/hull_area if hull_area > 0 else 0

            # Calculate extent
            rect_area = w * h
            extent = float(area)/rect_area if rect_area > 0 else 0

            # Apply filters
            if (ink_density > min_ink_density and
                min_solidity < solidity < max_solidity and
                min_extent < extent < max_extent):
                 potential_signatures.append(contour)


    if not potential_signatures:
        raise Exception("No potential signature contours found after filtering based on multiple criteria.")

    # Select the contour with the highest area among potential signatures
    # You might need a more sophisticated selection method here if multiple candidates exist
    signature_contour = max(potential_signatures, key=cv2.contourArea)
    x, y, w, h = cv2.boundingRect(signature_contour)
    signature = open_cv_image[y:y+h, x:y+h] # Fixed slicing here: should be y:y+h, x:x+w

    sig_image = Image.fromarray(cv2.cvtColor(signature, cv2.COLOR_BGR2RGB))
    sig_io = io.BytesIO()
    sig_image.save(sig_io, format='PNG')
    return sig_io.getvalue()

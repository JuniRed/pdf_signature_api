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

    potential_signatures_contours = []

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
                 potential_signatures_contours.append(contour)


    if not potential_signatures_contours:
        raise Exception("No potential signature contours found after filtering based on multiple criteria.")

    # Now, compare the image content within the bounding boxes of potential signatures
    # to find unique instances.
    unique_signatures_regions = []
    unique_signatures_contours = []
    similarity_threshold = 0.95 # Threshold for considering images as duplicates (adjust as needed)
    compare_size = (100, 50) # Resize images for comparison (adjust as needed)


    for contour in potential_signatures_contours:
        x, y, w, h = cv2.boundingRect(contour)
        # Extract the image region from the original color image
        potential_sig_img = open_cv_image[y:y+h, x:x+w]

        # Resize for comparison
        if potential_sig_img.shape[0] > 0 and potential_sig_img.shape[1] > 0:
             resized_sig_img = cv2.resize(potential_sig_img, compare_size)
             # Convert to grayscale for simpler comparison
             resized_sig_img_gray = cv2.cvtColor(resized_sig_img, cv2.COLOR_BGR2GRAY)
        else:
             continue # Skip empty regions

        is_duplicate = False
        for unique_region_gray in unique_signatures_regions:
            # Use structural similarity index (SSIM) or a simple pixel comparison
            # For exact duplicates, simple equality check after flattening can work if resized to same size
            if resized_sig_img_gray.shape == unique_region_gray.shape:
                 # Using correlation for similarity - 1.0 means perfect match
                 similarity = cv2.matchTemplate(resized_sig_img_gray, unique_region_gray, cv2.TM_CCOEFF_NORMED)[0,0]
                 if similarity > similarity_threshold:
                     is_duplicate = True
                     break

        if not is_duplicate:
            unique_signatures_regions.append(resized_sig_img_gray)
            unique_signatures_contours.append(contour)

    if not unique_signatures_contours:
        raise Exception("No unique signature instances found after duplicate filtering.")

    # Select the first unique signature found (you could add logic to pick based on position if needed)
    final_signature_contour = unique_signatures_contours[0]
    x, y, w, h = cv2.boundingRect(final_signature_contour)
    signature = open_cv_image[y:y+h, x:x+w]


    sig_image = Image.fromarray(cv2.cvtColor(signature, cv2.COLOR_BGR2RGB))
    sig_io = io.BytesIO()
    sig_image.save(sig_io, format='PNG')
    return sig_io.getvalue()

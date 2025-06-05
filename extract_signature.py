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

    # Filter and score contours based on potential signature characteristics
    potential_signatures_with_scores = []

    # Define ideal ranges and weights for scoring (THESE NEED TUNING FOR YOUR IMAGES)
    area_range = (500, 50000) # Example range, adjust based on typical signature size
    density_range = (0.2, 0.6) # Example range for ink density
    solidity_range = (0.3, 0.7) # Example range for solidity
    extent_range = (0.3, 0.7) # Example range for extent
    aspect_ratio_range = (0.5, 5.0) # Example range for aspect ratio

    # Weights for each criterion (adjust to emphasize more important factors)
    weight_area = 1.0
    weight_density = 1.5 # Density might be a stronger indicator
    weight_solidity = 1.0
    weight_extent = 1.0
    weight_aspect_ratio = 0.8 # Aspect ratio might be less critical than density/solidity


    for contour in contours:
        area = cv2.contourArea(contour)
        x, y, w, h = cv2.boundingRect(contour)

        # Basic filter to skip very small contours quickly
        if area > 100 and w > 0 and h > 0:
            # Calculate properties
            roi = thresh_cleaned[y:y+h, x:x+w]
            ink_density = cv2.countNonZero(roi) / (w * h)

            hull = cv2.convexHull(contour)
            hull_area = cv2.contourArea(hull)
            solidity = float(area)/hull_area if hull_area > 0 else 0

            rect_area = w * h
            extent = float(area)/rect_area if rect_area > 0 else 0

            aspect_ratio = w / h

            # Calculate a score based on how well the properties fit the defined ranges
            score = 0

            # Scoring logic (prioritize being within the defined range)
            if area_range[0] <= area <= area_range[1]:
                score += weight_area * (1 - abs(area - (area_range[0] + area_range[1])/2) / ((area_range[1] - area_range[0])/2))
            elif area > area_range[1]: score += weight_area * 0.1 # Small score for larger areas

            if density_range[0] <= ink_density <= density_range[1]:
                 score += weight_density * (1 - abs(ink_density - (density_range[0] + density_range[1])/2) / ((density_range[1] - density_range[0])/2))
            elif ink_density > density_range[1]: score += weight_density * 0.2 # Small score for higher density

            if solidity_range[0] <= solidity <= solidity_range[1]:
                 score += weight_solidity * (1 - abs(solidity - (solidity_range[0] + solidity_range[1])/2) / ((solidity_range[1] - solidity_range[0])/2))

            if extent_range[0] <= extent <= extent_range[1]:
                 score += weight_extent * (1 - abs(extent - (extent_range[0] + extent_range[1])/2) / ((extent_range[1] - extent_range[0])/2))

            if aspect_ratio_range[0] <= aspect_ratio <= aspect_ratio_range[1]:
                 score += weight_aspect_ratio * (1 - abs(aspect_ratio - (aspect_ratio_range[0] + aspect_ratio_range[1])/2) / ((aspect_ratio_range[1] - aspect_ratio_range[0])/2))


            # Add contour and its score if score is above a minimum threshold (adjust as needed)
            # This threshold should be low enough to include potential candidates for duplicate check
            min_score_for_consideration = (weight_area + weight_density + weight_solidity + weight_extent + weight_aspect_ratio) * 0.1 # Example: 10% of max possible score
            if score > min_score_for_consideration:
                 potential_signatures_with_scores.append({'contour': contour, 'score': score, 'area': area})


    if not potential_signatures_with_scores:
        # Fallback: if scoring found nothing, try basic area filtering again
        large_contours = [cnt for cnt in contours if cv2.contourArea(cnt) > 500] # Basic area filter fallback
        if large_contours:
             # Just take the largest if no scored candidates
             largest_contour = max(large_contours, key=cv2.contourArea)
             potential_signatures_with_scores.append({'contour': largest_contour, 'score': 0, 'area': cv2.contourArea(largest_contour)})
        else:
            raise Exception("No potential signature contours found after filtering or scoring.")


    # Sort potential signatures by score in descending order for better priority in unique selection
    potential_signatures_with_scores.sort(key=lambda x: x['score'], reverse=True)

    # Find unique instances based on image content comparison
    unique_signatures_regions = []
    final_signature_contour = None
    similarity_threshold = 0.9 # Threshold for considering images as duplicates (adjust as needed)
    compare_size = (150, 75) # Resize images for comparison (adjust as needed)


    for potential_sig_info in potential_signatures_with_scores:
        contour = potential_sig_info['contour']
        x, y, w, h = cv2.boundingRect(contour)

        # Extract the image region from the original color image
        # Ensure bounding box is within image bounds
        if y+h <= open_cv_image.shape[0] and x+w <= open_cv_image.shape[1]:
             potential_sig_img = open_cv_image[y:y+h, x:x+w]
        else:
             continue # Skip invalid bounding boxes


        # Resize for comparison
        if potential_sig_img.shape[0] > 0 and potential_sig_img.shape[1] > 0:
             try:
                 resized_sig_img = cv2.resize(potential_sig_img, compare_size, interpolation=cv2.INTER_AREA)
                 resized_sig_img_gray = cv2.cvtColor(resized_sig_img, cv2.COLOR_BGR2GRAY)
             except cv2.error as e:
                 print(f"Error resizing image for comparison: {e}")
                 continue # Skip if resize fails
        else:
             continue # Skip empty regions

        is_duplicate = False
        for unique_region_gray in unique_signatures_regions:
            if resized_sig_img_gray.shape == unique_region_gray.shape:
                 try:
                     # Using correlation for similarity
                     similarity = cv2.matchTemplate(resized_sig_img_gray, unique_region_gray, cv2.TM_CCOEFF_NORMED)[0,0]
                     if similarity > similarity_threshold:
                         is_duplicate = True
                         break
                 except cv2.error as e:
                     print(f"Error comparing images: {e}")
                     # Continue checking other unique regions despite the error with this one
                     pass


        if not is_duplicate:
            unique_signatures_regions.append(resized_sig_img_gray)
            # This is the highest scoring unique signature found so far, select it
            final_signature_contour = contour
            break # Stop after finding the best scoring unique signature


    if final_signature_contour is None:
         raise Exception("No unique signature instances found after duplicate filtering based on scoring.")


    # Extract the final signature using the bounding box of the selected contour
    x, y, w, h = cv2.boundingRect(final_signature_contour)
    # Ensure final bounding box is within image bounds before slicing
    if y+h <= open_cv_image.shape[0] and x+w <= open_cv_image.shape[1]:
        signature = open_cv_image[y:y+h, x:x+w]
    else:
        # This case should ideally not happen if the contour came from the image
        raise Exception("Selected signature bounding box is out of image bounds.")


    sig_image = Image.fromarray(cv2.cvtColor(signature, cv2.COLOR_BGR2RGB))
    sig_io = io.BytesIO()
    sig_image.save(sig_io, format='PNG')
    return sig_io.getvalue()

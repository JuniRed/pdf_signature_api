import cv2
import numpy as np
from PIL import Image
import io

def extract_signature_from_image_bytes(image_bytes):
    try:
        image = Image.open(io.BytesIO(image_bytes)).convert("RGB")
        open_cv_image = np.array(image)
        open_cv_image = open_cv_image[:, :, ::-1].copy()  # RGB to BGR

        gray = cv2.cvtColor(open_cv_image, cv2.COLOR_BGR2GRAY)

        # Use Otsu's binarization for better thresholding
        # Added a check for image content before applying Otsu's
        if gray.shape[0] == 0 or gray.shape[1] == 0:
             raise Exception("Empty or invalid image data.")

        _, thresh = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)

        # Apply morphological operations to clean up the image
        # Dilate to connect broken parts, then erode to remove small noise
        kernel = np.ones((3,3),np.uint8)
        thresh_cleaned = cv2.dilate(thresh, kernel, iterations = 1)
        thresh_cleaned = cv2.erode(thresh_cleaned, kernel, iterations = 1)

        contours, _ = cv2.findContours(thresh_cleaned, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        if not contours:
            raise Exception("No contours found in the image.")

        # Filter and score contours based on potential signature characteristics
        potential_signatures_with_scores = []

        # Define ideal ranges and weights for scoring (THESE NEED TUNING FOR YOUR IMAGES)
        # These are example values, adjust based on the size, density, and shape of signatures you expect.
        area_range = (800, 60000) # Adjusted example range
        density_range = (0.25, 0.7) # Adjusted example range
        solidity_range = (0.3, 0.85) # Adjusted example range
        extent_range = (0.25, 0.85) # Adjusted example range
        aspect_ratio_range = (0.4, 6.0) # Adjusted example range

        # Weights for each criterion (adjust to emphasize more important factors)
        weight_area = 1.0
        weight_density = 2.0 # Giving more weight to density
        weight_solidity = 1.5
        weight_extent = 1.5
        weight_aspect_ratio = 0.8


        for contour in contours:
            area = cv2.contourArea(contour)
            x, y, w, h = cv2.boundingRect(contour)

            # Basic filter to skip very small or degenerate contours
            if area > 150 and w > 5 and h > 5:
                try:
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
                    elif area > area_range[1]: score += weight_area * 0.2 # Small score for larger areas

                    if density_range[0] <= ink_density <= density_range[1]:
                         score += weight_density * (1 - abs(ink_density - (density_range[0] + density_range[1])/2) / ((density_range[1] - density_range[0])/2))
                    elif ink_density > density_range[1]: score += weight_density * 0.3 # Small score for higher density

                    if solidity_range[0] <= solidity <= solidity_range[1]:
                         score += weight_solidity * (1 - abs(solidity - (solidity_range[0] + solidity_range[1])/2) / ((solidity_range[1] - solidity_range[0])/2))

                    if extent_range[0] <= extent <= extent_range[1]:
                         score += weight_extent * (1 - abs(extent - (extent_range[0] + extent_range[1])/2) / ((extent_range[1] - extent_range[0])/2))

                    if aspect_ratio_range[0] <= aspect_ratio <= aspect_ratio_range[1]:
                         score += weight_aspect_ratio * (1 - abs(aspect_ratio - (aspect_ratio_range[0] + aspect_ratio_range[1])/2) / ((aspect_ratio_range[1] - aspect_ratio_range[0])/2))


                    # Add contour and its score if score is above a minimum threshold (adjust as needed)
                    # This threshold should be low enough to include potential candidates for duplicate check
                    # Example: requiring at least a moderate fit in density and area, plus something else
                    min_score_for_consideration = (weight_density * 0.4 + weight_area * 0.3) # Example threshold based on key factors
                    if score > min_score_for_consideration:
                         potential_signatures_with_scores.append({'contour': contour, 'score': score, 'area': area, 'bbox': (x,y,w,h)})

                except Exception as prop_calc_error:
                    # Log error for calculating properties of a specific contour
                    print(f"Error calculating properties for a contour: {prop_calc_error}")
                    # Continue to the next contour
                    pass


        if not potential_signatures_with_scores:
            # Fallback: if scoring found nothing, try basic area and aspect ratio filtering as a last resort
            fallback_candidates = []
            fallback_min_area = 800 # Fallback min area
            fallback_aspect_ratio_range = (0.3, 7.0) # Fallback aspect ratio range

            for contour in contours:
                 area = cv2.contourArea(contour)
                 x, y, w, h = cv2.boundingRect(contour)
                 if area > fallback_min_area and w > 0 and h > 0:
                     aspect_ratio = w / h
                     if fallback_aspect_ratio_range[0] <= aspect_ratio <= fallback_aspect_ratio_range[1]:
                          fallback_candidates.append({'contour': contour, 'score': 0, 'area': area, 'bbox': (x,y,w,h)})

            if fallback_candidates:
                 # Sort fallback candidates by area and take the largest as a last resort
                 fallback_candidates.sort(key=lambda x: x['area'], reverse=True)
                 potential_signatures_with_scores = fallback_candidates # Use fallback candidates if scoring found none
            else:
                raise Exception("No potential signature contours found after filtering, scoring, or fallback.")


        # Sort potential signatures by score in descending order for better priority in unique selection
        potential_signatures_with_scores.sort(key=lambda x: x['score'], reverse=True)

        # Find unique instances based on image content comparison
        unique_signatures_regions_gray = []
        final_signature_contour = None
        similarity_threshold = 0.92 # Adjusted threshold for considering images as duplicates (tune this!)
        compare_size = (150, 75) # Resize images for comparison (tune this!)


        for potential_sig_info in potential_signatures_with_scores:
            contour = potential_sig_info['contour']
            x, y, w, h = potential_sig_info['bbox']

            # Extract the image region from the original color image
            # Ensure bounding box is within image bounds before slicing
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
            for unique_region_gray in unique_signatures_regions_gray:
                if resized_sig_img_gray.shape == unique_region_gray.shape:
                     try:
                         # Using correlation for similarity
                         # Increased the requirement for similarity
                         similarity = cv2.matchTemplate(resized_sig_img_gray, unique_region_gray, cv2.TM_CCOEFF_NORMED)[0,0]
                         if similarity > similarity_threshold:
                             is_duplicate = True
                             break
                     except cv2.error as e:
                         print(f"Error comparing images: {e}")
                         # Continue checking other unique regions despite the error with this one
                         pass


            if not is_duplicate:
                unique_signatures_regions_gray.append(resized_sig_img_gray)
                # We found a unique signature candidate, the first one from the sorted list is the best scoring unique one
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

    except Exception as e:
        # Re-raise the exception after printing, so it can be caught by the Flask app
        print(f"Error in extract_signature_from_image_bytes: {e}")
        raise e

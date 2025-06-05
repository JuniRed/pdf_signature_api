import cv2
import numpy as np
from PIL import Image
import io

def extract_signature_from_image_bytes(image_bytes):
    try:
        # 1. Load and convert image
        image = Image.open(io.BytesIO(image_bytes)).convert("RGB")
        open_cv_image = np.array(image)
        # Check if image is valid before processing
        if open_cv_image is None or open_cv_image.size == 0 or open_cv_image.shape[0] <= 1 or open_cv_image.shape[1] <= 1:
             raise ValueError("Could not load valid image from bytes or image is too small.")

        open_cv_image = open_cv_image[:, :, ::-1].copy()  # RGB to BGR

        gray = cv2.cvtColor(open_cv_image, cv2.COLOR_BGR2GRAY)

        # 2. Binarization
        if gray.shape[0] == 0 or gray.shape[1] == 0:
             raise ValueError("Empty or invalid grayscale image data.")
        # Added check if all pixels are the same, which can cause issues with Otsu
        if np.all(gray == gray[0, 0]):
             # Fallback to a simple binary threshold if image is uniform
             _, thresh = cv2.threshold(gray, 128, 255, cv2.THRESH_BINARY_INV | cv2.THRESH_OTSU) # Still include OSTU flag for consistency, though it might not change anything
        else:
            try:
                _, thresh = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)
            except cv2.error as e:
                 # Fallback if Otsu's fails for some reason
                 print(f"Warning: Otsu's thresholding failed ({e}), falling back to simple threshold.")
                 _, thresh = cv2.threshold(gray, 128, 255, cv2.THRESH_BINARY_INV)


        # 3. Morphological Operations
        # Use a slightly larger kernel for potentially better connection/noise removal
        kernel = np.ones((5,5),np.uint8)
        thresh_cleaned = cv2.dilate(thresh, kernel, iterations = 1)
        thresh_cleaned = cv2.erode(thresh_cleaned, kernel, iterations = 1)


        # 4. Find Contours
        contours, _ = cv2.findContours(thresh_cleaned, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        if not contours:
            raise ValueError("No contours found in the cleaned image after morphological operations.")

        # 5. Filter and Score Contours
        potential_signatures_with_scores = []

        # Define ranges and weights for scoring (THESE ARE EXAMPLES - TUNE FOR YOUR IMAGES)
        # Adjusted ranges to be less strict and cover a wider variety
        area_range = (300, 100000) # Wider example range
        density_range = (0.1, 0.9) # Wider example range
        solidity_range = (0.1, 0.98) # Wider example range
        extent_range = (0.1, 0.98) # Wider example range
        aspect_ratio_range = (0.1, 15.0) # Wider example range

        # Weights for each criterion (adjust to emphasize more important factors)
        weight_area = 0.8 # Slightly reduced weight
        weight_density = 2.5 # Keeping higher weight for density
        weight_solidity = 2.0 # Keeping higher weight for solidity
        weight_extent = 1.5
        weight_aspect_ratio = 0.9


        for contour in contours:
            # Basic filter based on area and minimum dimensions - Made even less strict
            area = cv2.contourArea(contour)
            x, y, w, h = cv2.boundingRect(contour)

            # Lowered minimum area and dimensions significantly
            if area > 50 and w > 3 and h > 3:
                try:
                    # Calculate properties
                    roi = thresh_cleaned[y:y+h, x:x+w]
                    ink_density = cv2.countNonZero(roi) / (w * h) if (w * h) > 0 else 0

                    hull = cv2.convexHull(contour)
                    hull_area = cv2.contourArea(hull)
                    solidity = float(area)/hull_area if hull_area > 0 else 0

                    rect_area = w * h
                    extent = float(area)/rect_area if rect_area > 0 else 0

                    aspect_ratio = w / h if h > 0 else 0

                    # Calculate a score based on how well the properties fit the defined ranges
                    score = 0

                    # Scoring logic: higher score for properties within the preferred ranges
                    # Adjusted scoring to be more forgiving outside the exact center of ranges
                    if area_range[0] <= area <= area_range[1]:
                        score += weight_area * (1 - abs(area - (area_range[0] + area_range[1])/2) / ((area_range[1] - area_range[0])/2 + 1e-6)) # Added epsilon to avoid division by zero
                    else: score += weight_area * 0.1 # Small score even if outside range


                    if density_range[0] <= ink_density <= density_range[1]:
                         score += weight_density * (1 - abs(ink_density - (density_range[0] + density_range[1])/2) / ((density_range[1] - density_range[0])/2 + 1e-6))
                    else: score += weight_density * 0.1


                    if solidity_range[0] <= solidity <= solidity_range[1]:
                         score += weight_solidity * (1 - abs(solidity - (solidity_range[0] + solidity_range[1])/2) / ((solidity_range[1] - solidity_range[0])/2 + 1e-6))
                    # Less harsh penalty for solidity outside the expected range
                    else: score -= weight_solidity * 0.2


                    if extent_range[0] <= extent <= extent_range[1]:
                         score += weight_extent * (1 - abs(extent - (extent_range[0] + extent_range[1])/2) / ((extent_range[1] - extent_range[0])/2 + 1e-6))
                    # Less harsh penalty for extent outside the expected range
                    else: score -= weight_extent * 0.1


                    if aspect_ratio_range[0] <= aspect_ratio <= aspect_ratio_range[1]:
                         score += weight_aspect_ratio * (1 - abs(aspect_ratio - (aspect_ratio_range[0] + aspect_ratio_range[1])/2) / ((aspect_ratio_range[1] - aspect_ratio_range[0])/2 + 1e-6))
                    # Less harsh penalty for extreme aspect ratios
                    else: score -= weight_aspect_ratio * 0.2


                    # Add contour and its score if score is above a minimum threshold
                    # Lowered the minimum score required to be considered a candidate significantly
                    min_candidate_score = (weight_density * 0.2 + weight_solidity * 0.2) # Example: needs only a slight fit in density and solidity
                    if score > min_candidate_score:
                         potential_signatures_with_scores.append({'contour': contour, 'score': score, 'bbox': (x,y,w,h)})

                except Exception as prop_calc_error:
                    # Log error for calculating properties of a specific contour
                    print(f"Error calculating properties for a contour (skipping): {prop_calc_error}")
                    # Continue to the next contour
                    pass


        if not potential_signatures_with_scores:
            # This is the error message you received, made it more specific
            raise ValueError("No contours passed the property filtering and scoring. Consider adjusting area, dimension, range, weight, and min_candidate_score parameters.")


        # Sort potential signatures by score in descending order
        potential_signatures_with_scores.sort(key=lambda x: x['score'], reverse=True)

        # 6. Find Unique Instances based on image content
        unique_signatures_regions_gray = []
        final_signature_contour = None
        # Tune similarity_threshold: lower means more things considered duplicates
        # Tune compare_size: affects performance and precision of comparison
        # Adjusted similarity threshold and compare size again
        similarity_threshold = 0.85 # Lowered threshold for considering images as duplicates (TUNE THIS!)
        compare_size = (80, 40) # Smaller resize dimensions for faster comparison (TUNE THIS!)


        for potential_sig_info in potential_signatures_with_scores:
            contour = potential_sig_info['contour']
            x, y, w, h = potential_sig_info['bbox']

            # Extract the image region from the original color image
            # Ensure bounding box is within image bounds before slicing
            if y >= 0 and y+h <= open_cv_image.shape[0] and x >= 0 and x+w <= open_cv_image.shape[1]:
                 potential_sig_img = open_cv_image[y:y+h, x:x+w]
            else:
                 print(f"Warning: Potential signature bounding box out of image bounds: {potential_sig_info['bbox']} (skipping contour)")
                 continue # Skip invalid bounding boxes


            # Resize for comparison
            if potential_sig_img.shape[0] > 0 and potential_sig_img.shape[1] > 0:
                 try:
                     interpolation_method = cv2.INTER_AREA if potential_sig_img.shape[0] > compare_size[1] or potential_sig_img.shape[1] > compare_size[0] else cv2.INTER_LINEAR
                     resized_sig_img = cv2.resize(potential_sig_img, compare_size, interpolation=interpolation_method)
                     resized_sig_img_gray = cv2.cvtColor(resized_sig_img, cv2.COLOR_BGR2GRAY)
                 except cv2.error as e:
                     print(f"Error resizing image for comparison (skipping contour): {e}")
                     continue # Skip if resize fails
            else:
                 continue # Skip empty regions after slicing

            is_duplicate = False
            # Compare against existing unique regions
            for unique_region_gray in unique_signatures_regions_gray:
                if resized_sig_img_gray.shape == unique_region_gray.shape:
                     try:
                         # Using correlation for similarity (TM_CCOEFF_NORMED is robust to lighting)
                         similarity = cv2.matchTemplate(resized_sig_img_gray, unique_region_gray, cv2.TM_CCOEFF_NORMED)[0,0]
                         if similarity > similarity_threshold:
                             is_duplicate = True
                             break
                     except cv2.error as e:
                         print(f"Error comparing images (skipping comparison): {e}")
                         # Continue checking other unique regions despite the error with this one
                         pass


            if not is_duplicate:
                unique_signatures_regions_gray.append(resized_sig_img_gray)
                # The first unique contour from the sorted list is the best scoring unique one
                final_signature_contour = contour
                # We found the best unique candidate, no need to check lower scoring ones
                break # Exit the loop once the best unique candidate is found


        if final_signature_contour is None:
             # More specific error message if no unique signature found
             raise ValueError(f"No unique signature instances found among {len(potential_signatures_with_scores)} candidates after duplicate filtering. Adjust similarity_threshold or candidate filtering parameters.")


        # 7. Extract the final signature
        x, y, w, h = cv2.boundingRect(final_signature_contour)
        # Ensure final bounding box is within image bounds before slicing
        if y >= 0 and y+h <= open_cv_image.shape[0] and x >= 0 and x+w <= open_cv_image.shape[1]:
            signature = open_cv_image[y:y+h, x:x+w]
        else:
            # This indicates an issue if the contour was valid earlier but bbox is now invalid
            raise RuntimeError("Selected signature bounding box is out of image bounds during final extraction.")


        # 8. Convert extracted signature to bytes
        if signature is None or signature.size == 0:
             raise ValueError("Extracted signature image is empty.")

        sig_image = Image.fromarray(cv2.cvtColor(signature, cv2.COLOR_BGR2RGB))
        sig_io = io.BytesIO()
        # Added error handling for saving the image
        try:
            sig_image.save(sig_io, format='PNG')
        except Exception as save_error:
            raise RuntimeError(f"Failed to save extracted signature image: {save_error}")

        return sig_io.getvalue()

    except Exception as e:
        # Catch any unexpected errors and raise with a more informative message
        print(f"An unexpected error occurred during signature extraction: {e}")
        # Re-raise the exception so the Flask app can catch it and return 500
        raise RuntimeError(f"Signature extraction failed: {e}")

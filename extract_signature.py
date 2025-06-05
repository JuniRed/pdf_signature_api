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
             # Use a mid-range threshold like 128
             _, thresh = cv2.threshold(gray, 128, 255, cv2.THRESH_BINARY_INV)
        else:
            try:
                # Attempt Otsu's thresholding
                _, thresh = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)
            except cv2.error as e:
                 # Fallback if Otsu's fails for some reason (e.g., image too small or uniform)
                 print(f"Warning: Otsu's thresholding failed ({e}), falling back to simple threshold.")
                 _, thresh = cv2.threshold(gray, 128, 255, cv2.THRESH_BINARY_INV)


        # 3. Morphological Operations
        # Use a slightly larger kernel for potentially better connection/noise removal
        kernel = np.ones((3,3),np.uint8) # Keeping kernel size reasonable
        thresh_cleaned = cv2.dilate(thresh, kernel, iterations = 1)
        thresh_cleaned = cv2.erode(thresh_cleaned, kernel, iterations = 1)


        # 4. Find Contours
        contours, _ = cv2.findContours(thresh_cleaned, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        if not contours:
            raise ValueError("No contours found in the cleaned image after morphological operations.")

        # 5. Filter and Score Contours - Parameters adjusted for better scoring precision
        potential_signatures_with_scores = []

        # Define ranges and weights for scoring (THESE ARE EXAMPLES - TUNE FOR YOUR IMAGES)
        # Ranges are still relatively wide, but weights emphasize key signature properties
        area_range = (100, 150000) # Moderately wide area range
        density_range = (0.2, 0.8) # Focus on a typical ink density range
        solidity_range = (0.15, 0.7) # Focus on signatures being less solid
        extent_range = (0.2, 0.8) # Focus on extent range
        aspect_ratio_range = (0.1, 12.0) # Wide aspect ratio range

        # Weights for each criterion - Significantly increased emphasis on density and solidity
        weight_area = 1.0
        weight_density = 4.0 # High weight for density
        weight_solidity = 3.5 # High weight for solidity
        weight_extent = 1.5
        weight_aspect_ratio = 1.0


        for contour in contours:
            # Basic filter based on area and minimum dimensions - Keep it lenient but prevent tiny noise
            area = cv2.contourArea(contour)
            x, y, w, h = cv2.boundingRect(contour)

            if area > 30 and w > 3 and h > 3: # Low minimums, slightly higher than before
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
                    # Added more nuanced scoring based on position within the range
                    def range_score(value, lower, upper):
                        if lower <= value <= upper:
                            mid = (lower + upper) / 2
                            # Score decreases linearly as value moves away from the middle
                            return 1.0 - abs(value - mid) / ((upper - lower)/2 + 1e-6)
                        # Significantly penalize values far outside the range
                        return -0.5 # Negative score for being outside the primary range


                    score += weight_area * range_score(area, area_range[0], area_range[1])
                    score += weight_density * range_score(ink_density, density_range[0], density_range[1])
                    score += weight_solidity * range_score(solidity, solidity_range[0], solidity_range[1])
                    score += weight_extent * range_score(extent, extent_range[0], extent_range[1])
                    score += weight_aspect_ratio * range_score(aspect_ratio, aspect_ratio_range[0], aspect_ratio_range[1])

                    # Additional penalty for very high solidity, which often indicates text blocks
                    if solidity > 0.8:
                         score -= weight_solidity * 1.0 # Stronger penalty

                    # Add contour and its score if score is above a minimum threshold
                    # Slightly increased the minimum score to filter more noise early
                    min_candidate_score = (weight_density * 0.5 + weight_solidity * 0.5) # Example: requires a solid moderate fit in density and solidity
                    if score > min_candidate_score:
                         potential_signatures_with_scores.append({'contour': contour, 'score': score, 'bbox': (x,y,w,h)})

                except Exception as prop_calc_error:
                    # Log error for calculating properties of a specific contour
                    print(f"Error calculating properties for a contour (skipping): {prop_calc_error}")
                    # Continue to the next contour
                    pass


        if not potential_signatures_with_scores:
            # Error message if no contours passed filtering/scoring
            raise ValueError("No contours passed the property filtering and scoring with current parameters. Consider adjusting area, dimension, range, weight, and min_candidate_score parameters.")


        # Sort potential signatures by score in descending order
        potential_signatures_with_scores.sort(key=lambda x: x['score'], reverse=True)

        # 6. Find Unique Instances based on image content - CRITICAL for selecting the correct unique one
        unique_signatures_regions_gray = []
        final_signature_contour = None
        # Tune similarity_threshold and compare_size - These are crucial for identifying duplicate signatures
        similarity_threshold = 0.96 # High threshold for considering images as duplicates (TUNE THIS!)
        compare_size = (180, 80) # Increased resize dimensions for potentially better comparison (TUNE THIS!)


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
                         # Needs high similarity to be considered a duplicate
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
                # The first unique contour from the sorted list (by score) is the best unique candidate
                final_signature_contour = contour
                # We found the best unique candidate based on score and uniqueness, exit loop
                break


        if final_signature_contour is None:
             # Error message if no unique signature found
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

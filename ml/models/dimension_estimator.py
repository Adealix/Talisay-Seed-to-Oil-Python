"""
Advanced Dimension Estimation for Talisay Fruit
Uses reference object detection for accurate real-world measurements
"""

import cv2
import numpy as np
from pathlib import Path
from typing import Tuple, Dict, Optional, Union
from PIL import Image


# Known reference object sizes (in cm)
REFERENCE_OBJECTS = {
    # Philippine Peso Coins - New Generation Currency (NGC) Series 2017-present
    "peso_1_new": {"diameter": 2.0, "name": "₱1 Coin (Silver, New)", "color": "silver"},
    "peso_5_new": {"diameter": 2.4, "name": "₱5 Coin (Silver, New)", "color": "silver"},  # RECOMMENDED
    "peso_10_new": {"diameter": 2.45, "name": "₱10 Coin (Silver/Gold, New)", "color": "bimetallic"},
    "peso_20_new": {"diameter": 2.8, "name": "₱20 Coin (Silver/Gold, New)", "color": "bimetallic"},
    
    # Philippine Peso Coins - Old Series (pre-2017)
    "peso_1_old": {"diameter": 2.4, "name": "₱1 Coin (Brass, Old)", "color": "gold"},
    "peso_5_old": {"diameter": 2.5, "name": "₱5 Coin (Brass, Old)", "color": "gold"},
    "peso_10_old": {"diameter": 2.7, "name": "₱10 Coin (Bimetallic, Old)", "color": "bimetallic"},
    
    # Aliases for convenience
    "peso_1": {"diameter": 2.0, "name": "₱1 Coin (New Silver)", "color": "silver"},
    "peso_5": {"diameter": 2.4, "name": "₱5 Coin (New Silver)", "color": "silver"},
    "peso_10": {"diameter": 2.45, "name": "₱10 Coin (New)", "color": "bimetallic"},
    "peso_20": {"diameter": 2.8, "name": "₱20 Coin (New)", "color": "bimetallic"},
    
    # Other reference objects
    "credit_card": {"width": 8.56, "height": 5.398, "name": "Credit/ID Card"},
    "a4_paper": {"width": 21.0, "height": 29.7, "name": "A4 Paper"},
    "aruco_4x4": {"size": 5.0, "name": "ArUco Marker 5cm"},  # Default ArUco size
}


class DimensionEstimator:
    """
    Estimates real-world dimensions of Talisay fruit from images.
    
    Methods:
    1. Reference Object Method - Uses known-size object (coin, card) in image
    2. ArUco Marker Method - Uses printed ArUco marker for precise calibration
    3. Contour Analysis Method - Estimates relative size (less accurate)
    4. ML-Based Method - Trained regression model (requires training data)
    """
    
    def __init__(self, reference_type: str = "peso_5"):
        """
        Initialize the dimension estimator.
        
        Args:
            reference_type: Type of reference object used
                           ("peso_1", "peso_5", "peso_10", "peso_20", 
                            "credit_card", "aruco_4x4")
        """
        self.reference_type = reference_type
        self.reference_info = REFERENCE_OBJECTS.get(reference_type, REFERENCE_OBJECTS["peso_5"])
        self.pixels_per_cm = None
        self.aruco_dict = None
        self.aruco_params = None
        
        # Initialize ArUco detector if available
        try:
            self.aruco_dict = cv2.aruco.getPredefinedDictionary(cv2.aruco.DICT_4X4_50)
            self.aruco_params = cv2.aruco.DetectorParameters()
        except:
            pass
    
    def estimate_from_image(
        self,
        image: Union[str, Path, np.ndarray, Image.Image],
        reference_method: str = "auto"
    ) -> Dict:
        """
        Estimate fruit dimensions from an image.
        
        Args:
            image: Input image (path, numpy array, or PIL Image)
            reference_method: Detection method
                             "auto" - Try all methods
                             "coin" - Look for circular reference (coin)
                             "card" - Look for rectangular reference (card)
                             "aruco" - Look for ArUco markers
                             "contour" - Contour-based estimation (no reference)
        
        Returns:
            Dictionary with estimated dimensions
        """
        # Load image
        img = self._load_image(image)
        if img is None:
            return {"error": "Could not load image", "success": False}
        
        result = {
            "success": False,
            "method_used": None,
            "pixels_per_cm": None,
            "length_cm": None,
            "width_cm": None,
            "area_cm2": None,
            "fruit_contour": None,
            "reference_detected": False,
            "confidence": 0.0
        }
        
        # Try different methods based on preference
        if reference_method == "auto":
            # Try ArUco first (most accurate)
            aruco_result = self._detect_aruco(img)
            if aruco_result["detected"]:
                result.update(aruco_result)
                result["method_used"] = "aruco"
                result["reference_detected"] = True
            else:
                # Try coin detection
                coin_result = self._detect_coin_reference(img)
                if coin_result["detected"]:
                    result.update(coin_result)
                    result["method_used"] = "coin"
                    result["reference_detected"] = True
                else:
                    # Try card detection
                    card_result = self._detect_card_reference(img)
                    if card_result["detected"]:
                        result.update(card_result)
                        result["method_used"] = "card"
                        result["reference_detected"] = True
                    else:
                        # Fall back to contour estimation
                        contour_result = self._estimate_from_contour(img)
                        result.update(contour_result)
                        result["method_used"] = "contour_estimation"
        
        elif reference_method == "aruco":
            aruco_result = self._detect_aruco(img)
            result.update(aruco_result)
            result["method_used"] = "aruco"
            result["reference_detected"] = aruco_result["detected"]
            
        elif reference_method == "coin":
            coin_result = self._detect_coin_reference(img)
            result.update(coin_result)
            result["method_used"] = "coin"
            result["reference_detected"] = coin_result["detected"]
            
        elif reference_method == "card":
            card_result = self._detect_card_reference(img)
            result.update(card_result)
            result["method_used"] = "card"
            result["reference_detected"] = card_result["detected"]
            
        else:  # contour
            contour_result = self._estimate_from_contour(img)
            result.update(contour_result)
            result["method_used"] = "contour_estimation"
        
        # Estimate fruit dimensions if we have pixels_per_cm
        if result.get("pixels_per_cm"):
            fruit_dims = self._measure_fruit(img, result["pixels_per_cm"])
            result.update(fruit_dims)
            result["success"] = True
        
        return result
    
    def _load_image(self, image) -> Optional[np.ndarray]:
        """Load image from various sources."""
        if isinstance(image, (str, Path)):
            img = cv2.imread(str(image))
            return img
        elif isinstance(image, Image.Image):
            return cv2.cvtColor(np.array(image), cv2.COLOR_RGB2BGR)
        elif isinstance(image, np.ndarray):
            if len(image.shape) == 3 and image.shape[2] == 3:
                return image
            return cv2.cvtColor(image, cv2.COLOR_GRAY2BGR)
        return None
    
    def _detect_aruco(self, img: np.ndarray) -> Dict:
        """Detect ArUco markers for precise calibration."""
        result = {"detected": False}
        
        if self.aruco_dict is None:
            return result
        
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        
        try:
            detector = cv2.aruco.ArucoDetector(self.aruco_dict, self.aruco_params)
            corners, ids, rejected = detector.detectMarkers(gray)
        except:
            # Fallback for older OpenCV versions
            corners, ids, rejected = cv2.aruco.detectMarkers(
                gray, self.aruco_dict, parameters=self.aruco_params
            )
        
        if ids is not None and len(ids) > 0:
            # Get marker size in pixels
            marker_corners = corners[0][0]
            marker_width_px = np.linalg.norm(marker_corners[0] - marker_corners[1])
            marker_height_px = np.linalg.norm(marker_corners[1] - marker_corners[2])
            marker_size_px = (marker_width_px + marker_height_px) / 2
            
            # Calculate pixels per cm
            marker_size_cm = self.reference_info.get("size", 5.0)
            pixels_per_cm = marker_size_px / marker_size_cm
            
            result["detected"] = True
            result["pixels_per_cm"] = pixels_per_cm
            result["marker_id"] = int(ids[0][0])
            result["marker_corners"] = marker_corners
            result["confidence"] = 0.95  # ArUco is very accurate
        
        return result
    
    def _detect_coin_reference(self, img: np.ndarray) -> Dict:
        """
        Detect ₱5 Silver Coin (NEW - 25mm diameter) as size reference.
        
        DETECTION RULES:
        - Circle on LEFT side of image (x < 45% of width)
        - Silver/metallic appearance (low-medium saturation, neutral hue)
        - Very uniform surface (std_gray < 45) - KEY differentiator from fruits
        - Reasonable size (6-20% of image width)
        
        COIN SPECIFICATIONS:
        - ₱5 New Silver Coin: 25mm diameter
        
        RECOMMENDED PHOTO SETUP:
        - Coin on LEFT side, fruit on RIGHT side
        - Fill 60-80% of frame with coin and fruit
        - Use plain white/neutral background
        - Take photo from directly above (top-down view)
        """
        result = {"detected": False, "coin_type": None}
        
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
        
        # Apply bilateral filter to reduce noise while preserving edges
        blurred = cv2.bilateralFilter(gray, 9, 75, 75)
        
        h, w = img.shape[:2]
        
        # Calculate radius range based on image size
        # Coin should be 6-20% of image width for proper detection
        min_radius = max(20, int(w * 0.03))
        max_radius = min(250, int(w * 0.12))
        
        # Try multiple Hough parameters for robustness
        all_circles = []
        for param2 in [35, 45, 55]:
            circles = cv2.HoughCircles(
                blurred,
                cv2.HOUGH_GRADIENT,
                dp=1.2,
                minDist=80,
                param1=100,
                param2=param2,
                minRadius=min_radius,
                maxRadius=max_radius
            )
            if circles is not None:
                for c in circles[0]:
                    all_circles.append(c)
        
        if not all_circles:
            return result
        
        # Remove duplicate circles (same center within 20px)
        unique_circles = []
        for c in all_circles:
            is_duplicate = False
            for uc in unique_circles:
                dist = np.sqrt((c[0] - uc[0])**2 + (c[1] - uc[1])**2)
                if dist < 20:
                    is_duplicate = True
                    break
            if not is_duplicate:
                unique_circles.append(c)
        
        # Find the best coin candidate
        best_circle = None
        best_score = 0
        
        for circle in unique_circles:
            x, y, r = int(circle[0]), int(circle[1]), int(circle[2])
            diameter_px = 2 * r
            
            # RULE 1: Circle must be fully within image bounds (with margin)
            margin = 10
            if x - r < margin or x + r >= w - margin or y - r < margin or y + r >= h - margin:
                continue
            
            # RULE 2: Must be on LEFT 45% of image
            position_x_ratio = x / w
            if position_x_ratio > 0.45:
                continue
            
            # RULE 3: Size must be reasonable (6-20% of image width)
            size_ratio = diameter_px / w
            if not (0.06 < size_ratio < 0.22):
                continue
            
            # Analyze the circular region
            mask = np.zeros(gray.shape, dtype=np.uint8)
            cv2.circle(mask, (x, y), r, 255, -1)
            region_gray = gray[mask > 0]
            region_hsv = hsv[mask > 0]
            
            if len(region_gray) < 300:
                continue
            
            mean_hue = np.mean(region_hsv[:, 0])
            mean_sat = np.mean(region_hsv[:, 1])
            mean_val = np.mean(region_hsv[:, 2])
            std_gray = np.std(region_gray)
            
            # Compute gradient magnitude (coins have texture from minting)
            gradient = cv2.Laplacian(gray, cv2.CV_64F)
            gradient_std = np.std(gradient[mask > 0])
            
            # ===== SILVER COIN DETECTION =====
            # Silver coin characteristics (under typical indoor/warm lighting):
            # 1. Medium saturation (50-85) - metallic with warm reflection
            # 2. Neutral hue (< 30 or > 165) - gray/silver base tones
            # 3. Very uniform (std_gray < 45) - KEY differentiator from fruit
            # 4. Moderate brightness (45 < V < 165) - not washed out
            # 5. Gradient std > 35 - real coins have embossing/texture
            
            # REJECT: Definite green (plants/fruits)
            if 35 < mean_hue < 90 and mean_sat > 25:
                continue
            
            # REJECT: Too much texture (fruits have more texture)
            if std_gray > 50:
                continue
            
            # REJECT: Washed-out background areas (too bright)
            if mean_val > 165:
                continue
            
            # REJECT: Too dark areas
            if mean_val < 45:
                continue
            
            # REJECT: Featureless areas (no coin embossing/details)
            # Real coins have gradient_std > 35 from minting patterns
            if gradient_std < 35:
                continue
            
            # REJECT: Low saturation areas (washed out gray backgrounds)
            # Real coins under warm lighting have saturation > 55
            if mean_sat < 55:
                continue
            
            # CHECK: Is this silver/metallic coin?
            is_neutral_hue = (mean_hue < 30) or (mean_hue > 165)
            is_medium_saturation = 55 < mean_sat < 90  # Coins have moderate saturation
            is_uniform = std_gray < 45
            is_reasonable_brightness = 45 < mean_val < 165  # Not too bright or dark
            is_textured_coin = gradient_std > 35  # Has coin embossing
            
            if not (is_neutral_hue and is_medium_saturation and is_uniform and 
                    is_reasonable_brightness and is_textured_coin):
                continue
            
            # Calculate confidence score
            score = 0.0
            
            # Position score: prefer left side (0.20 max)
            position_score = (0.45 - position_x_ratio) / 0.45
            score += position_score * 0.20
            
            # Uniformity score: lower std is better (0.25 max)
            uniformity_score = max(0, 1.0 - std_gray / 50)
            score += uniformity_score * 0.25
            
            # Metallic score: lower saturation is more metallic (0.20 max)
            metallic_score = max(0, 1.0 - mean_sat / 100)
            score += metallic_score * 0.20
            
            # Texture score: coins have gradient_std 40-100 (0.15 max)
            if 40 <= gradient_std <= 100:
                score += 0.15
            elif gradient_std > 35:
                score += 0.08
            
            # Size score: optimal is 8-15% of width (0.20 max)
            if 0.08 < size_ratio < 0.16:
                score += 0.20
            elif 0.06 < size_ratio < 0.22:
                score += 0.12
            else:
                score += 0.05
            
            if score > best_score:
                best_score = score
                best_circle = (x, y, r)
        
        # Require minimum score of 0.45 to confirm detection
        if best_circle is not None and best_score >= 0.45:
            x, y, r = best_circle
            diameter_px = 2 * r
            
            # ₱5 Silver Coin (NEW): 25mm diameter
            coin_diameter_mm = 25
            coin_diameter_cm = 2.5
            
            pixels_per_mm = diameter_px / coin_diameter_mm
            pixels_per_cm = pixels_per_mm * 10
            
            result["detected"] = True
            result["pixels_per_cm"] = pixels_per_cm
            result["coin_center"] = (x, y)
            result["coin_radius"] = r
            result["coin_diameter_px"] = diameter_px
            result["coin_type"] = "silver_new"
            result["coin_name"] = "₱5 Silver Coin (25mm)"
            result["coin_diameter_cm"] = coin_diameter_cm
            result["detection_score"] = best_score
            result["confidence"] = min(0.95, 0.5 + best_score * 0.5)
        
        return result
    
    def _detect_card_reference(self, img: np.ndarray) -> Dict:
        """Detect rectangular reference object (credit card, ID card)."""
        result = {"detected": False}
        
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        blurred = cv2.GaussianBlur(gray, (5, 5), 0)
        edges = cv2.Canny(blurred, 50, 150)
        
        # Find contours
        contours, _ = cv2.findContours(edges, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        
        # Look for rectangle with credit card aspect ratio (~1.586)
        card_ratio = 8.56 / 5.398  # Standard card ratio
        
        best_rect = None
        best_score = 0
        
        for contour in contours:
            # Approximate contour to polygon
            peri = cv2.arcLength(contour, True)
            approx = cv2.approxPolyDP(contour, 0.02 * peri, True)
            
            # Check if it's a quadrilateral
            if len(approx) == 4:
                # Get bounding rectangle
                rect = cv2.minAreaRect(contour)
                width, height = rect[1]
                
                if width > 0 and height > 0:
                    # Ensure width > height
                    if width < height:
                        width, height = height, width
                    
                    ratio = width / height
                    ratio_diff = abs(ratio - card_ratio)
                    
                    # Check if ratio is close to card ratio
                    if ratio_diff < 0.2:
                        # Score based on size and ratio match
                        area = width * height
                        area_ratio = area / (img.shape[0] * img.shape[1])
                        
                        if 0.01 < area_ratio < 0.3:  # Reasonable size
                            score = (1 - ratio_diff) * 0.7 + (1 - abs(area_ratio - 0.05)) * 0.3
                            
                            if score > best_score:
                                best_score = score
                                best_rect = rect
        
        if best_rect is not None and best_score > 0.5:
            width_px, height_px = best_rect[1]
            if width_px < height_px:
                width_px, height_px = height_px, width_px
            
            # Calculate pixels per cm using card width
            card_width_cm = self.reference_info.get("width", 8.56)
            pixels_per_cm = width_px / card_width_cm
            
            result["detected"] = True
            result["pixels_per_cm"] = pixels_per_cm
            result["card_rect"] = best_rect
            result["confidence"] = min(0.80, 0.4 + best_score * 0.5)
        
        return result
    
    def _estimate_from_contour(self, img: np.ndarray) -> Dict:
        """
        Estimate dimensions without reference object.
        Uses typical Talisay fruit sizes as prior.
        Less accurate - provides range estimate.
        """
        result = {
            "detected": False,
            "warning": "No reference object detected. Using statistical estimation."
        }
        
        # Segment the fruit
        fruit_contour, fruit_mask = self._segment_fruit(img)
        
        if fruit_contour is not None:
            # Get bounding ellipse
            if len(fruit_contour) >= 5:
                ellipse = cv2.fitEllipse(fruit_contour)
                (cx, cy), (minor_axis, major_axis), angle = ellipse
                
                # Fruit typically fills 30-70% of a well-framed photo
                # Use this to estimate rough scale
                img_diagonal = np.sqrt(img.shape[0]**2 + img.shape[1]**2)
                fruit_major_ratio = major_axis / img_diagonal
                
                # Typical Talisay: 3.5-7.0 cm length
                # If fruit is ~50% of diagonal, assume typical framing
                estimated_length = 5.0  # Default average
                
                if 0.2 < fruit_major_ratio < 0.8:
                    # Scale based on how fruit fills frame
                    # Assume typical phone photo framing
                    estimated_length = 5.0 * (fruit_major_ratio / 0.4)
                    estimated_length = np.clip(estimated_length, 3.5, 7.0)
                
                # Estimate pixels per cm
                pixels_per_cm = major_axis / estimated_length
                
                result["detected"] = True
                result["pixels_per_cm"] = pixels_per_cm
                result["fruit_contour"] = fruit_contour
                result["confidence"] = 0.40  # Low confidence without reference
                result["estimation_method"] = "statistical_prior"
                result["note"] = "For accurate measurements, include a coin or reference object"
        
        return result
    
    def _segment_fruit(self, img: np.ndarray) -> Tuple[Optional[np.ndarray], Optional[np.ndarray]]:
        """Segment the fruit from background."""
        # Convert to HSV
        hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
        
        # Create mask for green/yellow/brown colors (fruit colors)
        # Green range
        mask_green = cv2.inRange(hsv, (25, 30, 30), (90, 255, 255))
        # Yellow range
        mask_yellow = cv2.inRange(hsv, (15, 50, 50), (35, 255, 255))
        # Brown range
        mask_brown = cv2.inRange(hsv, (5, 30, 30), (25, 200, 200))
        
        # Combine masks
        fruit_mask = mask_green | mask_yellow | mask_brown
        
        # Morphological operations
        kernel = np.ones((7, 7), np.uint8)
        fruit_mask = cv2.morphologyEx(fruit_mask, cv2.MORPH_CLOSE, kernel)
        fruit_mask = cv2.morphologyEx(fruit_mask, cv2.MORPH_OPEN, kernel)
        
        # Find contours
        contours, _ = cv2.findContours(fruit_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        
        if contours:
            # Get largest contour (likely the fruit)
            largest_contour = max(contours, key=cv2.contourArea)
            
            # Minimum area threshold
            min_area = img.shape[0] * img.shape[1] * 0.01  # At least 1% of image
            
            if cv2.contourArea(largest_contour) > min_area:
                return largest_contour, fruit_mask
        
        return None, None
    
    def _measure_fruit(self, img: np.ndarray, pixels_per_cm: float) -> Dict:
        """Measure fruit dimensions using the calibrated scale."""
        result = {}
        
        # Segment fruit
        fruit_contour, fruit_mask = self._segment_fruit(img)
        
        if fruit_contour is not None and len(fruit_contour) >= 5:
            # Fit ellipse to get major/minor axes
            ellipse = cv2.fitEllipse(fruit_contour)
            (cx, cy), (minor_axis_px, major_axis_px), angle = ellipse
            
            # Ensure major > minor
            if major_axis_px < minor_axis_px:
                major_axis_px, minor_axis_px = minor_axis_px, major_axis_px
            
            # Convert to cm
            length_cm = major_axis_px / pixels_per_cm
            width_cm = minor_axis_px / pixels_per_cm
            
            # Clip to valid Talisay ranges
            length_cm = np.clip(length_cm, 3.0, 8.0)
            width_cm = np.clip(width_cm, 1.5, 6.0)
            
            # Calculate area
            area_cm2 = np.pi * (length_cm / 2) * (width_cm / 2)
            
            # Estimate weight from dimensions (empirical formula)
            # Talisay fruits are roughly ellipsoidal
            volume_cm3 = (4/3) * np.pi * (length_cm/2) * (width_cm/2) * (width_cm/2 * 0.8)
            estimated_weight_g = volume_cm3 * 0.85  # Approximate density
            estimated_weight_g = np.clip(estimated_weight_g, 15.0, 60.0)
            
            # Estimate kernel mass (correlates with fruit size)
            kernel_mass_g = 0.1 + (length_cm * width_cm / 35) * 0.6
            kernel_mass_g = np.clip(kernel_mass_g, 0.1, 0.9)
            
            result["length_cm"] = round(length_cm, 2)
            result["width_cm"] = round(width_cm, 2)
            result["area_cm2"] = round(area_cm2, 2)
            result["estimated_weight_g"] = round(estimated_weight_g, 1)
            result["estimated_kernel_mass_g"] = round(kernel_mass_g, 3)
            result["fruit_contour"] = fruit_contour
            result["ellipse"] = ellipse
        
        return result
    
    def visualize_measurement(
        self,
        image: Union[str, Path, np.ndarray],
        result: Dict,
        output_path: str = None
    ) -> np.ndarray:
        """
        Create visualization of the measurement.
        
        Args:
            image: Input image
            result: Result from estimate_from_image()
            output_path: Optional path to save visualization
            
        Returns:
            Annotated image as numpy array
        """
        img = self._load_image(image)
        if img is None:
            return None
        
        vis = img.copy()
        
        # Draw reference object if detected
        if result.get("method_used") == "coin" and "coin_center" in result:
            x, y = result["coin_center"]
            r = result["coin_radius"]
            cv2.circle(vis, (x, y), r, (0, 255, 255), 3)
            cv2.putText(vis, f"Reference: {self.reference_info['name']}", 
                       (x - r, y - r - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 255), 2)
        
        elif result.get("method_used") == "aruco" and "marker_corners" in result:
            corners = result["marker_corners"].astype(int)
            cv2.polylines(vis, [corners], True, (0, 255, 255), 3)
            cv2.putText(vis, f"ArUco #{result.get('marker_id', '?')}", 
                       (corners[0][0], corners[0][1] - 10), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 255), 2)
        
        # Draw fruit contour and measurements
        if "fruit_contour" in result and result["fruit_contour"] is not None:
            cv2.drawContours(vis, [result["fruit_contour"]], -1, (0, 255, 0), 2)
            
            if "ellipse" in result:
                cv2.ellipse(vis, result["ellipse"], (255, 0, 255), 2)
        
        # Add measurement text
        y_offset = 30
        texts = []
        
        if result.get("length_cm"):
            texts.append(f"Length: {result['length_cm']:.2f} cm")
        if result.get("width_cm"):
            texts.append(f"Width: {result['width_cm']:.2f} cm")
        if result.get("estimated_weight_g"):
            texts.append(f"Est. Weight: {result['estimated_weight_g']:.1f} g")
        if result.get("confidence"):
            texts.append(f"Confidence: {result['confidence']*100:.0f}%")
        if result.get("method_used"):
            texts.append(f"Method: {result['method_used']}")
        
        for text in texts:
            cv2.putText(vis, text, (10, y_offset), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
            y_offset += 30
        
        if output_path:
            cv2.imwrite(output_path, vis)
        
        return vis


def create_aruco_reference(
    marker_id: int = 0,
    marker_size_cm: float = 5.0,
    output_path: str = "aruco_reference.png",
    dpi: int = 300
) -> str:
    """
    Generate a printable ArUco marker for reference.
    
    Args:
        marker_id: ArUco marker ID (0-49 for 4x4_50 dictionary)
        marker_size_cm: Physical size of marker in cm
        output_path: Where to save the marker image
        dpi: Print resolution
        
    Returns:
        Path to generated marker image
    """
    # Calculate pixel size for given cm and dpi
    inches = marker_size_cm / 2.54
    pixels = int(inches * dpi)
    
    # Generate marker
    aruco_dict = cv2.aruco.getPredefinedDictionary(cv2.aruco.DICT_4X4_50)
    marker_img = cv2.aruco.generateImageMarker(aruco_dict, marker_id, pixels)
    
    # Add white border
    border = pixels // 10
    marker_with_border = cv2.copyMakeBorder(
        marker_img, border, border, border, border,
        cv2.BORDER_CONSTANT, value=255
    )
    
    # Add size label
    label = f"{marker_size_cm}cm ArUco Marker (ID: {marker_id})"
    cv2.putText(marker_with_border, label,
               (border, marker_with_border.shape[0] - border // 2),
               cv2.FONT_HERSHEY_SIMPLEX, 0.5 * (pixels / 300), 0, 2)
    
    cv2.imwrite(output_path, marker_with_border)
    print(f"ArUco reference marker saved to: {output_path}")
    print(f"Print at 100% scale for {marker_size_cm}cm marker")
    
    return output_path


if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description="Estimate Talisay fruit dimensions")
    parser.add_argument("image", help="Path to fruit image")
    parser.add_argument("--reference", default="peso_5", 
                       choices=list(REFERENCE_OBJECTS.keys()),
                       help="Type of reference object in image")
    parser.add_argument("--method", default="auto",
                       choices=["auto", "coin", "card", "aruco", "contour"],
                       help="Detection method")
    parser.add_argument("--output", help="Output visualization path")
    parser.add_argument("--generate-aruco", action="store_true",
                       help="Generate printable ArUco marker")
    
    args = parser.parse_args()
    
    if args.generate_aruco:
        create_aruco_reference(marker_id=0, marker_size_cm=5.0)
    else:
        estimator = DimensionEstimator(reference_type=args.reference)
        result = estimator.estimate_from_image(args.image, reference_method=args.method)
        
        print("\n=== Dimension Estimation Results ===")
        print(f"Method: {result.get('method_used', 'N/A')}")
        print(f"Reference Detected: {result.get('reference_detected', False)}")
        print(f"Confidence: {result.get('confidence', 0)*100:.1f}%")
        print()
        
        if result.get("success"):
            print(f"Length: {result.get('length_cm', 'N/A')} cm")
            print(f"Width: {result.get('width_cm', 'N/A')} cm")
            print(f"Area: {result.get('area_cm2', 'N/A')} cm²")
            print(f"Est. Weight: {result.get('estimated_weight_g', 'N/A')} g")
            print(f"Est. Kernel Mass: {result.get('estimated_kernel_mass_g', 'N/A')} g")
        else:
            print("Could not estimate dimensions")
            if "warning" in result:
                print(f"Warning: {result['warning']}")
        
        if args.output:
            vis = estimator.visualize_measurement(args.image, result, args.output)
            print(f"\nVisualization saved to: {args.output}")

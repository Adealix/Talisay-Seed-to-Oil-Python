"""
Talisay Fruit Validator
Validates whether an image contains a Talisay (Terminalia catappa) fruit

This module handles:
1. Detection: Is there any fruit in the image?
2. Identification: Is the detected fruit a Talisay fruit?
3. Validation: Does it match expected Talisay characteristics?

Talisay Fruit Characteristics:
- Colors: Green (immature), Yellow (mature), Brown (fully ripe)
- Shape: Almond-shaped / elliptical with pointed ends
- Size: Typically 3.5-7.0 cm in length
- Surface: Relatively smooth with possible dark spots
"""

import cv2
import numpy as np
from pathlib import Path
from typing import Dict, Optional, Tuple, Union, List
from PIL import Image
from enum import Enum


class FruitDetectionResult(Enum):
    """Result of fruit detection."""
    NO_FRUIT = "no_fruit"           # No fruit detected in image
    TALISAY_FRUIT = "talisay"       # Valid Talisay fruit
    UNKNOWN_FRUIT = "unknown"       # Fruit detected but not Talisay
    UNCERTAIN = "uncertain"          # Cannot determine with confidence


class TalisayValidator:
    """
    Validates whether an image contains a Talisay fruit.
    
    Uses multiple detection strategies:
    1. Color analysis (Talisay-specific HSV ranges)
    2. Shape analysis (almond/elliptical shape)
    3. Texture analysis (smooth surface, possible spots)
    4. Size ratio analysis (reasonable fruit proportions)
    """
    
    def __init__(self):
        """Initialize the Talisay validator."""
        # Talisay-specific color ranges (HSV)
        # These colors distinguish Talisay from other objects
        self.talisay_colors = {
            "green": {
                "lower": np.array([25, 30, 30]),
                "upper": np.array([90, 255, 255]),
                "description": "Immature Talisay"
            },
            "yellow": {
                "lower": np.array([15, 50, 50]),
                "upper": np.array([35, 255, 255]),
                "description": "Mature Talisay"
            },
            "brown": {
                "lower": np.array([5, 30, 30]),
                "upper": np.array([25, 200, 200]),
                "description": "Fully Ripe Talisay"
            }
        }
        
        # Non-Talisay colors that might appear
        self.exclude_colors = {
            "blue": {"lower": np.array([100, 50, 50]), "upper": np.array([130, 255, 255])},
            "red": {"lower": np.array([0, 100, 100]), "upper": np.array([10, 255, 255])},
            "red_wrap": {"lower": np.array([160, 100, 100]), "upper": np.array([180, 255, 255])},
            "purple": {"lower": np.array([130, 50, 50]), "upper": np.array([160, 255, 255])},
            "white": {"lower": np.array([0, 0, 200]), "upper": np.array([180, 30, 255])},
            "black": {"lower": np.array([0, 0, 0]), "upper": np.array([180, 255, 40])},
        }
        
        # Shape parameters for Talisay
        self.shape_params = {
            "min_circularity": 0.35,    # Almond shape is less circular than a circle
            "max_circularity": 0.95,    # But not too irregular
            "min_aspect_ratio": 1.1,    # Slightly elongated
            "max_aspect_ratio": 2.5,    # Not too stretched
            "min_convexity": 0.85,      # Relatively convex (no big concavities)
        }
        
        # Size constraints (relative to image)
        self.size_constraints = {
            "min_area_ratio": 0.02,     # At least 2% of image
            "max_area_ratio": 0.70,     # At most 70% of image
        }
    
    def validate(
        self, 
        image: Union[str, Path, np.ndarray, Image.Image],
        segmentation_mask: Optional[np.ndarray] = None,
        return_details: bool = True
    ) -> Dict:
        """
        Validate whether the image contains a Talisay fruit.
        
        Args:
            image: Input image
            segmentation_mask: Optional pre-computed mask
            return_details: Include detailed analysis in result
            
        Returns:
            Dictionary with:
            - is_talisay: bool - True if Talisay fruit detected
            - result: FruitDetectionResult enum value
            - confidence: float - Detection confidence (0-1)
            - message: str - Human-readable result message
            - details: dict - Detailed analysis (if return_details=True)
        """
        # Load image
        img = self._load_image(image)
        if img is None:
            return {
                "is_talisay": False,
                "result": FruitDetectionResult.NO_FRUIT.value,
                "confidence": 0.0,
                "message": "Could not load image",
                "details": {}
            }
        
        h, w = img.shape[:2]
        
        # Step 1: Detect potential fruit regions
        fruit_mask, fruit_info = self._detect_fruit_regions(img, segmentation_mask)
        
        if fruit_mask is None or np.sum(fruit_mask) < 100:
            return {
                "is_talisay": False,
                "result": FruitDetectionResult.NO_FRUIT.value,
                "confidence": 0.95,
                "message": "No fruit detected in image. Please take a photo with a Talisay fruit clearly visible.",
                "details": {"detection_step": "no_fruit_regions"}
            }
        
        # Step 2: Analyze color distribution
        color_result = self._analyze_color(img, fruit_mask)
        
        # Step 3: Analyze shape
        shape_result = self._analyze_shape(fruit_mask)
        
        # Step 4: Check size ratio
        size_result = self._analyze_size(fruit_mask, h, w)
        
        # Step 5: Compute overall validation score
        validation_score = self._compute_validation_score(
            color_result, shape_result, size_result
        )
        
        # Determine final result
        if validation_score >= 0.70:
            result = FruitDetectionResult.TALISAY_FRUIT
            is_talisay = True
            message = f"Talisay fruit detected ({color_result['dominant_color']} - {color_result['maturity']})"
        elif validation_score >= 0.40:
            result = FruitDetectionResult.UNCERTAIN
            is_talisay = False
            message = "Object detected but uncertain if it's a Talisay fruit. Ensure the fruit is clearly visible with good lighting."
        elif color_result["has_fruit_colors"] and not color_result["is_talisay_color"]:
            result = FruitDetectionResult.UNKNOWN_FRUIT
            is_talisay = False
            message = f"Fruit detected but it doesn't appear to be a Talisay fruit. Talisay should be green, yellow, or brown colored."
        else:
            result = FruitDetectionResult.NO_FRUIT
            is_talisay = False
            message = "No valid Talisay fruit found. Please provide an image with a clearly visible Talisay fruit."
        
        response = {
            "is_talisay": is_talisay,
            "result": result.value,
            "confidence": round(validation_score, 3),
            "message": message
        }
        
        if return_details:
            response["details"] = {
                "color_analysis": color_result,
                "shape_analysis": shape_result,
                "size_analysis": size_result,
                "validation_score": validation_score,
                "fruit_mask_area": int(np.sum(fruit_mask > 0))
            }
        
        return response
    
    def _load_image(self, image) -> Optional[np.ndarray]:
        """Load image from various sources."""
        if isinstance(image, (str, Path)):
            return cv2.imread(str(image))
        elif isinstance(image, Image.Image):
            return cv2.cvtColor(np.array(image), cv2.COLOR_RGB2BGR)
        elif isinstance(image, np.ndarray):
            if len(image.shape) == 3 and image.shape[2] == 3:
                return image.copy()
            elif len(image.shape) == 2:
                return cv2.cvtColor(image, cv2.COLOR_GRAY2BGR)
        return None
    
    def _detect_fruit_regions(
        self, 
        img: np.ndarray, 
        provided_mask: Optional[np.ndarray]
    ) -> Tuple[Optional[np.ndarray], Dict]:
        """Detect potential fruit regions in the image."""
        h, w = img.shape[:2]
        info = {}
        
        if provided_mask is not None:
            return provided_mask, {"source": "provided"}
        
        # Convert to HSV for color analysis
        hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
        
        # Create mask for all Talisay colors
        fruit_mask = np.zeros((h, w), dtype=np.uint8)
        
        for color_name, ranges in self.talisay_colors.items():
            color_mask = cv2.inRange(hsv, ranges["lower"], ranges["upper"])
            fruit_mask = cv2.bitwise_or(fruit_mask, color_mask)
        
        # Remove excluded colors (background, non-fruit objects)
        for exc_name, exc_range in self.exclude_colors.items():
            exclude_mask = cv2.inRange(hsv, exc_range["lower"], exc_range["upper"])
            fruit_mask = cv2.bitwise_and(fruit_mask, cv2.bitwise_not(exclude_mask))
        
        # Morphological operations to clean up
        kernel = np.ones((7, 7), np.uint8)
        fruit_mask = cv2.morphologyEx(fruit_mask, cv2.MORPH_CLOSE, kernel)
        fruit_mask = cv2.morphologyEx(fruit_mask, cv2.MORPH_OPEN, kernel)
        
        # Find largest contour (main fruit)
        contours, _ = cv2.findContours(fruit_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        
        if contours:
            # Keep only significant contours
            min_area = h * w * 0.01
            valid_contours = [c for c in contours if cv2.contourArea(c) > min_area]
            
            if valid_contours:
                # Create mask from valid contours
                clean_mask = np.zeros((h, w), dtype=np.uint8)
                cv2.drawContours(clean_mask, valid_contours, -1, 255, -1)
                info["source"] = "color_detection"
                info["num_regions"] = len(valid_contours)
                return clean_mask, info
        
        return None, {"source": "failed"}
    
    def _analyze_color(self, img: np.ndarray, mask: np.ndarray) -> Dict:
        """Analyze color distribution in the fruit region."""
        hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
        
        # Get pixels within fruit mask
        fruit_pixels = hsv[mask > 0]
        
        if len(fruit_pixels) == 0:
            return {
                "has_fruit_colors": False,
                "is_talisay_color": False,
                "dominant_color": "unknown",
                "maturity": "unknown",
                "color_confidence": 0.0,
                "color_distribution": {}
            }
        
        # Count pixels matching each Talisay color
        color_counts = {}
        for color_name, ranges in self.talisay_colors.items():
            in_range = np.all(
                (fruit_pixels >= ranges["lower"]) & (fruit_pixels <= ranges["upper"]),
                axis=1
            )
            color_counts[color_name] = np.sum(in_range)
        
        total_pixels = len(fruit_pixels)
        total_talisay_pixels = sum(color_counts.values())
        
        # Calculate percentages
        color_distribution = {
            color: round(count / total_pixels * 100, 1)
            for color, count in color_counts.items()
        }
        
        talisay_coverage = total_talisay_pixels / total_pixels if total_pixels > 0 else 0
        
        # Determine dominant color
        if total_talisay_pixels > 0:
            dominant_color = max(color_counts, key=color_counts.get)
            maturity_map = {
                "green": "Immature",
                "yellow": "Mature (Optimal)",
                "brown": "Fully Ripe"
            }
            maturity = maturity_map.get(dominant_color, "Unknown")
        else:
            dominant_color = "unknown"
            maturity = "Unknown"
        
        # Determine if colors match Talisay
        is_talisay_color = talisay_coverage >= 0.30  # At least 30% Talisay colors
        has_fruit_colors = talisay_coverage >= 0.15  # At least 15% fruit-like colors
        
        return {
            "has_fruit_colors": has_fruit_colors,
            "is_talisay_color": is_talisay_color,
            "dominant_color": dominant_color,
            "maturity": maturity,
            "color_confidence": round(talisay_coverage, 3),
            "color_distribution": color_distribution,
            "talisay_coverage_percent": round(talisay_coverage * 100, 1)
        }
    
    def _analyze_shape(self, mask: np.ndarray) -> Dict:
        """Analyze the shape of the detected fruit region."""
        contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        
        if not contours:
            return {
                "is_valid_shape": False,
                "shape_confidence": 0.0,
                "circularity": 0.0,
                "aspect_ratio": 0.0,
                "convexity": 0.0
            }
        
        # Get largest contour
        contour = max(contours, key=cv2.contourArea)
        area = cv2.contourArea(contour)
        perimeter = cv2.arcLength(contour, True)
        
        # Calculate circularity: 4π * area / perimeter²
        circularity = (4 * np.pi * area) / (perimeter ** 2) if perimeter > 0 else 0
        
        # Calculate aspect ratio from fitted ellipse
        if len(contour) >= 5:
            ellipse = cv2.fitEllipse(contour)
            (_, (minor, major), _) = ellipse
            aspect_ratio = major / minor if minor > 0 else 0
        else:
            # Fallback to bounding rect
            x, y, w, h = cv2.boundingRect(contour)
            aspect_ratio = max(w, h) / min(w, h) if min(w, h) > 0 else 0
        
        # Calculate convexity: area / convex hull area
        hull = cv2.convexHull(contour)
        hull_area = cv2.contourArea(hull)
        convexity = area / hull_area if hull_area > 0 else 0
        
        # Validate shape parameters
        is_valid_shape = (
            self.shape_params["min_circularity"] <= circularity <= self.shape_params["max_circularity"]
            and self.shape_params["min_aspect_ratio"] <= aspect_ratio <= self.shape_params["max_aspect_ratio"]
            and convexity >= self.shape_params["min_convexity"]
        )
        
        # Calculate shape confidence
        shape_score = 0.0
        
        # Circularity score (0-0.4)
        if self.shape_params["min_circularity"] <= circularity <= self.shape_params["max_circularity"]:
            shape_score += 0.4
        elif circularity > 0.2:
            shape_score += 0.2
        
        # Aspect ratio score (0-0.35)
        if self.shape_params["min_aspect_ratio"] <= aspect_ratio <= self.shape_params["max_aspect_ratio"]:
            shape_score += 0.35
        elif 1.0 <= aspect_ratio <= 3.0:
            shape_score += 0.15
        
        # Convexity score (0-0.25)
        if convexity >= self.shape_params["min_convexity"]:
            shape_score += 0.25
        elif convexity >= 0.7:
            shape_score += 0.1
        
        return {
            "is_valid_shape": is_valid_shape,
            "shape_confidence": round(shape_score, 3),
            "circularity": round(circularity, 3),
            "aspect_ratio": round(aspect_ratio, 3),
            "convexity": round(convexity, 3)
        }
    
    def _analyze_size(self, mask: np.ndarray, h: int, w: int) -> Dict:
        """Analyze the size of the fruit relative to image."""
        total_pixels = h * w
        fruit_pixels = np.sum(mask > 0)
        area_ratio = fruit_pixels / total_pixels if total_pixels > 0 else 0
        
        is_valid_size = (
            self.size_constraints["min_area_ratio"] <= area_ratio <= self.size_constraints["max_area_ratio"]
        )
        
        # Size confidence
        if is_valid_size:
            # Optimal is 5-40% of image
            if 0.05 <= area_ratio <= 0.40:
                size_confidence = 1.0
            else:
                size_confidence = 0.7
        else:
            size_confidence = 0.3 if area_ratio > 0.01 else 0.0
        
        return {
            "is_valid_size": is_valid_size,
            "size_confidence": round(size_confidence, 3),
            "area_ratio": round(area_ratio, 4),
            "area_percent": round(area_ratio * 100, 2)
        }
    
    def _compute_validation_score(
        self, 
        color_result: Dict, 
        shape_result: Dict, 
        size_result: Dict
    ) -> float:
        """Compute overall validation score."""
        weights = {
            "color": 0.50,   # Color is most important for Talisay ID
            "shape": 0.30,   # Shape helps confirm
            "size": 0.20     # Size is a sanity check
        }
        
        color_score = color_result["color_confidence"] if color_result["is_talisay_color"] else 0.0
        shape_score = shape_result["shape_confidence"]
        size_score = size_result["size_confidence"] if size_result["is_valid_size"] else 0.0
        
        total = (
            weights["color"] * color_score +
            weights["shape"] * shape_score +
            weights["size"] * size_score
        )
        
        return round(total, 3)


def get_coin_mask(
    image: np.ndarray,
    coin_center: Tuple[int, int],
    coin_radius: int,
    expand_ratio: float = 1.05
) -> np.ndarray:
    """
    Create a simple circle mask for detected coin.
    
    This provides a clean segmentation of the coin reference
    for visualization and exclusion from fruit analysis.
    
    Args:
        image: Source image (for dimensions)
        coin_center: (x, y) center of coin
        coin_radius: Radius of coin in pixels
        expand_ratio: Slight expansion to fully cover coin edge
        
    Returns:
        Binary mask with coin region as white (255)
    """
    h, w = image.shape[:2]
    mask = np.zeros((h, w), dtype=np.uint8)
    
    # Draw filled circle at coin location
    expanded_radius = int(coin_radius * expand_ratio)
    cv2.circle(mask, coin_center, expanded_radius, 255, -1)
    
    return mask


def exclude_coin_from_mask(
    fruit_mask: np.ndarray,
    coin_mask: np.ndarray
) -> np.ndarray:
    """
    Remove coin region from fruit segmentation mask.
    
    This ensures the coin doesn't interfere with fruit analysis.
    
    Args:
        fruit_mask: Binary mask of fruit region
        coin_mask: Binary mask of coin region
        
    Returns:
        Fruit mask with coin region excluded
    """
    return cv2.bitwise_and(fruit_mask, cv2.bitwise_not(coin_mask))


if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description="Validate Talisay fruit in image")
    parser.add_argument("image", help="Path to image")
    parser.add_argument("--verbose", "-v", action="store_true", help="Show detailed output")
    
    args = parser.parse_args()
    
    validator = TalisayValidator()
    result = validator.validate(args.image, return_details=args.verbose)
    
    print("\n" + "=" * 50)
    print("TALISAY FRUIT VALIDATION")
    print("=" * 50)
    print(f"Result: {result['result'].upper()}")
    print(f"Is Talisay: {'Yes ✓' if result['is_talisay'] else 'No ✗'}")
    print(f"Confidence: {result['confidence']*100:.1f}%")
    print(f"Message: {result['message']}")
    
    if args.verbose and "details" in result:
        details = result["details"]
        print("\n--- Color Analysis ---")
        color = details.get("color_analysis", {})
        print(f"Dominant Color: {color.get('dominant_color', 'N/A')}")
        print(f"Maturity Stage: {color.get('maturity', 'N/A')}")
        print(f"Talisay Coverage: {color.get('talisay_coverage_percent', 0):.1f}%")
        print(f"Color Distribution: {color.get('color_distribution', {})}")
        
        print("\n--- Shape Analysis ---")
        shape = details.get("shape_analysis", {})
        print(f"Circularity: {shape.get('circularity', 0):.3f}")
        print(f"Aspect Ratio: {shape.get('aspect_ratio', 0):.2f}")
        print(f"Convexity: {shape.get('convexity', 0):.3f}")
        
        print("\n--- Size Analysis ---")
        size = details.get("size_analysis", {})
        print(f"Area: {size.get('area_percent', 0):.2f}% of image")

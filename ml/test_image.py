"""
Test Script for Talisay Fruit Image Analysis
Run this script to test the prediction system with your images.

Usage:
    python test_image.py <image_path>
    python test_image.py <image_path> --length 5.5 --width 3.8

Examples:
    python test_image.py "photos/talisay_yellow.jpg"
    python test_image.py "test_images/green_talisay.png"
"""

import sys
import os
from pathlib import Path

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent))

def test_image(image_path: str, length_cm: float = None, width_cm: float = None):
    """
    Test the prediction system with an image.
    
    Args:
        image_path: Path to the Talisay fruit image
        length_cm: Optional known length in cm
        width_cm: Optional known width in cm
    """
    from predict import TalisayPredictor
    from PIL import Image
    import cv2
    import numpy as np
    
    # Check if file exists
    if not os.path.exists(image_path):
        print(f"❌ Error: Image not found at: {image_path}")
        return
    
    print("\n" + "=" * 60)
    print("🌰 TALISAY FRUIT OIL YIELD ANALYSIS")
    print("=" * 60)
    print(f"\n📁 Image: {image_path}")
    
    # Load and display image info
    try:
        img = Image.open(image_path)
        print(f"📐 Image Size: {img.size[0]} x {img.size[1]} pixels")
        print(f"🎨 Image Mode: {img.mode}")
    except Exception as e:
        print(f"❌ Error loading image: {e}")
        return
    
    # Initialize predictor
    print("\n⏳ Analyzing image...")
    predictor = TalisayPredictor(use_simple_color=True)
    
    # Prepare known dimensions if provided
    known_dimensions = None
    if length_cm and width_cm:
        known_dimensions = {
            "length_cm": length_cm,
            "width_cm": width_cm
        }
        print(f"📏 Using provided dimensions: {length_cm} x {width_cm} cm")
    
    # Analyze
    result = predictor.analyze_image(img, known_dimensions)
    
    # Display results
    print("\n" + "-" * 60)
    print("📊 ANALYSIS RESULTS")
    print("-" * 60)
    
    if result.get("analysis_complete"):
        # Color Classification
        print(f"\n🎨 DETECTED COLOR: {result['color'].upper()}")
        print(f"   Confidence: {result['color_confidence'] * 100:.1f}%")
        print(f"   Maturity Stage: {result['maturity_stage']}")
        
        # Show if spots were detected
        if result.get('has_spots'):
            print(f"   ⚠️ Fruit has visible spots ({result.get('spot_coverage_percent', 0):.1f}% coverage)")
            print(f"      Spots excluded from color analysis for accuracy")
        
        # Color probabilities
        print("\n   Color Probabilities:")
        for color, prob in result['color_probabilities'].items():
            bar = "█" * int(prob * 20)
            print(f"   - {color.capitalize():8}: {prob * 100:5.1f}% {bar}")
        
        # Coin/Reference Detection
        print(f"\n💰 REFERENCE OBJECT:")
        if result.get('reference_detected'):
            print(f"   ✅ ₱5 Coin Detected - using for accurate measurements")
        else:
            print(f"   ⚠️ No coin detected - using estimated dimensions")
            print(f"   💡 Tip: Place a ₱5 coin (25mm) on the LEFT side for accurate sizing")
        
        # Dimensions
        print(f"\n📏 DIMENSIONS:")
        dims = result['dimensions']
        print(f"   Length: {dims['length_cm']} cm")
        print(f"   Width: {dims['width_cm']} cm")
        print(f"   Kernel Mass: {dims['kernel_mass_g']} g")
        print(f"   Fruit Weight: {dims['whole_fruit_weight_g']} g")
        if dims.get('note'):
            print(f"   ⚠️ Note: {dims['note']}")
        
        # Oil Yield Prediction
        print(f"\n🛢️ OIL YIELD PREDICTION:")
        print(f"   Predicted Yield: {result['oil_yield_percent']}%")
        print(f"   Category: {result['yield_category']}")
        print(f"   Confidence: {result['oil_confidence'] * 100:.1f}%")
        
        # Interpretation
        print(f"\n💡 INTERPRETATION:")
        print(f"   {result['interpretation']}")
        
        # Recommendation based on color
        print(f"\n📋 RECOMMENDATION:")
        recommendations = {
            "green": "⏳ This fruit is IMMATURE. Wait for it to turn yellow for higher oil yield.",
            "yellow": "✅ This fruit is at OPTIMAL maturity! Best time to harvest for oil extraction.",
            "brown": "⚠️ This fruit is OVERRIPE. Still good for extraction but slightly lower yield than yellow."
        }
        print(f"   {recommendations.get(result['color'], 'Unable to determine.')}")
        
    else:
        print(f"\n❌ Analysis failed: {result.get('error', 'Unknown error')}")
    
    print("\n" + "=" * 60)
    
    return result


def show_color_analysis(image_path: str, prediction_result: dict = None):
    """Show detailed color analysis of the image."""
    import cv2
    import numpy as np
    from PIL import Image
    
    print("\n" + "-" * 60)
    print("🔬 DETAILED COLOR ANALYSIS")
    print("-" * 60)
    
    # Show debug info from prediction if available
    if prediction_result and "debug_info" in prediction_result:
        debug = prediction_result["debug_info"]
        
        # Show white background detection status
        if debug.get('white_background_detected'):
            print(f"\n   ⬜ WHITE BACKGROUND DETECTED!")
            print(f"   - White coverage: {debug.get('white_coverage_percent', 0):.1f}%")
            print(f"   - Using white-exclusion segmentation for better accuracy")
        
        print(f"\n   🎯 Fruit Segmentation Results:")
        print(f"   - Fruit pixels detected: {debug.get('fruit_pixels', 'N/A'):,}")
        print(f"   - Clean pixels (no spots): {debug.get('clean_pixels', 'N/A'):,}")
        print(f"   - Total image pixels: {debug.get('total_pixels', 'N/A'):,}")
        if debug.get('fruit_pixels') and debug.get('total_pixels'):
            pct = debug['fruit_pixels'] / debug['total_pixels'] * 100
            print(f"   - Fruit coverage: {pct:.1f}%")
        
        # Show spot detection results
        spot_info = debug.get('spot_detection', {})
        if spot_info.get('spots_detected'):
            print(f"\n   🔵 SPOT DETECTION:")
            print(f"   - Spots detected: YES")
            print(f"   - Spot coverage: {spot_info.get('spot_coverage_percent', 0):.1f}% of fruit")
            details = spot_info.get('spot_details', {})
            if details:
                print(f"   - Black spots: {details.get('black_spots', 0):,} px")
                print(f"   - Brown spots: {details.get('brown_spots', 0):,} px")
                print(f"   - Purple spots: {details.get('purple_spots', 0):,} px")
                print(f"   - Dark patches: {details.get('dark_patches', 0):,} px")
            print(f"   📝 Note: Spots excluded from base color analysis for accuracy")
        
        # Show zone distribution if available
        color_dist = debug.get('color_distribution', {})
        if 'zone_distribution' in color_dist:
            zone = color_dist['zone_distribution']
            print(f"\n   🎨 COLOR ZONE DISTRIBUTION (excluding spots):")
            print(f"   - Green zone:  {zone.get('green_percent', 0):5.1f}%")
            print(f"   - Yellow zone: {zone.get('yellow_percent', 0):5.1f}%")
            print(f"   - Brown zone:  {zone.get('brown_percent', 0):5.1f}%")
        
        # Show scoring breakdown
        print(f"\n   📊 SCORING BREAKDOWN:")
        if debug.get('rgb_scores'):
            print(f"   RGB Analysis: Green={debug['rgb_scores'].get('green', 0)*100:.1f}%, Yellow={debug['rgb_scores'].get('yellow', 0)*100:.1f}%, Brown={debug['rgb_scores'].get('brown', 0)*100:.1f}%")
        if debug.get('hsv_scores'):
            print(f"   HSV Analysis: Green={debug['hsv_scores'].get('green', 0)*100:.1f}%, Yellow={debug['hsv_scores'].get('yellow', 0)*100:.1f}%, Brown={debug['hsv_scores'].get('brown', 0)*100:.1f}%")
        if debug.get('zone_scores'):
            print(f"   Zone Scores:  Green={debug['zone_scores'].get('green', 0)*100:.1f}%, Yellow={debug['zone_scores'].get('yellow', 0)*100:.1f}%, Brown={debug['zone_scores'].get('brown', 0)*100:.1f}%")
        
        print(f"\n   🎨 Dominant Color (Extracted from Fruit):")
        if debug.get('dominant_hsv'):
            h, s, v = debug['dominant_hsv']
            print(f"   - HSV: H={h}, S={s}, V={v}")
            # Interpret HSV
            if 35 <= h <= 90:
                hue_name = "GREEN"
            elif 15 <= h <= 35:
                hue_name = "YELLOW"
            elif 5 <= h <= 20:
                hue_name = "BROWN/RED"
            else:
                hue_name = "OTHER"
            print(f"   - Hue indicates: {hue_name} tones")
        
        if debug.get('dominant_bgr'):
            b, g, r = debug['dominant_bgr']
            print(f"   - BGR: B={b}, G={g}, R={r}")
            print(f"   - RGB: R={r}, G={g}, B={b}")
    
    # Load image for additional analysis
    img = cv2.imread(image_path)
    if img is None:
        print("   Could not load image for additional analysis")
        return
    
    # Convert to HSV
    hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
    
    # Calculate average color values (whole image)
    avg_h = np.mean(hsv[:, :, 0])
    avg_s = np.mean(hsv[:, :, 1])
    avg_v = np.mean(hsv[:, :, 2])
    
    print(f"\n   📊 Whole Image Average HSV:")
    print(f"   - Hue: {avg_h:.1f} (0-180 scale)")
    print(f"   - Saturation: {avg_s:.1f} (0-255)")
    print(f"   - Value: {avg_v:.1f} (0-255)")
    
    # Compare dominant vs average
    if prediction_result and "debug_info" in prediction_result:
        debug = prediction_result["debug_info"]
        if debug.get('dominant_hsv'):
            dom_h = debug['dominant_hsv'][0]
            print(f"\n   ⚡ Analysis:")
            print(f"   - Dominant hue (fruit): {dom_h}")
            print(f"   - Average hue (whole image): {avg_h:.1f}")
            print(f"   - The segmentation {'successfully' if abs(dom_h - avg_h) > 5 else 'minimally'} separated fruit from background")


if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(
        description="Test Talisay fruit prediction with an image",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python test_image.py "path/to/image.jpg"
  python test_image.py "image.jpg" --length 5.5 --width 3.8
  python test_image.py "image.jpg" --detailed
        """
    )
    
    parser.add_argument("image", help="Path to the Talisay fruit image")
    parser.add_argument("--length", type=float, help="Known fruit length in cm")
    parser.add_argument("--width", type=float, help="Known fruit width in cm")
    parser.add_argument("--detailed", action="store_true", help="Show detailed color analysis")
    
    args = parser.parse_args()
    
    # Run test
    result = test_image(args.image, args.length, args.width)
    
    # Show detailed analysis if requested
    if args.detailed and result:
        show_color_analysis(args.image, result)
    
    print("\n✅ Analysis complete!")

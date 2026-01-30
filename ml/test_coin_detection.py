"""Test coin detection after code changes."""
import sys
import os
sys.path.insert(0, 'models')

import cv2
from dimension_estimator import DimensionEstimator

# Test on Kaggle image (no coin)
estimator = DimensionEstimator()

print("=" * 60)
print("TEST 1: Kaggle image (should NOT detect coin)")
print("=" * 60)
img = cv2.imread('data/kaggle_talisay/fruit/Photo 08-20-22, 4 38 22 PM.jpg')

if img is None:
    print("Could not load Kaggle image!")
else:
    result = estimator._detect_coin_reference(img)
    print(f"  Detected: {result.get('detected')}")
    print(f"  Coin type: {result.get('coin_type')}")
    print(f"  Score: {result.get('detection_score')}")
    
    full_result = estimator.estimate_from_image(img)
    print(f"  Full method: {full_result.get('method_used')}")
    print(f"  Reference detected: {full_result.get('reference_detected')}")

# Test on coin+fruit image if available
print("\n" + "=" * 60)
print("TEST 2: Coin + Fruit image (should detect coin)")
print("=" * 60)

coin_test_paths = [
    'test_images/talisay5.png',
    'test_images/talisay_with_coin.jpg',
    'test_images/talisay_with_coin.png',
    'test_images/coin_fruit_test.jpg',
    'test_images/coin_fruit_test.png',
]

coin_img = None
for path in coin_test_paths:
    if os.path.exists(path):
        coin_img = cv2.imread(path)
        print(f"  Loaded: {path}")
        break

if coin_img is None:
    print("  No coin+fruit test image found.")
    print("  Please save an image with ₱5 coin (left) and fruit (right) as:")
    print("    test_images/talisay_with_coin.jpg")
else:
    result = estimator._detect_coin_reference(coin_img)
    print(f"  Detected: {result.get('detected')}")
    print(f"  Coin type: {result.get('coin_type')}")
    print(f"  Coin name: {result.get('coin_name')}")
    print(f"  Score: {result.get('detection_score')}")
    print(f"  Confidence: {result.get('confidence')}")
    
    full_result = estimator.estimate_from_image(coin_img)
    print(f"\n  Full estimation:")
    print(f"    Reference detected: {full_result.get('reference_detected')}")
    print(f"    Method: {full_result.get('method_used')}")
    print(f"    Dimensions: {full_result.get('length_cm')}cm x {full_result.get('width_cm')}cm")
    print(f"    Confidence: {full_result.get('confidence')}")

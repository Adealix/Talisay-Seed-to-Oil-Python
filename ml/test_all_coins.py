"""Test all coin images."""
import cv2
import sys
sys.path.insert(0, 'models')
from dimension_estimator import DimensionEstimator

estimator = DimensionEstimator()

test_images = [
    ('data/kaggle_talisay/fruit/Photo 08-20-22, 4 38 22 PM.jpg', 'Kaggle (NO COIN)'),
    ('test_images/talisay5.png', 'Talisay5 (COIN)'),
    ('test_images/talisay6_a4.png', 'Talisay6 A4 (COIN)'),
    ('test_images/talisay6_zoomed.png', 'Talisay6 Zoomed (COIN)'),
]

for path, name in test_images:
    img = cv2.imread(path)
    if img is None:
        print(f"{name}: Could not load")
        continue
    
    result = estimator._detect_coin_reference(img)
    status = "✅ DETECTED" if result['detected'] else "❌ NOT DETECTED"
    print(f"{name}: {status}")
    if result['detected']:
        print(f"  Score: {result['detection_score']:.3f}")
        print(f"  Center: {result['coin_center']}")
        print(f"  Radius: {result['coin_radius']}px")

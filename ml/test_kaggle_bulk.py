"""Test coin detection on all Kaggle images (should all be NO COIN)."""
import cv2
import os
import sys
sys.path.insert(0, 'models')
from dimension_estimator import DimensionEstimator

estimator = DimensionEstimator()

kaggle_dir = 'data/kaggle_talisay/fruit'
false_positives = []
total = 0

for filename in os.listdir(kaggle_dir):
    if filename.endswith('.jpg') or filename.endswith('.png'):
        path = os.path.join(kaggle_dir, filename)
        img = cv2.imread(path)
        if img is None:
            continue
        
        total += 1
        result = estimator._detect_coin_reference(img)
        
        if result['detected']:
            false_positives.append((filename, result['detection_score'], result['coin_center']))

print(f"Tested {total} Kaggle images")
print(f"False positives: {len(false_positives)}")

if false_positives:
    print("\nFalse positives:")
    for fn, score, center in false_positives:
        print(f"  {fn}: score={score:.3f} at {center}")
else:
    print("✅ All Kaggle images correctly show NO COIN")

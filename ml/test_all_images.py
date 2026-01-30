"""
Comprehensive Test Script for Talisay ML Model
Tests all available images to verify model accuracy and improvements.
"""

import sys
import os
from pathlib import Path
import time

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent))

from predict import TalisayPredictor
from PIL import Image

def test_single_image(predictor, image_path, category=""):
    """Test a single image and return results."""
    try:
        img = Image.open(image_path)
        start_time = time.time()
        result = predictor.analyze_image(img)
        elapsed = time.time() - start_time
        
        return {
            "path": image_path,
            "category": category,
            "success": result.get("analysis_complete", False),
            "color": result.get("color", "unknown"),
            "color_confidence": result.get("color_confidence", 0),
            "maturity": result.get("maturity_stage", "unknown"),
            "has_spots": result.get("has_spots", False),
            "spot_coverage": result.get("spot_coverage_percent", 0),
            "reference_detected": result.get("reference_detected", False),
            "measurement_mode": result.get("measurement_mode", "unknown"),
            "dimensions": result.get("dimensions", {}),
            "oil_yield": result.get("oil_yield_percent", 0),
            "yield_category": result.get("yield_category", "unknown"),
            "elapsed_ms": elapsed * 1000,
            "error": None
        }
    except Exception as e:
        return {
            "path": image_path,
            "category": category,
            "success": False,
            "error": str(e)
        }

def print_result_summary(result):
    """Print a single result in compact format."""
    name = Path(result["path"]).name[:35]
    if result.get("error"):
        print(f"  ❌ {name}: ERROR - {result['error'][:40]}")
    else:
        color = result["color"].upper()
        conf = result["color_confidence"] * 100
        ref = "🪙" if result["reference_detected"] else "📐"
        oil = result["oil_yield"]
        spots = "🔵" if result["has_spots"] else ""
        time_ms = result["elapsed_ms"]
        print(f"  ✅ {name:<35} | {color:<6} {conf:>5.1f}% | {ref} | Oil: {oil:>5.1f}% | {spots} {time_ms:>6.0f}ms")

def main():
    print("\n" + "=" * 80)
    print("🧪 COMPREHENSIVE ML MODEL TEST")
    print("=" * 80)
    print("\nInitializing predictor...")
    
    predictor = TalisayPredictor(use_simple_color=True)
    
    # Define test categories
    test_categories = [
        {
            "name": "Test Images (Mixed)",
            "path": "D:/talisay_ml/ml/test_images",
            "patterns": ["*.jpg", "*.png"],
            "expected_coin": "varies"
        },
        {
            "name": "Kaggle Dataset (No Coins)",
            "path": "D:/talisay_ml/ml/data/kaggle_talisay/fruit",
            "patterns": ["*.jpg"],
            "expected_coin": False
        }
    ]
    
    all_results = []
    category_stats = {}
    
    for category in test_categories:
        print(f"\n{'─' * 80}")
        print(f"📁 Testing: {category['name']}")
        print(f"   Path: {category['path']}")
        print(f"   Expected coin detection: {category['expected_coin']}")
        print(f"{'─' * 80}")
        
        # Get all images
        images = []
        for pattern in category["patterns"]:
            images.extend(list(Path(category["path"]).glob(pattern)))
        
        if not images:
            print(f"   ⚠️ No images found!")
            continue
        
        print(f"   Found {len(images)} images\n")
        print(f"   {'Image':<35} | {'Color':<12} | Ref | {'Oil Yield':<10} | Time")
        print(f"   {'-' * 35}-+-{'-' * 12}-+-----+-{'-' * 10}-+-------")
        
        category_results = []
        false_positives = []
        
        for img_path in sorted(images):
            result = test_single_image(predictor, str(img_path), category["name"])
            category_results.append(result)
            all_results.append(result)
            print_result_summary(result)
            
            # Check for false positives (coin detected when not expected)
            if category["expected_coin"] == False and result.get("reference_detected"):
                false_positives.append(result)
        
        # Category statistics
        successful = [r for r in category_results if r.get("success")]
        category_stats[category["name"]] = {
            "total": len(images),
            "successful": len(successful),
            "colors": {},
            "false_positives": len(false_positives),
            "avg_time_ms": sum(r.get("elapsed_ms", 0) for r in successful) / max(len(successful), 1)
        }
        
        for r in successful:
            color = r.get("color", "unknown")
            category_stats[category["name"]]["colors"][color] = \
                category_stats[category["name"]]["colors"].get(color, 0) + 1
        
        # Report false positives
        if false_positives:
            print(f"\n   ⚠️ FALSE POSITIVES ({len(false_positives)}):")
            for fp in false_positives:
                print(f"      - {Path(fp['path']).name}")
    
    # ========== SUMMARY REPORT ==========
    print("\n" + "=" * 80)
    print("📊 TEST SUMMARY REPORT")
    print("=" * 80)
    
    total_images = len(all_results)
    total_successful = len([r for r in all_results if r.get("success")])
    total_false_positives = sum(s["false_positives"] for s in category_stats.values())
    
    print(f"\n📈 OVERALL STATISTICS:")
    print(f"   Total images tested: {total_images}")
    print(f"   Successful analyses: {total_successful} ({total_successful/max(total_images,1)*100:.1f}%)")
    print(f"   False positive coin detections: {total_false_positives}")
    
    print(f"\n🎨 COLOR DISTRIBUTION (All Images):")
    all_colors = {}
    for r in all_results:
        if r.get("success"):
            color = r.get("color", "unknown")
            all_colors[color] = all_colors.get(color, 0) + 1
    
    for color, count in sorted(all_colors.items(), key=lambda x: -x[1]):
        pct = count / max(total_successful, 1) * 100
        bar = "█" * int(pct / 5)
        print(f"   {color.capitalize():<10}: {count:>3} ({pct:>5.1f}%) {bar}")
    
    print(f"\n💰 REFERENCE DETECTION:")
    ref_detected = len([r for r in all_results if r.get("reference_detected")])
    no_ref = total_successful - ref_detected
    print(f"   Coin detected: {ref_detected}")
    print(f"   No coin (estimation used): {no_ref}")
    
    print(f"\n🛢️ OIL YIELD DISTRIBUTION:")
    yields = [r.get("oil_yield", 0) for r in all_results if r.get("success")]
    if yields:
        print(f"   Min: {min(yields):.1f}%")
        print(f"   Max: {max(yields):.1f}%")
        print(f"   Average: {sum(yields)/len(yields):.1f}%")
    
    print(f"\n🔵 SPOT DETECTION:")
    spots_detected = len([r for r in all_results if r.get("has_spots")])
    print(f"   Images with spots: {spots_detected}")
    print(f"   Images without spots: {total_successful - spots_detected}")
    
    print(f"\n⏱️ PERFORMANCE:")
    avg_time = sum(r.get("elapsed_ms", 0) for r in all_results if r.get("success")) / max(total_successful, 1)
    print(f"   Average analysis time: {avg_time:.0f}ms per image")
    
    print(f"\n📋 PER-CATEGORY BREAKDOWN:")
    for cat_name, stats in category_stats.items():
        print(f"\n   {cat_name}:")
        print(f"      Images: {stats['total']}")
        print(f"      Successful: {stats['successful']}")
        print(f"      False positives: {stats['false_positives']}")
        print(f"      Avg time: {stats['avg_time_ms']:.0f}ms")
        print(f"      Colors: {stats['colors']}")
    
    # ========== VALIDATION CHECKS ==========
    print("\n" + "=" * 80)
    print("✅ VALIDATION CHECKS")
    print("=" * 80)
    
    checks_passed = 0
    total_checks = 5
    
    # Check 1: No false positives in Kaggle dataset
    kaggle_fp = category_stats.get("Kaggle Dataset (No Coins)", {}).get("false_positives", 0)
    if kaggle_fp == 0:
        print("   ✅ PASS: No false positive coin detections in Kaggle images")
        checks_passed += 1
    else:
        print(f"   ❌ FAIL: {kaggle_fp} false positive coin detections in Kaggle images")
    
    # Check 2: High success rate
    success_rate = total_successful / max(total_images, 1) * 100
    if success_rate >= 95:
        print(f"   ✅ PASS: Success rate {success_rate:.1f}% >= 95%")
        checks_passed += 1
    else:
        print(f"   ❌ FAIL: Success rate {success_rate:.1f}% < 95%")
    
    # Check 3: Color detection working
    if len(all_colors) >= 2:
        print(f"   ✅ PASS: Multiple colors detected ({list(all_colors.keys())})")
        checks_passed += 1
    else:
        print(f"   ❌ FAIL: Only {len(all_colors)} color(s) detected")
    
    # Check 4: Oil yield in reasonable range
    if yields and 30 <= sum(yields)/len(yields) <= 70:
        print(f"   ✅ PASS: Average oil yield in expected range (30-70%)")
        checks_passed += 1
    else:
        print(f"   ❌ FAIL: Average oil yield outside expected range")
    
    # Check 5: Performance acceptable
    if avg_time < 5000:
        print(f"   ✅ PASS: Average analysis time < 5 seconds")
        checks_passed += 1
    else:
        print(f"   ❌ FAIL: Average analysis time too slow")
    
    print(f"\n{'=' * 80}")
    if checks_passed == total_checks:
        print("🎉 ALL CHECKS PASSED! Model is ready for production use.")
    else:
        print(f"⚠️ {checks_passed}/{total_checks} checks passed. Review failed checks above.")
    print("=" * 80 + "\n")

if __name__ == "__main__":
    main()

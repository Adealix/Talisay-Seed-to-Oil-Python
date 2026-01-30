"""
Complete Integration Training for Talisay Oil Yield Prediction System
======================================================================

This script integrates and trains all components of the ML system:
1. Oil Yield Predictor - Trained on synthetic data
2. Color Classifier - Deep learning on Kaggle images
3. Dimension Estimator - Uses ₱5 coin reference (25mm)
4. Advanced Segmenter - Background removal

Run: python integrate_and_train.py
"""

import sys
from pathlib import Path
import json
import shutil
from datetime import datetime

sys.path.append(str(Path(__file__).parent))

from config import DATA_DIR, MODELS_DIR


def create_system_config():
    """Create/update system configuration with all learned parameters."""
    config = {
        "version": "2.1.0",
        "last_updated": datetime.now().isoformat(),
        
        # Coin Reference Configuration
        "coin_reference": {
            "default_coin": "peso_5_new",
            "coins": {
                "peso_5_new": {
                    "name": "₱5 Silver Coin (NEW)",
                    "diameter_mm": 25,
                    "color_type": "silver"
                },
                "peso_5_old": {
                    "name": "₱5 Brass Coin (OLD)",
                    "diameter_mm": 24,
                    "color_type": "brass"
                }
            },
            "detection_rules": {
                "position": "left_45_percent",
                "size_ratio": {"min": 0.06, "max": 0.22},
                "hue_range": {"neutral_max": 30, "neutral_min_wrap": 165},
                "saturation_range": {"min": 55, "max": 90},
                "brightness_range": {"min": 45, "max": 165},
                "uniformity_threshold": 45,
                "gradient_threshold": 35
            }
        },
        
        # Color Classification Configuration
        "color_classification": {
            "classes": ["green", "yellow", "brown"],
            "maturity_mapping": {
                "green": "Immature",
                "yellow": "Mature (Optimal)",
                "brown": "Fully Ripe"
            },
            "deep_learning_model": "color_classifier_kaggle.keras",
            "hsv_ranges": {
                "green": {"h_min": 25, "h_max": 90, "s_min": 30},
                "yellow": {"h_min": 15, "h_max": 35, "s_min": 50},
                "brown": {"h_min": 0, "h_max": 25, "s_min": 30}
            }
        },
        
        # Oil Yield Parameters (from research)
        "oil_yield": {
            "by_color": {
                "green": {"mean": 47.0, "std": 1.5, "min": 45.0, "max": 49.0},
                "yellow": {"mean": 58.5, "std": 1.2, "min": 57.0, "max": 60.0},
                "brown": {"mean": 55.5, "std": 1.3, "min": 54.0, "max": 57.0}
            },
            "categories": {
                "excellent": {"min": 58, "label": "Excellent"},
                "good": {"min": 55, "label": "Good"},
                "average": {"min": 50, "label": "Average"},
                "below_average": {"min": 0, "label": "Below Average"}
            }
        },
        
        # Dimension Ranges (Talisay fruit)
        "dimensions": {
            "length_cm": {"min": 3.5, "max": 7.0, "typical": 5.0},
            "width_cm": {"min": 2.0, "max": 5.5, "typical": 3.5},
            "kernel_mass_g": {"min": 0.1, "max": 0.9, "typical": 0.4},
            "whole_fruit_weight_g": {"min": 15.0, "max": 60.0, "typical": 35.0}
        },
        
        # Photo Guidelines
        "photo_guidelines": {
            "coin_position": "left",
            "fruit_position": "right",
            "background": "white or neutral (A4 paper recommended)",
            "angle": "top-down view",
            "fill_ratio": "60-80% of frame",
            "zoom": "zoomed preferred over full view"
        }
    }
    
    config_path = MODELS_DIR / "system_config.json"
    with open(config_path, 'w') as f:
        json.dump(config, f, indent=2)
    
    print(f"✓ System configuration saved: {config_path}")
    return config


def train_oil_yield_model():
    """Train the oil yield prediction model."""
    import pandas as pd
    from models.oil_yield_predictor import OilYieldPredictor
    
    print("\n" + "="*60)
    print("TRAINING: Oil Yield Predictor")
    print("="*60)
    
    # Check for dataset
    dataset_path = DATA_DIR / "synthetic_dataset.csv"
    
    if not dataset_path.exists():
        print("Generating synthetic dataset...")
        from data.generate_dataset import generate_synthetic_dataset
        df = generate_synthetic_dataset(num_samples=1000, save_csv=True)
    else:
        print(f"Loading existing dataset: {dataset_path}")
        df = pd.read_csv(dataset_path)
    
    print(f"Dataset size: {len(df)} samples")
    
    # Split data
    from sklearn.model_selection import train_test_split
    train_df, temp_df = train_test_split(df, test_size=0.3, random_state=42)
    val_df, test_df = train_test_split(temp_df, test_size=0.5, random_state=42)
    
    print(f"Train: {len(train_df)}, Val: {len(val_df)}, Test: {len(test_df)}")
    
    # Train model
    predictor = OilYieldPredictor()
    predictor.train(train_df, val_df)
    
    # Evaluate
    from sklearn.metrics import mean_absolute_error, r2_score
    import numpy as np
    
    test_predictions = predictor.predict_batch(test_df)
    y_test = test_df["oil_yield_percent"].values
    
    mae = mean_absolute_error(y_test, test_predictions)
    r2 = r2_score(y_test, test_predictions)
    
    print(f"\nTest Results:")
    print(f"  MAE: {mae:.4f}%")
    print(f"  R²: {r2:.4f}")
    
    # Save model
    model_path = MODELS_DIR / "oil_yield_predictor.joblib"
    predictor.save_model(model_path)
    print(f"✓ Model saved: {model_path}")
    
    return predictor, {"mae": mae, "r2": r2}


def verify_color_classifier():
    """Verify the deep learning color classifier is ready."""
    print("\n" + "="*60)
    print("VERIFYING: Deep Learning Color Classifier")
    print("="*60)
    
    model_path = MODELS_DIR / "color_classifier_kaggle.keras"
    
    if model_path.exists():
        print(f"✓ Model found: {model_path}")
        
        try:
            import tensorflow as tf
            model = tf.keras.models.load_model(str(model_path))
            print(f"✓ Model loaded successfully")
            print(f"  Input shape: {model.input_shape}")
            print(f"  Output shape: {model.output_shape}")
            
            # Check class indices
            indices_path = MODELS_DIR / "class_indices.json"
            if indices_path.exists():
                with open(indices_path) as f:
                    indices = json.load(f)
                print(f"  Classes: {indices.get('idx_to_class', {})}")
            
            return True
        except Exception as e:
            print(f"⚠ Error loading model: {e}")
            return False
    else:
        print(f"⚠ Model not found: {model_path}")
        print("  Run data/train_on_kaggle.py to train the color classifier")
        return False


def verify_dimension_estimator():
    """Verify dimension estimator with coin detection."""
    print("\n" + "="*60)
    print("VERIFYING: Dimension Estimator (Coin Detection)")
    print("="*60)
    
    import cv2
    from models.dimension_estimator import DimensionEstimator
    
    estimator = DimensionEstimator()
    
    # Test on images if available
    test_images = [
        ("test_images/talisay6_zoomed.png", True, "Zoomed with coin"),
        ("test_images/talisay5.png", True, "Standard with coin"),
        ("data/kaggle_talisay/fruit/Photo 08-20-22, 4 38 22 PM.jpg", False, "Kaggle (no coin)"),
    ]
    
    results = []
    for path, expected_coin, name in test_images:
        full_path = Path(__file__).parent / path
        if full_path.exists():
            img = cv2.imread(str(full_path))
            result = estimator._detect_coin_reference(img)
            detected = result.get("detected", False)
            status = "✓" if detected == expected_coin else "✗"
            results.append((name, detected, expected_coin, status))
            print(f"  {status} {name}: detected={detected}, expected={expected_coin}")
    
    # Check coin parameters
    print(f"\n  Coin Reference: ₱5 Silver (25mm)")
    print(f"  Detection rules:")
    print(f"    - Position: Left 45% of image")
    print(f"    - Size: 6-22% of image width")
    print(f"    - Saturation: 55-90 (warm lighting)")
    print(f"    - Gradient: >35 (coin texture)")
    
    return all(r[3] == "✓" for r in results) if results else True


def test_full_pipeline():
    """Test the complete prediction pipeline."""
    print("\n" + "="*60)
    print("TESTING: Full Prediction Pipeline")
    print("="*60)
    
    import cv2
    from predict import TalisayPredictor
    
    # Initialize predictor with deep learning
    print("Initializing TalisayPredictor...")
    predictor = TalisayPredictor(
        use_deep_learning_color=True,
        use_simple_color=True,
        enable_segmentation=True
    )
    
    # Test on zoomed image
    test_path = Path(__file__).parent / "test_images/talisay6_zoomed.png"
    
    if test_path.exists():
        print(f"\nAnalyzing: {test_path.name}")
        result = predictor.analyze_image(str(test_path))
        
        print(f"\nResults:")
        print(f"  Color: {result.get('color', 'N/A')} ({result.get('color_confidence', 0)*100:.1f}%)")
        print(f"  Maturity: {result.get('maturity_stage', 'N/A')}")
        print(f"  Reference Detected: {result.get('reference_detected', False)}")
        
        if result.get('reference_detected'):
            dims = result.get('dimensions', {})
            print(f"  Dimensions: {dims.get('length_cm', 0):.2f}cm x {dims.get('width_cm', 0):.2f}cm")
            print(f"  Est. Weight: {dims.get('whole_fruit_weight_g', 0):.1f}g")
        
        print(f"  Oil Yield: {result.get('oil_yield_percent', 0):.2f}%")
        print(f"  Category: {result.get('yield_category', 'N/A')}")
        print(f"  Overall Confidence: {result.get('overall_confidence', 0)*100:.1f}%")
        
        return result.get('analysis_complete', False)
    else:
        print("  No test image available")
        return True


def create_model_summary():
    """Create a summary of all trained models."""
    print("\n" + "="*60)
    print("MODEL SUMMARY")
    print("="*60)
    
    summary = {
        "system_version": "2.1.0",
        "created": datetime.now().isoformat(),
        "models": {}
    }
    
    # Check each model
    models = [
        ("Oil Yield Predictor", "oil_yield_predictor.joblib"),
        ("Color Classifier (DL)", "color_classifier_kaggle.keras"),
        ("Color Classifier (Best)", "color_classifier_best.keras"),
        ("System Config", "system_config.json"),
        ("Class Indices", "class_indices.json"),
    ]
    
    for name, filename in models:
        path = MODELS_DIR / filename
        if path.exists():
            size = path.stat().st_size
            summary["models"][name] = {
                "file": filename,
                "size_bytes": size,
                "exists": True
            }
            size_str = f"{size/1024:.1f}KB" if size < 1024*1024 else f"{size/1024/1024:.1f}MB"
            print(f"  ✓ {name}: {filename} ({size_str})")
        else:
            summary["models"][name] = {"file": filename, "exists": False}
            print(f"  ✗ {name}: {filename} (not found)")
    
    # Save summary
    summary_path = MODELS_DIR / "model_summary.json"
    with open(summary_path, 'w') as f:
        json.dump(summary, f, indent=2)
    
    return summary


def main():
    """Run complete integration and training."""
    print("="*60)
    print("TALISAY ML SYSTEM - COMPLETE INTEGRATION")
    print("="*60)
    print(f"Started: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    
    # Step 1: Create/update system configuration
    print("\n[1/5] Creating system configuration...")
    create_system_config()
    
    # Step 2: Train oil yield model
    print("\n[2/5] Training oil yield model...")
    try:
        oil_model, oil_metrics = train_oil_yield_model()
        print(f"✓ Oil yield model trained (MAE: {oil_metrics['mae']:.3f}%)")
    except Exception as e:
        print(f"⚠ Oil yield training failed: {e}")
    
    # Step 3: Verify color classifier
    print("\n[3/5] Verifying color classifier...")
    color_ok = verify_color_classifier()
    
    # Step 4: Verify dimension estimator
    print("\n[4/5] Verifying dimension estimator...")
    dim_ok = verify_dimension_estimator()
    
    # Step 5: Test full pipeline
    print("\n[5/5] Testing full pipeline...")
    pipeline_ok = test_full_pipeline()
    
    # Summary
    print("\n" + "="*60)
    create_model_summary()
    
    print("\n" + "="*60)
    print("INTEGRATION COMPLETE")
    print("="*60)
    print(f"Finished: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    
    status = all([color_ok, dim_ok, pipeline_ok])
    print(f"\nOverall Status: {'✓ SUCCESS' if status else '⚠ PARTIAL SUCCESS'}")
    
    print("\nNext Steps:")
    print("  1. Test with: python predict.py --image <your_image.jpg>")
    print("  2. View guide: python predict.py --guide")
    print("  3. Start API: python api.py")
    
    return status


if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)

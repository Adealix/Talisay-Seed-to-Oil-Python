"""
Training Script for Talisay Oil Yield Prediction Models
"""

import sys
from pathlib import Path

sys.path.append(str(Path(__file__).parent))

from config import DATA_DIR, MODELS_DIR, TRAINING_CONFIG


def train_oil_yield_model():
    """Train the oil yield prediction model using synthetic data."""
    import pandas as pd
    from models.oil_yield_predictor import OilYieldPredictor
    from data.generate_dataset import generate_synthetic_dataset, create_training_validation_split
    
    print("=" * 60)
    print("TALISAY OIL YIELD MODEL TRAINING")
    print("=" * 60)
    
    # Check if synthetic dataset exists, if not generate it
    dataset_path = DATA_DIR / "synthetic_dataset.csv"
    
    if dataset_path.exists():
        print(f"\nLoading existing dataset from: {dataset_path}")
        df = pd.read_csv(dataset_path)
    else:
        print("\nGenerating synthetic dataset...")
        df = generate_synthetic_dataset(num_samples=1000, save_csv=True)
    
    print(f"Dataset size: {len(df)} samples")
    
    # Create train/val/test split
    print("\nSplitting dataset...")
    train_df, val_df, test_df = create_training_validation_split(
        df,
        validation_ratio=TRAINING_CONFIG["validation_split"],
        test_ratio=0.1
    )
    
    # Initialize and train model
    print("\n" + "-" * 40)
    print("Training Oil Yield Predictor...")
    print("-" * 40)
    
    predictor = OilYieldPredictor()
    predictor.train(train_df, val_df)
    
    # Evaluate on test set
    print("\n" + "-" * 40)
    print("Evaluating on Test Set...")
    print("-" * 40)
    
    from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
    import numpy as np
    
    test_predictions = predictor.predict_batch(test_df)
    y_test = test_df["oil_yield_percent"].values
    
    mae = mean_absolute_error(y_test, test_predictions)
    rmse = np.sqrt(mean_squared_error(y_test, test_predictions))
    r2 = r2_score(y_test, test_predictions)
    
    print(f"Test Results:")
    print(f"  Mean Absolute Error: {mae:.4f}%")
    print(f"  Root Mean Squared Error: {rmse:.4f}%")
    print(f"  R² Score: {r2:.4f}")
    
    # Save model
    print("\n" + "-" * 40)
    print("Saving Model...")
    print("-" * 40)
    
    model_path = MODELS_DIR / "oil_yield_predictor.joblib"
    predictor.save_model(model_path)
    
    # Show sample predictions
    print("\n" + "-" * 40)
    print("Sample Predictions vs Actual:")
    print("-" * 40)
    
    sample_indices = test_df.sample(5).index
    for idx in sample_indices:
        row = test_df.loc[idx]
        pred = predictor.predict({
            "color": row["color"],
            "length_cm": row["length_cm"],
            "width_cm": row["width_cm"],
            "kernel_mass_g": row["kernel_mass_g"],
            "whole_fruit_weight_g": row["whole_fruit_weight_g"]
        })
        actual = row["oil_yield_percent"]
        predicted = pred["oil_yield_percent"]
        diff = abs(actual - predicted)
        
        print(f"  {row['color'].capitalize()}: Actual={actual:.2f}%, Predicted={predicted:.2f}%, Diff={diff:.2f}%")
    
    print("\n" + "=" * 60)
    print("TRAINING COMPLETE!")
    print("=" * 60)
    
    return predictor


def train_color_classifier():
    """Train the color classification model (requires image dataset)."""
    print("=" * 60)
    print("COLOR CLASSIFIER TRAINING")
    print("=" * 60)
    
    print("\nNOTE: Color classifier training requires an image dataset.")
    print("Please organize your training images as follows:")
    print(f"  {DATA_DIR}/training/green/  - Green Talisay fruit images")
    print(f"  {DATA_DIR}/training/yellow/ - Yellow Talisay fruit images")
    print(f"  {DATA_DIR}/training/brown/  - Brown Talisay fruit images")
    
    # Check if training data exists
    training_dirs = [
        DATA_DIR / "training" / "green",
        DATA_DIR / "training" / "yellow",
        DATA_DIR / "training" / "brown"
    ]
    
    has_data = all(d.exists() and len(list(d.glob("*"))) > 0 for d in training_dirs)
    
    if not has_data:
        print("\nNo training images found. Creating directory structure...")
        for d in training_dirs:
            d.mkdir(parents=True, exist_ok=True)
        print("Please add training images and run this script again.")
        return None
    
    # Count images
    for d in training_dirs:
        count = len(list(d.glob("*")))
        print(f"  {d.name}: {count} images")
    
    # Import and train
    try:
        import tensorflow as tf
        from models.color_classifier import ColorClassifier
        
        print("\nPreparing image data generators...")
        
        # Create data generators
        datagen = tf.keras.preprocessing.image.ImageDataGenerator(
            rescale=1./255,
            rotation_range=20,
            width_shift_range=0.2,
            height_shift_range=0.2,
            horizontal_flip=True,
            validation_split=0.2
        )
        
        train_generator = datagen.flow_from_directory(
            DATA_DIR / "training",
            target_size=(224, 224),
            batch_size=32,
            class_mode="categorical",
            subset="training"
        )
        
        val_generator = datagen.flow_from_directory(
            DATA_DIR / "training",
            target_size=(224, 224),
            batch_size=32,
            class_mode="categorical",
            subset="validation"
        )
        
        # Train model
        classifier = ColorClassifier()
        classifier.build_model()
        
        history = classifier.train(
            train_generator,
            val_generator,
            epochs=TRAINING_CONFIG["epochs"]
        )
        
        # Save model
        model_path = MODELS_DIR / "color_classifier.keras"
        classifier.save_model(str(model_path))
        
        print("\n" + "=" * 60)
        print("COLOR CLASSIFIER TRAINING COMPLETE!")
        print("=" * 60)
        
        return classifier
        
    except ImportError:
        print("\nTensorFlow not installed. Install with: pip install tensorflow")
        return None


if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description="Train Talisay prediction models")
    parser.add_argument(
        "--model",
        choices=["oil", "color", "all"],
        default="oil",
        help="Which model to train (oil, color, or all)"
    )
    
    args = parser.parse_args()
    
    if args.model in ["oil", "all"]:
        train_oil_yield_model()
    
    if args.model in ["color", "all"]:
        train_color_classifier()

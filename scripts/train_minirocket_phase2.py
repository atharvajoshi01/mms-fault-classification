"""
Phase 2: Train MiniRocket on 8-class combined fault dataset.

8 classes:
1. Normal
2. Bearing Fault
3. Misalignment Fault
4. Unbalance Fault
5. Bearing + Misalignment
6. Bearing + Unbalance
7. Misalignment + Unbalance
8. Bearing + Misalignment + Unbalance (Triple Fault)
"""

import logging
import sys
import time
from pathlib import Path
from datetime import datetime

sys.path.insert(0, str(Path(__file__).parent.parent))

import numpy as np
import joblib
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix

from src.data_loader import build_dataset
from src.preprocessing import normalize_data
from src.models.minirocket import build_minirocket

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


def main():
    """Train MiniRocket model on 8-class dataset."""
    logger.info("=" * 80)
    logger.info("Phase 2: MiniRocket Training (8-Class Combined Faults)")
    logger.info("=" * 80)

    # Configuration
    data_dir = Path('dataset/phase_2_4_combined')
    output_dir = Path('models/minirocket_phase2')
    output_dir.mkdir(parents=True, exist_ok=True)

    # File mapping (8 classes)
    file_map = {
        'normal': 'normal.csv',
        'bearing_fault': 'bearing_fault.csv',
        'misalignment_fault': 'misalignment_fault.csv',
        'unbalance_fault': 'unbalance_fault.csv',
        'bearing_misalignment_fault': 'bearing_misalignment_fault.csv',
        'bearing_unbalance_fault': 'bearing_unbalance_fault.csv',
        'misalignment_unbalance_fault': 'misalignment_unbalance_fault.csv',
        'bearing_misalignment_unbalance_fault': 'bearing_misalignment_unbalance_fault.csv'
    }

    # Load data
    logger.info("\n" + "=" * 80)
    logger.info("STEP 1: Loading Data (8 Classes)")
    logger.info("=" * 80)

    start_time = time.time()
    X, y = build_dataset(data_dir, file_map)
    load_time = time.time() - start_time

    logger.info(f"✓ Data loaded in {load_time:.2f}s")
    logger.info(f"  Total samples: {len(X)}")
    logger.info(f"  Shape: {X.shape}")
    logger.info(f"  Classes: {np.unique(y)}")

    # Encode labels
    logger.info("\n" + "=" * 80)
    logger.info("STEP 2: Encoding Labels")
    logger.info("=" * 80)

    label_encoder = LabelEncoder()
    y_encoded = label_encoder.fit_transform(y)

    logger.info(f"✓ Labels encoded")
    logger.info(f"  Class mapping:")
    for i, class_name in enumerate(label_encoder.classes_):
        logger.info(f"    {i}: {class_name}")

    # Save label encoder early (in case training fails)
    output_dir.mkdir(parents=True, exist_ok=True)
    encoder_path = output_dir / 'label_encoder.pkl'
    joblib.dump(label_encoder, encoder_path)
    logger.info(f"✓ Label encoder saved to: {encoder_path}")

    # Preprocess data
    logger.info("\n" + "=" * 80)
    logger.info("STEP 3: Preprocessing Data")
    logger.info("=" * 80)

    start_time = time.time()
    X_normalized, scaler = normalize_data(X, fit=True)
    preprocess_time = time.time() - start_time

    logger.info(f"✓ Data preprocessed in {preprocess_time:.2f}s")
    logger.info(f"  Normalization: standard (z-score)")

    # Save scaler
    scaler_path = output_dir / 'scaler.pkl'
    joblib.dump(scaler, scaler_path)
    logger.info(f"✓ Scaler saved to: {scaler_path}")

    # Split data
    logger.info("\n" + "=" * 80)
    logger.info("STEP 4: Splitting Data")
    logger.info("=" * 80)

    X_train, X_test, y_train, y_test = train_test_split(
        X_normalized, y_encoded,
        test_size=0.2,
        random_state=42,
        stratify=y_encoded
    )

    logger.info(f"✓ Data split complete")
    logger.info(f"  Train set: {X_train.shape[0]} samples")
    logger.info(f"  Test set: {X_test.shape[0]} samples")

    # Build model
    logger.info("\n" + "=" * 80)
    logger.info("STEP 5: Building MiniRocket Model")
    logger.info("=" * 80)

    model = build_minirocket(num_kernels=10000, random_state=42)

    logger.info(f"✓ Model built")
    logger.info(f"  Num kernels: 10000")

    # Train model
    logger.info("\n" + "=" * 80)
    logger.info("STEP 6: Training Model (This may take 2-3 hours)")
    logger.info("=" * 80)

    start_time = time.time()
    model.fit(X_train, y_train)
    train_time = time.time() - start_time

    logger.info(f"✓ Training complete in {train_time:.2f}s ({train_time/60:.2f} minutes)")

    # Evaluate
    logger.info("\n" + "=" * 80)
    logger.info("STEP 7: Evaluating Model")
    logger.info("=" * 80)

    # Train accuracy
    train_acc = model.score(X_train, y_train)
    logger.info(f"  Train accuracy: {train_acc:.4f} ({train_acc*100:.2f}%)")

    # Test accuracy
    test_acc = model.score(X_test, y_test)
    logger.info(f"  Test accuracy: {test_acc:.4f} ({test_acc*100:.2f}%)")

    # Detailed metrics
    y_pred = model.predict(X_test)
    y_pred_proba = model.predict_proba(X_test)

    logger.info("\n" + "-" * 80)
    logger.info("Classification Report:")
    logger.info("-" * 80)
    print(classification_report(
        y_test, y_pred,
        target_names=label_encoder.classes_,
        digits=4
    ))

    logger.info("\n" + "-" * 80)
    logger.info("Confusion Matrix:")
    logger.info("-" * 80)
    cm = confusion_matrix(y_test, y_pred)
    print(cm)

    # Save model
    logger.info("\n" + "=" * 80)
    logger.info("STEP 8: Saving Model")
    logger.info("=" * 80)

    model_path = output_dir / 'minirocket_model.pkl'
    model.save(model_path)

    logger.info(f"✓ Model saved to: {model_path}")

    # Save metadata
    metadata = {
        'model_type': 'MiniRocket',
        'phase': 2,
        'num_classes': 8,
        'num_kernels': 10000,
        'train_samples': len(X_train),
        'test_samples': len(X_test),
        'train_accuracy': float(train_acc),
        'test_accuracy': float(test_acc),
        'training_time_seconds': train_time,
        'classes': label_encoder.classes_.tolist(),
        'timestamp': datetime.now().isoformat()
    }

    import json
    metadata_path = output_dir / 'metadata.json'
    with open(metadata_path, 'w') as f:
        json.dump(metadata, f, indent=2)

    logger.info(f"✓ Metadata saved to: {metadata_path}")

    # Save confusion matrix and classification report for dashboard
    results_data = {
        'confusion_matrix': cm.tolist(),
        'classification_report': classification_report(
            y_test, y_pred,
            target_names=label_encoder.classes_,
            output_dict=True
        ),
        'classes': label_encoder.classes_.tolist()
    }

    results_path = output_dir / 'results.json'
    with open(results_path, 'w') as f:
        json.dump(results_data, f, indent=2)

    logger.info(f"✓ Results saved to: {results_path}")

    # Summary
    logger.info("\n" + "=" * 80)
    logger.info("PHASE 2 TRAINING COMPLETE - SUMMARY")
    logger.info("=" * 80)
    logger.info(f"  Model: MiniRocket (8-class)")
    logger.info(f"  Classes: {len(label_encoder.classes_)}")
    logger.info(f"  Train Accuracy: {train_acc*100:.2f}%")
    logger.info(f"  Test Accuracy: {test_acc*100:.2f}%")
    logger.info(f"  Training Time: {train_time:.2f}s ({train_time/60:.2f} min)")
    logger.info(f"  Model saved: {model_path}")
    logger.info("=" * 80)


if __name__ == "__main__":
    main()

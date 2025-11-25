"""
Preprocessing module for MMS vibration data.

Includes normalization, train/val/test splitting, and data caching.
"""

import logging
import pickle
from pathlib import Path
from typing import Tuple, Optional, Dict, Any

import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, LabelEncoder

logger = logging.getLogger(__name__)


def normalize_data(
    X: np.ndarray,
    scaler: Optional[StandardScaler] = None,
    fit: bool = True
) -> Tuple[np.ndarray, StandardScaler]:
    """
    Normalize data using StandardScaler across all channels.

    Args:
        X: Input data of shape (num_samples, time_steps, num_channels)
        scaler: Pre-fitted scaler (optional). If None, a new scaler is created.
        fit: Whether to fit the scaler (True for train, False for val/test)

    Returns:
        Tuple of:
        - X_normalized: Normalized data with same shape as input
        - scaler: The StandardScaler object (fitted if fit=True)

    Example:
        >>> X_train = np.random.randn(100, 1024, 3)
        >>> X_train_norm, scaler = normalize_data(X_train, fit=True)
        >>> X_test_norm, _ = normalize_data(X_test, scaler=scaler, fit=False)
    """
    original_shape = X.shape
    num_samples, time_steps, num_channels = original_shape

    # Reshape to 2D: (num_samples * time_steps, num_channels)
    X_reshaped = X.reshape(-1, num_channels)

    if scaler is None:
        scaler = StandardScaler()

    if fit:
        logger.info("Fitting StandardScaler on data...")
        X_normalized = scaler.fit_transform(X_reshaped)
        logger.info(f"  Mean: {scaler.mean_}")
        logger.info(f"  Scale (std): {scaler.scale_}")
    else:
        X_normalized = scaler.transform(X_reshaped)

    # Reshape back to original shape
    X_normalized = X_normalized.reshape(original_shape)

    return X_normalized.astype(np.float32), scaler


def encode_labels(
    y: np.ndarray,
    encoder: Optional[LabelEncoder] = None,
    fit: bool = True
) -> Tuple[np.ndarray, LabelEncoder]:
    """
    Encode string labels to integers.

    Args:
        y: String labels of shape (num_samples,)
        encoder: Pre-fitted encoder (optional)
        fit: Whether to fit the encoder

    Returns:
        Tuple of:
        - y_encoded: Integer labels
        - encoder: The LabelEncoder object
    """
    if encoder is None:
        encoder = LabelEncoder()

    if fit:
        y_encoded = encoder.fit_transform(y)
        logger.info(f"Label encoding:")
        for idx, class_name in enumerate(encoder.classes_):
            logger.info(f"  {class_name}: {idx}")
    else:
        y_encoded = encoder.transform(y)

    return y_encoded, encoder


def split_data(
    X: np.ndarray,
    y: np.ndarray,
    test_size: float = 0.2,
    val_size: float = 0.5,
    random_state: int = 42,
    stratify: bool = True
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """
    Stratified train/val/test split.

    Args:
        X: Features of shape (num_samples, time_steps, num_channels)
        y: Labels of shape (num_samples,)
        test_size: Proportion of data for test set (default: 0.2)
        val_size: Proportion of remaining data for validation (default: 0.5)
                 Actual val size = (1 - test_size) * val_size
        random_state: Random seed for reproducibility
        stratify: Whether to use stratified splitting

    Returns:
        Tuple of (X_train, X_val, X_test, y_train, y_val, y_test)

    Example:
        With test_size=0.2 and val_size=0.5:
        - Test: 20% of total
        - Val: 40% of total (50% of remaining 80%)
        - Train: 40% of total
    """
    logger.info("Splitting data into train/val/test sets...")
    logger.info(f"  Test size: {test_size:.1%}")
    logger.info(f"  Val size (of remaining): {val_size:.1%}")
    logger.info(f"  Random state: {random_state}")
    logger.info(f"  Stratified: {stratify}")

    stratify_param = y if stratify else None

    # First split: separate test set
    X_temp, X_test, y_temp, y_test = train_test_split(
        X, y,
        test_size=test_size,
        random_state=random_state,
        stratify=stratify_param
    )

    # Second split: separate train and validation
    stratify_param_temp = y_temp if stratify else None

    X_train, X_val, y_train, y_val = train_test_split(
        X_temp, y_temp,
        test_size=val_size,
        random_state=random_state,
        stratify=stratify_param_temp
    )

    logger.info(f"Split complete:")
    logger.info(f"  Train: {len(X_train)} samples ({len(X_train)/len(X)*100:.1f}%)")
    logger.info(f"  Val:   {len(X_val)} samples ({len(X_val)/len(X)*100:.1f}%)")
    logger.info(f"  Test:  {len(X_test)} samples ({len(X_test)/len(X)*100:.1f}%)")

    # Show class distribution
    if stratify:
        logger.info("\nClass distribution:")
        unique_classes = np.unique(y)

        for split_name, split_y in [('Train', y_train), ('Val', y_val), ('Test', y_test)]:
            logger.info(f"  {split_name}:")
            for cls in unique_classes:
                count = np.sum(split_y == cls)
                logger.info(f"    {cls}: {count} ({count/len(split_y)*100:.1f}%)")

    return X_train, X_val, X_test, y_train, y_val, y_test


def create_sliding_windows(
    X: np.ndarray,
    window_size: int = 1024,
    step_size: int = 512,
    min_samples: Optional[int] = None
) -> np.ndarray:
    """
    Create sliding windows for data augmentation.

    Args:
        X: Input data of shape (num_samples, time_steps, num_channels)
        window_size: Size of each window
        step_size: Step size for sliding (overlap = window_size - step_size)
        min_samples: Minimum number of samples required (None = no minimum)

    Returns:
        Windowed data of shape (new_num_samples, window_size, num_channels)

    Note:
        If time_steps < window_size, the original samples are returned.
        If time_steps > window_size, multiple windows are extracted per sample.
    """
    num_samples, time_steps, num_channels = X.shape

    if time_steps < window_size:
        logger.warning(
            f"Time steps ({time_steps}) < window size ({window_size}). "
            "Returning original data."
        )
        return X

    if time_steps == window_size:
        return X

    logger.info(f"Creating sliding windows:")
    logger.info(f"  Window size: {window_size}")
    logger.info(f"  Step size: {step_size}")
    logger.info(f"  Overlap: {window_size - step_size}")

    all_windows = []

    for sample in X:
        # Extract windows from this sample
        start_indices = range(0, time_steps - window_size + 1, step_size)

        for start in start_indices:
            window = sample[start:start + window_size]
            all_windows.append(window)

    result = np.array(all_windows, dtype=np.float32)

    logger.info(f"  Original samples: {num_samples}")
    logger.info(f"  Windowed samples: {len(result)}")
    logger.info(f"  Augmentation factor: {len(result) / num_samples:.1f}x")

    if min_samples is not None and len(result) < min_samples:
        logger.warning(
            f"Generated {len(result)} samples, but {min_samples} required. "
            "Consider reducing step size."
        )

    return result


def save_preprocessed_data(
    data_dict: Dict[str, Any],
    output_path: Path
) -> None:
    """
    Save preprocessed data to disk for faster loading.

    Args:
        data_dict: Dictionary containing preprocessed data, e.g.:
                  {
                      'X_train': ...,
                      'X_val': ...,
                      'X_test': ...,
                      'y_train': ...,
                      'y_val': ...,
                      'y_test': ...,
                      'scaler': ...,
                      'encoder': ...
                  }
        output_path: Path to save the data (will be saved as pickle)
    """
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    logger.info(f"Saving preprocessed data to {output_path}")

    with open(output_path, 'wb') as f:
        pickle.dump(data_dict, f, protocol=pickle.HIGHEST_PROTOCOL)

    file_size_mb = output_path.stat().st_size / (1024 * 1024)
    logger.info(f"  Saved successfully ({file_size_mb:.2f} MB)")


def load_preprocessed_data(input_path: Path) -> Dict[str, Any]:
    """
    Load preprocessed data from disk.

    Args:
        input_path: Path to the preprocessed data file

    Returns:
        Dictionary containing the preprocessed data
    """
    input_path = Path(input_path)

    if not input_path.exists():
        raise FileNotFoundError(f"Preprocessed data not found: {input_path}")

    logger.info(f"Loading preprocessed data from {input_path}")

    with open(input_path, 'rb') as f:
        data_dict = pickle.load(f)

    logger.info("  Loaded successfully")
    logger.info(f"  Keys: {list(data_dict.keys())}")

    return data_dict


def preprocess_pipeline(
    X: np.ndarray,
    y: np.ndarray,
    test_size: float = 0.2,
    val_size: float = 0.5,
    random_state: int = 42,
    normalize: bool = True,
    save_path: Optional[Path] = None
) -> Dict[str, Any]:
    """
    Complete preprocessing pipeline.

    Args:
        X: Raw features
        y: Raw labels (string)
        test_size: Test set proportion
        val_size: Validation set proportion (of remaining data)
        random_state: Random seed
        normalize: Whether to normalize the data
        save_path: Optional path to save preprocessed data

    Returns:
        Dictionary containing all preprocessed data and preprocessing objects
    """
    logger.info("=" * 60)
    logger.info("Starting preprocessing pipeline")
    logger.info("=" * 60)

    # 1. Split data
    X_train, X_val, X_test, y_train, y_val, y_test = split_data(
        X, y,
        test_size=test_size,
        val_size=val_size,
        random_state=random_state,
        stratify=True
    )

    # 2. Normalize data
    scaler = None
    if normalize:
        logger.info("\nNormalizing data...")
        X_train, scaler = normalize_data(X_train, fit=True)
        X_val, _ = normalize_data(X_val, scaler=scaler, fit=False)
        X_test, _ = normalize_data(X_test, scaler=scaler, fit=False)
        logger.info("  Normalization complete")

    # 3. Encode labels
    logger.info("\nEncoding labels...")
    y_train_encoded, encoder = encode_labels(y_train, fit=True)
    y_val_encoded, _ = encode_labels(y_val, encoder=encoder, fit=False)
    y_test_encoded, _ = encode_labels(y_test, encoder=encoder, fit=False)

    # Prepare result dictionary
    result = {
        'X_train': X_train,
        'X_val': X_val,
        'X_test': X_test,
        'y_train': y_train,
        'y_val': y_val,
        'y_test': y_test,
        'y_train_encoded': y_train_encoded,
        'y_val_encoded': y_val_encoded,
        'y_test_encoded': y_test_encoded,
        'scaler': scaler,
        'encoder': encoder,
        'num_classes': len(encoder.classes_),
        'class_names': list(encoder.classes_)
    }

    # 4. Save if requested
    if save_path is not None:
        save_preprocessed_data(result, save_path)

    logger.info("\n" + "=" * 60)
    logger.info("Preprocessing pipeline complete")
    logger.info("=" * 60)

    return result


if __name__ == "__main__":
    # Example usage
    print("=" * 60)
    print("MMS Preprocessing - Example Usage")
    print("=" * 60)

    # Create dummy data for testing
    print("\nCreating dummy data...")
    num_samples = 1000
    time_steps = 1024
    num_channels = 3

    X_dummy = np.random.randn(num_samples, time_steps, num_channels).astype(np.float32)
    y_dummy = np.random.choice(
        ['normal', 'unbalance_fault', 'misalignment_fault', 'bearing_fault'],
        size=num_samples
    )

    print(f"Dummy data: X={X_dummy.shape}, y={y_dummy.shape}")

    # Run preprocessing pipeline
    print("\n" + "=" * 60)
    result = preprocess_pipeline(
        X_dummy,
        y_dummy,
        test_size=0.2,
        val_size=0.5,
        random_state=42,
        normalize=True
    )

    print("\nPreprocessing results:")
    for key, value in result.items():
        if isinstance(value, np.ndarray):
            print(f"  {key}: {value.shape}")
        else:
            print(f"  {key}: {value}")

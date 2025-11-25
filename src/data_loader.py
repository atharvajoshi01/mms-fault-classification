"""
Data loader for MMS vibration data with unique row-based CSV format.

The CSV files have a unique structure:
- Each timestamp has 6 rows (2 X, 2 Y, 2 Z axes - duplicates are averaged)
- Columns 3-1026 contain 1024 vibration samples
- Data must be grouped by timestamp to create (1024, 3) samples
"""

import logging
from pathlib import Path
from typing import Dict, List, Tuple, Optional

import numpy as np
import pandas as pd

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def load_mms_csv(csv_path: Path) -> np.ndarray:
    """
    Load MMS CSV and return array of shape (num_samples, 1024, 3).

    The CSV format is unique with duplicate axis readings per timestamp:
    ```
    timestamp,axis,1,2,3,4,5,...,1024
    2025-11-14 17:57:46,X,137.89,226.89,-17.11,...
    2025-11-14 17:57:46,Y,450.99,421.99,430.99,...
    2025-11-14 17:57:46,Z,69.44,76.44,207.44,...
    2025-11-14 17:57:46,X,138.12,227.01,-16.89,...  (duplicate X)
    2025-11-14 17:57:46,Y,451.22,422.13,431.12,...  (duplicate Y)
    2025-11-14 17:57:46,Z,69.67,76.58,207.68,...    (duplicate Z)
    ```

    Process:
    1. Read CSV
    2. Group by timestamp
    3. Extract X, Y, Z rows for each timestamp (averaging duplicates)
    4. Stack as (1024, 3) per timestamp
    5. Return stacked array

    Args:
        csv_path: Path to the CSV file

    Returns:
        np.ndarray: Array of shape (num_samples, 1024, 3) where:
                   - num_samples = number of unique timestamps
                   - 1024 = number of samples per signal
                   - 3 = channels (X, Y, Z)

    Raises:
        FileNotFoundError: If CSV file doesn't exist
        ValueError: If CSV format is invalid or data is incomplete
    """
    csv_path = Path(csv_path)

    if not csv_path.exists():
        raise FileNotFoundError(f"CSV file not found: {csv_path}")

    logger.info(f"Loading CSV file: {csv_path}")

    try:
        # Read CSV file
        df = pd.read_csv(csv_path)

        # Validate structure
        if 'timestamp' not in df.columns or 'axis' not in df.columns:
            raise ValueError(
                "Invalid CSV format: missing 'timestamp' or 'axis' columns"
            )

        # Get the signal columns (all columns after 'axis')
        signal_columns = [col for col in df.columns if col not in ['timestamp', 'axis']]

        if len(signal_columns) != 1024:
            logger.warning(
                f"Expected 1024 signal columns, found {len(signal_columns)}. "
                "Adjusting to available columns."
            )

        num_signal_cols = len(signal_columns)

        # Group by timestamp
        grouped = df.groupby('timestamp')

        samples = []
        invalid_count = 0

        for timestamp, group in grouped:
            # Each timestamp typically has 6 rows (2X, 2Y, 2Z)
            # We'll average duplicates for each axis
            if len(group) < 3:
                logger.warning(
                    f"Timestamp {timestamp} has only {len(group)} rows, need at least 3. Skipping."
                )
                invalid_count += 1
                continue

            # Collect all readings for each axis (may have duplicates)
            axes_readings = {'X': [], 'Y': [], 'Z': []}

            for _, row in group.iterrows():
                axis = row['axis']
                if axis not in ['X', 'Y', 'Z']:
                    logger.warning(f"Unknown axis '{axis}' at timestamp {timestamp}")
                    continue

                # Extract signal values (columns 3 onwards in original CSV)
                signal_values = row[signal_columns].values.astype(np.float32)
                axes_readings[axis].append(signal_values)

            # Average duplicate readings for each axis
            axes_data = {}
            for axis in ['X', 'Y', 'Z']:
                if not axes_readings[axis]:
                    logger.warning(
                        f"Timestamp {timestamp} missing axis {axis}. Skipping timestamp."
                    )
                    invalid_count += 1
                    break

                # Average all readings for this axis
                axes_data[axis] = np.mean(axes_readings[axis], axis=0)

            # Skip if we broke out of the loop (missing axis)
            if len(axes_data) != 3:
                continue

            # Stack X, Y, Z as columns to create shape (1024, 3)
            # Note: We stack in order X, Y, Z
            sample = np.column_stack([
                axes_data['X'],
                axes_data['Y'],
                axes_data['Z']
            ])  # Shape: (num_signal_cols, 3)

            samples.append(sample)

        if invalid_count > 0:
            logger.warning(f"Skipped {invalid_count} invalid timestamp groups")

        if not samples:
            raise ValueError(f"No valid samples found in {csv_path}")

        # Stack all samples
        result = np.array(samples, dtype=np.float32)  # Shape: (num_samples, num_signal_cols, 3)

        logger.info(
            f"Loaded {len(samples)} samples from {csv_path}. "
            f"Shape: {result.shape}"
        )

        return result

    except pd.errors.EmptyDataError:
        raise ValueError(f"CSV file is empty: {csv_path}")
    except Exception as e:
        logger.error(f"Error loading CSV file {csv_path}: {str(e)}")
        raise


def build_dataset(
    data_dir: Path,
    file_map: Dict[str, str],
    expected_shape: Optional[Tuple[int, int]] = (1024, 3)
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Load all CSV files and create labeled dataset.

    Args:
        data_dir: Directory containing CSV files
        file_map: Mapping of class names to CSV filenames
                 e.g., {'normal': 'normal.csv', 'unbalance_fault': 'unbalance_fault.csv'}
        expected_shape: Expected shape of each sample (default: (1024, 3))

    Returns:
        Tuple of:
        - X: np.ndarray of shape (total_samples, 1024, 3) - all vibration signals
        - y: np.ndarray of shape (total_samples,) - string labels

    Example:
        >>> file_map = {
        ...     'normal': 'normal.csv',
        ...     'unbalance_fault': 'unbalance_fault.csv',
        ...     'misalignment_fault': 'misalignment_fault.csv',
        ...     'bearing_fault': 'bearing_fault.csv'
        ... }
        >>> X, y = build_dataset(Path('data/raw'), file_map)
        >>> print(X.shape, y.shape)
        (1200, 1024, 3) (1200,)
    """
    data_dir = Path(data_dir)

    if not data_dir.exists():
        raise FileNotFoundError(f"Data directory not found: {data_dir}")

    logger.info(f"Building dataset from {data_dir}")
    logger.info(f"Classes to load: {list(file_map.keys())}")

    all_samples = []
    all_labels = []

    for class_name, filename in file_map.items():
        csv_path = data_dir / filename

        if not csv_path.exists():
            logger.warning(f"File not found: {csv_path}. Skipping class '{class_name}'.")
            continue

        try:
            # Load samples for this class
            samples = load_mms_csv(csv_path)

            # Validate shape if specified
            if expected_shape is not None:
                if samples.shape[1:] != expected_shape:
                    logger.warning(
                        f"Class '{class_name}' has unexpected shape {samples.shape[1:]}. "
                        f"Expected {expected_shape}. Continuing anyway."
                    )

            # Create labels
            labels = np.array([class_name] * len(samples))

            all_samples.append(samples)
            all_labels.append(labels)

            logger.info(f"Class '{class_name}': loaded {len(samples)} samples")

        except Exception as e:
            logger.error(f"Error loading class '{class_name}': {str(e)}")
            raise

    if not all_samples:
        raise ValueError("No valid data loaded from any class")

    # Concatenate all samples and labels
    X = np.concatenate(all_samples, axis=0)
    y = np.concatenate(all_labels, axis=0)

    logger.info(f"Dataset built successfully:")
    logger.info(f"  Total samples: {len(X)}")
    logger.info(f"  Shape: {X.shape}")
    logger.info(f"  Classes: {np.unique(y)}")
    logger.info(f"  Class distribution:")

    unique, counts = np.unique(y, return_counts=True)
    for class_name, count in zip(unique, counts):
        logger.info(f"    {class_name}: {count} samples ({count/len(y)*100:.1f}%)")

    return X, y


def validate_data(X: np.ndarray, y: np.ndarray) -> bool:
    """
    Validate the loaded data.

    Args:
        X: Feature array of shape (num_samples, 1024, 3)
        y: Label array of shape (num_samples,)

    Returns:
        bool: True if data is valid

    Raises:
        ValueError: If data is invalid
    """
    # Check shapes
    if X.ndim != 3:
        raise ValueError(f"X should be 3D, got shape {X.shape}")

    if y.ndim != 1:
        raise ValueError(f"y should be 1D, got shape {y.shape}")

    if len(X) != len(y):
        raise ValueError(
            f"Number of samples mismatch: X has {len(X)}, y has {len(y)}"
        )

    # Check for NaN or Inf
    if np.isnan(X).any():
        logger.warning("Data contains NaN values")
        nan_count = np.isnan(X).sum()
        logger.warning(f"  NaN count: {nan_count} ({nan_count/X.size*100:.4f}%)")

    if np.isinf(X).any():
        logger.warning("Data contains Inf values")
        inf_count = np.isinf(X).sum()
        logger.warning(f"  Inf count: {inf_count} ({inf_count/X.size*100:.4f}%)")

    # Check data range
    logger.info(f"Data statistics:")
    logger.info(f"  Min: {X.min():.2f}")
    logger.info(f"  Max: {X.max():.2f}")
    logger.info(f"  Mean: {X.mean():.2f}")
    logger.info(f"  Std: {X.std():.2f}")

    return True


def load_single_sample_from_csv(
    csv_path: Path,
    timestamp_idx: int = 0
) -> np.ndarray:
    """
    Load a single sample from a CSV file (useful for testing/debugging).

    Args:
        csv_path: Path to CSV file
        timestamp_idx: Index of the timestamp to extract (default: 0 for first)

    Returns:
        np.ndarray: Single sample of shape (1024, 3)
    """
    samples = load_mms_csv(csv_path)

    if timestamp_idx >= len(samples):
        raise IndexError(
            f"Timestamp index {timestamp_idx} out of range. "
            f"File has {len(samples)} samples."
        )

    return samples[timestamp_idx]


if __name__ == "__main__":
    # Example usage and testing
    print("=" * 60)
    print("MMS Data Loader - Example Usage")
    print("=" * 60)

    # Example 1: Load a single CSV file
    print("\nExample 1: Load single CSV file")
    print("-" * 60)
    example_csv = Path("data/raw/normal.csv")

    if example_csv.exists():
        try:
            data = load_mms_csv(example_csv)
            print(f"Loaded data shape: {data.shape}")
            print(f"First sample shape: {data[0].shape}")
            print(f"Data type: {data.dtype}")
        except Exception as e:
            print(f"Error: {e}")
    else:
        print(f"Example file not found: {example_csv}")

    # Example 2: Build complete dataset
    print("\nExample 2: Build complete dataset")
    print("-" * 60)

    file_map = {
        'normal': 'normal.csv',
        'unbalance_fault': 'unbalance_fault.csv',
        'misalignment_fault': 'misalignment_fault.csv',
        'bearing_fault': 'bearing_fault.csv'
    }

    data_dir = Path("data/raw")

    if data_dir.exists():
        try:
            X, y = build_dataset(data_dir, file_map)
            print(f"\nDataset Summary:")
            print(f"  X shape: {X.shape}")
            print(f"  y shape: {y.shape}")
            validate_data(X, y)
        except Exception as e:
            print(f"Error: {e}")
    else:
        print(f"Data directory not found: {data_dir}")

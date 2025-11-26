"""
Create combined fault dataset for Phase 2: Multi-fault detection.

This script generates combined fault conditions by adding vibration signals:
- bearing_fault + misalignment_fault
- bearing_fault + unbalance_fault
- misalignment_fault + unbalance_fault
- bearing_fault + misalignment_fault + unbalance_fault (triple fault)

The combined signals simulate real-world scenarios where multiple faults occur simultaneously.
"""

import pandas as pd
import numpy as np
from pathlib import Path
import logging
from datetime import datetime

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


def load_fault_data(file_path: Path) -> pd.DataFrame:
    """Load fault data from CSV file."""
    logger.info(f"Loading: {file_path.name}")
    df = pd.read_csv(file_path)
    logger.info(f"  Loaded {len(df)} rows")
    return df


def create_combined_fault(df1: pd.DataFrame, df2: pd.DataFrame,
                          fault1_name: str, fault2_name: str,
                          num_samples: int = None) -> pd.DataFrame:
    """
    Create combined fault by adding two fault signals.

    Args:
        df1: First fault dataframe
        df2: Second fault dataframe
        fault1_name: Name of first fault
        fault2_name: Name of second fault
        num_samples: Number of combined samples to generate

    Returns:
        DataFrame with combined fault samples
    """
    # Determine number of samples
    if num_samples is None:
        num_samples = min(len(df1) // 3, len(df2) // 3)  # Each sample has 3 rows (X,Y,Z)

    logger.info(f"Creating {num_samples} samples of {fault1_name} + {fault2_name}")

    # Get value columns (1 to 1024)
    value_cols = [str(i) for i in range(1, 1025)]

    combined_rows = []

    for i in range(num_samples):
        # Randomly select a sample from each fault
        sample1_idx = np.random.randint(0, len(df1) // 3) * 3
        sample2_idx = np.random.randint(0, len(df2) // 3) * 3

        # Get the 3 rows (X, Y, Z axes) for each sample
        sample1 = df1.iloc[sample1_idx:sample1_idx+3].copy()
        sample2 = df2.iloc[sample2_idx:sample2_idx+3].copy()

        # Create combined sample by adding the signals
        for axis_idx in range(3):
            row1 = sample1.iloc[axis_idx]
            row2 = sample2.iloc[axis_idx]

            # Add the vibration signals
            combined_values = row1[value_cols].values + row2[value_cols].values

            # Create new row
            new_row = {
                'timestamp': datetime.now().isoformat(),
                'axis': row1['axis']
            }

            # Add combined values
            for col, val in zip(value_cols, combined_values):
                new_row[col] = val

            combined_rows.append(new_row)

    combined_df = pd.DataFrame(combined_rows)
    logger.info(f"  Created {len(combined_df)} rows ({num_samples} samples × 3 axes)")

    return combined_df


def create_triple_fault(df1: pd.DataFrame, df2: pd.DataFrame, df3: pd.DataFrame,
                       fault1_name: str, fault2_name: str, fault3_name: str,
                       num_samples: int = None) -> pd.DataFrame:
    """
    Create triple combined fault by adding three fault signals.

    Args:
        df1: First fault dataframe
        df2: Second fault dataframe
        df3: Third fault dataframe
        fault1_name: Name of first fault
        fault2_name: Name of second fault
        fault3_name: Name of third fault
        num_samples: Number of combined samples to generate

    Returns:
        DataFrame with combined fault samples
    """
    # Determine number of samples
    if num_samples is None:
        num_samples = min(len(df1) // 3, len(df2) // 3, len(df3) // 3)

    logger.info(f"Creating {num_samples} samples of {fault1_name} + {fault2_name} + {fault3_name}")

    # Get value columns (1 to 1024)
    value_cols = [str(i) for i in range(1, 1025)]

    combined_rows = []

    for i in range(num_samples):
        # Randomly select a sample from each fault
        sample1_idx = np.random.randint(0, len(df1) // 3) * 3
        sample2_idx = np.random.randint(0, len(df2) // 3) * 3
        sample3_idx = np.random.randint(0, len(df3) // 3) * 3

        # Get the 3 rows (X, Y, Z axes) for each sample
        sample1 = df1.iloc[sample1_idx:sample1_idx+3].copy()
        sample2 = df2.iloc[sample2_idx:sample2_idx+3].copy()
        sample3 = df3.iloc[sample3_idx:sample3_idx+3].copy()

        # Create combined sample by adding the signals
        for axis_idx in range(3):
            row1 = sample1.iloc[axis_idx]
            row2 = sample2.iloc[axis_idx]
            row3 = sample3.iloc[axis_idx]

            # Add the vibration signals
            combined_values = row1[value_cols].values + row2[value_cols].values + row3[value_cols].values

            # Create new row
            new_row = {
                'timestamp': datetime.now().isoformat(),
                'axis': row1['axis']
            }

            # Add combined values
            for col, val in zip(value_cols, combined_values):
                new_row[col] = val

            combined_rows.append(new_row)

    combined_df = pd.DataFrame(combined_rows)
    logger.info(f"  Created {len(combined_df)} rows ({num_samples} samples × 3 axes)")

    return combined_df


def main():
    """Generate combined fault dataset."""
    logger.info("=" * 80)
    logger.info("Phase 2: Creating Combined Fault Dataset")
    logger.info("=" * 80)

    # Paths
    input_dir = Path('dataset/phase_2_3')
    output_dir = Path('dataset/phase_2_4_combined')
    output_dir.mkdir(parents=True, exist_ok=True)

    logger.info(f"\nInput directory: {input_dir}")
    logger.info(f"Output directory: {output_dir}")

    # Load individual fault datasets
    logger.info("\n" + "=" * 80)
    logger.info("STEP 1: Loading Individual Fault Data")
    logger.info("=" * 80)

    bearing_df = load_fault_data(input_dir / 'bearing_fault.csv')
    misalignment_df = load_fault_data(input_dir / 'misalignment_fault.csv')
    unbalance_df = load_fault_data(input_dir / 'unbalance_fault.csv')
    normal_df = load_fault_data(input_dir / 'normal.csv')

    # Copy individual faults to new directory
    logger.info("\n" + "=" * 80)
    logger.info("STEP 2: Copying Individual Fault Files")
    logger.info("=" * 80)

    for filename in ['bearing_fault.csv', 'misalignment_fault.csv', 'unbalance_fault.csv', 'normal.csv']:
        src = input_dir / filename
        dst = output_dir / filename
        logger.info(f"Copying: {filename}")
        pd.read_csv(src).to_csv(dst, index=False)

    # Generate combined faults
    logger.info("\n" + "=" * 80)
    logger.info("STEP 3: Generating Combined Fault Data")
    logger.info("=" * 80)

    # Determine sample count (use ~9000 samples for balance)
    target_samples = 9000

    # 1. Bearing + Misalignment
    logger.info("\n1. Bearing + Misalignment Fault")
    bearing_misalignment_df = create_combined_fault(
        bearing_df, misalignment_df,
        'bearing_fault', 'misalignment_fault',
        num_samples=target_samples
    )
    output_path = output_dir / 'bearing_misalignment_fault.csv'
    bearing_misalignment_df.to_csv(output_path, index=False)
    logger.info(f"✓ Saved: {output_path}")

    # 2. Bearing + Unbalance
    logger.info("\n2. Bearing + Unbalance Fault")
    bearing_unbalance_df = create_combined_fault(
        bearing_df, unbalance_df,
        'bearing_fault', 'unbalance_fault',
        num_samples=target_samples
    )
    output_path = output_dir / 'bearing_unbalance_fault.csv'
    bearing_unbalance_df.to_csv(output_path, index=False)
    logger.info(f"✓ Saved: {output_path}")

    # 3. Misalignment + Unbalance
    logger.info("\n3. Misalignment + Unbalance Fault")
    misalignment_unbalance_df = create_combined_fault(
        misalignment_df, unbalance_df,
        'misalignment_fault', 'unbalance_fault',
        num_samples=target_samples
    )
    output_path = output_dir / 'misalignment_unbalance_fault.csv'
    misalignment_unbalance_df.to_csv(output_path, index=False)
    logger.info(f"✓ Saved: {output_path}")

    # 4. Bearing + Misalignment + Unbalance (Triple Fault)
    logger.info("\n4. Bearing + Misalignment + Unbalance Fault (Triple)")
    triple_fault_df = create_triple_fault(
        bearing_df, misalignment_df, unbalance_df,
        'bearing_fault', 'misalignment_fault', 'unbalance_fault',
        num_samples=target_samples
    )
    output_path = output_dir / 'bearing_misalignment_unbalance_fault.csv'
    triple_fault_df.to_csv(output_path, index=False)
    logger.info(f"✓ Saved: {output_path}")

    # Summary
    logger.info("\n" + "=" * 80)
    logger.info("✅ PHASE 2 DATASET CREATED SUCCESSFULLY")
    logger.info("=" * 80)

    logger.info(f"\n📁 Output directory: {output_dir}")
    logger.info(f"\nDataset structure (8 classes):")
    logger.info(f"  Individual Faults (4):")
    logger.info(f"    - normal.csv")
    logger.info(f"    - bearing_fault.csv")
    logger.info(f"    - misalignment_fault.csv")
    logger.info(f"    - unbalance_fault.csv")
    logger.info(f"  Combined Faults (4):")
    logger.info(f"    - bearing_misalignment_fault.csv")
    logger.info(f"    - bearing_unbalance_fault.csv")
    logger.info(f"    - misalignment_unbalance_fault.csv")
    logger.info(f"    - bearing_misalignment_unbalance_fault.csv")

    # Calculate total samples
    total_samples = 0
    for file in output_dir.glob('*.csv'):
        df = pd.read_csv(file)
        num_samples = len(df) // 3
        total_samples += num_samples
        logger.info(f"    {file.name}: {num_samples:,} samples")

    logger.info(f"\n  Total: {total_samples:,} samples across 8 classes")
    logger.info("=" * 80)


if __name__ == "__main__":
    main()

"""
Create FFT visualization for all 4 fault conditions in a 2x2 grid.
"""

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from pathlib import Path
from scipy.fft import fft, fftfreq

# Set style for publication-quality plots
plt.style.use('seaborn-v0_8-poster')
plt.rcParams['figure.dpi'] = 300
plt.rcParams['savefig.dpi'] = 300
plt.rcParams['font.size'] = 12
plt.rcParams['font.family'] = 'sans-serif'

project_root = Path(__file__).parent.parent
output_dir = project_root / "visualizations"
output_dir.mkdir(exist_ok=True)

print("=" * 80)
print("Creating FFT Visualization for All Fault Conditions")
print("=" * 80)

# File mapping
data_dir = project_root / 'dataset/phase_2_2'
file_map = {
    'Normal': 'normal.csv',
    'Bearing Fault': 'bearing_fault.csv',
    'Misalignment': 'misalignment_fault.csv',
    'Unbalance': 'unbalance_fault.csv'
}

# Sampling parameters (adjust based on your actual sampling rate)
sampling_rate = 12800  # Hz (typical for vibration monitoring)
# If you know the actual sampling rate, update this value

# Create figure with 2x2 subplots
fig, axes = plt.subplots(2, 2, figsize=(16, 14))
axes = axes.flatten()

colors = ['#2ecc71', '#e74c3c', '#f39c12', '#3498db']
condition_order = ['Normal', 'Bearing Fault', 'Misalignment', 'Unbalance']

for idx, (condition, color) in enumerate(zip(condition_order, colors)):
    ax = axes[idx]

    print(f"\nProcessing {condition}...")

    # Load one sample from this condition
    file_path = data_dir / file_map[condition]

    # Read first 3 rows (one sample = X, Y, Z axes)
    df = pd.read_csv(file_path, nrows=3)

    # Extract the numerical columns (columns 1-1024)
    # The columns are named: timestamp, axis, 1, 2, 3, ..., 1024
    value_cols = [str(i) for i in range(1, 1025)]

    # Get X, Y, Z axis data
    x_data = df[df['axis'] == 'X'][value_cols].values.flatten()
    y_data = df[df['axis'] == 'Y'][value_cols].values.flatten()
    z_data = df[df['axis'] == 'Z'][value_cols].values.flatten()

    # Compute magnitude (resultant) of 3-axis vibration
    signal = np.sqrt(x_data**2 + y_data**2 + z_data**2)

    # Compute FFT
    N = len(signal)
    yf = fft(signal)
    xf = fftfreq(N, 1/sampling_rate)

    # Only take positive frequencies
    positive_freq_idx = xf > 0
    xf = xf[positive_freq_idx]
    yf = np.abs(yf[positive_freq_idx])

    # Normalize
    yf = yf / np.max(yf)

    # Plot FFT
    ax.plot(xf, yf, color=color, linewidth=1.5, alpha=0.8)
    ax.fill_between(xf, yf, alpha=0.3, color=color)

    # Styling
    ax.set_xlabel('Frequency (Hz)', fontsize=14, fontweight='bold')
    ax.set_ylabel('Normalized Amplitude', fontsize=14, fontweight='bold')
    ax.set_title(f'{condition}', fontsize=16, fontweight='bold', pad=15)
    ax.grid(True, alpha=0.3, linestyle='--')
    ax.set_xlim(0, sampling_rate/2)  # Nyquist frequency
    ax.set_ylim(0, 1.1)

    # Add background color
    ax.set_facecolor('#f8f9fa')

    print(f"✓ {condition} FFT computed")
    print(f"  Peak frequency: {xf[np.argmax(yf)]:.2f} Hz")

# Overall title
fig.suptitle('Frequency Domain Analysis - FFT of Vibration Signals',
             fontsize=20, fontweight='bold', y=0.995)

# Adjust layout
plt.tight_layout(rect=[0, 0, 1, 0.99])

# Save
output_path = output_dir / 'fft_all_conditions.png'
plt.savefig(output_path, bbox_inches='tight', facecolor='white')
plt.close()

print("\n" + "=" * 80)
print("✅ FFT visualization created successfully!")
print(f"📁 Saved to: {output_path}")
print("=" * 80)

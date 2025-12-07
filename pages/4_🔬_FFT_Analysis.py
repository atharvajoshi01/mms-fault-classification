"""
FFT Analysis page - Visualize frequency signatures of different fault types.

Shows the characteristic frequency patterns for each fault type.
"""

import streamlit as st
import numpy as np
import pandas as pd
from pathlib import Path
import sys
import plotly.graph_objects as go
from plotly.subplots import make_subplots

project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))


def compute_fft(signal: np.ndarray, sampling_rate: float = 1024) -> tuple:
    """
    Compute FFT of a signal.

    Args:
        signal: Time-domain signal
        sampling_rate: Sampling rate in Hz

    Returns:
        Tuple of (frequencies, magnitudes)
    """
    n = len(signal)
    fft_vals = np.fft.fft(signal)
    fft_mag = np.abs(fft_vals[:n//2]) * 2 / n
    freqs = np.fft.fftfreq(n, 1/sampling_rate)[:n//2]
    return freqs, fft_mag


def generate_synthetic_signal(fault_type: str, n_samples: int = 1024) -> np.ndarray:
    """
    Generate synthetic vibration signal for demonstration.

    Creates realistic frequency signatures for each fault type:
    - Normal: Low amplitude noise
    - Unbalance: Distinct frequency pattern
    - Misalignment: Multiple frequency peaks
    - Bearing: High-frequency noise components
    """
    t = np.linspace(0, 1, n_samples)
    running_speed = 20  # Hz (data collected at 15, 20, 25 Hz)

    # Base noise for all signals
    noise = 0.05 * np.random.randn(n_samples, 3)

    if fault_type == 'normal':
        # Low amplitude, mostly noise
        signal = 0.1 * np.sin(2 * np.pi * running_speed * t)[:, np.newaxis] * np.array([1, 0.8, 0.6])
        signal += noise

    elif fault_type == 'unbalance_fault':
        # Distinct frequency component
        signal = np.zeros((n_samples, 3))
        signal[:, 0] = 0.8 * np.sin(2 * np.pi * running_speed * t)  # Strong X
        signal[:, 1] = 0.6 * np.sin(2 * np.pi * running_speed * t + np.pi/4)  # Y phase shift
        signal[:, 2] = 0.3 * np.sin(2 * np.pi * running_speed * t + np.pi/2)  # Weaker Z
        signal += noise

    elif fault_type == 'misalignment_fault':
        # Multiple frequency peaks pattern
        signal = np.zeros((n_samples, 3))
        # First frequency component
        signal += 0.3 * np.sin(2 * np.pi * running_speed * t)[:, np.newaxis] * np.array([1, 0.8, 0.6])
        # Second frequency component (characteristic of misalignment)
        signal += 0.7 * np.sin(2 * np.pi * 2 * running_speed * t)[:, np.newaxis] * np.array([1, 0.9, 0.7])
        # Third frequency component
        signal += 0.4 * np.sin(2 * np.pi * 3 * running_speed * t)[:, np.newaxis] * np.array([0.8, 1, 0.6])
        signal += noise

    elif fault_type == 'bearing_fault':
        # High-frequency components
        signal = np.zeros((n_samples, 3))
        # Some low frequency
        signal += 0.2 * np.sin(2 * np.pi * running_speed * t)[:, np.newaxis] * np.array([1, 0.8, 0.6])
        # High-frequency bearing defect frequencies
        for freq in [45, 67, 89, 112]:
            amplitude = 0.3 * np.random.uniform(0.5, 1.0)
            phase = np.random.uniform(0, 2*np.pi)
            signal += amplitude * np.sin(2 * np.pi * freq * t + phase)[:, np.newaxis] * np.array([1, 0.9, 0.8])
        signal += 2 * noise  # More noise for bearing faults
    else:
        signal = noise

    return signal.astype(np.float32)


def load_sample_data(fault_type: str, num_samples: int = 5) -> np.ndarray:
    """Load sample data for a fault type from pre-extracted samples."""
    sample_dir = project_root / "sample_data"
    file_map = {
        'normal': 'normal_samples.npy',
        'unbalance_fault': 'unbalance_fault_samples.npy',
        'misalignment_fault': 'misalignment_fault_samples.npy',
        'bearing_fault': 'bearing_fault_samples.npy'
    }

    filepath = sample_dir / file_map.get(fault_type, 'normal_samples.npy')

    # Load pre-extracted real samples
    if filepath.exists():
        data = np.load(filepath)
        if len(data) > num_samples:
            indices = np.random.choice(len(data), num_samples, replace=False)
            return data[indices]
        return data

    # Fall back to synthetic data only if sample files missing
    return np.array([generate_synthetic_signal(fault_type) for _ in range(num_samples)])


def plot_fft_comparison(signal1: np.ndarray, signal2: np.ndarray,
                        label1: str, label2: str, axis: int = 0) -> go.Figure:
    """
    Create side-by-side FFT comparison plot.

    Args:
        signal1: First signal (timesteps, channels)
        signal2: Second signal (timesteps, channels)
        label1: Label for first signal
        label2: Label for second signal
        axis: Which axis to plot (0=X, 1=Y, 2=Z)
    """
    axis_names = ['X', 'Y', 'Z']

    # Compute FFTs
    freq1, mag1 = compute_fft(signal1[:, axis])
    freq2, mag2 = compute_fft(signal2[:, axis])

    # Create subplot
    fig = make_subplots(
        rows=2, cols=2,
        subplot_titles=(
            f'{label1} - Time Domain', f'{label2} - Time Domain',
            f'{label1} - Frequency Domain (FFT)', f'{label2} - Frequency Domain (FFT)'
        ),
        vertical_spacing=0.15,
        horizontal_spacing=0.1
    )

    # Time domain signals
    time = np.arange(len(signal1))
    fig.add_trace(
        go.Scatter(x=time, y=signal1[:, axis], name=f'{label1} Time',
                   line=dict(color='#FF6B6B', width=1)),
        row=1, col=1
    )
    fig.add_trace(
        go.Scatter(x=time, y=signal2[:, axis], name=f'{label2} Time',
                   line=dict(color='#4ECDC4', width=1)),
        row=1, col=2
    )

    # FFT - focus on low frequencies where fault signatures are
    freq_mask1 = freq1 < 50  # Show up to 50 Hz
    freq_mask2 = freq2 < 50

    fig.add_trace(
        go.Scatter(x=freq1[freq_mask1], y=mag1[freq_mask1], name=f'{label1} FFT',
                   fill='tozeroy', fillcolor='rgba(255, 107, 107, 0.3)',
                   line=dict(color='#FF6B6B', width=2)),
        row=2, col=1
    )
    fig.add_trace(
        go.Scatter(x=freq2[freq_mask2], y=mag2[freq_mask2], name=f'{label2} FFT',
                   fill='tozeroy', fillcolor='rgba(78, 205, 196, 0.3)',
                   line=dict(color='#4ECDC4', width=2)),
        row=2, col=2
    )

    fig.update_layout(
        height=700,
        showlegend=False,
        title_text=f"FFT Comparison - {axis_names[axis]}-Axis"
    )

    fig.update_xaxes(title_text="Sample", row=1, col=1)
    fig.update_xaxes(title_text="Sample", row=1, col=2)
    fig.update_xaxes(title_text="Frequency (Hz)", row=2, col=1)
    fig.update_xaxes(title_text="Frequency (Hz)", row=2, col=2)
    fig.update_yaxes(title_text="Amplitude", row=1, col=1)
    fig.update_yaxes(title_text="Magnitude", row=2, col=1)

    return fig


def plot_harmonic_analysis(signal: np.ndarray, label: str) -> go.Figure:
    """
    Create detailed frequency analysis plot for all axes.
    """
    fig = make_subplots(
        rows=1, cols=3,
        subplot_titles=('X-Axis FFT', 'Y-Axis FFT', 'Z-Axis FFT'),
        horizontal_spacing=0.08
    )

    colors = ['#FF6B6B', '#4ECDC4', '#45B7D1']

    for i, (color, axis_name) in enumerate(zip(colors, ['X', 'Y', 'Z'])):
        freq, mag = compute_fft(signal[:, i])
        freq_mask = freq < 100

        fig.add_trace(
            go.Scatter(
                x=freq[freq_mask],
                y=mag[freq_mask],
                name=f'{axis_name}-Axis',
                fill='tozeroy',
                fillcolor=f'rgba{tuple(list(int(color[i:i+2], 16) for i in (1, 3, 5)) + [0.3])}',
                line=dict(color=color, width=2)
            ),
            row=1, col=i+1
        )

    fig.update_layout(
        height=400,
        title_text=f"Frequency Analysis - {label}",
        showlegend=False
    )

    for i in range(1, 4):
        fig.update_xaxes(title_text="Frequency (Hz)", row=1, col=i)
        fig.update_yaxes(title_text="Magnitude" if i == 1 else "", row=1, col=i)

    return fig


def show():
    """Display FFT analysis page."""

    st.markdown("# FFT Analysis - Fault Frequency Signatures")

    st.markdown("""
    ### Understanding Vibration Fault Patterns

    Different mechanical faults create characteristic frequency signatures in vibration data:

    | Fault Type | Frequency Pattern | Description |
    |------------|-------------------|-------------|
    | **Unbalance** | Distinct low-frequency peak | Mass imbalance causes characteristic vibration |
    | **Misalignment** | Multiple frequency peaks | Shaft misalignment creates complex patterns |
    | **Bearing Fault** | High-frequency content | Bearing defects create high-frequency noise |
    """)

    st.markdown("---")

    # Fault comparison selector
    st.markdown("## Compare Fault Signatures")

    col1, col2 = st.columns(2)

    with col1:
        fault1 = st.selectbox(
            "First Fault Type",
            ['unbalance_fault', 'misalignment_fault', 'bearing_fault', 'normal'],
            format_func=lambda x: x.replace('_', ' ').title()
        )

    with col2:
        fault2 = st.selectbox(
            "Second Fault Type",
            ['misalignment_fault', 'unbalance_fault', 'bearing_fault', 'normal'],
            format_func=lambda x: x.replace('_', ' ').title()
        )

    axis_select = st.radio("Select Axis", ['X', 'Y', 'Z'], horizontal=True)
    axis_idx = ['X', 'Y', 'Z'].index(axis_select)

    # Check if sample data is available
    sample_dir = project_root / "sample_data"
    has_sample_data = (sample_dir / "normal_samples.npy").exists()

    if has_sample_data:
        st.success("**Using real vibration samples** extracted from the training dataset.")

    if st.button("Generate FFT Comparison", type="primary"):
        with st.spinner("Computing FFT..."):
            try:
                # Load samples (will use synthetic if real data unavailable)
                data1 = load_sample_data(fault1, num_samples=1)
                data2 = load_sample_data(fault2, num_samples=1)

                signal1 = data1[0]
                signal2 = data2[0]

                # Create comparison plot
                fig = plot_fft_comparison(
                    signal1, signal2,
                    fault1.replace('_', ' ').title(),
                    fault2.replace('_', ' ').title(),
                    axis=axis_idx
                )
                st.plotly_chart(fig, use_container_width=True)

                # Key observations
                st.markdown("### Key Observations")

                if 'unbalance' in fault1 or 'unbalance' in fault2:
                    st.info("""
                    **Unbalance Signature:** Look for a distinct low-frequency peak.
                    This indicates mass imbalance - one side of the rotating element is heavier.
                    """)

                if 'misalignment' in fault1 or 'misalignment' in fault2:
                    st.warning("""
                    **Misalignment Signature:** Look for multiple frequency peaks in the spectrum.
                    Misaligned shafts create complex vibration patterns.
                    """)

                if 'bearing' in fault1 or 'bearing' in fault2:
                    st.error("""
                    **Bearing Fault Signature:** Look for high-frequency components (>40 Hz).
                    Bearing defects create characteristic frequencies based on bearing geometry.
                    """)

            except Exception as e:
                st.error(f"Error: {e}")
                import traceback
                st.code(traceback.format_exc())

    st.markdown("---")

    # Individual fault analysis
    st.markdown("## Detailed Frequency Analysis")

    fault_type = st.selectbox(
        "Select Fault Type for Analysis",
        ['unbalance_fault', 'misalignment_fault', 'bearing_fault', 'normal'],
        format_func=lambda x: x.replace('_', ' ').title(),
        key="harmonic_select"
    )

    if st.button("Analyze Frequencies"):
        with st.spinner("Computing frequency analysis..."):
            try:
                data = load_sample_data(fault_type, num_samples=1)

                fig = plot_harmonic_analysis(
                    data[0],
                    fault_type.replace('_', ' ').title()
                )
                st.plotly_chart(fig, use_container_width=True)

                # Add interpretation
                st.markdown("### Interpretation")
                if fault_type == 'unbalance_fault':
                    st.success("Notice the distinct low-frequency peak - this is "
                              "the characteristic signature of mass unbalance.")
                elif fault_type == 'misalignment_fault':
                    st.success("Notice the multiple frequency peaks - this complex pattern "
                              "indicates shaft misalignment.")
                elif fault_type == 'bearing_fault':
                    st.success("Notice the broad high-frequency content - bearing defects create "
                              "characteristic high-frequency patterns.")

            except Exception as e:
                st.error(f"Error: {e}")

    st.markdown("---")

    # Educational content
    with st.expander("Learn More: How MiniRocket Uses These Patterns"):
        st.markdown("""
        ### MiniRocket Feature Extraction

        MiniRocket doesn't directly compute FFT. Instead, it uses **random convolutional kernels**
        that automatically learn to detect these frequency patterns:

        1. **10,000 Random Kernels**: Each kernel captures different frequency components
        2. **PPV Features**: Proportion of Positive Values extracts signal characteristics
        3. **RidgeClassifier**: Learns which kernel responses indicate each fault type

        ### Why This Works

        - Unbalance creates periodic signals → specific kernels fire consistently
        - Misalignment creates complex patterns → different kernel combinations activate
        - Bearing faults create high-frequency noise → yet another kernel response pattern

        The model learns these patterns from data without explicit FFT computation,
        making it faster while achieving similar or better accuracy.
        """)


show()

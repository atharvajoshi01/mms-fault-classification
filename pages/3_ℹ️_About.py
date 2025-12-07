"""
About page - MiniRocket algorithm and fault detection explanation.
"""

import streamlit as st
import plotly.graph_objects as go
import numpy as np
from pathlib import Path

project_root = Path(__file__).parent.parent


def create_fft_comparison():
    """Create FFT visualization showing fault frequency signatures."""
    # Simulate frequency domain signatures
    freq = np.linspace(0, 500, 1000)
    rotation_freq = 60  # 60 Hz = 3600 RPM

    # Normal - low flat spectrum
    normal = 0.1 * np.ones_like(freq) + 0.05 * np.random.randn(len(freq))
    normal = np.maximum(normal, 0)

    # Unbalance - strong 1x peak
    unbalance = 0.1 * np.ones_like(freq)
    unbalance += 0.8 * np.exp(-((freq - rotation_freq) ** 2) / 50)  # 1x peak
    unbalance += 0.05 * np.random.randn(len(freq))
    unbalance = np.maximum(unbalance, 0)

    # Misalignment - strong 2x peak, some 1x and 3x
    misalignment = 0.1 * np.ones_like(freq)
    misalignment += 0.3 * np.exp(-((freq - rotation_freq) ** 2) / 50)  # 1x
    misalignment += 0.9 * np.exp(-((freq - 2*rotation_freq) ** 2) / 50)  # 2x peak
    misalignment += 0.4 * np.exp(-((freq - 3*rotation_freq) ** 2) / 50)  # 3x
    misalignment += 0.05 * np.random.randn(len(freq))
    misalignment = np.maximum(misalignment, 0)

    # Bearing - high frequency harmonics
    bearing = 0.1 * np.ones_like(freq)
    for harmonic in [150, 200, 250, 300, 350, 400]:
        bearing += 0.5 * np.exp(-((freq - harmonic) ** 2) / 100)
    bearing += 0.08 * np.random.randn(len(freq))
    bearing = np.maximum(bearing, 0)

    fig = go.Figure()

    fig.add_trace(go.Scatter(x=freq, y=normal, name='Normal',
                             line=dict(color='green', width=2)))
    fig.add_trace(go.Scatter(x=freq, y=unbalance, name='Unbalance (1×)',
                             line=dict(color='orange', width=2)))
    fig.add_trace(go.Scatter(x=freq, y=misalignment, name='Misalignment (2×)',
                             line=dict(color='red', width=2)))
    fig.add_trace(go.Scatter(x=freq, y=bearing, name='Bearing (HF)',
                             line=dict(color='blue', width=2)))

    # Add vertical lines for reference frequencies
    fig.add_vline(x=rotation_freq, line_dash="dash", line_color="gray",
                  annotation_text="1× (60Hz)")
    fig.add_vline(x=2*rotation_freq, line_dash="dash", line_color="gray",
                  annotation_text="2× (120Hz)")
    fig.add_vline(x=3*rotation_freq, line_dash="dash", line_color="gray",
                  annotation_text="3× (180Hz)")

    fig.update_layout(
        title="Frequency Signatures by Fault Type",
        xaxis_title="Frequency (Hz)",
        yaxis_title="Amplitude",
        height=400,
        legend=dict(orientation="h", yanchor="bottom", y=1.02),
        margin=dict(l=50, r=50, t=80, b=50)
    )

    return fig


def show():
    """Display about page."""

    st.markdown("# About MMS Fault Classification")

    # MiniRocket Section
    st.markdown("## 🧠 MiniRocket Algorithm")

    col1, col2 = st.columns([1, 1])

    with col1:
        st.markdown("""
        **MiniRocket** (MINImally RandOm Convolutional KErnel Transform) is a
        state-of-the-art time series classification algorithm.

        ### How It Works

        1. **Random Kernels**: Generate 10,000 random convolutional kernels
        2. **Feature Extraction**: Apply kernels to time series data
        3. **PPV Pooling**: Calculate Proportion of Positive Values
        4. **Classification**: Ridge Regression on extracted features

        ### Why MiniRocket?

        | Advantage | Benefit |
        |-----------|---------|
        | **75× faster** | Than original ROCKET |
        | **No GPU needed** | CPU-only training |
        | **No tuning** | Works with defaults |
        | **Lightweight** | 1.7 MB model size |
        """)

    with col2:
        st.markdown("""
        ### Architecture Diagram
        ```
        Input Signal (1024 × 3)
               ↓
        ┌─────────────────────┐
        │  10,000 Random      │
        │  Convolutional      │
        │  Kernels            │
        └─────────────────────┘
               ↓
        ┌─────────────────────┐
        │  PPV Pooling        │
        │  (29,988 features)  │
        └─────────────────────┘
               ↓
        ┌─────────────────────┐
        │  StandardScaler     │
        └─────────────────────┘
               ↓
        ┌─────────────────────┐
        │  Ridge Classifier   │
        │  (α = 1.0)          │
        └─────────────────────┘
               ↓
        Output: Fault Class
        ```
        """)

    st.markdown("---")

    # Fault Frequency Analysis
    st.markdown("## 📊 Fault Frequency Signatures")

    st.markdown("""
    Different fault types produce distinct **vibration frequency patterns**:

    - **Unbalance**: Dominant peak at **1× rotation frequency** (synchronous vibration)
    - **Misalignment**: Strong peaks at **2× and 3× rotation frequency** (harmonics)
    - **Bearing**: **High-frequency** harmonics (defect frequencies)
    """)

    # FFT Comparison Chart
    fig = create_fft_comparison()
    st.plotly_chart(fig, use_container_width=True)

    st.info("""
    **Key Insight**: At 3600 RPM (60 Hz rotation), unbalance shows 1× peak at 60 Hz,
    while misalignment shows dominant 2× peak at 120 Hz. MiniRocket learns these
    patterns automatically from raw time-domain data.
    """)

    st.markdown("---")

    # Performance Summary
    st.markdown("## 📈 Model Performance")

    col1, col2, col3 = st.columns(3)

    with col1:
        st.metric("Test Accuracy", "99.96%")
        st.metric("Training Samples", "28,842")

    with col2:
        st.metric("Precision", "99.96%")
        st.metric("Recall", "99.96%")

    with col3:
        st.metric("F1-Score", "99.96%")
        st.metric("Model Size", "1.7 MB")

    st.markdown("""
    | Fault Type | F1-Score | Detection |
    |------------|----------|-----------|
    | Bearing | 100.00% | Perfect |
    | Misalignment | 100.00% | Perfect |
    | Normal | 99.92% | Excellent |
    | Unbalance | 99.92% | Excellent |
    """)

    st.markdown("---")

    # Future Directions
    st.markdown("## 🚀 Future Directions")

    st.markdown("""
    - **Speed Variation Testing**: Train with varying RPM conditions
    - **Edge Deployment**: Port to STM32 microcontroller
    - **Additional Faults**: Looseness, cavitation, resonance
    - **Attention Mechanisms**: Explore transformer-based models
    - **Real-time Monitoring**: Continuous streaming classification
    """)

    st.markdown("---")

    # Citation
    st.markdown("## 📚 Reference")
    st.code("""
Dempster, A., Schmidt, D. F., & Webb, G. I. (2021).
MiniRocket: A Very Fast (Almost) Deterministic Transform
for Time Series Classification.
KDD 2021.
    """)


show()

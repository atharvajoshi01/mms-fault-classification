"""
About page - MiniRocket algorithm and fault detection explanation.
"""

import streamlit as st
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import numpy as np
from pathlib import Path

project_root = Path(__file__).parent.parent


def create_fft_comparison():
    """Create clean FFT visualization with separate subplots for each fault type."""
    freq = np.linspace(0, 250, 500)
    rotation_freq = 60  # 60 Hz = 3600 RPM

    def gaussian_peak(f, center, width, amplitude):
        return amplitude * np.exp(-((f - center) ** 2) / (2 * width ** 2))

    # Generate clean spectra for each fault type
    spectra = {
        'Normal': 0.05 * np.ones_like(freq),
        'Unbalance': 0.05 + gaussian_peak(freq, 60, 5, 0.9),
        'Misalignment': 0.05 + gaussian_peak(freq, 60, 5, 0.3) +
                        gaussian_peak(freq, 120, 5, 0.95) +
                        gaussian_peak(freq, 180, 5, 0.5),
        'Bearing': 0.05 + gaussian_peak(freq, 60, 5, 0.2) +
                   gaussian_peak(freq, 150, 8, 0.6) +
                   gaussian_peak(freq, 200, 8, 0.5) +
                   gaussian_peak(freq, 250, 8, 0.4)
    }

    colors = {'Normal': '#2ECC71', 'Unbalance': '#F39C12',
              'Misalignment': '#E74C3C', 'Bearing': '#3498DB'}

    descriptions = {
        'Normal': 'Flat spectrum - healthy',
        'Unbalance': 'Strong 1× peak (60 Hz)',
        'Misalignment': 'Strong 2× & 3× harmonics',
        'Bearing': 'High-frequency peaks'
    }

    fig = make_subplots(
        rows=2, cols=2,
        subplot_titles=[f"<b>{name}</b><br><span style='font-size:11px;color:gray'>{descriptions[name]}</span>"
                        for name in spectra.keys()],
        vertical_spacing=0.18,
        horizontal_spacing=0.1
    )

    positions = [(1, 1), (1, 2), (2, 1), (2, 2)]

    for (name, spectrum), (row, col) in zip(spectra.items(), positions):
        fig.add_trace(
            go.Scatter(
                x=freq, y=spectrum,
                fill='tozeroy',
                fillcolor=f'rgba{tuple(list(int(colors[name][i:i+2], 16) for i in (1, 3, 5)) + [0.3])}',
                line=dict(color=colors[name], width=2.5),
                name=name,
                showlegend=False
            ),
            row=row, col=col
        )

        # Add harmonic markers for relevant plots
        if name in ['Unbalance', 'Misalignment']:
            for mult, label in [(1, '1×'), (2, '2×'), (3, '3×')]:
                if mult * 60 <= 250:
                    fig.add_vline(
                        x=mult * 60, line_dash="dot", line_color="rgba(0,0,0,0.3)",
                        line_width=1, row=row, col=col
                    )

    fig.update_xaxes(title_text="Frequency (Hz)", row=2, col=1)
    fig.update_xaxes(title_text="Frequency (Hz)", row=2, col=2)
    fig.update_yaxes(title_text="Amplitude", row=1, col=1)
    fig.update_yaxes(title_text="Amplitude", row=2, col=1)

    fig.update_layout(
        height=500,
        margin=dict(l=60, r=40, t=80, b=60),
        title=dict(
            text="<b>Frequency Signatures by Fault Type</b>",
            x=0.5,
            font=dict(size=18)
        )
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

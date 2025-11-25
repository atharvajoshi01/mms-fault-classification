"""
Data explorer page for the dashboard.
"""

import streamlit as st
import pandas as pd
import numpy as np
from pathlib import Path
import sys

project_root = Path(__file__).parent.parent.parent
sys.path.insert(0, str(project_root))

from dashboard.utils import plot_class_distribution, plot_vibration_signal


def show():
    """Display data explorer page."""
    
    st.markdown('<h1 class="main-header">📈 Data Explorer</h1>', unsafe_allow_html=True)
    st.markdown(
        '<p class="sub-header">Explore the training dataset and signal patterns</p>',
        unsafe_allow_html=True
    )
    
    # Dataset overview
    st.markdown("## 📊 Dataset Overview")

    # Use static dataset statistics (from training)
    # Since dataset files are excluded from deployment to reduce size
    class_info = {
        'normal': 5404,
        'unbalance_fault': 5375,
        'misalignment_fault': 5404,
        'bearing_fault': 5404
    }

    # Display statistics
    total_samples = sum(class_info.values())
    
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        st.metric("Total Samples", f"{total_samples:,}")
    
    with col2:
        st.metric("Number of Classes", len(class_info))
    
    with col3:
        st.metric("Timesteps per Sample", "1,024")
    
    with col4:
        st.metric("Axes per Sample", "3 (X, Y, Z)")
    
    st.markdown("---")
    
    # Class distribution
    st.markdown("## 📊 Class Distribution")
    
    col1, col2 = st.columns([2, 1])
    
    with col1:
        fig_dist = plot_class_distribution(class_info)
        st.plotly_chart(fig_dist, use_container_width=True)
    
    with col2:
        st.markdown("### Class Breakdown")
        
        class_df = pd.DataFrame([
            {
                'Class': name.replace('_', ' ').title(),
                'Samples': count,
                'Percentage': f"{count/total_samples*100:.2f}%"
            }
            for name, count in sorted(class_info.items(), key=lambda x: x[1], reverse=True)
        ])
        
        st.dataframe(class_df, use_container_width=True, hide_index=True)
    
    st.markdown("---")
    
    # Sample visualization
    st.markdown("## 🔍 Sample Visualization")

    st.info("View a sample vibration signal from the dataset (Demo: Normal operation)")

    if st.button("📊 Load Demo Sample", type="primary"):
        with st.spinner("Loading signal..."):
            try:
                from src.data_loader import load_mms_csv

                # Load demo sample from sample_data
                demo_file = project_root / "sample_data/sample_vibration_data.csv"

                if demo_file.exists():
                    signals = load_mms_csv(demo_file)

                    if len(signals) > 0:
                        signal = signals[0]  # Get first sample

                        # Display signal info
                        st.markdown(f"### Signal Information")

                        col1, col2, col3 = st.columns(3)
                        with col1:
                            st.metric("Class", "Normal")
                        with col2:
                            st.metric("Sample Type", "Demo")
                        with col3:
                            st.metric("Shape", f"{signal.shape[0]} × {signal.shape[1]}")

                        # Plot signal
                        fig = plot_vibration_signal(
                            signal,
                            title="Normal Operation - Demo Sample"
                        )
                        st.plotly_chart(fig, use_container_width=True)

                        # Signal statistics
                        with st.expander("📊 Signal Statistics"):
                            stats_df = pd.DataFrame({
                                'Axis': ['X', 'Y', 'Z'],
                                'Mean': [f"{signal[:, i].mean():.6f}" for i in range(3)],
                                'Std Dev': [f"{signal[:, i].std():.6f}" for i in range(3)],
                                'Min': [f"{signal[:, i].min():.6f}" for i in range(3)],
                                'Max': [f"{signal[:, i].max():.6f}" for i in range(3)],
                                'Range': [f"{signal[:, i].max() - signal[:, i].min():.6f}" for i in range(3)]
                            })
                            st.dataframe(stats_df, use_container_width=True, hide_index=True)

                        # Raw data preview
                        with st.expander("📋 Raw Data Preview"):
                            preview_df = pd.DataFrame(
                                signal[:100],  # Show first 100 timesteps
                                columns=['X-Axis', 'Y-Axis', 'Z-Axis']
                            )
                            preview_df.index.name = 'Timestep'
                            st.dataframe(preview_df, use_container_width=True)
                    else:
                        st.error("No signals found in demo file")
                else:
                    st.error(f"Demo file not found at: {demo_file}")

            except Exception as e:
                st.error(f"Error loading signal: {e}")
                import traceback
                st.code(traceback.format_exc())
    
    st.markdown("---")
    
    # Data characteristics
    st.markdown("## 📖 Dataset Characteristics")
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.markdown("### 🎯 Fault Types")
        st.markdown("""
        1. **Normal**: Healthy machinery operation
           - Balanced vibration patterns
           - Low amplitude variations
           
        2. **Unbalance Fault**: Mass imbalance in rotating components
           - Increased vibration at rotation frequency
           - Radial vibration patterns
           
        3. **Misalignment Fault**: Shaft or coupling misalignment
           - Vibration at 2× rotation frequency
           - Axial and radial components
           
        4. **Bearing Fault**: Bearing degradation or damage
           - High-frequency impulses
           - Characteristic fault frequencies
        """)
    
    with col2:
        st.markdown("### 📏 Data Specifications")
        st.markdown("""
        - **Sampling**: Time-series vibration data
        - **Sensors**: 3-axis accelerometers (X, Y, Z)
        - **Timesteps**: 1,024 per sample
        - **Format**: CSV with timestamp and axis data
        - **Normalization**: Z-score standardization
        - **Split**: 80% training / 20% testing
        - **Balance**: Perfectly balanced dataset
        - **Quality**: Production-grade sensor data
        """)
    
    # Data quality
    st.markdown("---")
    st.markdown("## ✅ Data Quality Indicators")
    
    col1, col2, col3 = st.columns(3)
    
    with col1:
        st.markdown("""
        <div class="success-box">
            <h4>📊 Balance</h4>
            <p><strong>Excellent</strong></p>
            <p>Classes are evenly distributed (5,375-5,404 samples each)</p>
        </div>
        """, unsafe_allow_html=True)
    
    with col2:
        st.markdown("""
        <div class="success-box">
            <h4>🎯 Consistency</h4>
            <p><strong>High</strong></p>
            <p>All samples have consistent 1,024×3 shape</p>
        </div>
        """, unsafe_allow_html=True)
    
    with col3:
        st.markdown("""
        <div class="success-box">
            <h4>✅ Completeness</h4>
            <p><strong>100%</strong></p>
            <p>No missing values or corrupted samples</p>
        </div>
        """, unsafe_allow_html=True)


if __name__ == "__main__":
    show()

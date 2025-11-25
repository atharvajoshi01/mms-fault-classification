"""
MMS Fault Classification Dashboard - Home Page
"""

import streamlit as st
import json
from pathlib import Path
import sys

# Add project root to path
project_root = Path(__file__).parent
sys.path.insert(0, str(project_root))

# Page configuration
st.set_page_config(
    page_title="MMS Fault Classification",
    page_icon="🔧",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS for better styling
st.markdown("""
<style>
    .main-header {
        font-size: 3rem;
        font-weight: bold;
        color: #0068C9;
        text-align: center;
        margin-bottom: 1rem;
    }
    .sub-header {
        font-size: 1.5rem;
        color: #31333F;
        text-align: center;
        margin-bottom: 2rem;
    }
    .metric-card {
        background-color: #f0f2f6;
        padding: 20px;
        border-radius: 10px;
        text-align: center;
        box-shadow: 0 2px 4px rgba(0,0,0,0.1);
    }
    .success-box {
        padding: 1rem;
        border-radius: 0.5rem;
        background-color: #d4edda;
        border: 1px solid #c3e6cb;
        color: #155724;
        margin: 1rem 0;
    }
    .info-box {
        padding: 1rem;
        border-radius: 0.5rem;
        background-color: #d1ecf1;
        border: 1px solid #bee5eb;
        color: #0c5460;
        margin: 1rem 0;
    }
    .stButton>button {
        width: 100%;
        background-color: #0068C9;
        color: white;
        font-weight: bold;
        padding: 0.5rem 1rem;
        border-radius: 0.5rem;
    }
    .stButton>button:hover {
        background-color: #0056b3;
        border-color: #0056b3;
    }
</style>
""", unsafe_allow_html=True)

# Header
st.markdown('<h1 class="main-header">🔧 MMS Fault Classification System</h1>', unsafe_allow_html=True)
st.markdown(
    '<p class="sub-header">State-of-the-art vibration fault detection using MiniRocket</p>',
    unsafe_allow_html=True
)

# Load metadata
metadata_path = project_root / "models/minirocket/metadata.json"
try:
    with open(metadata_path, 'r') as f:
        metadata = json.load(f)
except Exception as e:
    st.error(f"Error loading model metadata: {e}")
    metadata = {}

# Key metrics
st.markdown("## 📊 Key Performance Metrics")

col1, col2, col3, col4 = st.columns(4)

with col1:
    st.markdown(
        f"""
        <div class="metric-card">
            <h4 style="margin: 0; color: #31333F;">Test Accuracy</h4>
            <h2 style="margin: 10px 0; color: #0068C9;">{metadata.get('test_accuracy', 0)*100:.2f}%</h2>
            <p style="color: green; font-size: 14px; margin: 0;">World-class</p>
        </div>
        """,
        unsafe_allow_html=True
    )

with col2:
    st.markdown(
        f"""
        <div class="metric-card">
            <h4 style="margin: 0; color: #31333F;">Training Time</h4>
            <h2 style="margin: 10px 0; color: #0068C9;">{metadata.get('training_time_seconds', 0):.0f}s</h2>
            <p style="color: green; font-size: 14px; margin: 0;">Ultra-fast</p>
        </div>
        """,
        unsafe_allow_html=True
    )

with col3:
    st.markdown(
        f"""
        <div class="metric-card">
            <h4 style="margin: 0; color: #31333F;">Total Samples</h4>
            <h2 style="margin: 10px 0; color: #0068C9;">{metadata.get('train_samples', 0) + metadata.get('test_samples', 0):,}</h2>
            <p style="color: green; font-size: 14px; margin: 0;">Robust dataset</p>
        </div>
        """,
        unsafe_allow_html=True
    )

with col4:
    st.markdown(
        f"""
        <div class="metric-card">
            <h4 style="margin: 0; color: #31333F;">Model Size</h4>
            <h2 style="margin: 10px 0; color: #0068C9;">1.7 MB</h2>
            <p style="color: green; font-size: 14px; margin: 0;">Lightweight</p>
        </div>
        """,
        unsafe_allow_html=True
    )

st.markdown("---")

# Feature highlights
col1, col2 = st.columns(2)

with col1:
    st.markdown("## ✨ Key Features")
    st.markdown("""
    - **🎯 99.98% Accuracy**: State-of-the-art performance on vibration fault classification
    - **⚡ Fast Training**: Trains in under 3 minutes
    - **🚀 Real-time Inference**: Instant predictions on new data
    - **📊 Multi-axis Support**: Handles X, Y, Z vibration data
    - **🔍 4-class Detection**: Normal, Unbalance, Misalignment, Bearing faults
    - **💾 Lightweight**: Only 1.7 MB model size
    """)

with col2:
    st.markdown("## 🔬 Technical Specs")
    st.markdown(f"""
    - **Model**: MiniRocket (Random Convolutional Kernel Transform)
    - **Kernels**: {metadata.get('num_kernels', 10000):,} random kernels
    - **Classifier**: Ridge Regression (α=1.0)
    - **Input Shape**: (1024, 3) - 1024 timesteps × 3 channels
    - **Feature Space**: 29,988 features
    - **Preprocessing**: Z-score standardization
    """)

st.markdown("---")

# Fault classes
st.markdown("## 🔧 Fault Classes Detected")

col1, col2, col3, col4 = st.columns(4)

with col1:
    st.markdown("""
    <div style="text-align: center; padding: 20px; background-color: #e8f4f8; border-radius: 10px;">
        <h3>✅ Normal</h3>
        <p>Healthy operation</p>
    </div>
    """, unsafe_allow_html=True)

with col2:
    st.markdown("""
    <div style="text-align: center; padding: 20px; background-color: #fff3cd; border-radius: 10px;">
        <h3>⚠️ Unbalance</h3>
        <p>Mass imbalance fault</p>
    </div>
    """, unsafe_allow_html=True)

with col3:
    st.markdown("""
    <div style="text-align: center; padding: 20px; background-color: #f8d7da; border-radius: 10px;">
        <h3>🔄 Misalignment</h3>
        <p>Shaft misalignment</p>
    </div>
    """, unsafe_allow_html=True)

with col4:
    st.markdown("""
    <div style="text-align: center; padding: 20px; background-color: #d1ecf1; border-radius: 10px;">
        <h3>⚙️ Bearing Fault</h3>
        <p>Bearing degradation</p>
    </div>
    """, unsafe_allow_html=True)

st.markdown("---")

# Quick start guide
st.markdown("## 🚀 Quick Start Guide")

st.markdown("""
1. **📤 Upload Data**: Go to "Prediction" page and upload a CSV file
2. **🎯 Get Predictions**: Receive instant fault classification
3. **📊 Analyze Results**: View detailed metrics and visualizations
4. **💾 Export**: Download predictions for further analysis
""")

# Info box
st.markdown("""
<div class="info-box">
    <h4>📝 CSV Format Requirements</h4>
    <p>Your CSV file should contain vibration data with columns for timestamp, axis (X/Y/Z), and amplitude values.
    The system automatically handles multi-axis data and generates predictions for each sample.</p>
</div>
""", unsafe_allow_html=True)

# Call to action
st.markdown("---")
col1, col2, col3 = st.columns([1, 2, 1])
with col2:
    st.markdown("""
    <div style="text-align: center;">
        <h3>Ready to start detecting faults?</h3>
        <p>Navigate to <strong>Prediction</strong> to begin!</p>
    </div>
    """, unsafe_allow_html=True)

"""
MMS Fault Classification Dashboard
Main application entry point
"""

import sys
from pathlib import Path

# Add project root to path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

import streamlit as st
import json
from datetime import datetime

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

# Load model metadata
@st.cache_data
def load_model_metadata():
    """Load model metadata."""
    metadata_path = project_root / "models/minirocket/metadata.json"
    try:
        with open(metadata_path, 'r') as f:
            return json.load(f)
    except Exception as e:
        st.error(f"Error loading model metadata: {e}")
        return {}

# Sidebar navigation
st.sidebar.title("🔧 MMS Fault Classification")
st.sidebar.markdown("---")

page = st.sidebar.radio(
    "Navigation",
    ["🏠 Home", "🎯 Live Prediction", "📊 Model Analytics", "📈 Data Explorer", "ℹ️ About"]
)

st.sidebar.markdown("---")

# Load metadata
metadata = load_model_metadata()

# Display model stats in sidebar
if metadata:
    st.sidebar.markdown("### Model Statistics")
    st.sidebar.metric("Test Accuracy", f"{metadata.get('test_accuracy', 0)*100:.2f}%")
    st.sidebar.metric("Training Samples", f"{metadata.get('train_samples', 0):,}")
    st.sidebar.markdown(f"**Last Trained:** {metadata.get('timestamp', 'Unknown')[:10]}")

# Page routing
if page == "🏠 Home":
    from pages import home
    home.show()
elif page == "🎯 Live Prediction":
    from pages import prediction
    prediction.show()
elif page == "📊 Model Analytics":
    from pages import analytics
    analytics.show()
elif page == "📈 Data Explorer":
    from pages import data_explorer
    data_explorer.show()
elif page == "ℹ️ About":
    from pages import about
    about.show()

# Footer
st.sidebar.markdown("---")
st.sidebar.markdown(
    """
    <div style='text-align: center; color: #666; font-size: 0.8rem;'>
        <p>Built with Streamlit</p>
        <p>© 2025 MMS Fault Classification</p>
    </div>
    """,
    unsafe_allow_html=True
)

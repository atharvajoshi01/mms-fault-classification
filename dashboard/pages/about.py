"""
About page for the dashboard.
"""

import streamlit as st
from pathlib import Path
import sys

project_root = Path(__file__).parent.parent.parent
sys.path.insert(0, str(project_root))


def show():
    """Display about page."""
    
    st.markdown('<h1 class="main-header">ℹ️ About</h1>', unsafe_allow_html=True)
    st.markdown(
        '<p class="sub-header">Learn more about the MMS Fault Classification System</p>',
        unsafe_allow_html=True
    )
    
    # Project overview
    st.markdown("## 🔧 Project Overview")
    
    st.markdown("""
    The **MMS Fault Classification System** is a state-of-the-art machine learning solution for 
    real-time vibration fault detection in industrial machinery. Using advanced time series 
    classification algorithms, the system achieves **99.98% accuracy** in identifying four types 
    of machinery faults from multi-axis vibration data.
    
    ### Key Capabilities
    
    - **Real-time Fault Detection**: Instant classification of machinery health status
    - **Multi-axis Analysis**: Comprehensive analysis of X, Y, Z vibration data
    - **High Accuracy**: 99.98% test accuracy with minimal misclassifications
    - **Production Ready**: Lightweight, fast, and scalable solution
    """)
    
    st.markdown("---")
    
    # Technology stack
    st.markdown("## 🛠️ Technology Stack")
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.markdown("### Core Technologies")
        st.markdown("""
        - **Python 3.8+**: Primary programming language
        - **scikit-learn**: Machine learning framework
        - **sktime**: Time series analysis library
        - **NumPy & Pandas**: Data processing
        - **Streamlit**: Web dashboard framework
        """)
    
    with col2:
        st.markdown("### Visualization")
        st.markdown("""
        - **Plotly**: Interactive visualizations
        - **Matplotlib**: Statistical plots
        - **Seaborn**: Enhanced graphics
        - **Streamlit Components**: Custom UI elements
        """)
    
    st.markdown("---")
    
    # Algorithm details
    st.markdown("## 🧠 Algorithm: MiniRocket")
    
    st.markdown("""
    ### What is MiniRocket?
    
    **MiniRocket** (MINImally RandOm Convolutional KErnel Transform) is a highly efficient 
    time series classification algorithm developed by Angus Dempster et al. It represents 
    a significant breakthrough in time series analysis.
    
    ### How It Works
    
    1. **Random Kernel Generation**: Creates 10,000 random convolutional kernels
    2. **Feature Extraction**: Applies kernels to extract time series features
    3. **Pooling**: Uses PPV (Proportion of Positive Values) pooling
    4. **Classification**: Linear classifier (Ridge Regression) on extracted features
    
    ### Advantages
    
    - ⚡ **Ultra-fast**: 75× faster than original ROCKET
    - 🎯 **Accurate**: Matches deep learning performance
    - 💾 **Lightweight**: Minimal memory footprint
    - 🔧 **No Tuning**: Works well with default parameters
    - 📈 **Scalable**: Handles large datasets efficiently
    
    ### Citation
    
    ```
    Dempster, A., Schmidt, D. F., & Webb, G. I. (2021).
    MiniRocket: A Very Fast (Almost) Deterministic Transform for Time Series Classification.
    In Proceedings of the 27th ACM SIGKDD Conference on Knowledge Discovery & Data Mining.
    ```
    """)
    
    st.markdown("---")
    
    # Fault types
    st.markdown("## 🔍 Fault Types Detected")
    
    tab1, tab2, tab3, tab4 = st.tabs(["✅ Normal", "⚠️ Unbalance", "🔄 Misalignment", "⚙️ Bearing Fault"])
    
    with tab1:
        st.markdown("""
        ### Normal Operation
        
        **Characteristics:**
        - Smooth, consistent vibration patterns
        - Low amplitude variations
        - Balanced across all axes
        - No dominant frequencies
        
        **Indicators:**
        - Regular waveform patterns
        - Minimal harmonic components
        - Stable amplitude levels
        
        **Action:** Continue routine monitoring and preventive maintenance
        """)
    
    with tab2:
        st.markdown("""
        ### Unbalance Fault
        
        **Characteristics:**
        - Increased vibration at rotation frequency
        - Predominantly radial vibration
        - Sinusoidal waveform pattern
        - Phase difference between measurement points
        
        **Causes:**
        - Mass imbalance in rotor
        - Uneven weight distribution
        - Eccentric mounting
        - Material buildup
        
        **Action:** Schedule balancing procedure to correct mass distribution
        """)
    
    with tab3:
        st.markdown("""
        ### Misalignment Fault
        
        **Characteristics:**
        - Vibration at 2× rotation frequency
        - Both axial and radial components
        - High axial vibration levels
        - Phase relationships between bearings
        
        **Causes:**
        - Shaft misalignment
        - Coupling misalignment
        - Improper installation
        - Thermal growth effects
        
        **Action:** Immediate alignment correction required to prevent bearing damage
        """)
    
    with tab4:
        st.markdown("""
        ### Bearing Fault
        
        **Characteristics:**
        - High-frequency impulses
        - Characteristic fault frequencies
        - Modulated patterns
        - Increasing trend over time
        
        **Causes:**
        - Bearing wear or damage
        - Inadequate lubrication
        - Contamination
        - Excessive loading
        
        **Action:** Schedule bearing replacement as soon as possible to avoid catastrophic failure
        """)
    
    st.markdown("---")
    
    # Model performance
    st.markdown("## 📊 Model Performance Summary")
    
    col1, col2, col3 = st.columns(3)
    
    with col1:
        st.markdown("""
        ### Accuracy Metrics
        - **Train Accuracy**: 100.00%
        - **Test Accuracy**: 99.98%
        - **Precision**: 99.98%
        - **Recall**: 99.98%
        - **F1-Score**: 99.98%
        """)
    
    with col2:
        st.markdown("""
        ### Performance Stats
        - **Training Time**: ~2.5 minutes
        - **Inference Time**: < 1 second
        - **Model Size**: 1.7 MB
        - **Total Samples**: 21,559
        - **Test Samples**: 4,312
        """)
    
    with col3:
        st.markdown("""
        ### Per-Class Results
        - **Bearing**: 100.00% F1
        - **Misalignment**: 100.00% F1
        - **Normal**: 99.95% F1
        - **Unbalance**: 99.95% F1
        """)
    
    st.markdown("---")
    
    # Use cases
    st.markdown("## 🏭 Use Cases & Applications")
    
    st.markdown("""
    This fault classification system can be deployed in various industrial settings:
    
    ### Manufacturing
    - Production line equipment monitoring
    - Quality control systems
    - Preventive maintenance scheduling
    
    ### Energy Sector
    - Turbine health monitoring
    - Generator fault detection
    - Pump performance analysis
    
    ### Transportation
    - Railway equipment monitoring
    - Automotive testing
    - Aviation maintenance
    
    ### Industrial Facilities
    - HVAC system monitoring
    - Conveyor belt systems
    - Rotating machinery health
    """)
    
    st.markdown("---")
    
    # Future enhancements
    st.markdown("## 🚀 Future Enhancements")
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.markdown("""
        ### Planned Features
        - [ ] Real-time streaming data support
        - [ ] Multi-sensor fusion
        - [ ] Trend analysis and forecasting
        - [ ] Automated alert system
        - [ ] Mobile app integration
        """)
    
    with col2:
        st.markdown("""
        ### Advanced Capabilities
        - [ ] Anomaly detection
        - [ ] Remaining useful life prediction
        - [ ] Root cause analysis
        - [ ] Maintenance schedule optimization
        - [ ] Integration with SCADA systems
        """)
    
    st.markdown("---")
    
    # Contact and support
    st.markdown("## 📬 Contact & Support")
    
    st.markdown("""
    For questions, issues, or feature requests:
    
    - **Documentation**: See README.md in project root
    - **Issues**: Report bugs or request features
    - **Email**: Contact your system administrator
    
    ### Getting Started
    
    1. Upload your vibration data in CSV format
    2. Get instant fault predictions
    3. Analyze results and take action
    4. Download reports for documentation
    
    For detailed usage instructions, refer to the **Live Prediction** page.
    """)
    
    # Version info
    st.markdown("---")
    st.markdown("""
    <div style="text-align: center; color: #666; padding: 20px;">
        <p><strong>MMS Fault Classification System</strong></p>
        <p>Version 1.0.0 | © 2025 | Built with ❤️ using Streamlit</p>
    </div>
    """, unsafe_allow_html=True)


if __name__ == "__main__":
    show()

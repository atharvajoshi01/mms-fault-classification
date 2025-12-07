"""
Live prediction page for the dashboard.
"""

import streamlit as st
import pandas as pd
import numpy as np
from pathlib import Path
import sys
import io

project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from dashboard.utils import FaultPredictor, plot_probability_bars, plot_vibration_signal


@st.cache_resource(show_spinner="Loading MiniRocket model...")
def load_predictor():
    """Load predictor model (includes warm-up for fast inference)."""
    try:
        predictor = FaultPredictor(
            model_path="models/minirocket/minirocket_model.pkl",
            label_encoder_path="models/minirocket/label_encoder.pkl",
            scaler_path="models/minirocket/scaler.pkl"
        )
        return predictor
    except Exception as e:
        st.error(f"Error loading model: {e}")
        return None


def show():
    """Display prediction page."""

    st.markdown("# 🎯 Live Fault Prediction")
    st.markdown("*Upload vibration data for instant fault classification*")

    # Load predictor
    predictor = load_predictor()

    if predictor is None:
        st.error("Failed to load prediction model. Please check the model files.")
        return
    
    # Upload section
    st.markdown("## 📤 Upload Vibration Data")
    
    uploaded_file = st.file_uploader(
        "Choose a CSV file",
        type=['csv'],
        help="Upload a CSV file containing vibration data with timestamp, axis, and amplitude columns"
    )
    
    # Demo data option
    col1, col2 = st.columns([3, 1])
    with col2:
        use_demo = st.button("🎲 Use Demo Data", help="Load sample data for testing")
    
    if use_demo:
        st.info("Loading demo data...")
        # Load sample data
        try:
            from src.data_loader import load_mms_csv
            demo_file = project_root / "sample_data/sample_vibration_data.csv"
            if demo_file.exists():
                signal = load_mms_csv(demo_file)
                if len(signal) > 0:
                    signal = signal[0]  # Get first sample
                    st.session_state['demo_signal'] = signal
                    st.success("✅ Demo data loaded successfully!")
            else:
                st.error(f"Demo file not found at: {demo_file}")
        except Exception as e:
            st.error(f"Error loading demo data: {e}")
            import traceback
            st.code(traceback.format_exc())
    
    # Process uploaded file or demo data
    signal = None
    
    if uploaded_file is not None:
        try:
            # Save uploaded file temporarily
            temp_path = project_root / "temp_upload.csv"
            with open(temp_path, 'wb') as f:
                f.write(uploaded_file.getbuffer())
            
            # Load signal
            from src.data_loader import load_mms_csv
            signal = load_mms_csv(temp_path)
            
            # Clean up
            temp_path.unlink()
            
            st.success(f"✅ Loaded {len(signal)} samples from file")
            
        except Exception as e:
            st.error(f"Error processing file: {e}")
            st.info("Make sure your CSV file has the correct format with timestamp, axis, and amplitude columns.")
    
    elif 'demo_signal' in st.session_state:
        signal = st.session_state['demo_signal']
        signal = signal[np.newaxis, :]  # Add batch dimension
    
    # Prediction section
    if signal is not None and len(signal) > 0:
        st.markdown("---")
        st.markdown("## 🔍 Prediction Results")
        
        # Select sample to predict
        sample_idx = 0
        if len(signal) > 1:
            sample_idx = st.slider("Select sample to analyze", 0, len(signal)-1, 0)
        
        selected_signal = signal[sample_idx]
        
        # Display signal
        with st.expander("📊 View Vibration Signal", expanded=True):
            fig = plot_vibration_signal(selected_signal, title=f"Sample {sample_idx + 1}")
            st.plotly_chart(fig, use_container_width=True)
        
        # Predict button
        if st.button("🎯 Predict Fault", type="primary"):
            with st.spinner("Analyzing vibration signal..."):
                try:
                    # Make prediction
                    result = predictor.predict(selected_signal)
                    
                    # Display results
                    st.markdown("### 🎯 Prediction Result")
                    
                    # Main prediction
                    predicted_class = result['predicted_class']
                    confidence = result['confidence']
                    
                    # Color code based on fault type
                    color_map = {
                        'normal': 'green',
                        'unbalance_fault': 'orange',
                        'misalignment_fault': 'red',
                        'bearing_fault': 'blue'
                    }
                    color = color_map.get(predicted_class, 'gray')
                    
                    # Display prediction with styling
                    st.markdown(
                        f"""
                        <div style="padding: 30px; background-color: #f0f2f6; border-radius: 10px; border-left: 5px solid {color};">
                            <h2 style="margin: 0; color: {color}; text-transform: capitalize;">
                                {predicted_class.replace('_', ' ')}
                            </h2>
                            <p style="font-size: 1.2rem; margin: 10px 0 0 0;">
                                Confidence: <strong>{confidence*100:.2f}%</strong>
                            </p>
                        </div>
                        """,
                        unsafe_allow_html=True
                    )
                    
                    # Probability distribution
                    st.markdown("### 📊 Class Probabilities")
                    fig_probs = plot_probability_bars(result['probabilities'])
                    st.plotly_chart(fig_probs, use_container_width=True)
                    
                    # Detailed probabilities table
                    with st.expander("📋 Detailed Probabilities"):
                        prob_df = pd.DataFrame([
                            {
                                'Class': class_name.replace('_', ' ').title(),
                                'Probability': f"{prob*100:.4f}%",
                                'Raw Score': f"{prob:.6f}"
                            }
                            for class_name, prob in sorted(
                                result['probabilities'].items(),
                                key=lambda x: x[1],
                                reverse=True
                            )
                        ])
                        st.dataframe(prob_df, use_container_width=True, hide_index=True)
                    
                    # Recommendation based on prediction
                    st.markdown("### 💡 Recommendation")

                    if predicted_class == 'normal':
                        st.success("✅ Equipment is operating normally. Continue routine monitoring.")
                    elif predicted_class == 'unbalance_fault':
                        st.warning("⚠️ Unbalance fault detected. Schedule balancing maintenance.")
                    elif predicted_class == 'misalignment_fault':
                        st.error("🔴 Misalignment fault detected. Schedule alignment correction immediately.")
                    elif predicted_class == 'bearing_fault':
                        st.error("🔴 Bearing fault detected. Schedule bearing replacement as soon as possible.")
                    
                except Exception as e:
                    st.error(f"Prediction error: {e}")
        
        # Batch prediction option
        if len(signal) > 1:
            st.markdown("---")
            st.markdown("## 📊 Batch Prediction")
            
            if st.button("🔄 Predict All Samples"):
                with st.spinner(f"Processing {len(signal)} samples..."):
                    try:
                        # Batch predict
                        results_df = predictor.batch_predict(signal)
                        
                        # Display summary
                        st.success(f"✅ Processed {len(results_df)} samples")
                        
                        # Summary statistics
                        col1, col2 = st.columns(2)
                        with col1:
                            st.markdown("### 📊 Fault Distribution")
                            fault_counts = results_df['predicted_class'].value_counts()
                            st.bar_chart(fault_counts)
                        
                        with col2:
                            st.markdown("### 📈 Average Confidence")
                            avg_confidence = results_df.groupby('predicted_class')['confidence'].mean()
                            st.bar_chart(avg_confidence)
                        
                        # Full results table
                        st.markdown("### 📋 All Predictions")
                        display_df = results_df.copy()
                        display_df['predicted_class'] = display_df['predicted_class'].str.replace('_', ' ').str.title()
                        display_df['confidence'] = display_df['confidence'].apply(lambda x: f"{x*100:.2f}%")
                        st.dataframe(display_df, use_container_width=True, hide_index=True)
                        
                        # Download button
                        csv = results_df.to_csv(index=False)
                        st.download_button(
                            label="📥 Download Results (CSV)",
                            data=csv,
                            file_name="fault_predictions.csv",
                            mime="text/csv"
                        )
                        
                    except Exception as e:
                        st.error(f"Batch prediction error: {e}")
    
    else:
        st.info("👆 Please upload a CSV file or use demo data to start prediction")
        
        # Format guide
        with st.expander("📘 CSV Format Guide"):
            st.markdown("""
            ### Expected CSV Format
            
            Your CSV file should contain vibration data with the following structure:
            
            ```
            timestamp, axis, amplitude_1, amplitude_2, ..., amplitude_1024
            1234567890, X, 0.123, -0.456, ...
            1234567890, Y, 0.234, -0.567, ...
            1234567890, Z, 0.345, -0.678, ...
            ```
            
            **Requirements:**
            - Each sample requires data for X, Y, and Z axes
            - 1024 amplitude values per axis
            - Timestamps should be consistent across axes for the same sample
            
            The system will automatically:
            - Group readings by timestamp
            - Average duplicate readings
            - Stack X, Y, Z data into (1024, 3) format
            """)


# Auto-run for Streamlit multi-page
show()

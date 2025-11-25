"""
Test script to verify dashboard components.
"""

import sys
from pathlib import Path

# Add project to path
project_root = Path(__file__).parent
sys.path.insert(0, str(project_root))

def test_imports():
    """Test that all imports work."""
    print("Testing imports...")
    
    try:
        import streamlit
        print("✓ Streamlit imported")
    except ImportError as e:
        print(f"✗ Streamlit import failed: {e}")
        return False
    
    try:
        import plotly
        print("✓ Plotly imported")
    except ImportError as e:
        print(f"✗ Plotly import failed: {e}")
        return False
    
    try:
        from dashboard.utils import FaultPredictor
        print("✓ FaultPredictor imported")
    except ImportError as e:
        print(f"✗ FaultPredictor import failed: {e}")
        return False
    
    return True


def test_model_loading():
    """Test that model can be loaded."""
    print("\nTesting model loading...")
    
    try:
        from dashboard.utils import FaultPredictor
        
        predictor = FaultPredictor()
        print("✓ Model loaded successfully")
        
        # Get model info
        info = predictor.get_model_info()
        print(f"  Model type: {info['model_type']}")
        print(f"  Test accuracy: {info['test_accuracy']*100:.2f}%")
        print(f"  Classes: {info['classes']}")
        
        return True
    
    except Exception as e:
        print(f"✗ Model loading failed: {e}")
        return False


def test_prediction():
    """Test prediction functionality."""
    print("\nTesting prediction...")
    
    try:
        from dashboard.utils import FaultPredictor
        import numpy as np
        
        predictor = FaultPredictor()
        
        # Create dummy signal
        dummy_signal = np.random.randn(1024, 3).astype(np.float32)
        
        # Predict
        result = predictor.predict(dummy_signal)
        
        print("✓ Prediction successful")
        print(f"  Predicted class: {result['predicted_class']}")
        print(f"  Confidence: {result['confidence']*100:.2f}%")
        
        return True
    
    except Exception as e:
        print(f"✗ Prediction failed: {e}")
        import traceback
        traceback.print_exc()
        return False


def test_visualizations():
    """Test visualization functions."""
    print("\nTesting visualizations...")
    
    try:
        from dashboard.utils import (
            plot_confusion_matrix,
            plot_probability_bars,
            plot_vibration_signal
        )
        import numpy as np
        
        # Test confusion matrix
        cm = np.array([[100, 0, 0, 0], [0, 100, 0, 0], [0, 0, 99, 1], [0, 0, 0, 100]])
        classes = ['bearing_fault', 'misalignment_fault', 'normal', 'unbalance_fault']
        fig = plot_confusion_matrix(cm, classes)
        print("✓ Confusion matrix visualization works")
        
        # Test probability bars
        probs = {'normal': 0.8, 'unbalance_fault': 0.15, 'misalignment_fault': 0.03, 'bearing_fault': 0.02}
        fig = plot_probability_bars(probs)
        print("✓ Probability bars visualization works")
        
        # Test signal plot
        signal = np.random.randn(1024, 3)
        fig = plot_vibration_signal(signal)
        print("✓ Signal visualization works")
        
        return True
    
    except Exception as e:
        print(f"✗ Visualization failed: {e}")
        import traceback
        traceback.print_exc()
        return False


def main():
    """Run all tests."""
    print("=" * 60)
    print("MMS Fault Classification Dashboard - Component Tests")
    print("=" * 60)
    
    all_passed = True
    
    if not test_imports():
        all_passed = False
    
    if not test_model_loading():
        all_passed = False
    
    if not test_prediction():
        all_passed = False
    
    if not test_visualizations():
        all_passed = False
    
    print("\n" + "=" * 60)
    if all_passed:
        print("✓ All tests passed! Dashboard is ready to use.")
        print("\nTo launch the dashboard, run:")
        print("  streamlit run dashboard/app.py")
    else:
        print("✗ Some tests failed. Please check the errors above.")
    print("=" * 60)
    
    return all_passed


if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)

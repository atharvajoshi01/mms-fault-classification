"""
Prediction utility for MiniRocket fault classification.
"""

import logging
from pathlib import Path
from typing import Dict, Any, Tuple, Optional
import numpy as np
import pandas as pd
from src.models.minirocket import MiniRocketClassifier

logger = logging.getLogger(__name__)


class FaultPredictor:
    """Fault classification predictor for MiniRocket model."""
    
    def __init__(
        self,
        model_path: str = "models/minirocket/minirocket_model.pkl",
        label_encoder_path: str = "models/minirocket/label_encoder.pkl",
        scaler_path: str = "models/minirocket/scaler.pkl"
    ):
        """
        Initialize predictor.
        
        Args:
            model_path: Path to trained MiniRocket model
            label_encoder_path: Path to label encoder
            scaler_path: Path to data scaler
        """
        self.model_path = Path(model_path)
        self.label_encoder_path = Path(label_encoder_path)
        self.scaler_path = Path(scaler_path)
        
        # Load model
        logger.info(f"Loading model from: {self.model_path}")
        self.model = MiniRocketClassifier.load(self.model_path)
        
        # Load label encoder
        import joblib
        self.label_encoder = joblib.load(self.label_encoder_path)
        self.scaler = joblib.load(self.scaler_path)
        
        self.class_names = list(self.label_encoder.classes_)
        logger.info(f"Predictor initialized. Classes: {self.class_names}")
    
    def preprocess_signal(self, signal: np.ndarray) -> np.ndarray:
        """
        Preprocess signal for prediction.
        
        Args:
            signal: Raw vibration signal
            
        Returns:
            Preprocessed signal
        """
        # Ensure correct shape (samples, timesteps, channels)
        if signal.ndim == 2:
            signal = signal[np.newaxis, :]  # Add batch dimension
        
        # Normalize using saved scaler
        from src.preprocessing import normalize_data
        signal_normalized, _ = normalize_data(signal, scaler=self.scaler, fit=False)
        
        return signal_normalized
    
    def predict(self, signal: np.ndarray) -> Dict[str, Any]:
        """
        Predict fault class for a signal.
        
        Args:
            signal: Vibration signal of shape (timesteps, channels) or (samples, timesteps, channels)
            
        Returns:
            Dictionary with prediction results
        """
        # Preprocess
        signal_preprocessed = self.preprocess_signal(signal)
        
        # Predict
        predictions = self.model.predict(signal_preprocessed)
        probabilities = self.model.predict_proba(signal_preprocessed)
        
        # Get class names
        predicted_classes = [self.class_names[p] for p in predictions]
        
        # Build result
        results = []
        for i, (pred_class, probs) in enumerate(zip(predicted_classes, probabilities)):
            result = {
                'predicted_class': pred_class,
                'predicted_index': int(predictions[i]),
                'confidence': float(probs[predictions[i]]),
                'probabilities': {
                    class_name: float(prob)
                    for class_name, prob in zip(self.class_names, probs)
                }
            }
            results.append(result)
        
        # Return single result if single input
        if len(results) == 1:
            return results[0]
        
        return results
    
    def predict_from_csv(self, csv_path: str) -> Dict[str, Any]:
        """
        Predict from CSV file.
        
        Args:
            csv_path: Path to CSV file
            
        Returns:
            Prediction results
        """
        from src.data_loader import load_mms_csv
        
        # Load data
        signal = load_mms_csv(Path(csv_path))
        
        # Predict
        return self.predict(signal)
    
    def batch_predict(self, signals: np.ndarray) -> pd.DataFrame:
        """
        Batch prediction on multiple signals.
        
        Args:
            signals: Array of signals (n_samples, timesteps, channels)
            
        Returns:
            DataFrame with predictions
        """
        results = self.predict(signals)
        
        if not isinstance(results, list):
            results = [results]
        
        # Convert to DataFrame
        df_data = []
        for i, result in enumerate(results):
            row = {
                'sample_id': i,
                'predicted_class': result['predicted_class'],
                'confidence': result['confidence']
            }
            # Add probabilities
            for class_name, prob in result['probabilities'].items():
                row[f'prob_{class_name}'] = prob
            df_data.append(row)
        
        return pd.DataFrame(df_data)
    
    def get_model_info(self) -> Dict[str, Any]:
        """Get model information."""
        import json
        
        metadata_path = self.model_path.parent / 'metadata.json'
        with open(metadata_path, 'r') as f:
            metadata = json.load(f)
        
        return {
            'model_type': metadata.get('model_type', 'MiniRocket'),
            'num_kernels': metadata.get('num_kernels', 10000),
            'train_accuracy': metadata.get('train_accuracy', 0),
            'test_accuracy': metadata.get('test_accuracy', 0),
            'train_samples': metadata.get('train_samples', 0),
            'test_samples': metadata.get('test_samples', 0),
            'classes': metadata.get('classes', []),
            'training_time': metadata.get('training_time_seconds', 0),
            'timestamp': metadata.get('timestamp', 'Unknown')
        }


if __name__ == "__main__":
    # Test predictor
    print("Testing FaultPredictor...")
    
    predictor = FaultPredictor()
    
    # Get model info
    info = predictor.get_model_info()
    print(f"\nModel Info:")
    print(f"  Accuracy: {info['test_accuracy']*100:.2f}%")
    print(f"  Classes: {info['classes']}")
    
    # Create dummy signal
    dummy_signal = np.random.randn(1024, 3).astype(np.float32)
    result = predictor.predict(dummy_signal)
    
    print(f"\nPrediction Result:")
    print(f"  Class: {result['predicted_class']}")
    print(f"  Confidence: {result['confidence']:.4f}")

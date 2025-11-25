"""
MiniRocket + RidgeClassifier for vibration fault classification.

MiniRocket is a fast and accurate time series classification method
that uses random convolutional kernels.
"""

import logging
import pickle
from pathlib import Path
from typing import Tuple, Optional

import numpy as np
from sklearn.linear_model import RidgeClassifier
from sklearn.preprocessing import StandardScaler

logger = logging.getLogger(__name__)


class MiniRocketClassifier:
    """
    MiniRocket + RidgeClassifier wrapper for fault classification.

    MiniRocket transforms time series into feature vectors using
    random convolutional kernels, then uses Ridge Classifier for classification.
    """

    def __init__(
        self,
        num_kernels: int = 10000,
        random_state: Optional[int] = None
    ):
        """
        Initialize MiniRocket classifier.

        Args:
            num_kernels: Number of random kernels (default: 10000)
            random_state: Random seed for reproducibility
        """
        self.num_kernels = num_kernels
        self.random_state = random_state
        self.minirocket = None
        self.classifier = None
        self.scaler = StandardScaler()

        logger.info("MiniRocketClassifier initialized")
        logger.info(f"  Num kernels: {num_kernels}")

    def _reshape_data(self, X: np.ndarray) -> np.ndarray:
        """
        Reshape data for MiniRocket.

        MiniRocket expects (n_samples, n_channels, n_timesteps)
        Our data is (n_samples, n_timesteps, n_channels)

        Args:
            X: Input data of shape (n_samples, n_timesteps, n_channels)

        Returns:
            Reshaped data of shape (n_samples, n_channels, n_timesteps)
        """
        # Transpose from (samples, time, channels) to (samples, channels, time)
        return np.transpose(X, (0, 2, 1))

    def fit(self, X: np.ndarray, y: np.ndarray) -> 'MiniRocketClassifier':
        """
        Fit MiniRocket transformer and Ridge classifier.

        Args:
            X: Training data of shape (n_samples, n_timesteps, n_channels)
            y: Training labels

        Returns:
            self
        """
        try:
            from sktime.transformations.panel.rocket import MiniRocket
        except ImportError:
            raise ImportError(
                "sktime is required for MiniRocket. "
                "Install it with: pip install sktime"
            )

        logger.info("Fitting MiniRocket transformer...")
        logger.info(f"  Input shape: {X.shape}")

        # Reshape data for MiniRocket
        X_reshaped = self._reshape_data(X)
        logger.info(f"  Reshaped to: {X_reshaped.shape}")

        # Initialize and fit MiniRocket transformer
        self.minirocket = MiniRocket(
            num_kernels=self.num_kernels,
            random_state=self.random_state
        )

        # Transform data
        logger.info("  Transforming data with MiniRocket...")
        X_transformed = self.minirocket.fit_transform(X_reshaped)
        logger.info(f"  Transformed shape: {X_transformed.shape}")

        # Scale features
        logger.info("  Scaling features...")
        X_scaled = self.scaler.fit_transform(X_transformed)

        # Train Ridge classifier
        logger.info("  Training Ridge classifier...")
        self.classifier = RidgeClassifier(alpha=1.0)
        self.classifier.fit(X_scaled, y)

        logger.info("MiniRocket training complete!")

        return self

    def predict(self, X: np.ndarray) -> np.ndarray:
        """
        Predict class labels.

        Args:
            X: Input data of shape (n_samples, n_timesteps, n_channels)

        Returns:
            Predicted labels
        """
        if self.minirocket is None or self.classifier is None:
            raise ValueError("Model must be fitted before prediction")

        # Reshape and transform
        X_reshaped = self._reshape_data(X)
        X_transformed = self.minirocket.transform(X_reshaped)
        X_scaled = self.scaler.transform(X_transformed)

        # Predict
        return self.classifier.predict(X_scaled)

    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        """
        Predict class probabilities (decision function normalized).

        Args:
            X: Input data of shape (n_samples, n_timesteps, n_channels)

        Returns:
            Class probabilities of shape (n_samples, n_classes)
        """
        if self.minirocket is None or self.classifier is None:
            raise ValueError("Model must be fitted before prediction")

        # Reshape and transform
        X_reshaped = self._reshape_data(X)
        X_transformed = self.minirocket.transform(X_reshaped)
        X_scaled = self.scaler.transform(X_transformed)

        # Get decision function
        decision = self.classifier.decision_function(X_scaled)

        # Convert to probabilities using softmax
        from scipy.special import softmax
        probas = softmax(decision, axis=1)

        return probas

    def score(self, X: np.ndarray, y: np.ndarray) -> float:
        """
        Compute accuracy score.

        Args:
            X: Input data
            y: True labels

        Returns:
            Accuracy score
        """
        predictions = self.predict(X)
        return np.mean(predictions == y)

    def save(self, filepath: Path) -> None:
        """
        Save the trained model.

        Args:
            filepath: Path to save the model
        """
        filepath = Path(filepath)
        filepath.parent.mkdir(parents=True, exist_ok=True)

        model_data = {
            'minirocket': self.minirocket,
            'classifier': self.classifier,
            'scaler': self.scaler,
            'num_kernels': self.num_kernels,
            'random_state': self.random_state
        }

        with open(filepath, 'wb') as f:
            pickle.dump(model_data, f, protocol=pickle.HIGHEST_PROTOCOL)

        logger.info(f"Model saved to: {filepath}")

    @classmethod
    def load(cls, filepath: Path) -> 'MiniRocketClassifier':
        """
        Load a trained model.

        Args:
            filepath: Path to the saved model

        Returns:
            Loaded MiniRocketClassifier
        """
        with open(filepath, 'rb') as f:
            model_data = pickle.load(f)

        # Create instance
        instance = cls(
            num_kernels=model_data['num_kernels'],
            random_state=model_data['random_state']
        )

        # Load components
        instance.minirocket = model_data['minirocket']
        instance.classifier = model_data['classifier']
        instance.scaler = model_data['scaler']

        logger.info(f"Model loaded from: {filepath}")

        return instance


def build_minirocket(
    num_kernels: int = 10000,
    random_state: Optional[int] = 42
) -> MiniRocketClassifier:
    """
    Build MiniRocket classifier.

    Args:
        num_kernels: Number of random kernels (more = better but slower)
        random_state: Random seed

    Returns:
        MiniRocketClassifier instance

    Example:
        >>> model = build_minirocket(num_kernels=10000)
        >>> model.fit(X_train, y_train)
        >>> accuracy = model.score(X_test, y_test)
    """
    return MiniRocketClassifier(
        num_kernels=num_kernels,
        random_state=random_state
    )


if __name__ == "__main__":
    # Test MiniRocket
    print("=" * 60)
    print("Testing MiniRocket Classifier")
    print("=" * 60)

    # Create dummy data
    np.random.seed(42)
    X_train = np.random.randn(100, 1024, 3).astype(np.float32)
    y_train = np.random.randint(0, 4, size=100)
    X_test = np.random.randn(20, 1024, 3).astype(np.float32)
    y_test = np.random.randint(0, 4, size=20)

    print(f"\nTrain data: {X_train.shape}")
    print(f"Test data: {X_test.shape}")

    # Build and train
    print("\nBuilding MiniRocket model...")
    model = build_minirocket(num_kernels=1000)  # Small for testing

    print("\nTraining...")
    model.fit(X_train, y_train)

    print("\nEvaluating...")
    train_acc = model.score(X_train, y_train)
    test_acc = model.score(X_test, y_test)

    print(f"\nResults:")
    print(f"  Train accuracy: {train_acc:.4f}")
    print(f"  Test accuracy: {test_acc:.4f}")

    # Test prediction
    predictions = model.predict(X_test[:5])
    probas = model.predict_proba(X_test[:5])

    print(f"\nSample predictions:")
    print(f"  Predictions: {predictions}")
    print(f"  Probabilities shape: {probas.shape}")

    print("\nTest complete!")

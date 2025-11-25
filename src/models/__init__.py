"""Model architectures for fault classification."""

from .minirocket import build_minirocket, MiniRocketClassifier

__all__ = [
    'build_minirocket',
    'MiniRocketClassifier'
]

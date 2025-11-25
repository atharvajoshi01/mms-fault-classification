"""Dashboard utilities."""

from .predictor import FaultPredictor
from .visualizations import (
    plot_confusion_matrix,
    plot_class_distribution,
    plot_probability_bars,
    plot_vibration_signal,
    plot_metrics_gauge,
    plot_performance_comparison,
    create_metric_card
)

__all__ = [
    'FaultPredictor',
    'plot_confusion_matrix',
    'plot_class_distribution',
    'plot_probability_bars',
    'plot_vibration_signal',
    'plot_metrics_gauge',
    'plot_performance_comparison',
    'create_metric_card'
]

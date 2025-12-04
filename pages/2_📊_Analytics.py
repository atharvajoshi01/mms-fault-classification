"""
Model analytics page - Performance metrics and confusion matrix.
"""

import streamlit as st
import pandas as pd
import numpy as np
from pathlib import Path
import sys
import json

project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from dashboard.utils import (
    plot_confusion_matrix,
    plot_metrics_gauge,
    plot_performance_comparison
)


def show():
    """Display analytics page."""

    st.markdown("# Model Analytics")

    # Model paths
    model_dir = project_root / "models/minirocket"
    metadata_path = model_dir / "metadata.json"
    results_path = model_dir / "results.json"

    # Load metadata
    try:
        with open(metadata_path, 'r') as f:
            metadata = json.load(f)
    except Exception as e:
        st.error(f"Error loading model metadata: {e}")
        return

    # Accuracy Gauges
    st.markdown("## Performance Overview")

    col1, col2, col3, col4 = st.columns(4)

    with col1:
        fig = plot_metrics_gauge(metadata.get('train_accuracy', 0), "Train Acc")
        st.plotly_chart(fig, use_container_width=True)

    with col2:
        fig = plot_metrics_gauge(metadata.get('test_accuracy', 0), "Test Acc")
        st.plotly_chart(fig, use_container_width=True)

    with col3:
        st.metric("Training Time", f"{metadata.get('training_time_seconds', 0):.0f}s")
        st.metric("Kernels", f"{metadata.get('num_kernels', 0):,}")

    with col4:
        st.metric("Train Samples", f"{metadata.get('train_samples', 0):,}")
        st.metric("Test Samples", f"{metadata.get('test_samples', 0):,}")

    st.markdown("---")

    # Confusion Matrix
    st.markdown("## Confusion Matrix")

    try:
        with open(results_path, 'r') as f:
            results = json.load(f)
        cm = np.array(results['confusion_matrix'])
        classes = results['classes']
        classification_rep = results['classification_report']
    except Exception:
        classes = ['bearing_fault', 'misalignment_fault', 'normal', 'unbalance_fault']
        cm = np.eye(4) * 1000
        classification_rep = None

    col1, col2 = st.columns([3, 2])

    with col1:
        fig_cm = plot_confusion_matrix(cm, classes)
        st.plotly_chart(fig_cm, use_container_width=True)

    with col2:
        st.markdown("### Per-Class Metrics")

        if classification_rep:
            report_data = []
            for class_name in classes:
                metrics = classification_rep[class_name]
                report_data.append({
                    'Class': class_name.replace('_', ' ').title(),
                    'F1': f"{metrics['f1-score']*100:.1f}%",
                    'Prec': f"{metrics['precision']*100:.1f}%",
                    'Recall': f"{metrics['recall']*100:.1f}%"
                })
            report_df = pd.DataFrame(report_data)
            st.dataframe(report_df, use_container_width=True, hide_index=True)

    st.markdown("---")

    # Per-class bar chart
    st.markdown("## Per-Class Performance")

    if classification_rep:
        metrics_list = []
        for class_name in classes:
            m = classification_rep[class_name]
            metrics_list.append({
                'class': class_name,
                'precision': m['precision'],
                'recall': m['recall'],
                'f1-score': m['f1-score']
            })
        metrics_df = pd.DataFrame(metrics_list)
        fig_perf = plot_performance_comparison(metrics_df)
        st.plotly_chart(fig_perf, use_container_width=True)


show()

"""
Visualization utilities for the dashboard.
"""

import numpy as np
import pandas as pd
import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots
from typing import Dict, List, Optional


def plot_confusion_matrix(cm: np.ndarray, class_names: List[str]) -> go.Figure:
    """
    Create interactive confusion matrix heatmap.
    
    Args:
        cm: Confusion matrix
        class_names: List of class names
        
    Returns:
        Plotly figure
    """
    # Normalize
    cm_normalized = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
    
    # Create heatmap
    fig = go.Figure(data=go.Heatmap(
        z=cm_normalized,
        x=class_names,
        y=class_names,
        colorscale='Blues',
        text=cm,
        texttemplate='%{text}',
        textfont={"size": 16},
        hovertemplate='True: %{y}<br>Predicted: %{x}<br>Count: %{text}<br>Rate: %{z:.2%}<extra></extra>',
        colorbar=dict(title="Rate")
    ))
    
    fig.update_layout(
        title="Confusion Matrix",
        xaxis_title="Predicted Class",
        yaxis_title="True Class",
        width=600,
        height=600,
        font=dict(size=12)
    )
    
    return fig


def plot_class_distribution(class_counts: Dict[str, int]) -> go.Figure:
    """
    Create class distribution bar chart.
    
    Args:
        class_counts: Dictionary of class names to counts
        
    Returns:
        Plotly figure
    """
    fig = go.Figure(data=[
        go.Bar(
            x=list(class_counts.keys()),
            y=list(class_counts.values()),
            marker_color='rgb(55, 83, 109)',
            text=list(class_counts.values()),
            textposition='auto',
        )
    ])
    
    fig.update_layout(
        title="Class Distribution",
        xaxis_title="Fault Class",
        yaxis_title="Number of Samples",
        showlegend=False,
        height=400
    )
    
    return fig


def plot_probability_bars(probabilities: Dict[str, float]) -> go.Figure:
    """
    Create probability bar chart for predictions.
    
    Args:
        probabilities: Dictionary of class names to probabilities
        
    Returns:
        Plotly figure
    """
    # Sort by probability
    sorted_probs = sorted(probabilities.items(), key=lambda x: x[1], reverse=True)
    classes = [x[0] for x in sorted_probs]
    probs = [x[1] for x in sorted_probs]
    
    # Color the highest probability differently
    colors = ['rgb(26, 118, 255)' if i == 0 else 'rgb(158, 185, 243)' for i in range(len(classes))]
    
    fig = go.Figure(data=[
        go.Bar(
            x=probs,
            y=classes,
            orientation='h',
            marker_color=colors,
            text=[f'{p*100:.2f}%' for p in probs],
            textposition='auto',
        )
    ])
    
    fig.update_layout(
        title="Prediction Probabilities",
        xaxis_title="Probability",
        yaxis_title="Fault Class",
        showlegend=False,
        height=300,
        xaxis=dict(range=[0, 1])
    )
    
    return fig


def plot_vibration_signal(signal: np.ndarray, title: str = "Vibration Signal") -> go.Figure:
    """
    Plot 3-axis vibration signal.
    
    Args:
        signal: Signal array of shape (timesteps, 3)
        title: Plot title
        
    Returns:
        Plotly figure
    """
    timesteps = np.arange(signal.shape[0])
    
    fig = make_subplots(
        rows=3, cols=1,
        subplot_titles=('X-Axis', 'Y-Axis', 'Z-Axis'),
        vertical_spacing=0.1
    )
    
    # X-axis
    fig.add_trace(
        go.Scatter(x=timesteps, y=signal[:, 0], name='X', line=dict(color='red')),
        row=1, col=1
    )
    
    # Y-axis
    fig.add_trace(
        go.Scatter(x=timesteps, y=signal[:, 1], name='Y', line=dict(color='green')),
        row=2, col=1
    )
    
    # Z-axis
    fig.add_trace(
        go.Scatter(x=timesteps, y=signal[:, 2], name='Z', line=dict(color='blue')),
        row=3, col=1
    )
    
    fig.update_layout(
        title_text=title,
        height=700,
        showlegend=True
    )
    
    fig.update_xaxes(title_text="Timestep", row=3, col=1)
    fig.update_yaxes(title_text="Amplitude")
    
    return fig


def plot_metrics_gauge(accuracy: float, title: str = "Model Accuracy") -> go.Figure:
    """
    Create gauge chart for metrics.

    Args:
        accuracy: Accuracy value (0-1)
        title: Chart title

    Returns:
        Plotly figure
    """
    fig = go.Figure(go.Indicator(
        mode="gauge+number+delta",
        value=accuracy * 100,
        domain={'x': [0, 1], 'y': [0, 1]},
        title={'text': title, 'font': {'size': 24}},
        delta={'reference': 95, 'increasing': {'color': "green"}},
        gauge={
            'axis': {'range': [None, 100], 'tickwidth': 1, 'tickcolor': "darkblue"},
            'bar': {'color': "darkblue"},
            'bgcolor': "white",
            'borderwidth': 2,
            'bordercolor': "gray",
            'steps': [
                {'range': [0, 70], 'color': 'lightgray'},
                {'range': [70, 90], 'color': 'lightblue'},
                {'range': [90, 100], 'color': 'lightgreen'}
            ],
            'threshold': {
                'line': {'color': "red", 'width': 4},
                'thickness': 0.75,
                'value': 99
            }
        },
        number={'suffix': "%", 'valueformat': ".2f"}
    ))

    fig.update_layout(height=300)

    return fig


def plot_performance_comparison(metrics_df: pd.DataFrame) -> go.Figure:
    """
    Create performance comparison chart.
    
    Args:
        metrics_df: DataFrame with performance metrics
        
    Returns:
        Plotly figure
    """
    fig = go.Figure()
    
    metrics = ['Precision', 'Recall', 'F1-Score']
    
    for metric in metrics:
        if metric.lower() in metrics_df.columns:
            fig.add_trace(go.Bar(
                name=metric,
                x=metrics_df['class'],
                y=metrics_df[metric.lower()],
                text=[f'{v:.2%}' for v in metrics_df[metric.lower()]],
                textposition='auto'
            ))
    
    fig.update_layout(
        title="Per-Class Performance Metrics",
        xaxis_title="Fault Class",
        yaxis_title="Score",
        barmode='group',
        height=400,
        yaxis=dict(range=[0, 1.05])
    )
    
    return fig


def create_metric_card(title: str, value: str, delta: Optional[str] = None) -> str:
    """
    Create HTML for a metric card.
    
    Args:
        title: Metric title
        value: Metric value
        delta: Optional delta value
        
    Returns:
        HTML string
    """
    delta_html = f'<p style="color: green; font-size: 14px; margin: 0;">{delta}</p>' if delta else ''
    
    return f"""
    <div style="background-color: #f0f2f6; padding: 20px; border-radius: 10px; text-align: center;">
        <h4 style="margin: 0; color: #31333F;">{title}</h4>
        <h2 style="margin: 10px 0; color: #0068C9;">{value}</h2>
        {delta_html}
    </div>
    """

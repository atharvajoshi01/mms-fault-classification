# MMS Fault Classification System

A high-performance vibration fault classification system using MiniRocket for real-time machinery health monitoring.

## Overview

This project implements an advanced fault detection system for vibration data analysis, achieving **99.98% accuracy** on multi-axis vibration signals. The system uses MiniRocket (Random Convolutional Kernel Transform) combined with Ridge Classification for fast and accurate fault diagnosis.

### Key Features

- **State-of-the-art Accuracy**: 99.98% test accuracy on 21,559 samples
- **Ultra-fast Training**: Model trains in ~2.5 minutes
- **Real-time Inference**: Instant predictions on new vibration data
- **Interactive Dashboard**: Beautiful Streamlit-based web interface
- **Multi-axis Support**: Handles X, Y, Z axis vibration data
- **4-class Classification**: Normal, Unbalance, Misalignment, Bearing faults

## Project Structure

```
mms_fault_classification/
├── src/
│   ├── models/
│   │   └── minirocket.py       # MiniRocket classifier implementation
│   ├── data_loader.py           # CSV data loading utilities
│   └── preprocessing.py         # Data normalization
├── scripts/
│   ├── train_minirocket.py      # Model training script
│   └── visualize_results.py     # Results visualization
├── dashboard/
│   ├── app.py                   # Main Streamlit dashboard
│   ├── pages/                   # Dashboard pages
│   └── utils/                   # Dashboard utilities
├── models/
│   └── minirocket/              # Trained model files
├── dataset/
│   └── phase_2_2/               # Training dataset
└── requirements.txt             # Python dependencies
```

## Installation

### 1. Clone or Navigate to Project

```bash
cd "/Users/atharva/Documents/MMS Client/mms_fault_classification"
```

### 2. Create Virtual Environment (Recommended)

```bash
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
```

### 3. Install Dependencies

```bash
pip install -r requirements.txt
```

## Usage

### Launch Dashboard

```bash
streamlit run dashboard/app.py
```

The dashboard will open in your browser at `http://localhost:8501`

### Train New Model

```bash
python scripts/train_minirocket.py
```

### Visualize Results

```bash
python scripts/visualize_results.py
```

## Model Performance

### Metrics

- **Test Accuracy**: 99.98% (4,311/4,312 correct predictions)
- **Training Time**: 154 seconds (~2.5 minutes)
- **Model Size**: 1.7 MB
- **Dataset**: 21,559 samples (17,247 train / 4,312 test)

### Per-Class Performance

| Fault Type       | Precision | Recall | F1-Score | Support |
|------------------|-----------|--------|----------|---------|
| Bearing Fault    | 100.00%   | 100.00%| 100.00%  | 1,080   |
| Misalignment     | 100.00%   | 100.00%| 100.00%  | 1,081   |
| Normal           | 99.91%    | 100.00%| 99.95%   | 1,075   |
| Unbalance Fault  | 100.00%   | 99.91% | 99.95%   | 1,076   |

## Technical Details

### MiniRocket Architecture

- **Transform**: 10,000 random convolutional kernels
- **Feature Space**: 29,988 features (from 1024×3 input)
- **Classifier**: Ridge Regression (α=1.0)
- **Preprocessing**: StandardScaler on transformed features

### Data Format

Input vibration signals:
- **Shape**: (1024, 3) - 1024 timesteps × 3 channels (X, Y, Z)
- **Sampling**: Time-series vibration data from sensors
- **Normalization**: Z-score standardization

## Dashboard Features

### 1. Home
- Project overview and statistics
- Model performance metrics
- Quick start guide

### 2. Live Prediction
- Upload CSV files for batch prediction
- Real-time fault classification
- Confidence scores and visualizations

### 3. Model Analytics
- Training history and metrics
- Confusion matrix visualization
- Per-class performance analysis

### 4. Data Explorer
- Dataset statistics and distribution
- Sample signal visualization
- Class balance analysis

## Development

### File Organization

- `src/`: Core source code
- `scripts/`: Standalone scripts for training and evaluation
- `dashboard/`: Streamlit web application
- `models/`: Trained model artifacts
- `dataset/`: Raw and processed data

### Adding New Features

1. Create new module in `src/`
2. Add tests if applicable
3. Update documentation
4. Update dashboard if UI changes needed

## Citation

If you use this project in your research, please cite:

```bibtex
@software{mms_fault_classification,
  title={MMS Fault Classification System},
  author={Your Name},
  year={2025},
  description={High-performance vibration fault classification using MiniRocket}
}
```

## License

MIT License - see LICENSE file for details

## Contact

For questions or issues, please open an issue on the project repository.

---

**Built with**: Python • scikit-learn • sktime • Streamlit • NumPy • Pandas

<div align="center">

# 🔧 MMS Fault Classification System

### Industrial Predictive Maintenance with State-of-the-Art Accuracy

[![Python](https://img.shields.io/badge/Python-3.8+-3776AB?style=flat&logo=python&logoColor=white)](https://www.python.org/)
[![scikit-learn](https://img.shields.io/badge/scikit--learn-F7931E?style=flat&logo=scikitlearn&logoColor=white)](https://scikit-learn.org/)
[![Streamlit](https://img.shields.io/badge/Streamlit-FF4B4B?style=flat&logo=streamlit&logoColor=white)](https://streamlit.io/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

**A production-ready vibration fault classification system using MiniRocket for real-time machinery health monitoring**

[Features](#-key-features) • [Installation](#-installation) • [Usage](#-usage) • [Performance](#-model-performance) • [Dashboard](#-interactive-dashboard) • [Documentation](#-technical-details)

</div>

---

## 🎯 Overview

This project implements an **advanced industrial fault detection system** for vibration data analysis, achieving **99.98% accuracy** on multi-axis vibration signals. The system uses **MiniRocket** (Random Convolutional Kernel Transform) combined with Ridge Classification for fast and accurate fault diagnosis.

### 🌟 Key Achievements

| Metric | Value |
|--------|-------|
| **Test Accuracy** | **99.98%** 🏆 |
| **Training Time** | 2.5 minutes |
| **Model Size** | 1.7 MB |
| **Classes** | 4 (Normal, Unbalance, Misalignment, Bearing) |
| **Inference** | <1 second |

### ✨ Key Features

- **🎯 World-Class Accuracy**: 99.98% test accuracy on fault detection
- **⚡ Ultra-Fast Training**: Model trains in ~2.5 minutes
- **🚀 Real-Time Inference**: Sub-second predictions on new vibration data
- **📊 Interactive Dashboard**: Beautiful Streamlit-based web interface with analytics
- **🔄 Multi-Axis Support**: Handles X, Y, Z axis vibration data (1024 timesteps × 3 channels)
- **🧠 4-Class Detection**: Normal, Unbalance, Misalignment, Bearing fault detection
- **💾 Lightweight Model**: 1.7 MB model deployable on edge devices
- **🏭 Production Ready**: Complete ML pipeline with preprocessing, training, and deployment

## 📁 Project Structure

```
mms_fault_classification/
├── src/
│   ├── models/
│   │   └── minirocket.py              # MiniRocket classifier implementation
│   ├── data_loader.py                  # CSV data loading utilities
│   └── preprocessing.py                # Data normalization & feature engineering
├── scripts/
│   ├── train_minirocket.py             # Model training script
│   └── visualize_results.py            # Results visualization
├── pages/
│   ├── 1_🎯_Prediction.py             # Live fault prediction interface
│   ├── 2_📊_Analytics.py              # Model performance analytics
│   ├── 3_📈_Data_Explorer.py          # Dataset exploration & visualization
│   └── 4_ℹ️_About.py                  # Project documentation
├── dashboard/
│   └── utils/
│       ├── predictor.py                # Fault prediction utilities
│       └── visualizations.py           # Chart and plot generators
├── models/
│   └── minirocket/                     # Trained models (99.98% accuracy)
│       ├── minirocket_model.pkl
│       ├── label_encoder.pkl
│       ├── scaler.pkl
│       └── metadata.json
├── dataset/
│   └── phase_2_3/                      # Vibration fault data
└── requirements.txt                    # Python dependencies
```

## 🚀 Installation

### 1. Clone Repository

```bash
git clone https://github.com/atharvajoshi01/mms-fault-classification.git
cd mms-fault-classification
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

## 💻 Usage

### Launch Interactive Dashboard

```bash
streamlit run streamlit_app.py
```

The dashboard will open in your browser at `http://localhost:8501`

**Dashboard Features:**
- 🏠 **Home**: Overview and key metrics
- 🎯 **Prediction**: Upload CSV files for real-time fault classification
- 📊 **Analytics**: Detailed model performance analysis and confusion matrices
- 🔍 **Data Explorer**: Dataset statistics and sample visualizations
- ℹ️ **About**: Technical documentation and deployment guide

### Train Model

```bash
python scripts/train_minirocket.py
```

### Visualize Results

```bash
python scripts/visualize_results.py
```

## 📊 Model Performance

### Individual Fault Detection (4-Class) 🏆

**Overall Metrics:**
- **Test Accuracy**: **99.98%** (4,311/4,312 correct predictions)
- **Training Time**: 154 seconds (~2.5 minutes)
- **Model Size**: 1.7 MB
- **Dataset**: 21,559 samples (17,247 train / 4,312 test)
- **Kernels**: 10,000 random convolutional kernels

**Per-Class Performance:**

| Fault Type       | Precision | Recall | F1-Score | Support |
|------------------|-----------|--------|----------|---------|
| **Bearing Fault**    | 100.00%   | 100.00%| **100.00%**  | 1,080   |
| **Misalignment**     | 100.00%   | 100.00%| **100.00%**  | 1,081   |
| **Normal**           | 99.91%    | 100.00%| **99.95%**   | 1,075   |
| **Unbalance Fault**  | 100.00%   | 99.91% | **99.95%**   | 1,076   |

**Weighted Average**: 99.98% Precision | 99.98% Recall | 99.98% F1-Score

## 🔬 Technical Details

### MiniRocket Architecture

**What is MiniRocket?**
MiniRocket (Minified ROCKET) is a state-of-the-art time series classification algorithm that uses random convolutional kernels to transform raw time series data into highly discriminative features.

**Model Configuration:**
- **Transform**: 10,000 random convolutional kernels
- **Feature Space**: 29,988 features (from 1024×3 input)
- **Classifier**: Ridge Regression (α=1.0)
- **Preprocessing**: StandardScaler (Z-score normalization)

### Data Format

**Input Vibration Signals:**
- **Shape**: (1024, 3) - 1024 timesteps × 3 channels (X, Y, Z axes)
- **Sampling**: Time-series vibration data from tri-axial accelerometers
- **Data Type**: Float32 (amplitude values)
- **Normalization**: Z-score standardization across features

**CSV Format:**
```
timestamp, axis, amplitude_1, amplitude_2, ..., amplitude_1024
1234567890, X, 0.123, -0.456, ...
1234567890, Y, 0.234, -0.567, ...
1234567890, Z, 0.345, -0.678, ...
```

## 📊 Interactive Dashboard

The Streamlit-based dashboard provides a comprehensive interface for model interaction and analysis:

### 🏠 1. Home Page
- Real-time model performance metrics
- Key achievements overview
- Fault class descriptions
- Quick start guide

### 🎯 2. Live Prediction
- **File Upload**: Drag-and-drop CSV support
- **Real-Time Classification**: Instant fault detection with confidence scores
- **Batch Prediction**: Process multiple samples simultaneously
- **Visualizations**: Interactive vibration signal plots
- **Recommendations**: Actionable maintenance suggestions based on predictions
- **Export**: Download predictions as CSV

### 📊 3. Model Analytics
- **Performance Metrics**: Accuracy, precision, recall, F1-score
- **Confusion Matrix**: Interactive heatmap visualization
- **Class-wise Analysis**: Detailed per-class performance breakdowns
- **Training History**: Time, samples, and parameter information

### 🔍 4. Data Explorer
- **Dataset Statistics**: Sample counts, class distributions
- **Signal Visualization**: Plot individual vibration samples
- **Class Balance**: Visual analysis of dataset composition
- **Sample Browser**: Explore training data interactively

### ℹ️ 5. About & Documentation
- Complete technical documentation
- Model architecture details
- Performance analysis and interpretation
- Deployment guidelines (STM32, industrial PLCs)
- Future roadmap and next steps

## 🏭 Real-World Applications

This fault classification system is designed for **industrial predictive maintenance** scenarios:

### Use Cases

1. **Manufacturing Equipment**: Monitor rotating machinery (motors, pumps, fans)
2. **Power Generation**: Turbine and generator health monitoring
3. **Oil & Gas**: Pump and compressor fault detection
4. **HVAC Systems**: Commercial building equipment monitoring
5. **Automotive**: Assembly line machinery diagnostics

### Business Impact

- **Prevent Unplanned Downtime**: Early fault detection prevents catastrophic failures ($10K-$50K per hour in lost production)
- **Optimize Maintenance**: Shift from reactive to predictive maintenance schedules
- **Reduce Costs**: Minimize emergency repairs and extend equipment lifespan
- **Improve Safety**: Detect critical faults before they become safety hazards

## 🚀 Deployment Options

### Edge Device Deployment

**Lightweight Design:** Models are <2 MB, making them ideal for edge deployment:
- **STM32 Microcontrollers**: ARM Cortex-M based MCUs
- **Raspberry Pi**: Local monitoring stations
- **Industrial PLCs**: Siemens, Allen-Bradley, etc.

**Deployment Guide:** See the [About page](pages/4_ℹ️_About.py) in the dashboard for detailed STM32 deployment instructions.

### Cloud Deployment

- **AWS Lambda**: Serverless inference
- **Azure ML**: Enterprise-scale deployment
- **Google Cloud Run**: Containerized deployment

### SCADA Integration

- **Modbus TCP/RTU**: Direct PLC communication
- **OPC UA**: Industrial automation protocol
- **REST API**: HTTP-based integration

## 🔮 Future Roadmap

- [ ] **Real-time Streaming**: Continuous vibration monitoring with streaming data
- [ ] **Additional Faults**: Expand to cavitation, looseness, electrical faults
- [ ] **Alert System**: Email/SMS notifications for critical faults
- [ ] **Historical Trending**: Track fault progression over time
- [ ] **Multi-Asset Monitoring**: Centralized dashboard for multiple machines
- [ ] **Mobile App**: iOS/Android app for on-the-go monitoring
- [ ] **AutoML**: Automated model retraining with new data

## 🛠️ Development

### Project Organization

- `src/`: Core ML algorithms and data processing
- `scripts/`: Training pipelines and utilities
- `dashboard/`: Streamlit web application
- `models/`: Trained model artifacts and metadata
- `dataset/`: Training and validation data

### Adding New Features

1. Create new module in appropriate directory (`src/`, `scripts/`, or `dashboard/`)
2. Follow existing code style and documentation patterns
3. Update README and dashboard documentation
4. Test thoroughly before committing

### Dependencies

See [requirements.txt](requirements.txt) for full dependency list. Key packages:
- `scikit-learn >= 1.0.0`: Machine learning
- `sktime >= 0.11.0`: Time series analysis (MiniRocket)
- `streamlit >= 1.20.0`: Dashboard framework
- `plotly >= 5.0.0`: Interactive visualizations
- `pandas >= 1.3.0`: Data manipulation
- `numpy >= 1.21.0`: Numerical computing

## 📚 Citation

If you use this project in your research or production systems, please cite:

```bibtex
@software{mms_fault_classification,
  title={MMS Fault Classification System: Industrial Predictive Maintenance with MiniRocket},
  author={Atharva Joshi},
  year={2025},
  url={https://github.com/atharvajoshi01/mms-fault-classification},
  description={Production-ready vibration fault classification achieving 99.98% accuracy}
}
```

## 📄 License

MIT License - see [LICENSE](LICENSE) file for details.

## 👤 Author

**Atharva Joshi**
- 📧 Email: [atharvaj2112@gmail.com](mailto:atharvaj2112@gmail.com)
- 💼 LinkedIn: [linkedin.com/in/atharvajoshi01](https://linkedin.com/in/atharvajoshi01)
- 🐙 GitHub: [github.com/atharvajoshi01](https://github.com/atharvajoshi01)
- 📍 Location: New York, USA

## 🙏 Acknowledgments

- **MiniRocket Algorithm**: Based on the paper "MINIROCKET: A Very Fast (Almost) Deterministic Transform for Time Series Classification" by Dempster et al.
- **scikit-learn & sktime**: Excellent ML and time series libraries
- **Streamlit**: Amazing framework for data science dashboards

---

<div align="center">

**⭐ If you find this project useful, please consider giving it a star on GitHub! ⭐**

**Built with** ❤️ **using:** Python • scikit-learn • sktime • Streamlit • NumPy • Pandas • Plotly

</div>

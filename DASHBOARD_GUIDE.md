# MMS Fault Classification Dashboard Guide

## Quick Start

### Launch the Dashboard

**Mac/Linux:**
```bash
./run_dashboard.sh
```

**Windows:**
```cmd
run_dashboard.bat
```

**Manual Launch:**
```bash
streamlit run dashboard/app.py
```

The dashboard will automatically open in your browser at `http://localhost:8501`

## Dashboard Pages

### 1. 🏠 Home
**Purpose:** Project overview and key metrics

**Features:**
- Display model performance statistics (99.98% accuracy)
- Show training time and dataset information
- Overview of fault types detected
- Quick start guide

### 2. 🎯 Live Prediction
**Purpose:** Real-time fault classification

**Features:**
- Upload CSV files for fault prediction
- Use demo data for testing
- View vibration signal visualization (X, Y, Z axes)
- Get instant predictions with confidence scores
- See probability distribution across all fault classes
- Batch prediction for multiple samples
- Download prediction results as CSV

**How to Use:**
1. Click "Choose a CSV file" to upload your data
2. OR click "Use Demo Data" to test with sample data
3. View the vibration signal visualization
4. Click "Predict Fault" to get classification
5. Review the prediction and recommendations
6. For multiple samples, use "Predict All Samples"

### 3. 📊 Model Analytics
**Purpose:** Comprehensive performance analysis

**Features:**
- Training and test accuracy gauges
- Efficiency score visualization
- Training statistics and metrics
- Confusion matrix (interactive)
- Per-class performance metrics
- Classification report
- Model architecture details

**Insights:**
- View which fault types are most accurately detected
- Understand model performance across different classes
- Analyze misclassifications (if any)

### 4. 📈 Data Explorer
**Purpose:** Dataset exploration and visualization

**Features:**
- Dataset overview and statistics
- Class distribution charts
- Sample visualization tool
- Signal statistics for each axis
- Raw data preview
- Data quality indicators

**How to Use:**
1. View overall dataset statistics
2. Select a fault class from dropdown
3. Choose a sample index
4. Click "Load and Visualize"
5. Explore signal patterns and statistics

### 5. ℹ️ About
**Purpose:** Project information and documentation

**Features:**
- Project overview and capabilities
- Technology stack details
- MiniRocket algorithm explanation
- Detailed fault type descriptions
- Model performance summary
- Use cases and applications
- Future enhancements roadmap

## CSV File Format

### Required Structure

Your CSV file should have the following format:

```csv
timestamp,axis,amplitude_1,amplitude_2,...,amplitude_1024
1234567890,X,0.123,-0.456,...,0.789
1234567890,Y,0.234,-0.567,...,0.890
1234567890,Z,0.345,-0.678,...,0.901
```

### Requirements

1. **Columns:**
   - `timestamp`: Sample identifier
   - `axis`: One of X, Y, or Z
   - `amplitude_1` to `amplitude_1024`: 1024 amplitude values

2. **Data:**
   - Each sample needs 3 rows (one for each axis: X, Y, Z)
   - Same timestamp for all three axes of a sample
   - 1024 amplitude values per axis

3. **Format:**
   - CSV format with comma delimiter
   - Numeric values for amplitudes
   - Consistent column count

### Sample Data

A sample CSV file is provided in `sample_data/sample_vibration_data.csv` for testing.

## Prediction Results

### Understanding the Output

**Predicted Class:**
- The fault type detected by the model
- Options: Normal, Unbalance Fault, Misalignment Fault, Bearing Fault

**Confidence:**
- Probability score for the predicted class (0-100%)
- Higher confidence = more certain prediction

**Class Probabilities:**
- Breakdown of probabilities for all fault types
- Sum of all probabilities = 100%

### Recommendations

Based on prediction:

- **Normal**: Continue routine monitoring
- **Unbalance Fault**: Schedule balancing maintenance
- **Misalignment Fault**: Immediate alignment correction needed
- **Bearing Fault**: Schedule bearing replacement ASAP

## Tips for Best Results

### Data Quality

1. **Ensure complete data**: All 1024 amplitude values for each axis
2. **Consistent sampling**: Regular time intervals
3. **Clean data**: Remove outliers and noise if possible
4. **Correct format**: Follow CSV structure exactly

### Using the Dashboard

1. **Start with demo data**: Test functionality before using real data
2. **Check visualizations**: Verify signal looks reasonable
3. **Review probabilities**: Check if prediction is confident
4. **Batch processing**: Use for multiple samples efficiently
5. **Download results**: Save predictions for documentation

## Troubleshooting

### Dashboard won't start

**Solution:**
```bash
pip install -r requirements.txt
streamlit run dashboard/app.py
```

### Model loading error

**Check:**
- Model files exist in `models/minirocket/`
- All required files present:
  - minirocket_model.pkl
  - label_encoder.pkl
  - scaler.pkl
  - metadata.json

### CSV upload error

**Verify:**
- Correct CSV format (see format guide above)
- File has timestamp, axis, and amplitude columns
- 1024 amplitude values per row
- X, Y, Z axes included

### Prediction error

**Possible causes:**
- Incorrect data shape
- Missing or corrupted model files
- Incompatible data format

**Solution:**
- Use demo data to verify system works
- Check CSV format matches requirements
- Re-upload CSV file

## Advanced Features

### Batch Prediction

Process multiple samples at once:
1. Upload CSV with multiple timestamps
2. Click "Predict All Samples"
3. View distribution of predictions
4. Download results as CSV

### Signal Visualization

Analyze vibration patterns:
- View X, Y, Z axes separately
- Identify patterns and anomalies
- Compare different fault types
- Study signal characteristics

### Export Results

Save predictions for documentation:
- Click "Download Results (CSV)"
- Include in maintenance reports
- Track equipment health over time
- Build historical database

## Performance

### Speed

- Single prediction: < 1 second
- Batch (100 samples): < 5 seconds
- Upload/processing: Depends on file size

### Accuracy

- Overall: 99.98%
- Bearing fault: 100%
- Misalignment: 100%
- Normal: 99.95%
- Unbalance: 99.95%

## Support

For issues or questions:

1. Check this guide
2. Review README.md
3. Run test script: `python test_dashboard.py`
4. Contact system administrator

## Version Information

- Dashboard Version: 1.0.0
- Model Type: MiniRocket + Ridge Classifier
- Training Date: 2025-11-22
- Last Updated: 2025-11-22

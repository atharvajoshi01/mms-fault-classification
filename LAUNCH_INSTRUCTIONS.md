# 🚀 Quick Launch Instructions

## Start the Dashboard in 3 Easy Steps

### Step 1: Install Dependencies (First Time Only)

```bash
pip install -r requirements.txt
```

### Step 2: Launch Dashboard

Choose your platform:

**Mac/Linux:**
```bash
chmod +x run_dashboard.sh
./run_dashboard.sh
```

**Windows:**
```cmd
run_dashboard.bat
```

**Or manually:**
```bash
streamlit run dashboard/app.py
```

### Step 3: Access Dashboard

The dashboard will automatically open in your browser at:
```
http://localhost:8501
```

---

## 🎯 What You Can Do

### 1. Home Page
- View project overview and key metrics
- See model performance (99.98% accuracy!)
- Learn about fault types

### 2. Live Prediction
- Upload CSV files for fault detection
- Use demo data to test the system
- Get instant predictions with confidence scores
- Download results as CSV

### 3. Model Analytics
- View detailed performance metrics
- Explore confusion matrix
- Analyze per-class performance

### 4. Data Explorer
- Browse the training dataset
- Visualize vibration signals
- Examine data quality

### 5. About
- Learn about MiniRocket algorithm
- Understand fault types
- View technology stack

---

## 📤 Upload Your Data

### CSV Format

Your file should have:
- `timestamp` column
- `axis` column (X, Y, or Z)
- `amplitude_1` through `amplitude_1024` columns

### Example

```csv
timestamp,axis,amplitude_1,amplitude_2,...,amplitude_1024
1234567890,X,0.123,-0.456,...,0.789
1234567890,Y,0.234,-0.567,...,0.890
1234567890,Z,0.345,-0.678,...,0.901
```

### Sample Data

Test with provided sample:
```
sample_data/sample_vibration_data.csv
```

---

## 🔍 Quick Test

Verify everything works:

```bash
python test_dashboard.py
```

Expected output:
```
✓ All tests passed! Dashboard is ready to use.
```

---

## 💡 Tips

1. **First Time?** Use demo data to explore features
2. **Have Data?** Upload your CSV for real predictions
3. **Multiple Samples?** Use batch prediction feature
4. **Need Help?** Check DASHBOARD_GUIDE.md

---

## 🛑 Troubleshooting

**Dashboard won't start?**
```bash
pip install --upgrade streamlit
streamlit run dashboard/app.py
```

**Can't upload file?**
- Check CSV format matches requirements
- Ensure all 1024 amplitude columns present
- Verify X, Y, Z axes included

**Prediction error?**
- Try demo data first
- Verify model files exist
- Run test script

---

## 📞 Need Help?

1. Read DASHBOARD_GUIDE.md for detailed instructions
2. Check README.md for project overview
3. Review PROJECT_SUMMARY.md for technical details
4. Run test suite: `python test_dashboard.py`

---

**Ready to detect faults? Let's go!** 🚀

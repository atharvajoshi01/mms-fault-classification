# Installation Guide for New Users

This guide explains how to set up and run the MMS Fault Classification Dashboard on a new device.

## 📋 Prerequisites

- **Python 3.8 or higher** ([Download here](https://www.python.org/downloads/))
- **pip** (comes with Python)
- **Git** (optional, for cloning)

### Check Python Version

```bash
python --version  # Should be 3.8+
# Or
python3 --version
```

---

## 🚀 Installation Steps

### Option 1: Download ZIP File (Easiest)

1. **Download the project ZIP file**
2. **Extract** to your desired location
3. **Open terminal/command prompt** in the extracted folder
4. **Continue to "Setup Dependencies"** below

### Option 2: Clone from Git Repository

```bash
git clone <repository-url>
cd mms_fault_classification
```

---

## 📦 Setup Dependencies

### Step 1: Create Virtual Environment (Recommended)

**Mac/Linux:**
```bash
python3 -m venv venv
source venv/bin/activate
```

**Windows:**
```cmd
python -m venv venv
venv\Scripts\activate
```

You should see `(venv)` in your terminal prompt.

### Step 2: Install Required Packages

```bash
pip install -r requirements.txt
```

This will install:
- streamlit (dashboard framework)
- scikit-learn (machine learning)
- sktime (time series analysis)
- plotly (visualizations)
- numpy, pandas, scipy (data processing)
- And other dependencies

**Installation time:** ~2-5 minutes depending on internet speed

### Step 3: Verify Installation

```bash
python test_dashboard.py
```

Expected output:
```
✓ All tests passed! Dashboard is ready to use.
```

---

## 🎯 Launch the Dashboard

### Mac/Linux

```bash
chmod +x run_dashboard.sh
./run_dashboard.sh
```

### Windows

```cmd
run_dashboard.bat
```

### Manual Launch (Any Platform)

```bash
streamlit run dashboard/app.py
```

The dashboard will open automatically in your browser at:
```
http://localhost:8501
```

---

## 📂 Project Structure

After extraction, you should have:

```
mms_fault_classification/
├── dashboard/          # Dashboard application
├── models/            # Trained model files (IMPORTANT!)
├── dataset/           # Training data
├── sample_data/       # Demo data for testing
├── requirements.txt   # Python dependencies
├── README.md         # Project overview
└── run_dashboard.sh  # Launch script
```

**Important:** The `models/minirocket/` folder contains the trained model files (~1.7 MB). These files are required for the dashboard to work.

---

## 🔧 Troubleshooting

### Issue: "Python not found"

**Solution:**
- Install Python 3.8+ from [python.org](https://www.python.org)
- Try `python3` instead of `python`
- Restart your terminal after installation

### Issue: "pip not found"

**Solution:**
```bash
python -m ensurepip --upgrade
# Or
curl https://bootstrap.pypa.io/get-pip.py -o get-pip.py
python get-pip.py
```

### Issue: "Module not found" errors

**Solution:**
1. Make sure virtual environment is activated
2. Reinstall requirements:
   ```bash
   pip install --upgrade -r requirements.txt
   ```

### Issue: "Streamlit command not found"

**Solution:**
```bash
pip install --upgrade streamlit
# Or use Python module syntax
python -m streamlit run dashboard/app.py
```

### Issue: Dashboard won't open in browser

**Solution:**
1. Check if port 8501 is already in use
2. Manually open: http://localhost:8501
3. Try a different port:
   ```bash
   streamlit run dashboard/app.py --server.port 8502
   ```

### Issue: Model loading error

**Solution:**
1. Verify `models/minirocket/` folder exists
2. Check these files are present:
   - minirocket_model.pkl
   - label_encoder.pkl
   - scaler.pkl
   - metadata.json
3. Re-run test: `python test_dashboard.py`

### Issue: CSV upload fails

**Solution:**
1. Use provided sample data first: `sample_data/sample_vibration_data.csv`
2. Check your CSV format matches the requirements
3. See DASHBOARD_GUIDE.md for CSV format details

---

## 💻 Platform-Specific Notes

### macOS

- You may need to use `python3` and `pip3` instead of `python` and `pip`
- If you get permission errors, try: `sudo python3 -m pip install -r requirements.txt`
- Make script executable: `chmod +x run_dashboard.sh`

### Windows

- Use Command Prompt or PowerShell (not Git Bash for launcher)
- If execution policy error occurs:
  ```powershell
  Set-ExecutionPolicy -ExecutionPolicy RemoteSigned -Scope CurrentUser
  ```
- Use `python` and `pip` (not `python3`)

### Linux

- May need to install additional dependencies:
  ```bash
  sudo apt-get update
  sudo apt-get install python3-pip python3-venv
  ```
- Use `python3` and `pip3`

---

## 🌐 Network Access (Optional)

To access the dashboard from other devices on your network:

1. **Find your IP address:**
   - Mac/Linux: `ifconfig | grep "inet "`
   - Windows: `ipconfig`

2. **Launch with network access:**
   ```bash
   streamlit run dashboard/app.py --server.address 0.0.0.0
   ```

3. **Access from other devices:**
   ```
   http://YOUR-IP-ADDRESS:8501
   ```

---

## 📱 System Requirements

### Minimum

- **CPU:** Dual-core 2.0 GHz
- **RAM:** 4 GB
- **Storage:** 500 MB free space
- **OS:** Windows 10, macOS 10.13+, or Linux (Ubuntu 18.04+)

### Recommended

- **CPU:** Quad-core 2.5 GHz+
- **RAM:** 8 GB+
- **Storage:** 1 GB free space
- **Browser:** Chrome, Firefox, Safari, or Edge (latest version)

---

## ✅ Quick Start Checklist

- [ ] Python 3.8+ installed
- [ ] Project files extracted/cloned
- [ ] Virtual environment created
- [ ] Dependencies installed (`pip install -r requirements.txt`)
- [ ] Tests passing (`python test_dashboard.py`)
- [ ] Dashboard launches successfully
- [ ] Browser opens to http://localhost:8501
- [ ] Can upload sample data and get predictions

---

## 📚 Next Steps

1. **Read the guides:**
   - LAUNCH_INSTRUCTIONS.md (quick start)
   - DASHBOARD_GUIDE.md (detailed usage)
   - PROJECT_SUMMARY.md (technical details)

2. **Try the dashboard:**
   - Start with demo data
   - Explore all 5 pages
   - Test with your own CSV files

3. **Learn more:**
   - About page in dashboard
   - Model Analytics page
   - Data Explorer page

---

## 🆘 Getting Help

If you encounter issues:

1. **Check troubleshooting section** above
2. **Run diagnostic test:** `python test_dashboard.py`
3. **Review error messages** carefully
4. **Check documentation** in the guides
5. **Verify all files** are present (especially model files)

---

## 🔄 Updating

To update to a new version:

1. **Backup your data** (if any custom data added)
2. **Download new version**
3. **Extract over old files** (or replace folder)
4. **Update dependencies:**
   ```bash
   pip install --upgrade -r requirements.txt
   ```
5. **Test:** `python test_dashboard.py`

---

## 📦 Sharing with Others

When sharing this project:

**Include:**
- ✅ All project files
- ✅ Model files in `models/minirocket/`
- ✅ Documentation (README, guides)
- ✅ requirements.txt
- ✅ Sample data

**Optional:**
- Dataset in `dataset/` (large files)
- Virtual environment folder (will be recreated)

**Recommended sharing method:**
- ZIP file with all necessary files
- Or Git repository with model files included

---

## 📝 License

MIT License - See LICENSE file for details

---

**Ready to go! If everything installed correctly, you should now have a working fault classification dashboard.** 🎉

For detailed usage instructions, see [DASHBOARD_GUIDE.md](DASHBOARD_GUIDE.md)

# 📋 Quick Reference Card

## 🎯 For You (Project Owner)

### Run Dashboard Locally
```bash
./run_dashboard.sh        # Mac/Linux
run_dashboard.bat         # Windows
```

### Create Package to Share
```bash
./create_distribution.sh  # Mac/Linux
create_distribution.bat   # Windows
```

### Test Everything
```bash
python test_dashboard.py
```

---

## 📤 For Sharing with Others

### What to Send
1. **The ZIP file** (created by distribution script)
2. **This message:**

```
Setup Instructions:
1. Extract ZIP file
2. Install Python 3.8+ (python.org)
3. Open terminal in project folder
4. Run: pip install -r requirements.txt
5. Run: ./run_dashboard.sh (Mac) or run_dashboard.bat (Windows)

Opens at: http://localhost:8501
See INSTALLATION_GUIDE.md for help.
```

### Sharing Options
- 📧 **Email** (< 25 MB)
- ☁️ **Cloud** (Google Drive, Dropbox)
- 💾 **USB Drive**
- 🌐 **Git Repo** (GitHub, GitLab)

---

## 👥 For Recipients (New Users)

### Setup Steps
```bash
# 1. Extract ZIP
# 2. Open terminal in folder
# 3. Install dependencies
pip install -r requirements.txt

# 4. Launch dashboard
./run_dashboard.sh        # Mac/Linux
run_dashboard.bat         # Windows
```

### Requirements
- Python 3.8+
- 500 MB disk space
- 5 minutes setup time

### Help
- See `INSTALLATION_GUIDE.md`
- Run `python test_dashboard.py`

---

## 🚀 Dashboard Features

### Pages
1. **🏠 Home** - Overview & metrics
2. **🎯 Prediction** - Upload CSV, get results
3. **📊 Analytics** - Model performance
4. **📈 Explorer** - Dataset visualization
5. **ℹ️ About** - Documentation

### Quick Actions
- Upload CSV → Get prediction
- Use demo data → Test system
- View analytics → Check accuracy
- Export results → Download CSV

---

## 🔧 Troubleshooting

### Dashboard won't start?
```bash
pip install --upgrade streamlit
streamlit run dashboard/app.py
```

### Tests failing?
```bash
pip install --upgrade -r requirements.txt
python test_dashboard.py
```

### Model not loading?
Check files exist:
- `models/minirocket/minirocket_model.pkl`
- `models/minirocket/label_encoder.pkl`
- `models/minirocket/scaler.pkl`
- `models/minirocket/metadata.json`

---

## 📊 Key Stats

- **Accuracy:** 99.98%
- **Speed:** < 1 second per prediction
- **Model Size:** 1.7 MB
- **Platforms:** Windows, Mac, Linux

---

## 📚 Documentation

| File | Purpose |
|------|---------|
| README.md | Project overview |
| INSTALLATION_GUIDE.md | Setup for new users |
| DASHBOARD_GUIDE.md | How to use dashboard |
| SHARING_GUIDE.md | How to share project |
| PROJECT_SUMMARY.md | Technical details |
| LAUNCH_INSTRUCTIONS.md | Quick start |

---

## 🌐 URLs

- **Local:** http://localhost:8501
- **Network:** http://YOUR-IP:8501 (if enabled)

---

## ⚡ Most Common Commands

```bash
# Install
pip install -r requirements.txt

# Test
python test_dashboard.py

# Run
./run_dashboard.sh

# Create package
./create_distribution.sh

# Manual launch
streamlit run dashboard/app.py
```

---

**Keep this card handy for quick reference!** 📌

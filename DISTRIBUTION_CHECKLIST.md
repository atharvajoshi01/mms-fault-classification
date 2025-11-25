# Distribution Checklist

Use this checklist when preparing to share the project with others.

## 📦 What to Share

### Essential Files (Required)

- [x] **Source Code**
  - `src/` folder
  - `dashboard/` folder
  - `scripts/` folder

- [x] **Model Files** (CRITICAL!)
  - `models/minirocket/minirocket_model.pkl` (1.7 MB)
  - `models/minirocket/label_encoder.pkl`
  - `models/minirocket/scaler.pkl`
  - `models/minirocket/metadata.json`

- [x] **Configuration**
  - `requirements.txt`
  - `dashboard/.streamlit/config.toml` (if exists)

- [x] **Documentation**
  - `README.md`
  - `INSTALLATION_GUIDE.md`
  - `DASHBOARD_GUIDE.md`
  - `LAUNCH_INSTRUCTIONS.md`
  - `PROJECT_SUMMARY.md`

- [x] **Scripts**
  - `run_dashboard.sh` (Mac/Linux)
  - `run_dashboard.bat` (Windows)
  - `test_dashboard.py`

- [x] **Sample Data**
  - `sample_data/sample_vibration_data.csv`

### Optional Files

- [ ] **Full Dataset** (Large - 21,559 samples)
  - `dataset/phase_2_2/` (~200+ MB)
  - Only include if recipient needs to retrain

- [ ] **Visualizations**
  - `visualizations/MiniRocket_Visualization/`
  - Pre-generated charts and plots

- [ ] **Additional Docs**
  - `DISTRIBUTION_CHECKLIST.md` (this file)

### Files to EXCLUDE

- [ ] **Virtual Environment**
  - `venv/` or `env/` folder (will be recreated)

- [ ] **Python Cache**
  - `__pycache__/` folders
  - `*.pyc` files
  - `.DS_Store` files

- [ ] **Temporary Files**
  - `temp_upload.csv`
  - Log files (optional)

- [ ] **Personal Data**
  - Any custom test data with sensitive info

---

## 📋 Pre-Distribution Checklist

### 1. Test Everything

```bash
# Run automated tests
python test_dashboard.py

# Launch dashboard manually
streamlit run dashboard/app.py

# Test all features:
- [ ] Home page loads
- [ ] Live prediction works with demo data
- [ ] CSV upload works
- [ ] Batch prediction works
- [ ] Analytics page displays correctly
- [ ] Data explorer works
- [ ] About page loads
```

### 2. Verify Model Files

```bash
# Check model files exist and have correct sizes
ls -lh models/minirocket/

# Should show:
# minirocket_model.pkl (~1.7 MB)
# label_encoder.pkl (~600 bytes)
# scaler.pkl (~600 bytes)
# metadata.json (~400 bytes)
```

### 3. Check Documentation

- [ ] README.md is up to date
- [ ] INSTALLATION_GUIDE.md is clear
- [ ] All file paths in docs are correct
- [ ] Screenshots/examples are included (if any)

### 4. Clean Up

```bash
# Remove unnecessary files
find . -name "__pycache__" -type d -exec rm -rf {} + 2>/dev/null
find . -name "*.pyc" -delete
find . -name ".DS_Store" -delete
rm -f temp_upload.csv
```

### 5. Create Distribution Package

Choose your method:

**Option A: ZIP File** (Recommended)
```bash
# Create ZIP excluding unwanted files
zip -r mms_fault_classification.zip . \
  -x "*.git*" \
  -x "*__pycache__*" \
  -x "*.pyc" \
  -x "*venv/*" \
  -x "*env/*" \
  -x "*.DS_Store"
```

**Option B: Git Repository**
```bash
# Create .gitignore
cat > .gitignore << 'GITIGNORE'
# Python
__pycache__/
*.py[cod]
*$py.class
*.so
.Python
venv/
env/
ENV/

# OS
.DS_Store
Thumbs.db

# IDE
.vscode/
.idea/
*.swp
*.swo

# Temporary
temp_upload.csv
*.log

# Large files (optional - remove if you want to include dataset)
# dataset/phase_2_2/*.csv
GITIGNORE

# Initialize and commit
git init
git add .
git commit -m "Initial commit: MMS Fault Classification System"
```

**Option C: Cloud Storage**
- Upload to Google Drive, Dropbox, or OneDrive
- Share link with recipient
- Include INSTALLATION_GUIDE.md link in email

---

## 📧 Sharing Instructions

### Email Template

```
Subject: MMS Fault Classification Dashboard - Installation Package

Hi [Name],

I'm sharing the MMS Fault Classification Dashboard project with you.

What's included:
- Complete dashboard with 5 interactive pages
- Trained model (99.98% accuracy)
- Sample data for testing
- Comprehensive documentation

Setup (5-10 minutes):
1. Extract the ZIP file
2. Install Python 3.8+ (if not already installed)
3. Open terminal in project folder
4. Run: pip install -r requirements.txt
5. Launch: ./run_dashboard.sh (Mac/Linux) or run_dashboard.bat (Windows)

Detailed instructions: See INSTALLATION_GUIDE.md

The dashboard will open at http://localhost:8501

Questions? Check DASHBOARD_GUIDE.md or reply to this email.

Best regards,
[Your Name]
```

---

## 🔍 Verification Before Sending

### Size Check

```bash
# Check total size
du -sh .

# Should be approximately:
# With dataset: ~250 MB
# Without dataset: ~50 MB
```

### Quick Test (On Clean System if Possible)

1. **Extract/clone to new location**
2. **Create fresh virtual environment**
3. **Install dependencies**
4. **Run tests**
5. **Launch dashboard**
6. **Test basic functionality**

---

## 🌐 Distribution Methods

### Method 1: Direct File Transfer

**Best for:** Small teams, local sharing

**Steps:**
1. Create ZIP file
2. Transfer via email, USB, or network share
3. Provide INSTALLATION_GUIDE.md

**Pros:** Simple, no internet required for use
**Cons:** Manual updates needed

### Method 2: Git Repository

**Best for:** Teams, version control, collaboration

**Steps:**
1. Create repository (GitHub, GitLab, Bitbucket)
2. Push code
3. Share repository URL
4. Recipients clone and setup

**Pros:** Easy updates, version control
**Cons:** Requires Git knowledge

### Method 3: Cloud Storage

**Best for:** Large files, multiple recipients

**Steps:**
1. Upload to cloud (Google Drive, Dropbox, etc.)
2. Generate sharing link
3. Send link + INSTALLATION_GUIDE.md

**Pros:** Easy access, good for large files
**Cons:** Requires internet, storage limits

### Method 4: Docker Container (Advanced)

**Best for:** Consistent deployment, servers

**Create Dockerfile:**
```dockerfile
FROM python:3.9-slim

WORKDIR /app
COPY requirements.txt .
RUN pip install -r requirements.txt

COPY . .

EXPOSE 8501

CMD ["streamlit", "run", "dashboard/app.py", "--server.address", "0.0.0.0"]
```

**Build and share:**
```bash
docker build -t mms-fault-classification .
docker save mms-fault-classification > mms-dashboard.tar
```

---

## 📝 Support Preparation

### Create FAQ Document

Common questions recipients might have:

1. **Q: What Python version do I need?**
   A: Python 3.8 or higher

2. **Q: How much disk space needed?**
   A: ~500 MB minimum (50 MB without dataset)

3. **Q: Can I run this without internet?**
   A: Yes, after initial setup

4. **Q: How do I update the model?**
   A: Run `python scripts/train_minirocket.py` with new data

5. **Q: Can multiple people use it simultaneously?**
   A: Each person needs their own instance (or use network sharing)

---

## ✅ Final Checklist Before Distribution

- [ ] All tests pass
- [ ] Model files included and working
- [ ] Documentation is complete and accurate
- [ ] Sample data included
- [ ] Launch scripts work on target platforms
- [ ] File size is reasonable
- [ ] Sensitive data removed
- [ ] License file included (if applicable)
- [ ] Installation guide reviewed
- [ ] Tested on clean system (if possible)

---

## 📊 Package Contents Summary

**Total Files:** ~100
**Total Size:** ~50 MB (without dataset)
**Model Files:** 4 files (~1.7 MB)
**Documentation:** 5 markdown files
**Code Files:** ~20 Python files
**Platform Support:** Mac, Linux, Windows

---

## 🚀 Ready to Share!

Once you've completed this checklist:

1. ✅ Create distribution package (ZIP or repository)
2. ✅ Include INSTALLATION_GUIDE.md
3. ✅ Send to recipient with instructions
4. ✅ Be available for questions
5. ✅ Gather feedback for improvements

**Your project is ready for distribution!** 🎉

# 🌐 Deploy Dashboard Online (Streamlit Cloud)

## FREE Deployment - Others Access via URL

### Option 1: Streamlit Cloud (Recommended - FREE)

**Best for:** Public/team access, no installation needed by users

#### Step 1: Prepare Your Project

1. **Create a GitHub account** (if you don't have one)
   - Go to: https://github.com/signup

2. **Create a new repository**
   - Click "New repository"
   - Name: `mms-fault-classification`
   - Make it **Public** (required for free Streamlit Cloud)
   - Don't initialize with README (we have one)

#### Step 2: Push Your Code to GitHub

```bash
# In your project folder
cd "/Users/atharva/Documents/MMS Client/mms_fault_classification"

# Initialize git (if not already done)
git init

# Create .gitignore
cat > .gitignore << 'GITIGNORE'
__pycache__/
*.pyc
venv/
env/
.DS_Store
temp_upload.csv
*.log
GITIGNORE

# Add all files
git add .

# Commit
git commit -m "Initial commit: MMS Fault Classification Dashboard"

# Add your GitHub repository (replace with your URL)
git remote add origin https://github.com/YOUR-USERNAME/mms-fault-classification.git

# Push to GitHub
git push -u origin main
```

#### Step 3: Deploy on Streamlit Cloud

1. **Go to:** https://streamlit.io/cloud

2. **Sign in** with your GitHub account

3. **Click "New app"**

4. **Configure deployment:**
   - Repository: `YOUR-USERNAME/mms-fault-classification`
   - Branch: `main`
   - Main file path: `dashboard/app.py`

5. **Click "Deploy"**

6. **Wait 2-3 minutes** for deployment

7. **Get your URL:** `https://YOUR-APP-NAME.streamlit.app`

#### Step 4: Share the URL

Send this to anyone:
```
Dashboard: https://YOUR-APP-NAME.streamlit.app

No installation needed - just click and use!
```

**That's it!** ✨

---

### Option 2: Local Network Sharing (Same WiFi)

**Best for:** Office/home network, same location

#### Step 1: Find Your IP Address

**Mac:**
```bash
ifconfig | grep "inet " | grep -v 127.0.0.1
```

**Windows:**
```cmd
ipconfig
```

Look for: `192.168.X.X` or `10.0.X.X`

#### Step 2: Launch Dashboard with Network Access

```bash
streamlit run dashboard/app.py --server.address 0.0.0.0 --server.port 8501
```

#### Step 3: Share Your IP

Others on the same WiFi can access:
```
http://YOUR-IP-ADDRESS:8501
```

Example: `http://192.168.1.100:8501`

**Note:** Your computer must stay on and running the dashboard.

---

### Option 3: Cloud Platforms (Advanced)

#### Heroku (Free Tier Available)

1. **Create Heroku account:** https://heroku.com

2. **Install Heroku CLI:** https://devcenter.heroku.com/articles/heroku-cli

3. **Create deployment files:**

```bash
# Create Procfile
echo "web: streamlit run dashboard/app.py --server.port=\$PORT --server.address=0.0.0.0" > Procfile

# Create setup.sh
cat > setup.sh << 'SETUP'
mkdir -p ~/.streamlit/

echo "\
[server]\n\
headless = true\n\
port = \$PORT\n\
enableCORS = false\n\
\n\
" > ~/.streamlit/config.toml
SETUP

chmod +x setup.sh
```

4. **Deploy:**

```bash
heroku login
heroku create your-app-name
git push heroku main
```

Access at: `https://your-app-name.herokuapp.com`

---

### Option 4: Docker + Cloud (Most Flexible)

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

**Deploy to:**
- Google Cloud Run
- AWS ECS
- Azure Container Apps
- DigitalOcean App Platform

---

## 🎯 Quick Comparison

| Method | Cost | Setup Time | Best For |
|--------|------|------------|----------|
| **Streamlit Cloud** | FREE | 10 min | Everyone (recommended) |
| **Local Network** | FREE | 2 min | Same office/home |
| **Heroku** | FREE tier | 20 min | Small teams |
| **Docker + Cloud** | Varies | 30+ min | Large scale |

---

## 🚀 Recommended: Streamlit Cloud

**Why?**
- ✅ Completely FREE
- ✅ 10 minute setup
- ✅ No server management
- ✅ Automatic HTTPS
- ✅ Auto-updates from GitHub
- ✅ Share via simple URL
- ✅ Works globally

**Limitations:**
- Repository must be public (for free tier)
- 1 GB RAM limit (your app uses ~200 MB - fine!)
- Community tier: limited resources

---

## 📱 What Users Will See

After deployment, users simply:
1. Click the URL you share
2. Dashboard loads in browser
3. Use immediately - no installation!

Works on:
- ✅ Desktop computers
- ✅ Laptops
- ✅ Tablets
- ✅ Smartphones
- ✅ Any device with a browser

---

## 🔒 Security Considerations

### Public Deployment (Streamlit Cloud)
- Anyone with the URL can access
- Don't include sensitive data
- Consider adding password protection (see below)

### Add Password Protection (Optional)

Add to top of `dashboard/app.py`:

```python
import streamlit as st

def check_password():
    """Returns `True` if the user had the correct password."""
    
    def password_entered():
        """Checks whether a password entered by the user is correct."""
        if st.session_state["password"] == "your-secret-password":
            st.session_state["password_correct"] = True
            del st.session_state["password"]  # don't store password
        else:
            st.session_state["password_correct"] = False

    if "password_correct" not in st.session_state:
        # First run, show input for password
        st.text_input(
            "Password", type="password", on_change=password_entered, key="password"
        )
        return False
    elif not st.session_state["password_correct"]:
        # Password incorrect
        st.text_input(
            "Password", type="password", on_change=password_entered, key="password"
        )
        st.error("😕 Password incorrect")
        return False
    else:
        # Password correct
        return True

if check_password():
    # Your existing dashboard code here
    # (rest of app.py content)
```

---

## ⚡ Quick Start (Streamlit Cloud)

```bash
# 1. Push to GitHub
git init
git add .
git commit -m "Dashboard"
git remote add origin https://github.com/YOUR-USERNAME/repo-name.git
git push -u origin main

# 2. Go to streamlit.io/cloud
# 3. Connect GitHub
# 4. Deploy dashboard/app.py
# 5. Share URL

# Done! 🎉
```

---

## 🆘 Troubleshooting

### Deployment fails?
- Check `requirements.txt` has all dependencies
- Ensure `dashboard/app.py` path is correct
- Check Streamlit Cloud logs for errors

### Slow loading?
- Model files are large? (yours are only 1.7 MB - should be fine)
- Too many users? (upgrade to paid tier)

### Can't access on network?
- Check firewall settings
- Ensure same WiFi network
- Try different port: `--server.port 8502`

---

## 📞 Need Help?

- **Streamlit Docs:** https://docs.streamlit.io/deploy
- **Community Forum:** https://discuss.streamlit.io
- **GitHub Issues:** Report deployment problems

---

**Recommendation:** Start with Streamlit Cloud (FREE, easiest) 🚀

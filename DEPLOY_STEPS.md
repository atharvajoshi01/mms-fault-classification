# Step-by-Step Deployment to Streamlit Cloud

## Part 1: Push to GitHub (5 minutes)

### Step 1: Create GitHub Repository

1. Go to: https://github.com/new
2. Fill in:
   - **Repository name:** `mms-fault-classification` (or any name you like)
   - **Description:** "Vibration fault classification dashboard with 99.98% accuracy"
   - **Visibility:** Select **Public** (required for free Streamlit Cloud)
   - **DO NOT** check "Initialize this repository with a README"
3. Click **"Create repository"**

### Step 2: Push Your Code

Copy the repository URL shown (looks like: `https://github.com/YOUR-USERNAME/mms-fault-classification.git`)

Then run these commands in your terminal:

```bash
cd "/Users/atharva/Documents/MMS Client/mms_fault_classification"

# Initialize git (if not already done)
git init

# Add all files
git add .

# Create first commit
git commit -m "Initial commit: MMS Fault Classification Dashboard"

# Connect to your GitHub repo (REPLACE with your actual URL)
git remote add origin https://github.com/YOUR-USERNAME/mms-fault-classification.git

# Push to GitHub
git branch -M main
git push -u origin main
```

You'll be asked to login to GitHub - enter your credentials.

✅ Your code is now on GitHub!

---

## Part 2: Deploy on Streamlit Cloud (5 minutes)

### Step 3: Go to Streamlit Cloud

1. Open: https://streamlit.io/cloud
2. Click **"Sign in"**
3. Choose **"Continue with GitHub"**
4. Authorize Streamlit to access your GitHub

### Step 4: Deploy Your App

1. Click **"New app"** (big button)

2. Fill in the form:
   - **Repository:** Select `YOUR-USERNAME/mms-fault-classification`
   - **Branch:** `main`
   - **Main file path:** `dashboard/app.py`

3. Click **"Deploy!"**

### Step 5: Wait for Deployment

- Deployment takes 2-3 minutes
- You'll see logs showing installation progress
- When done, you'll see: "Your app is live! 🎉"

### Step 6: Get Your URL

Your dashboard is now live at:
```
https://YOUR-APP-NAME.streamlit.app
```

**Share this URL with anyone!**

---

## Part 3: Share with Others

Send this to your team:

```
Hi team!

Access the MMS Fault Classification Dashboard here:
https://YOUR-APP-NAME.streamlit.app

Features:
- Upload CSV files for fault prediction
- Real-time analysis with 99.98% accuracy
- Interactive visualizations
- No installation needed

Just click and use!
```

---

## Troubleshooting

### If git push asks for credentials:

**Option 1: Use Personal Access Token (Recommended)**

1. Go to: https://github.com/settings/tokens
2. Click "Generate new token" → "Generate new token (classic)"
3. Give it a name: "MMS Dashboard"
4. Check: ✅ `repo` (all sub-items)
5. Click "Generate token"
6. **Copy the token** (you won't see it again!)
7. When git asks for password, paste the token

**Option 2: Use SSH**

```bash
# Generate SSH key
ssh-keygen -t ed25519 -C "your-email@example.com"

# Copy public key
cat ~/.ssh/id_ed25519.pub

# Add to GitHub: Settings → SSH and GPG keys → New SSH key
# Then change remote URL:
git remote set-url origin git@github.com:YOUR-USERNAME/mms-fault-classification.git
```

### If deployment fails:

1. Check Streamlit Cloud logs for errors
2. Ensure `requirements.txt` includes all dependencies
3. Verify `dashboard/app.py` path is correct
4. Check that model files are in the repository

### If "Repository too large" error:

The dataset might be too big. Create `.gitignore`:

```bash
echo "dataset/phase_2_2/*.csv" >> .gitignore
git rm -r --cached dataset/phase_2_2/*.csv
git add .gitignore
git commit -m "Remove large dataset files"
git push
```

The model files (1.7 MB) are fine and needed!

---

## Managing Your Deployment

### Update the Dashboard

When you make changes:

```bash
git add .
git commit -m "Updated dashboard"
git push
```

Streamlit Cloud auto-deploys updates!

### View Logs

- Go to: https://streamlit.io/cloud
- Click on your app
- Click "Manage app" → "Logs"

### Reboot App

If app is slow or stuck:
- Manage app → "Reboot app"

---

## ✅ Success Checklist

- [ ] Created GitHub repository
- [ ] Pushed code with `git push`
- [ ] Signed into Streamlit Cloud with GitHub
- [ ] Deployed app
- [ ] Got URL: `https://xxx.streamlit.app`
- [ ] Tested dashboard works
- [ ] Shared URL with team

---

**That's it! Your dashboard is now live and accessible to anyone!** 🎉

"""
Streamlit Cloud entry point for MMS Fault Classification Dashboard.
This file serves as the main entry point for Streamlit Cloud deployment.
"""

import sys
from pathlib import Path

# Add project root to path
project_root = Path(__file__).parent
sys.path.insert(0, str(project_root))

# Execute the dashboard app
# We use exec to run the dashboard/app.py code directly
dashboard_path = project_root / "dashboard" / "app.py"
with open(dashboard_path, 'r') as f:
    exec(f.read())

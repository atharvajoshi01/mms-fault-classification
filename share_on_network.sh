#!/bin/bash

# Quick Network Sharing Script
# Launches dashboard accessible to others on same WiFi

echo "==========================================="
echo "MMS Fault Classification Dashboard"
echo "Network Sharing Mode"
echo "==========================================="
echo ""

# Get IP address
echo "Finding your IP address..."
if [[ "$OSTYPE" == "darwin"* ]]; then
    # macOS
    IP=$(ifconfig | grep "inet " | grep -v 127.0.0.1 | head -1 | awk '{print $2}')
else
    # Linux
    IP=$(hostname -I | awk '{print $1}')
fi

if [ -z "$IP" ]; then
    echo "⚠️  Could not auto-detect IP address"
    echo "Please find your IP manually:"
    echo "  Mac: System Preferences → Network"
    echo "  Linux: ip addr show"
    echo ""
    read -p "Enter your IP address: " IP
fi

echo ""
echo "==========================================="
echo "Your IP Address: $IP"
echo "==========================================="
echo ""
echo "Others can access your dashboard at:"
echo ""
echo "  🌐 http://$IP:8501"
echo ""
echo "Share this URL with people on the same WiFi network."
echo ""
echo "⚠️  Important:"
echo "  • Keep this terminal window open"
echo "  • Keep your computer running"
echo "  • Everyone must be on the same WiFi"
echo ""
echo "Press Ctrl+C to stop sharing"
echo ""
echo "==========================================="
echo "Launching dashboard..."
echo "==========================================="
echo ""

# Launch Streamlit with network access
streamlit run dashboard/app.py --server.address 0.0.0.0 --server.port 8501

#!/bin/bash

# Distribution Package Creator
# Creates a clean ZIP file ready for sharing

echo "=========================================="
echo "MMS Fault Classification"
echo "Distribution Package Creator"
echo "=========================================="
echo ""

# Package name
PACKAGE_NAME="mms_fault_classification_v1.0"
TIMESTAMP=$(date +%Y%m%d_%H%M%S)
OUTPUT_FILE="${PACKAGE_NAME}_${TIMESTAMP}.zip"

echo "Creating distribution package..."
echo ""

# Clean up temporary files
echo "1. Cleaning temporary files..."
find . -name "__pycache__" -type d -exec rm -rf {} + 2>/dev/null
find . -name "*.pyc" -delete 2>/dev/null
find . -name ".DS_Store" -delete 2>/dev/null
rm -f temp_upload.csv 2>/dev/null
echo "   ✓ Cleanup complete"

# Verify model files exist
echo ""
echo "2. Verifying model files..."
if [ -f "models/minirocket/minirocket_model.pkl" ]; then
    MODEL_SIZE=$(du -h models/minirocket/minirocket_model.pkl | cut -f1)
    echo "   ✓ Model found (${MODEL_SIZE})"
else
    echo "   ✗ ERROR: Model file not found!"
    echo "   Make sure models/minirocket/minirocket_model.pkl exists"
    exit 1
fi

# Check required files
echo ""
echo "3. Checking required files..."
REQUIRED_FILES=(
    "requirements.txt"
    "README.md"
    "INSTALLATION_GUIDE.md"
    "dashboard/app.py"
    "src/models/minirocket.py"
    "test_dashboard.py"
)

for file in "${REQUIRED_FILES[@]}"; do
    if [ -f "$file" ]; then
        echo "   ✓ $file"
    else
        echo "   ✗ Missing: $file"
        exit 1
    fi
done

# Create ZIP file
echo ""
echo "4. Creating ZIP package..."

zip -r "$OUTPUT_FILE" . \
    -x "*.git*" \
    -x "*__pycache__*" \
    -x "*.pyc" \
    -x "*venv/*" \
    -x "*env/*" \
    -x "*.DS_Store" \
    -x "*Thumbs.db" \
    -x "dataset/phase_2/*" \
    -x "create_distribution.sh" \
    > /dev/null 2>&1

if [ $? -eq 0 ]; then
    PACKAGE_SIZE=$(du -h "$OUTPUT_FILE" | cut -f1)
    echo "   ✓ Package created"
else
    echo "   ✗ Failed to create package"
    exit 1
fi

# Summary
echo ""
echo "=========================================="
echo "Distribution Package Ready!"
echo "=========================================="
echo ""
echo "Package: $OUTPUT_FILE"
echo "Size: $PACKAGE_SIZE"
echo ""
echo "What's included:"
echo "  ✓ Dashboard application (5 pages)"
echo "  ✓ Trained model (99.98% accuracy)"
echo "  ✓ Complete documentation"
echo "  ✓ Sample data"
echo "  ✓ Test suite"
echo "  ✓ Launch scripts"
echo ""
echo "To share:"
echo "  1. Send $OUTPUT_FILE to recipient"
echo "  2. Include INSTALLATION_GUIDE.md"
echo "  3. They extract and run: pip install -r requirements.txt"
echo "  4. Launch: ./run_dashboard.sh"
echo ""
echo "=========================================="

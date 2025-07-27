#!/bin/bash
# Render build script to ensure latest code is used

echo "🚀 HazardEye Render Build Script"
echo "================================="

# Show current directory and files
echo "📂 Current directory: $(pwd)"
echo "📄 Files in directory:"
ls -la

# Show git status and latest commit
echo "📊 Git status:"
git log --oneline -3

# Install Python requirements
echo "📦 Installing Python requirements..."
pip install -r requirements.txt

# Run deployment test
echo "🧪 Running deployment test..."
if python test_deployment.py; then
    echo "✅ Deployment test passed!"
else
    echo "❌ Deployment test failed!"
    exit 1
fi

echo "✅ Build completed successfully!"

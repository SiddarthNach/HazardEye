#!/bin/bash

# HazardEye EC2 Startup Script
set -e

echo "🚀 Starting HazardEye deployment..."

# Update system
sudo yum update -y

# Install Python 3.9 and pip
sudo yum install -y python3 python3-pip git

# Install system dependencies for OpenCV
sudo yum install -y mesa-libGL

# Create application directory
sudo mkdir -p /opt/hazardeye
sudo chown ec2-user:ec2-user /opt/hazardeye
cd /opt/hazardeye

# Clone your repository (you'll need to make it public or use deploy keys)
# For now, we'll assume you upload files manually
echo "📁 Application directory ready at /opt/hazardeye"

# Install Python dependencies
pip3 install --user -r requirements.txt

# Create systemd service
sudo tee /etc/systemd/system/hazardeye.service > /dev/null <<EOF
[Unit]
Description=HazardEye Streamlit App
After=network.target

[Service]
Type=simple
User=ec2-user
WorkingDirectory=/opt/hazardeye
Environment=PATH=/home/ec2-user/.local/bin:/usr/local/bin:/usr/bin:/bin
Environment=PYTHONPATH=/opt/hazardeye
ExecStart=/home/ec2-user/.local/bin/streamlit run app.py --server.port=8501 --server.address=0.0.0.0
Restart=always
RestartSec=3

[Install]
WantedBy=multi-user.target
EOF

# Start and enable service
sudo systemctl daemon-reload
sudo systemctl enable hazardeye
sudo systemctl start hazardeye

echo "✅ HazardEye service started!"
echo "🌐 Application will be available at http://YOUR_EC2_IP:8501"

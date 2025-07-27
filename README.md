# HazardEye - AI-Powered Road Safety Analysis System

HazardEye is a comprehensive road safety analysis system that combines lane detection and pothole detection capabilities using computer vision and machine learning. The system provides real-time video analysis through a user-friendly Streamlit web interface.

## Features

- **Lane Detection**: Advanced lane line detection and analysis using OpenCV
- **Pothole Detection**: AI-powered pothole detection using computer vision techniques
- **Lane Deviation Analysis**: Real-time monitoring of lane keeping behavior
- **Curve Detection**: Identification and analysis of road curvature
- **User Management**: Secure user authentication and session management
- **Cloud Storage**: AWS S3 integration for video storage and processing
- **Web Interface**: Intuitive Streamlit-based dashboard

## System Architecture

- **Frontend**: Streamlit web application
- **Backend**: Python with OpenCV for computer vision processing
- **Storage**: AWS S3 for video files, SQLite for user data
- **Deployment**: AWS EC2 with systemd service management

## Prerequisites

- Python 3.7+
- AWS Account (for cloud deployment)
- OpenCV dependencies
- FFmpeg (for video processing)

## Local Development Setup

1. **Clone the repository**:
   ```bash
   git clone <your-repo-url>
   cd HazardEye
   ```

2. **Install dependencies**:
   ```bash
   pip install -r requirements.txt
   ```

3. **Set up environment variables**:
   Create a `.env` file in the root directory:
   ```env
   AWS_ACCESS_KEY_ID=your_access_key
   AWS_SECRET_ACCESS_KEY=your_secret_key
   AWS_DEFAULT_REGION=us-east-2
   S3_BUCKET_NAME=your-bucket-name
   ```

4. **Run the application**:
   ```bash
   streamlit run app.py
   ```

## AWS Deployment

### Prerequisites
- AWS CLI installed and configured
- AWS account with appropriate permissions

### Step 1: AWS CLI Configuration
```bash
aws configure
```
Enter your AWS credentials and set region to `us-east-2`.

### Step 2: Create S3 Bucket
```bash
aws s3 mb s3://your-bucket-name --region us-east-2
```

### Step 3: Apply S3 Bucket Policy
```bash
aws s3api put-bucket-policy --bucket your-bucket-name --policy file://s3-policy.json
```

### Step 4: Create IAM Role
```bash
aws iam create-role --role-name HazardEyeEC2Role --assume-role-policy-document file://trust-policy.json
aws iam attach-role-policy --role-name HazardEyeEC2Role --policy-arn arn:aws:iam::aws:policy/AmazonS3FullAccess
aws iam create-instance-profile --instance-profile-name HazardEyeProfile
aws iam add-role-to-instance-profile --instance-profile-name HazardEyeProfile --role-name HazardEyeEC2Role
```

### Step 5: Launch EC2 Instance
```bash
aws ec2 run-instances \
    --image-id ami-0c02fb55956c7d316 \
    --instance-type t3.medium \
    --key-name your-key-pair \
    --security-group-ids sg-your-security-group \
    --iam-instance-profile Name=HazardEyeProfile \
    --user-data file://setup_ec2.sh
```

### Step 6: Deploy Application
1. Package the application:
   ```bash
   tar -czf hazard-eye-app.tar.gz app.py utils_fallback.py requirements.txt .env aws_config.py users.db
   ```

2. Upload to EC2:
   ```bash
   scp -i your-key.pem hazard-eye-app.tar.gz ec2-user@your-ec2-ip:~
   ```

3. SSH into EC2 and extract:
   ```bash
   ssh -i your-key.pem ec2-user@your-ec2-ip
   tar -xzf hazard-eye-app.tar.gz
   ```

## Project Structure

```
HazardEye/
├── app.py                          # Main Streamlit application
├── utils.py                        # Utility functions (original)
├── utils_fallback.py              # Utility functions with fallback logic
├── aws_config.py                  # AWS S3 integration
├── requirements.txt               # Python dependencies
├── setup_ec2.sh                   # EC2 deployment script
├── s3-policy.json                 # S3 bucket policy
├── trust-policy.json              # IAM trust policy
├── users.db                       # SQLite user database
├── .env                           # Environment variables (not in git)
├── lane_analysis.py               # Lane detection algorithms
├── lane_analysis_new.py           # Enhanced lane detection
├── debug_scoring.py               # Debugging utilities
├── test_*.py                      # Test files
├── Pothole_Detection_Model_Training.ipynb  # ML model training
├── train/                         # Training data and results
├── temp/                          # Temporary files
└── lane_detection/                # Lane detection modules
```

## API Endpoints

The Streamlit app provides the following main pages:
- **Login/Signup**: User authentication
- **Dashboard**: Main analysis interface
- **Video Upload**: Upload videos for analysis
- **Results**: View analysis results and statistics

## Configuration

### Environment Variables
- `AWS_ACCESS_KEY_ID`: AWS access key
- `AWS_SECRET_ACCESS_KEY`: AWS secret key  
- `AWS_DEFAULT_REGION`: AWS region (default: us-east-2)
- `S3_BUCKET_NAME`: S3 bucket for video storage

### Application Settings
- Default port: 8501
- Maximum video file size: 200MB
- Supported formats: MP4, AVI, MOV

## Monitoring and Logs

### Service Status (on EC2)
```bash
sudo systemctl status hazard-eye
```

### View Logs
```bash
sudo journalctl -u hazard-eye -f
```

### Restart Service
```bash
sudo systemctl restart hazard-eye
```

## Troubleshooting

### Common Issues

1. **Import Errors**: Ensure all dependencies are installed
   ```bash
   pip install -r requirements.txt
   ```

2. **AWS Credentials**: Verify AWS configuration
   ```bash
   aws sts get-caller-identity
   ```

3. **Port Issues**: Check if port 8501 is available
   ```bash
   netstat -tlnp | grep 8501
   ```

4. **Memory Issues**: Consider upgrading EC2 instance type for large videos

### Performance Optimization

- Use t3.medium or larger EC2 instances for better performance
- Optimize video resolution for faster processing
- Consider GPU instances for enhanced ML capabilities

## Contributing

1. Fork the repository
2. Create a feature branch
3. Make your changes
4. Add tests if applicable
5. Submit a pull request

## License

This project is licensed under the MIT License - see the LICENSE file for details.

## Support

For questions or issues, please create an issue in the GitHub repository.

---

**Note**: This system uses OpenCV-based analysis for maximum compatibility. YOLO-based ML features are available but may require additional configuration in production environments.

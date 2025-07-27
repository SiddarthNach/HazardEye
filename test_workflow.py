#!/usr/bin/env python3
"""
Quick test script to verify video upload and analysis works
"""

import os
import sys
import tempfile

# Add current directory to path
current_dir = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, current_dir)

from utils import analyze_road_safety, save_uploaded_file

class MockUploadedFile:
    """Mock Streamlit uploaded file for testing"""
    def __init__(self, file_path):
        self.name = os.path.basename(file_path)
        with open(file_path, 'rb') as f:
            self.content = f.read()
    
    def getbuffer(self):
        return self.content

def test_file_upload_workflow():
    """Test the complete file upload and analysis workflow"""
    print("🧪 Testing complete upload and analysis workflow...")
    
    # Find a test video
    video_files = [f for f in os.listdir('.') if f.endswith(('.mp4', '.avi', '.mov'))]
    if not video_files:
        print("❌ No video files found for testing")
        return False
    
    test_video = video_files[0]
    print(f"📹 Using test video: {test_video}")
    
    try:
        # Step 1: Simulate file upload
        mock_file = MockUploadedFile(test_video)
        print(f"✅ Mock file created: {mock_file.name}")
        
        # Step 2: Save uploaded file
        saved_path = save_uploaded_file(mock_file)
        print(f"✅ File saved to: {saved_path}")
        
        # Step 3: Verify file exists and is readable
        if not os.path.exists(saved_path):
            print(f"❌ Saved file doesn't exist: {saved_path}")
            return False
        
        file_size = os.path.getsize(saved_path)
        print(f"✅ Saved file size: {file_size} bytes")
        
        # Step 4: Run analysis
        print("🔄 Starting analysis...")
        result = analyze_road_safety(saved_path)
        
        # Step 5: Check results
        print(f"✅ Analysis result type: {type(result)}")
        print(f"✅ Analysis keys: {list(result.keys()) if isinstance(result, dict) else 'Not a dict'}")
        print(f"✅ Analysis complete: {result.get('analysis_complete', 'Unknown')}")
        
        if result.get('analysis_complete'):
            print(f"🏆 Safety score: {result['safety_score']}")
            print("✅ Full workflow test PASSED!")
            return True
        else:
            print(f"❌ Analysis failed: {result.get('error', 'Unknown error')}")
            return False
            
    except Exception as e:
        print(f"❌ Workflow test failed: {e}")
        import traceback
        traceback.print_exc()
        return False
    finally:
        # Cleanup
        if 'saved_path' in locals() and os.path.exists(saved_path):
            try:
                os.remove(saved_path)
                print(f"🧹 Cleaned up: {saved_path}")
            except:
                pass

def main():
    print("🔍 File Upload and Analysis Workflow Test")
    print("=" * 50)
    
    success = test_file_upload_workflow()
    
    print("\n" + "=" * 50)
    if success:
        print("🎉 All tests passed! The upload and analysis workflow should work.")
    else:
        print("🚨 Workflow test failed. Check the error messages above.")

if __name__ == "__main__":
    main()

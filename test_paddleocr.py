import sys

def check_installations():
    """Check if all required packages are installed"""
    required_packages = {
        'paddleocr': 'paddleocr',
        'cv2': 'opencv-python',
        'pandas': 'pandas',
        'paddle': 'paddlepaddle'
    }
    
    missing_packages = []
    
    print("Checking installed packages...\n")
    
    for package, pip_name in required_packages.items():
        try:
            __import__(package)
            print(f"✓ {pip_name} is installed")
        except ImportError:
            print(f"✗ {pip_name} is NOT installed")
            missing_packages.append(pip_name)
    
    if missing_packages:
        print(f"\n⚠ Missing packages: {', '.join(missing_packages)}")
        print("\nTo install missing packages, run:")
        for package in missing_packages:
            print(f"  pip install {package}")
        return False
    else:
        print("\n✓ All packages are installed!")
        return True

def test_paddleocr(image_path):
    """Test PaddleOCR on a sample image"""
    try:
        from paddleocr import PaddleOCR
        import cv2
        
        print("\n" + "="*50)
        print("Testing PaddleOCR...")
        print("="*50)
        
        # Initialize PaddleOCR
        # use_angle_cls=True helps with rotated text
        # lang='en' for English (default)
        ocr = PaddleOCR(use_angle_cls=True, lang='en')
        
        # Read image
        img = cv2.imread(image_path)
        if img is None:
            print(f"✗ Could not read image at: {image_path}")
            print("Please check the file path and try again.")
            return False
        
        print(f"✓ Image loaded successfully: {img.shape}")
        
        # Perform OCR
        print("\nRunning OCR... (this may take a moment)")
        result = ocr.predict(image_path)
        
        # Display results
        print("\n" + "="*50)
        print("OCR Results:")
        print("="*50)


        if result and len(result) > 0:

            ocr_result = result[0]
                        
            # Print each key-value pair
            texts = ocr_result.get('rec_texts', [])
            scores = ocr_result.get('rec_scores', [])
           
            print(f"\nFound {len(texts)} text items:\n")
           
            for idx, (text, score) in enumerate(zip(texts, scores)):
                print(f"{idx+1}. Text: '{text}' (Confidence: {score:.3f})")
                
        print("\n✓ PaddleOCR test completed successfully!")
        return True
        
    except Exception as e:
        print(f"\n✗ Error during OCR test: {str(e)}")
        return False

if __name__ == "__main__":
    print("="*50)
    print("PaddleOCR Setup and Test Script")
    print("="*50 + "\n")
    
    # Check installations
    if not check_installations():
        print("\n⚠ Please install missing packages before continuing.")
        sys.exit(1)
    
    # Test with an image
    print("\n" + "="*50)
    print("Ready to test OCR!")
    print("="*50)
    
    # Update this path to your image
    image_path = "C:\Game Project\GameData\Guest\Subject[Guest]_Visit[5]_Trial[2]_[S3V868T1806].png"
    
    print(f"\nImage path: {image_path}")
    print("\n⚠ Make sure to update 'image_path' variable with your actual image path!")
    
    proceed = input("\nHave you updated the image path? (Y/n): ").strip().lower()
    
    if proceed == 'y':
        test_paddleocr(image_path)
    else:
        print("\nPlease update the 'image_path' variable in the script and run again.")
        print("Example: image_path = 'C:/Users/YourName/Documents/fitness_data/image1.png'")
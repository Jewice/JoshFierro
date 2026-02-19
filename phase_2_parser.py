# Phase 2: Extract and Parse Text from Single Image
# This script processes one image and structures the data

from paddleocr import PaddleOCR
import re
import pandas as pd

def extract_text_from_image(image_path):
    """Extract all text from image using PaddleOCR with positions"""
    ocr = PaddleOCR(use_angle_cls=True, lang='en')
    result = ocr.predict(image_path)
    
    if result and len(result) > 0:
        ocr_result = result[0]
        texts = ocr_result.get('rec_texts', [])
        scores = ocr_result.get('rec_scores', [])
        polys = ocr_result.get('rec_polys', [])  # Bounding box coordinates
        return texts, scores, polys
    return [], [], []

def parse_text_to_dict(texts, polys):
    """Parse extracted text into structured dictionary by spatial matching"""
    data = {}
    
    # Create list of text items with their positions
    items = []
    for text, poly in zip(texts, polys):
        x = poly[0][0]
        y = poly[0][1]
        items.append({'text': text.strip(), 'x': x, 'y': y})
    
    # First, group adjacent digits/decimals that are on the same horizontal line
    # This reconstructs numbers like "9.8" from separate "9", ".", "8"
    grouped_values = []
    items_sorted = sorted(items, key=lambda i: (i['y'], i['x']))  # Sort by y, then x
    
    current_group = []
    last_y = None
    last_x = None
    
    for item in items_sorted:
        text = item['text']
        x, y = item['x'], item['y']
        
        # Check if this could be part of a number (digit, decimal point, or certain symbols)
        is_number_part = text.replace('.', '').replace('•', '').isdigit() or text in ['.', '•']
        
        if is_number_part:
            # Check if it's on the same line and close horizontally to previous
            if last_y is not None and abs(y - last_y) < 30 and abs(x - last_x) < 100:
                # Continue current group
                current_group.append(item)
            else:
                # Save previous group if it exists
                if current_group:
                    combined_text = ''.join([i['text'].replace('•', '.') for i in current_group])
                    avg_x = sum([i['x'] for i in current_group]) / len(current_group)
                    avg_y = sum([i['y'] for i in current_group]) / len(current_group)
                    grouped_values.append({'text': combined_text, 'x': avg_x, 'y': avg_y})
                
                # Start new group
                current_group = [item]
            
            last_y = y
            last_x = x
    
    # Don't forget the last group
    if current_group:
        combined_text = ''.join([i['text'].replace('•', '.') for i in current_group])
        avg_x = sum([i['x'] for i in current_group]) / len(current_group)
        avg_y = sum([i['y'] for i in current_group]) / len(current_group)
        grouped_values.append({'text': combined_text, 'x': avg_x, 'y': avg_y})
    
    print(f"\nDebug - Grouped values: {[v['text'] for v in grouped_values]}")
    
    # Get labels
    labels = []
    for item in items:
        text = item['text']
        if ':' in text or text in ['Distance', 'Calories']:
            labels.append(item)
    
    print(f"Debug - Labels found: {len(labels)}")
    
    # Match each label with the closest grouped value above it
    for label_item in labels:
        label_text = label_item['text']
        label_x = label_item['x']
        label_y = label_item['y']
        
        # Find values above and near this label
        candidates = []
        for val_item in grouped_values:
            val_x = val_item['x']
            val_y = val_item['y']
            
            if val_y < label_y and abs(val_x - label_x) < 150:
                distance = label_y - val_y
                candidates.append((val_item, distance))
        
        if candidates:
            candidates.sort(key=lambda x: x[1])
            closest_value = candidates[0][0]['text']
            
            label_clean = label_text.replace(':', '').replace('#', '').strip()
            
            # Convert to number
            try:
                if '.' in closest_value:
                    closest_value = float(closest_value)
                else:
                    closest_value = int(closest_value)
            except (ValueError, AttributeError):
                pass
            
            data[label_clean] = closest_value
            print(f"Matched: {label_clean} = {closest_value}")
    
    return data

def process_single_image(image_path):
    """Complete pipeline: extract text and parse to structured data"""
    print(f"Processing: {image_path}")
    print("="*60)
    
    # Step 1: Extract text
    print("\nStep 1: Extracting text with OCR...")
    texts, scores, polys = extract_text_from_image(image_path)
    
    print(f"Found {len(texts)} text items:\n")
    for idx, (text, score, poly) in enumerate(zip(texts, scores, polys)):
        x, y = poly[0][0], poly[0][1]
        print(f"{idx+1}. '{text}' at position (x:{x:.0f}, y:{y:.0f}) (confidence: {score:.3f})")
    
    # Step 2: Parse into structured data
    print("\n" + "="*60)
    print("Step 2: Parsing text into structured data...")
    data_dict = parse_text_to_dict(texts, polys)
    
    print("\nParsed data:")
    for key, value in data_dict.items():
        print(f"  {key}: {value} ({type(value).__name__})")
    
    # Step 3: Show as DataFrame (CSV preview)
    print("\n" + "="*60)
    print("Step 3: Preview as CSV format (DataFrame):\n")
    df = pd.DataFrame([data_dict])
    print(df.to_string(index=False))
    
    return data_dict, df

if __name__ == "__main__":
    # Update this path to your image
    image_path = "C:\Game Project\GameData\Guest\Subject[Guest]_Visit[5]_Trial[2]_[S3V956T2114].png"
    
    print("="*60)
    print("Phase 2: Single Image Processing and Parsing")
    print("="*60 + "\n")
    
    # Process the image
    data_dict, df = process_single_image(image_path)
    
    print("\n" + "="*60)
    print("Processing complete!")
    print("="*60)
    print("\nNext steps:")
    print("1. Verify the parsed data looks correct")
    print("2. Check if any fields are missing or misread")
    print("3. Once confirmed, we'll move to batch processing")
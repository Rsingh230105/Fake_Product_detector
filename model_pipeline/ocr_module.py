"""
OCR module for extracting text from food packaging.
Uses Tesseract OCR with proper image preprocessing.
"""
import cv2
import numpy as np
import re
import os
import pytesseract

# Configure Tesseract path
pytesseract.pytesseract.tesseract_cmd = r'C:\Program Files\Tesseract-OCR\tesseract.exe'

def preprocess_for_ocr(image_path):
    """
    Preprocess image for optimal OCR results.
    
    Args:
        image_path: Path to image file
        
    Returns:
        Preprocessed grayscale image
    """
    # Read image
    img = cv2.imread(image_path)
    if img is None:
        return None
    
    # Convert to grayscale
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    
    # Apply CLAHE (Contrast Limited Adaptive Histogram Equalization)
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
    gray = clahe.apply(gray)
    
    # Denoise
    gray = cv2.fastNlMeansDenoising(gray, h=10)
    
    # Adaptive thresholding
    binary = cv2.adaptiveThreshold(
        gray, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, 
        cv2.THRESH_BINARY, 11, 2
    )
    
    # Morphological operations to remove noise
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (2, 2))
    binary = cv2.morphologyEx(binary, cv2.MORPH_CLOSE, kernel)
    
    return binary

def extract_text(image_path):
    """
    Extract text from image using Tesseract OCR.
    
    Args:
        image_path: Path to image file
        
    Returns:
        Dict with extracted text and confidence
    """
    try:
        import pytesseract
        
        # Preprocess image
        processed = preprocess_for_ocr(image_path)
        if processed is None:
            return {'text': '', 'confidence': 0.0, 'error': 'Failed to load image'}
        
        # Extract text with confidence
        data = pytesseract.image_to_data(processed, output_type=pytesseract.Output.DICT)
        
        # Filter out low confidence text
        texts = []
        confidences = []
        for i, conf in enumerate(data['conf']):
            if conf > 30:  # Confidence threshold
                text = data['text'][i].strip()
                if text:
                    texts.append(text)
                    confidences.append(conf)
        
        # Combine text
        full_text = ' '.join(texts).lower()
        avg_confidence = np.mean(confidences) / 100.0 if confidences else 0.0
        
        return {
            'text': full_text,
            'confidence': float(avg_confidence),
            'has_text': len(full_text) > 0,
            'word_count': len(texts)
        }
        
    except ImportError:
        return {
            'text': '', 
            'confidence': 0.0, 
            'error': 'pytesseract not installed. Install: pip install pytesseract'
        }
    except Exception as e:
        return {'text': '', 'confidence': 0.0, 'error': str(e)}

def extract_fssai(text):
    """
    Extract FSSAI license number (14 digits).
    
    Args:
        text: Extracted text string
        
    Returns:
        FSSAI number or None
    """
    # Pattern: 14 consecutive digits
    pattern = r'\b\d{14}\b'
    matches = re.findall(pattern, text)
    return matches[0] if matches else None

def extract_dates(text):
    """
    Extract dates from text (expiry, manufacturing).
    
    Args:
        text: Extracted text string
        
    Returns:
        List of date strings
    """
    patterns = [
        r'\b\d{2}[/-]\d{2}[/-]\d{4}\b',  # DD/MM/YYYY or DD-MM-YYYY
        r'\b\d{4}[/-]\d{2}[/-]\d{2}\b',  # YYYY/MM/DD or YYYY-MM-DD
        r'\b\d{2}[/-]\d{2}[/-]\d{2}\b'   # DD/MM/YY or DD-MM-YY
    ]
    
    dates = []
    for pattern in patterns:
        dates.extend(re.findall(pattern, text))
    
    return list(set(dates))  # Remove duplicates

def extract_batch_number(text):
    """
    Extract batch/lot number.
    
    Args:
        text: Extracted text string
        
    Returns:
        Batch number or None
    """
    # Look for patterns like "Batch: XXX" or "Lot: XXX"
    patterns = [
        r'batch[:\s]+([a-z0-9]+)',
        r'lot[:\s]+([a-z0-9]+)',
        r'b\.?no\.?[:\s]+([a-z0-9]+)'
    ]
    
    for pattern in patterns:
        match = re.search(pattern, text, re.IGNORECASE)
        if match:
            return match.group(1)
    
    return None

def analyze_packaging_text(image_path):
    """
    Complete OCR analysis of packaging.
    
    Args:
        image_path: Path to image file
        
    Returns:
        Dict with all extracted information
    """
    # Extract raw text
    ocr_result = extract_text(image_path)
    
    if not ocr_result['has_text']:
        return {
            'success': False,
            'confidence': 0.0,
            'text': '',
            'fssai': None,
            'dates': [],
            'batch': None
        }
    
    text = ocr_result['text']
    
    # Extract specific information
    fssai = extract_fssai(text)
    dates = extract_dates(text)
    batch = extract_batch_number(text)
    
    # Calculate overall confidence
    confidence = ocr_result['confidence']
    if fssai:
        confidence += 0.2
    if dates:
        confidence += 0.1
    if batch:
        confidence += 0.1
    
    confidence = min(confidence, 1.0)
    
    return {
        'success': True,
        'confidence': float(confidence),
        'text': text,
        'fssai': fssai,
        'dates': dates,
        'batch': batch,
        'word_count': ocr_result['word_count']
    }

"""
Barcode detection and validation module.
"""
import cv2
import numpy as np

def preprocess_for_barcode(image_path):
    """
    Preprocess image for better barcode detection.
    
    Args:
        image_path: Path to image file
        
    Returns:
        Preprocessed grayscale image
    """
    img = cv2.imread(image_path)
    if img is None:
        return None
    
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    
    # Increase contrast
    clahe = cv2.createCLAHE(clipLimit=3.0, tileGridSize=(8, 8))
    gray = clahe.apply(gray)
    
    # Sharpen
    kernel = np.array([[-1,-1,-1], [-1,9,-1], [-1,-1,-1]])
    gray = cv2.filter2D(gray, -1, kernel)
    
    return gray

def detect_barcode(image_path):
    """
    Detect and decode barcode from image.
    
    Args:
        image_path: Path to image file
        
    Returns:
        Dict with barcode data and validation result
    """
    try:
        from pyzbar import pyzbar
        
        img = cv2.imread(image_path)
        if img is None:
            return {'detected': False, 'data': None, 'confidence': 0.0}
        
        # Try original image
        barcodes = pyzbar.decode(img)
        
        # If not found, try preprocessed
        if not barcodes:
            processed = preprocess_for_barcode(image_path)
            if processed is not None:
                barcodes = pyzbar.decode(processed)
        
        if barcodes:
            barcode = barcodes[0]
            data = barcode.data.decode('utf-8')
            
            # Validate format
            is_valid = validate_barcode(data)
            confidence = 0.9 if is_valid else 0.5
            
            return {
                'detected': True,
                'data': data,
                'type': barcode.type,
                'valid': is_valid,
                'confidence': confidence
            }
        
        return {'detected': False, 'data': None, 'confidence': 0.0}
            
    except ImportError:
        return {'detected': False, 'data': None, 'confidence': 0.0, 'error': 'pyzbar not installed'}
    except Exception as e:
        return {'detected': False, 'data': None, 'confidence': 0.0, 'error': str(e)}

def validate_barcode(barcode_data):
    """
    Validate barcode format and checksum.
    
    Args:
        barcode_data: Barcode string
        
    Returns:
        Boolean indicating if barcode is valid
    """
    if not barcode_data:
        return False
    
    # Basic validation: check if numeric and proper length
    if barcode_data.isdigit() and len(barcode_data) in [8, 12, 13, 14]:
        return True
    
    return False

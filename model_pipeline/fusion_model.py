"""
Fusion model combining image ML, OCR, and barcode signals.
"""
import numpy as np

def fuse_predictions(ml_score, ocr_result, barcode_result):
    """
    Combine multiple signals into final prediction.
    
    Args:
        ml_score: ML model probability (0-1)
        ocr_result: Dict from OCR module
        barcode_result: Dict from barcode module
        
    Returns:
        Dict with final prediction and confidence
    """
    # Weights for each component
    ML_WEIGHT = 0.7
    OCR_WEIGHT = 0.15
    BARCODE_WEIGHT = 0.15
    
    # ML score (already 0-1)
    ml_component = ml_score * ML_WEIGHT
    
    # OCR score
    ocr_confidence = ocr_result.get('confidence', 0.0)
    ocr_component = ocr_confidence * OCR_WEIGHT
    
    # Barcode score
    barcode_confidence = barcode_result.get('confidence', 0.0)
    barcode_component = barcode_confidence * BARCODE_WEIGHT
    
    # Final score
    final_score = ml_component + ocr_component + barcode_component
    
    # Decision threshold
    THRESHOLD = 0.5
    prediction = 'real' if final_score >= THRESHOLD else 'fake'
    
    return {
        'prediction': prediction,
        'confidence': float(final_score),
        'components': {
            'ml': float(ml_score),
            'ocr': float(ocr_confidence),
            'barcode': float(barcode_confidence)
        },
        'is_fake': prediction == 'fake',
        'is_real': prediction == 'real'
    }

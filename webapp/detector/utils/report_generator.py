"""
User-Friendly Analysis Report Generator
Converts technical ML scores into clear explanations for end users
"""

from datetime import datetime
from typing import Dict, List

class UserFriendlyReportGenerator:
    """Generate user-friendly analysis reports from technical data"""
    
    def __init__(self):
        self.status_messages = {
            'barcode': {
                'valid': 'Barcode verified successfully',
                'copied': 'Barcode copied from another product',
                'not_detected': 'Barcode not found on packaging'
            },
            'fssai': {
                'valid': 'Valid FSSAI license detected',
                'expired': 'FSSAI license expired',
                'invalid': 'Invalid FSSAI number format',
                'not_found': 'FSSAI license not found'
            },
            'expiry': {
                'valid': 'Product is within expiry date',
                'expired': 'Product expiry date has passed',
                'missing': 'Expiry date not visible on package'
            },
            'logo': {
                'match': 'Brand logo verified successfully',
                'partial': 'Logo appears similar but has minor differences',
                'mismatch': 'Logo shape or alignment does not match official brand'
            },
            'packaging': {
                'good': 'Packaging quality appears authentic',
                'average': 'Packaging quality is acceptable',
                'poor': 'Packaging quality appears suspicious'
            }
        }
    
    def generate_report(self, product_data: Dict) -> Dict:
        """
        Generate user-friendly analysis report with simplified REAL/FAKE logic
        
        Args:
            product_data: Technical analysis data from ML pipeline
            
        Returns:
            Dict containing user-friendly report sections
        """
        # Map internal status to user-facing status
        internal_status = product_data.get('final_status', 'Unknown')
        user_status = self._map_to_user_status(internal_status)
        
        # Generate simple user message
        user_message = self._generate_simple_message(user_status)
        
        report = {
            'status': user_status,
            'message': user_message,
            'user_status': user_status,
            'internal_status': internal_status,  # For admin use
            'product_summary': self._generate_product_summary(product_data),
            'final_result': self._generate_final_result(user_status),
            'safety_message': self._generate_safety_message(user_status),
            'admin_details': self._generate_admin_details(product_data)  # For admin use
        }
        
        return report
    
    def _map_to_user_status(self, internal_status: str) -> str:
        """Map internal status to user-facing status"""
        if internal_status == 'Real':
            return 'REAL'
        else:  # Both 'Fake' and 'Suspicious' map to 'FAKE'
            return 'FAKE'
    
    def _generate_simple_message(self, user_status: str) -> str:
        """Generate simple one-line message for users"""
        if user_status == 'REAL':
            return "Product verified successfully."
        else:  # FAKE
            return "Product authenticity could not be confirmed."
    
    def _generate_product_summary(self, data: Dict) -> Dict:
        """Generate product summary section"""
        return {
            'product_name': data.get('brand_name', 'Unknown Product'),
            'analysis_date': datetime.now().strftime('%B %d, %Y at %I:%M %p'),
            'images_analyzed': len(data.get('images', [])),
            'processing_time': f"{data.get('processing_time', 0):.1f} seconds"
        }
    
    def _generate_final_result(self, user_status: str) -> Dict:
        """Generate final result with simplified REAL/FAKE logic"""
        if user_status == 'REAL':
            return {
                'status': 'REAL',
                'title': 'AUTHENTIC PRODUCT',
                'icon': 'shield-check',
                'color': 'green'
            }
        else:  # FAKE
            return {
                'status': 'FAKE',
                'title': 'COUNTERFEIT DETECTED',
                'icon': 'shield-times',
                'color': 'red'
            }
    
    def _generate_safety_message(self, user_status: str) -> Dict:
        """Generate simple safety message for users"""
        if user_status == 'REAL':
            return {
                'message': 'Product authenticity and safety checks passed.',
                'disclaimer': 'This result is based on automated verification.'
            }
        else:  # FAKE
            return {
                'message': 'Product failed important safety checks such as barcode or license validation.',
                'disclaimer': 'This result is based on automated verification.'
            }
    
    def _generate_safety_checks(self, data: Dict) -> List[Dict]:
        """Generate safety and authenticity checks section"""
        checks = []
        detailed_analysis = data.get('detailed_analysis', {})
        
        # Barcode Verification
        barcode_data = detailed_analysis.get('barcode', {})
        barcode_status = barcode_data.get('status', 'not_detected')
        checks.append({
            'check_name': 'Barcode Verification',
            'status': self._get_status_icon(barcode_status, 'barcode'),
            'message': self.status_messages['barcode'].get(barcode_status, 'Unknown status'),
            'importance': 'Critical'
        })
        
        # FSSAI License Check
        fssai_data = detailed_analysis.get('fssai', {})
        if fssai_data.get('extracted'):
            fssai_status = 'valid' if fssai_data.get('valid_format') else 'invalid'
        else:
            fssai_status = 'not_found'
        
        fssai_message = self.status_messages['fssai'].get(fssai_status)
        if fssai_data.get('number'):
            fssai_message += f" (License: {fssai_data['number']})"
        
        checks.append({
            'check_name': 'FSSAI License Check',
            'status': self._get_status_icon(fssai_status, 'fssai'),
            'message': fssai_message,
            'importance': 'Critical'
        })
        
        # Expiry Date
        expiry_data = detailed_analysis.get('expiry_date', {})
        if expiry_data.get('detected'):
            if expiry_data.get('is_expired'):
                expiry_status = 'expired'
            else:
                expiry_status = 'valid'
        else:
            expiry_status = 'missing'
        
        expiry_message = self.status_messages['expiry'].get(expiry_status)
        if expiry_data.get('date_value'):
            expiry_message += f" (Expires: {expiry_data['date_value']})"
        
        checks.append({
            'check_name': 'Expiry Date',
            'status': self._get_status_icon(expiry_status, 'expiry'),
            'message': expiry_message,
            'importance': 'Important'
        })
        
        # Brand Logo Verification
        logo_data = detailed_analysis.get('logo', {})
        logo_status = logo_data.get('status', 'not_detected')
        if logo_status == 'not_detected':
            logo_status = 'mismatch'
        
        checks.append({
            'check_name': 'Brand Logo Verification',
            'status': self._get_status_icon(logo_status, 'logo'),
            'message': self.status_messages['logo'].get(logo_status, 'Logo verification failed'),
            'importance': 'Important'
        })
        
        # Packaging Quality
        packaging_data = detailed_analysis.get('packaging', {})
        packaging_status = packaging_data.get('status', 'poor')
        if packaging_status == 'match':
            packaging_status = 'good'
        elif packaging_status == 'partial':
            packaging_status = 'average'
        else:
            packaging_status = 'poor'
        
        checks.append({
            'check_name': 'Packaging Quality',
            'status': self._get_status_icon(packaging_status, 'packaging'),
            'message': self.status_messages['packaging'].get(packaging_status),
            'importance': 'Moderate'
        })
        
        return checks
    
    def _generate_fake_reasons(self, data: Dict) -> Dict:
        """Generate explanation for fake products"""
        if data.get('final_status') != 'Fake':
            return None
        
        failure_reasons = data.get('failure_reasons', [])
        
        return {
            'title': 'Why this product is FAKE:',
            'reasons': failure_reasons,
            'explanation': 'Even though packaging design may look similar, critical safety verification failed.',
            'advice': 'Do not consume this product. Report to authorities if purchased from a store.'
        }
    
    def _generate_recommendation(self, data: Dict) -> Dict:
        """Generate user recommendation based on analysis"""
        final_status = data.get('final_status', 'Unknown')
        
        if final_status == 'Real':
            return {
                'action': 'SAFE TO USE',
                'message': 'This product passed our authenticity checks. You can use it with confidence.',
                'color': 'green'
            }
        elif final_status == 'Fake':
            return {
                'action': 'DO NOT USE',
                'message': 'This product failed critical safety checks. Do not consume and report if purchased.',
                'color': 'red'
            }
        else:  # Suspicious
            return {
                'action': 'VERIFY MANUALLY',
                'message': 'Some checks failed. Please verify with official brand sources before use.',
                'color': 'yellow'
            }
    
    def _generate_admin_details(self, data: Dict) -> Dict:
        """Generate detailed admin information"""
        detailed_analysis = data.get('detailed_analysis', {})
        component_scores = data.get('component_scores', {})
        
        return {
            'internal_status': data.get('final_status', 'Unknown'),
            'final_score': data.get('final_score', 0),
            'barcode_status': self._get_admin_status(detailed_analysis.get('barcode', {})),
            'fssai_status': self._get_admin_status(detailed_analysis.get('fssai', {})),
            'expiry_status': self._get_admin_status(detailed_analysis.get('expiry_date', {})),
            'logo_score': component_scores.get('logo_score', 0),
            'packaging_score': component_scores.get('packaging_score', 0),
            'barcode_score': component_scores.get('barcode_score', 0),
            'ocr_score': component_scores.get('ocr_score', 0),
            'failure_reasons': data.get('failure_reasons', []),
            'processing_time': data.get('processing_time', 0)
        }
    
    def _get_admin_status(self, component_data: Dict) -> str:
        """Get admin-friendly status for component"""
        status = component_data.get('status', 'unknown')
        if status == 'verified' or status == 'valid' or status == 'match':
            return 'Valid'
        elif status == 'copied':
            return 'Copied'
        elif status == 'not_detected' or status == 'missing':
            return 'Not Detected'
        elif status == 'expired':
            return 'Expired'
        elif status == 'invalid':
            return 'Invalid'
        else:
            return 'Unknown'

# Usage function for Django integration
def generate_user_friendly_report(analysis_data: Dict) -> Dict:
    """
    Main function to generate user-friendly report
    
    Args:
        analysis_data: Raw analysis data from ML pipeline
        
    Returns:
        Dict containing user-friendly report
    """
    generator = UserFriendlyReportGenerator()
    return generator.generate_report(analysis_data)
// src/api/detectApi.js
// Handles all communication with POST /api/detect/

const BASE_URL = process.env.REACT_APP_API_URL || 'https://fake-product-detector-6.onrender.com';

/**
 * Submit product images for fake/real detection.
 *
 * Backend expects:
 *   brand_name      → string
 *   images[]        → File (one per view)
 *   view_types[]    → string matching each image ('front'|'back'|'side'|'barcode')
 *
 * Backend returns (regular user):
 *   { id, brand_name, status: 'REAL' | 'FAKE' }
 *
 * @param {string} brandName
 * @param {{ front?: File, back?: File, side?: File, barcode?: File }} imageMap
 */
export async function detectProduct(brandName, imageMap) {
  const formData = new FormData();
  formData.append('brand_name', brandName);

  // Append images and view_types in the same order — backend zips them together
  Object.entries(imageMap).forEach(([viewType, file]) => {
    if (file) {
      formData.append('images[]', file);
      formData.append('view_types[]', viewType);
    }
  });

  const response = await fetch(`${BASE_URL}/api/detect/`, {
    method: 'POST',
    // Do NOT manually set Content-Type — browser adds the multipart boundary automatically
    credentials: 'include', // sends session cookie for authenticated users
    body: formData,
  });

  if (!response.ok) {
    const err = await response.json().catch(() => ({}));
    throw new Error(err.error || `Server error: ${response.status}`);
  }

  return response.json(); // { id, brand_name, status }
}

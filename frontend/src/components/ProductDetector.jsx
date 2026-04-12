// src/components/ProductDetector.jsx
import { useState } from 'react';
import { detectProduct } from '../api/detectApi';

const VIEW_TYPES = ['front', 'back', 'side', 'barcode'];

const INITIAL_IMAGES = { front: null, back: null, side: null, barcode: null };

export default function ProductDetector() {
  const [brandName, setBrandName]   = useState('');
  const [images, setImages]         = useState(INITIAL_IMAGES);
  const [loading, setLoading]       = useState(false);
  const [result, setResult]         = useState(null);   // { id, brand_name, status }
  const [error, setError]           = useState('');

  // ── handlers ──────────────────────────────────────────────────────────────

  function handleImageChange(viewType, file) {
    setImages(prev => ({ ...prev, [viewType]: file }));
  }

  async function handleSubmit(e) {
    e.preventDefault();
    setError('');
    setResult(null);

    // Validation
    if (!brandName.trim()) {
      setError('Please enter a brand name.');
      return;
    }
    const selectedImages = Object.values(images).filter(Boolean);
    if (selectedImages.length === 0) {
      setError('Please upload at least one image.');
      return;
    }

    setLoading(true);
    try {
      const data = await detectProduct(brandName.trim(), images);
      setResult(data);
    } catch (err) {
      setError(err.message || 'Something went wrong. Please try again.');
    } finally {
      setLoading(false);
    }
  }

  function handleReset() {
    setBrandName('');
    setImages(INITIAL_IMAGES);
    setResult(null);
    setError('');
  }

  // ── render ─────────────────────────────────────────────────────────────────

  return (
    <div style={styles.container}>
      <h2 style={styles.title}>🔍 Fake Product Detector</h2>

      {/* ── Upload Form ── */}
      {!result && (
        <form onSubmit={handleSubmit} style={styles.form}>

          {/* Brand Name */}
          <div style={styles.field}>
            <label style={styles.label}>Brand Name *</label>
            <input
              type="text"
              value={brandName}
              onChange={e => setBrandName(e.target.value)}
              placeholder="e.g. Maggi, Bourbon, Daawat"
              style={styles.input}
              disabled={loading}
            />
          </div>

          {/* Image Uploads */}
          <div style={styles.imageGrid}>
            {VIEW_TYPES.map(viewType => (
              <div key={viewType} style={styles.imageField}>
                <label style={styles.label}>
                  {viewType.charAt(0).toUpperCase() + viewType.slice(1)} View
                </label>
                <input
                  type="file"
                  accept="image/jpeg,image/png,image/webp"
                  onChange={e => handleImageChange(viewType, e.target.files[0] || null)}
                  style={styles.fileInput}
                  disabled={loading}
                />
                {images[viewType] && (
                  <img
                    src={URL.createObjectURL(images[viewType])}
                    alt={`${viewType} preview`}
                    style={styles.preview}
                  />
                )}
              </div>
            ))}
          </div>

          {/* Error */}
          {error && <p style={styles.error}>{error}</p>}

          {/* Submit */}
          <button type="submit" style={styles.button} disabled={loading}>
            {loading ? '⏳ Analysing...' : '🚀 Detect Product'}
          </button>
        </form>
      )}

      {/* ── Loading ── */}
      {loading && (
        <div style={styles.loadingBox}>
          <p style={styles.loadingText}>⏳ Analysing images, please wait...</p>
          <p style={styles.loadingSubtext}>This may take 30–60 seconds on first run.</p>
        </div>
      )}

      {/* ── Result ── */}
      {result && !loading && (
        <div style={{
          ...styles.resultBox,
          borderColor: result.status === 'REAL' ? '#22c55e' : '#ef4444',
          background: result.status === 'REAL' ? '#f0fdf4' : '#fef2f2',
        }}>
          <h3 style={styles.resultTitle}>
            {result.status === 'REAL' ? '✅ REAL Product' : '❌ FAKE Product'}
          </h3>
          <p style={styles.resultBrand}>Brand: <strong>{result.brand_name}</strong></p>
          <p style={styles.resultId}>Analysis ID: #{result.id}</p>
          <p style={{
            ...styles.resultStatus,
            color: result.status === 'REAL' ? '#16a34a' : '#dc2626',
          }}>
            {result.status === 'REAL'
              ? 'This product appears to be genuine.'
              : 'This product appears to be counterfeit. Avoid purchasing.'}
          </p>
          <button onClick={handleReset} style={styles.resetButton}>
            🔄 Analyse Another Product
          </button>
        </div>
      )}
    </div>
  );
}

// ── Inline styles (replace with your CSS/Tailwind as needed) ──────────────────

const styles = {
  container:      { maxWidth: 640, margin: '40px auto', padding: '0 16px', fontFamily: 'sans-serif' },
  title:          { textAlign: 'center', marginBottom: 24, fontSize: 24 },
  form:           { display: 'flex', flexDirection: 'column', gap: 20 },
  field:          { display: 'flex', flexDirection: 'column', gap: 6 },
  label:          { fontWeight: 600, fontSize: 14, textTransform: 'capitalize' },
  input:          { padding: '10px 12px', border: '1px solid #d1d5db', borderRadius: 8, fontSize: 15 },
  imageGrid:      { display: 'grid', gridTemplateColumns: '1fr 1fr', gap: 16 },
  imageField:     { display: 'flex', flexDirection: 'column', gap: 6 },
  fileInput:      { fontSize: 13 },
  preview:        { width: '100%', height: 120, objectFit: 'cover', borderRadius: 8, marginTop: 4, border: '1px solid #e5e7eb' },
  error:          { color: '#dc2626', fontSize: 14, margin: 0 },
  button:         { padding: '12px 0', background: '#2563eb', color: '#fff', border: 'none', borderRadius: 8, fontSize: 16, fontWeight: 600, cursor: 'pointer' },
  loadingBox:     { textAlign: 'center', padding: 32 },
  loadingText:    { fontSize: 18, margin: 0 },
  loadingSubtext: { fontSize: 13, color: '#6b7280', marginTop: 8 },
  resultBox:      { padding: 28, borderRadius: 12, border: '2px solid', textAlign: 'center' },
  resultTitle:    { fontSize: 26, margin: '0 0 12px' },
  resultBrand:    { fontSize: 16, margin: '4px 0' },
  resultId:       { fontSize: 13, color: '#6b7280', margin: '4px 0 16px' },
  resultStatus:   { fontSize: 15, fontWeight: 500, margin: '0 0 24px' },
  resetButton:    { padding: '10px 24px', background: '#374151', color: '#fff', border: 'none', borderRadius: 8, fontSize: 14, cursor: 'pointer' },
};

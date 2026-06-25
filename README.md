# Product Authenticity Detection System

AI-powered web application for detecting counterfeit food products using deep learning, OCR, and barcode verification.

## Overview

This system analyzes food product images to determine authenticity by examining multiple components:
- **Barcode verification** against official databases
- **Logo detection** and similarity matching
- **OCR text extraction** for FSSAI license, expiry dates, and batch numbers
- **Packaging analysis** for color and texture matching
- **ML classification** using MobileNetV2 architecture

## System Architecture

```
┌─────────────────┐
│  Image Upload   │
└────────┬────────┘
         │
    ┌────▼────┐
    │ Django  │
    │  API    │
    └────┬────┘
         │
    ┌────▼──────────────────────┐
    │  Multi-Component Analysis │
    ├───────────────────────────┤
    │ • MobileNetV2 CNN Model   │
    │ • Tesseract OCR Engine    │
    │ • OpenCV Image Processing │
    │ • Barcode Detection       │
    └────┬──────────────────────┘
         │
    ┌────▼────────┐
    │ Final Score │
    │ REAL / FAKE │
    └─────────────┘
```

## Tech Stack

**Backend:**
- Django 4.2.0
- Django REST Framework 3.14.0
- TensorFlow 2.20+
- Python 3.13

**ML/CV:**
- TensorFlow/Keras (MobileNetV2)
- OpenCV 4.13+
- Tesseract OCR
- NumPy 2.0+
- Pillow 12.0+

**Security:**
- python-magic (file validation)
- Django authentication system
- CSRF protection

## Setup Instructions

### 1. Clone Repository
```bash
git clone <repository-url>
cd Project_Major_food
```

### 2. Create Virtual Environment
```bash
python -m venv .venv
.venv\Scripts\activate  # Windows
source .venv/bin/activate  # Linux/Mac
```

### 3. Install Dependencies
```bash
pip install -r requirements.txt
```

### 4. Install Tesseract OCR
**Windows:**
```bash
# Download from: https://github.com/UB-Mannheim/tesseract/wiki
# Install to: C:\Program Files\Tesseract-OCR\
```

**Linux:**
```bash
sudo apt-get install tesseract-ocr
```

### 5. Configure Environment
```bash
# Create .env file
cp .env.example .env

# Edit .env with your settings:
SECRET_KEY=your-secret-key
DEBUG=False
ALLOWED_HOSTS=localhost,127.0.0.1
```

### 6. Run Migrations
```bash
cd webapp
python manage.py makemigrations
python manage.py migrate
```

### 7. Create Superuser
```bash
python manage.py createsuperuser
```

### 8. Train Model (Optional)
```bash
# Place your dataset in: data/train/ and data/test/
cd model_pipeline
python train_model.py
# Model saved to: models/mobilenet_v2_food_production.keras
```

### 9. Run Development Server
```bash
cd webapp
python manage.py runserver
```

Access at: `http://localhost:8000`

## Project Structure

```
Project_Major_food/
├── webapp/
│   ├── detector/           # Main Django app
│   │   ├── models.py       # Database models
│   │   ├── views.py        # API endpoints
│   │   ├── validators.py   # File upload security
│   │   └── utils/
│   │       ├── ml_utils.py # ML inference
│   │       └── report_generator.py
│   ├── templates/          # HTML templates
│   └── static/             # CSS/JS assets
├── models/                 # Trained ML models (not in repo)
├── data/                   # Training datasets (not in repo)
└── requirements.txt
```

## Security Features

✅ **File Upload Validation:**
- Magic byte verification
- 5MB size limit
- MIME type whitelist (jpeg, png, webp)
- Decompression bomb protection
- Image resolution limits

✅ **Authentication:**
- User registration/login system
- Password reset functionality
- Session management
- CSRF protection

✅ **Data Protection:**
- Environment variable configuration
- Secure credential storage
- SQL injection prevention (Django ORM)

## Model Training Notes

**Dataset Requirements:**
- Minimum 1000 images per class (Real/Fake)
- Multiple views: front, back, side, barcode
- High resolution (min 224x224)
- Balanced class distribution

**Training Configuration:**
- Architecture: MobileNetV2 (transfer learning)
- Input size: 224x224x3
- Optimizer: Adam
- Loss: Binary crossentropy
- Metrics: Accuracy, Precision, Recall

**Model not included in repository due to size. Train locally using provided pipeline.**

## API Endpoints

```
POST /api/detect/
- Upload images for analysis
- Returns: prediction, confidence, detailed report

GET /dashboard/
- User analysis history

GET /admin-report/<id>/
- Detailed admin analysis report
```

## Future Improvements

- [ ] Add real-time barcode database integration
- [ ] Implement blockchain for product verification
- [ ] Add mobile app support
- [ ] Integrate with government FSSAI database
- [ ] Add multi-language OCR support
- [ ] Implement batch processing API
- [ ] Add explainable AI visualizations
- [ ] Deploy to cloud (AWS/Azure)

## License

This project is for educational and research purposes.

## Contributors

Developed as part of Major Project - Product Authentication System.

---

**Note:** This system is a prototype. For production use, integrate with official databases and obtain necessary certifications.

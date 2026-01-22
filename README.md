# Fake Food Detection System 🛡️

[![Python](https://img.shields.io/badge/Python-3.8+-blue.svg)](https://www.python.org/)
[![Django](https://img.shields.io/badge/Django-5.2.3-green.svg)](https://www.djangoproject.com/)
[![TensorFlow](https://img.shields.io/badge/TensorFlow-2.20.0-orange.svg)](https://www.tensorflow.org/)
[![License](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

An AI-powered web application that detects counterfeit FMCG (Fast-Moving Consumer Goods) products using computer vision, OCR technology, and machine learning algorithms.

## 📋 Table of Contents
- [Problem Statement](#-problem-statement)
- [Proposed Solution](#-proposed-solution)
- [Key Features](#-key-features)
- [Technology Stack](#-technology-stack)
- [Project Architecture](#-project-architecture)
- [How It Works](#-how-it-works)
- [Installation & Setup](#-installation--setup)
- [Future Scope](#-future-scope)
- [Disclaimer](#-disclaimer)

## 🚨 Problem Statement

Counterfeit products pose a significant threat to consumer safety, brand reputation, and economic stability. According to industry reports, counterfeit goods account for approximately 3.3% of world trade, costing legitimate businesses over $500 billion annually. In the FMCG sector, fake products can contain harmful substances, incorrect dosages, or poor quality ingredients that endanger public health.

**Key Challenges:**
- Difficulty in distinguishing genuine from counterfeit products
- Lack of accessible verification tools for consumers
- Time-consuming manual inspection processes
- Limited technological solutions for small retailers

## 💡 Proposed Solution

The Fake Food Detection System leverages cutting-edge AI technologies to provide an accessible, user-friendly platform for product authenticity verification. By combining computer vision, optical character recognition (OCR), and machine learning, the system analyzes multiple aspects of product packaging to determine authenticity.

## ✨ Key Features

- **🔍 Multi-View Analysis**: Supports front, back, side, and barcode image analysis
- **📝 OCR Text Extraction**: Automatically extracts and analyzes text from product packaging
- **🤖 AI-Powered Classification**: Machine learning model trained to distinguish genuine from counterfeit products
- **🔍 Brand Verification**: Cross-references extracted text with known brand databases
- **👤 User Authentication**: Secure user registration and login system
- **📱 Responsive Web Interface**: Works seamlessly across desktop and mobile devices
- **🖼️ Gallery System**: Users can view and manage their analysis history
- **📊 Admin Dashboard**: Comprehensive admin panel for system management
- **🔒 Secure Architecture**: Implements industry-standard security practices

## 🛠️ Technology Stack

### Backend
- **Framework**: Django 5.2.3
- **API**: Django REST Framework 3.14.0
- **Authentication**: Django Allauth 0.63.3
- **Database**: SQLite (development) / PostgreSQL (production)

### Frontend
- **Templates**: Django Templates with Bootstrap
- **Styling**: Custom CSS with responsive design
- **JavaScript**: Vanilla JavaScript with jQuery

### AI/ML Components
- **Deep Learning**: TensorFlow 2.20.0
- **Computer Vision**: OpenCV 4.12.0
- **OCR Engine**: Tesseract 0.3.13
- **Text Matching**: FuzzyWuzzy 0.18.0
- **Model Architecture**: MobileNetV2 (fine-tuned for product classification)

### Additional Libraries
- **Image Processing**: Pillow 11.3.0
- **Phone Numbers**: django-phonenumber-field 7.3.0
- **Environment Management**: python-dotenv 1.0.0
- **Static Files**: WhiteNoise 6.9.0

## 🏗️ Project Architecture

```
Fake Food Detection System/
├── webapp/                          # Django Project Root
│   ├── food_detection/              # Main Django Project Settings
│   ├── detector/                    # Core Application
│   │   ├── models.py               # Database Models
│   │   ├── views.py                # View Controllers
│   │   ├── templates/              # HTML Templates
│   │   ├── static/                 # CSS, JS, Images
│   │   └── utils/
│   │       └── ml_utils.py         # ML Prediction Logic
│   ├── media/                      # User Uploads (excluded)
│   └── staticfiles/                # Collected Static Files (excluded)
├── src/                            # ML Training Code
│   ├── model_training/             # Training Scripts
│   └── data_preprocessing/         # Data Processing Utilities
├── models/                         # ML Model Files (excluded)
├── data/                           # Training Data (excluded)
├── requirements.txt                # Python Dependencies
├── manage.py                       # Django Management Script
└── README.md                       # Project Documentation
```

## 🔄 How It Works

### Detection Flow
1. **📤 Image Upload**: User uploads multiple images of the product (front, back, side, barcode)
2. **🔍 Preprocessing**: Images are resized, enhanced, and prepared for analysis
3. **📝 OCR Analysis**: Tesseract extracts text from packaging labels and barcodes
4. **🤖 ML Classification**: MobileNetV2 model analyzes visual features to predict authenticity
5. **🔍 Brand Verification**: Extracted text is compared against known brand databases
6. **📊 Result Generation**: System combines all analyses to provide confidence score
7. **💾 Result Storage**: Analysis results are saved to user profile for future reference

### AI Model Details
- **Architecture**: MobileNetV2 with custom classification head
- **Input**: 224x224 RGB images
- **Output**: Binary classification (FAKE/REAL) with confidence score
- **Training Data**: Balanced dataset of genuine and counterfeit product images
- **Accuracy**: >85% on validation set (varies by product category)

## 🚀 Installation & Setup

### Prerequisites
- Python 3.8 or higher
- pip package manager
- Git

### Local Development Setup

1. **Clone the Repository**
   ```bash
   git clone https://github.com/your-username/fake-food-detection.git
   cd fake-food-detection
   ```

2. **Create Virtual Environment**
   ```bash
   python -m venv venv
   # On Windows
   venv\Scripts\activate
   # On macOS/Linux
   source venv/bin/activate
   ```

3. **Install Dependencies**
   ```bash
   pip install -r requirements.txt
   ```

4. **Environment Configuration**
   ```bash
   # Create .env file in webapp/ directory
   cd webapp
   cp .env.example .env  # If provided, otherwise create manually
   ```

   Configure the following in `.env`:
   ```env
   SECRET_KEY=your-secret-key-here
   DEBUG=True
   DATABASE_URL=sqlite:///db.sqlite3
   # Add other required environment variables
   ```

5. **Database Setup**
   ```bash
   python manage.py migrate
   python manage.py createsuperuser
   ```

6. **Collect Static Files**
   ```bash
   python manage.py collectstatic --noinput
   ```

7. **Run Development Server**
   ```bash
   python manage.py runserver
   ```

8. **Access the Application**
   - Open browser and navigate to: `http://127.0.0.1:8000`
   - Admin panel: `http://127.0.0.1:8000/admin`

### Additional Setup for ML Features

**Tesseract OCR Installation:**
- **Windows**: Download from [GitHub releases](https://github.com/UB-Mannheim/tesseract/wiki)
- **macOS**: `brew install tesseract`
- **Linux**: `sudo apt-get install tesseract-ocr`

**ML Model Setup:**
- Place trained model file (`mobilenet_v2_food.h5`) in `models/` directory
- Ensure model file is not committed to version control

## 🔮 Future Scope

### Short-term Enhancements (3-6 months)
- **📱 Mobile Application**: Native iOS/Android apps
- **🌐 Multi-language Support**: OCR support for multiple languages
- **📊 Advanced Analytics**: Detailed reporting and insights dashboard
- **🔗 API Integration**: RESTful API for third-party integrations

### Medium-term Goals (6-12 months)
- **🧠 Improved AI Models**: Larger datasets and advanced architectures (ResNet, EfficientNet)
- **📷 Real-time Detection**: Camera integration for instant verification
- **🏪 Retail Integration**: POS system integration for automated scanning
- **🌍 Global Expansion**: Support for international brands and regulations

### Long-term Vision (1-2 years)
- **🔬 Laboratory Verification**: Integration with certified testing labs
- **🏛️ Regulatory Compliance**: Compliance with FDA, EU standards
- **🤝 Industry Partnerships**: Collaboration with brand manufacturers
- **📈 Enterprise Solutions**: B2B platform for large retailers

## ⚠️ Disclaimer

**Academic Project Notice**

This project was developed as part of an academic exercise and serves as a demonstration of AI/ML concepts in product authenticity verification. While the system incorporates industry-standard practices, it should not be used as the sole method for product verification in commercial or critical applications.

**Important Limitations:**
- The AI model is trained on limited datasets and may not cover all product categories
- OCR accuracy depends on image quality and text clarity
- Results should be verified through official channels when possible
- Not intended for legal or regulatory compliance purposes

**Recommendations:**
- Use as a supplementary tool alongside traditional verification methods
- Consult with domain experts for critical applications
- Regular model updates and validation are recommended for production use

---

**Developed with ❤️ for Consumer Safety and Brand Protection**

For questions or contributions, please open an issue or submit a pull request.

- Django
- TensorFlow
- OpenCV
- Pytesseract OCR
- TailwindCSS

## Contributors
- [Rajendra Singh]

## License
[Your chosen license]

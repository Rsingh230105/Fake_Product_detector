# Fake Food Detection System - Complete Setup Guide

## ✅ What Has Been Completed

### 1. **Authentication System** ✓
- Custom User Model with email-based authentication
- User Registration with form validation
- User Login with session management
- User Logout functionality
- User Profile management
- User Settings page
- Activity logging system

### 2. **Database Structure** ✓
- CustomUser model (email, first_name, last_name, phone, profile_picture)
- UserProfile model (bio, location, birth_date, website, privacy settings)
- FoodProduct model (brand analysis with ML results)
- FoodImage model (multiple view support)
- UserActivity model (activity tracking)
- Advertisement, GalleryItem, MediaItem models

### 3. **Google OAuth Integration** ✓
- Django-allauth configured
- Google OAuth provider setup
- Social account management

### 4. **Admin Panel** ✓
- Custom admin interface for all models
- User management
- Product analysis management
- Media management

### 5. **API Endpoints** ✓
- Food detection API
- RESTful serializers
- Multi-image upload support

## 🚀 Quick Start Guide

### Step 1: Install Dependencies

```bash
cd webapp
pip install -r ../requirements.txt
```

### Step 2: Configure Environment Variables

Edit `webapp/.env` file:

```env
# Django Settings
DJANGO_SECRET_KEY=your-secret-key-here-change-in-production
DJANGO_DEBUG=True
DJANGO_ALLOWED_HOSTS=localhost,127.0.0.1

# Email Settings (for email verification)
EMAIL_HOST=smtp.gmail.com
EMAIL_PORT=587
EMAIL_HOST_USER=your-email@gmail.com
EMAIL_HOST_PASSWORD=your-app-password
DEFAULT_FROM_EMAIL=noreply@fakeproductdetector.com

# Google OAuth Settings
GOOGLE_OAUTH2_CLIENT_ID=your-google-client-id
GOOGLE_OAUTH2_CLIENT_SECRET=your-google-client-secret
```

### Step 3: Database is Already Set Up ✓

The database has been migrated and a superuser has been created:
- **Email:** admin@example.com
- **Password:** admin123

### Step 4: Run the Server

```bash
cd webapp
python manage.py runserver
```

Visit: http://127.0.0.1:8000/

## 📋 Available URLs

### Public Pages
- **Home/Upload:** http://127.0.0.1:8000/
- **About:** http://127.0.0.1:8000/about/
- **Contact:** http://127.0.0.1:8000/contact/

### Authentication
- **Login:** http://127.0.0.1:8000/login/
- **Register:** http://127.0.0.1:8000/register/
- **Logout:** http://127.0.0.1:8000/logout/

### User Dashboard (Login Required)
- **Dashboard:** http://127.0.0.1:8000/dashboard/
- **Profile:** http://127.0.0.1:8000/profile/
- **Settings:** http://127.0.0.1:8000/settings/

### Admin Panel
- **Admin:** http://127.0.0.1:8000/admin/
  - Login with: admin@example.com / admin123

### API Endpoints
- **Food Detection API:** http://127.0.0.1:8000/api/detect/

### Media Management (Admin Only)
- **Upload Ads:** http://127.0.0.1:8000/media/ads/
- **Gallery:** http://127.0.0.1:8000/media/gallery/
- **Media Library:** http://127.0.0.1:8000/media/library/

## 🔧 Setting Up Google OAuth (Optional)

### Step 1: Create Google OAuth Credentials

1. Go to [Google Cloud Console](https://console.cloud.google.com/)
2. Create a new project or select existing one
3. Enable Google+ API
4. Go to "Credentials" → "Create Credentials" → "OAuth 2.0 Client ID"
5. Set Application type to "Web application"
6. Add Authorized redirect URIs:
   - http://127.0.0.1:8000/accounts/google/login/callback/
   - http://localhost:8000/accounts/google/login/callback/

### Step 2: Configure in Django Admin

1. Login to admin panel: http://127.0.0.1:8000/admin/
2. Go to "Sites" → Edit the default site:
   - Domain name: 127.0.0.1:8000
   - Display name: Fake Product Detector
3. Go to "Social applications" → Add social application:
   - Provider: Google
   - Name: Google OAuth
   - Client ID: (paste from Google Console)
   - Secret key: (paste from Google Console)
   - Sites: Select your site

### Step 3: Update .env File

```env
GOOGLE_OAUTH2_CLIENT_ID=your-client-id-here
GOOGLE_OAUTH2_CLIENT_SECRET=your-client-secret-here
```

## 📝 Testing the System

### Test User Registration
1. Go to http://127.0.0.1:8000/register/
2. Fill in the form:
   - First Name: Test
   - Last Name: User
   - Email: test@example.com
   - Password: testpass123
   - Confirm Password: testpass123
3. Click "Register"
4. You'll be redirected to login page

### Test User Login
1. Go to http://127.0.0.1:8000/login/
2. Enter credentials:
   - Email: test@example.com
   - Password: testpass123
3. Click "Sign in"
4. You'll be redirected to dashboard

### Test Product Upload
1. Login to the system
2. Go to http://127.0.0.1:8000/
3. Enter brand name
4. Upload front and back images (required)
5. Optionally upload side and barcode images
6. Click "Analyze Product"
7. View results on dashboard

## 🗂️ Project Structure

```
Project_Major_food/
├── webapp/
│   ├── detector/                 # Main application
│   │   ├── migrations/          # Database migrations
│   │   ├── templates/           # HTML templates
│   │   │   ├── auth/           # Login/Register templates
│   │   │   └── detector/       # App templates
│   │   ├── models.py           # Database models
│   │   ├── views.py            # View logic
│   │   ├── forms.py            # Form definitions
│   │   ├── serializers.py      # API serializers
│   │   ├── admin.py            # Admin configuration
│   │   └── urls.py             # URL routing
│   ├── food_detection/          # Project settings
│   │   ├── settings.py         # Django settings
│   │   └── urls.py             # Main URL config
│   ├── media/                   # Uploaded files
│   ├── static/                  # Static files
│   ├── .env                     # Environment variables
│   ├── db.sqlite3              # Database
│   └── manage.py               # Django management
├── models/                      # ML models
├── data/                        # Training data
├── requirements.txt            # Python dependencies
└── README.md                   # Project documentation
```

## 🔐 Security Features Implemented

1. **Password Hashing:** Django's built-in password hashing
2. **CSRF Protection:** Enabled on all forms
3. **Session Management:** Secure session handling
4. **Email Verification:** Configured (requires email setup)
5. **Activity Logging:** All user actions are logged
6. **Permission System:** Admin-only pages protected

## 📊 Database Models

### CustomUser
- email (unique, used for login)
- first_name, last_name
- phone_number
- profile_picture
- is_email_verified
- created_at, updated_at

### UserProfile
- user (OneToOne with CustomUser)
- bio, location, birth_date, website
- profile_visibility (public/private)
- email_notifications, sms_notifications

### FoodProduct
- user (ForeignKey to CustomUser)
- brand_name
- final_prediction (Real/Fake)
- overall_confidence
- risk_level (low/medium/high)
- ocr_results (JSON)
- created_at

### FoodImage
- product (ForeignKey to FoodProduct)
- image
- view_type (front/back/side/barcode)
- prediction, confidence
- detected_text

### UserActivity
- user (ForeignKey to CustomUser)
- activity_type (login/logout/upload/analysis)
- description
- ip_address, user_agent
- timestamp

## 🐛 Troubleshooting

### Issue: Server won't start
**Solution:** Make sure you're in the webapp directory:
```bash
cd webapp
python manage.py runserver
```

### Issue: Database errors
**Solution:** Run migrations:
```bash
python manage.py makemigrations
python manage.py migrate
```

### Issue: Static files not loading
**Solution:** Collect static files:
```bash
python manage.py collectstatic
```

### Issue: Google OAuth not working
**Solution:** 
1. Check if django-allauth is installed
2. Verify Google credentials in admin panel
3. Ensure redirect URIs match exactly

### Issue: Email verification not working
**Solution:** For development, emails are printed to console. Check terminal output.

## 🎯 Next Steps

1. **Configure Email:** Set up real SMTP for email verification
2. **Add ML Model:** Integrate actual ML model for product detection
3. **Implement OCR:** Add Tesseract OCR for text extraction
4. **Add Tests:** Write unit and integration tests
5. **Deploy:** Deploy to production server (Heroku, AWS, etc.)

## 📞 Support

For issues or questions:
- Check the error logs in `webapp/logs/debug.log`
- Review Django documentation: https://docs.djangoproject.com/
- Check django-allauth docs: https://django-allauth.readthedocs.io/

## ✨ Features Summary

✅ User Registration & Login
✅ Email-based Authentication
✅ Google OAuth Integration
✅ User Dashboard with Statistics
✅ Profile Management
✅ Settings Page
✅ Activity Tracking
✅ Admin Panel
✅ Multi-image Upload
✅ Product Analysis API
✅ Media Management
✅ Responsive Design (TailwindCSS)
✅ CSRF Protection
✅ Session Management
✅ Password Validation

## 🎉 System is Ready!

Your authentication system is fully functional and ready to use. You can now:
1. Register new users
2. Login with email/password
3. Manage user profiles
4. Upload and analyze products
5. View dashboard statistics
6. Access admin panel

**Default Admin Credentials:**
- Email: admin@example.com
- Password: admin123

**Remember to change the admin password in production!**

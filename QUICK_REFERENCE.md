# Quick Reference Guide

## 🚀 Common Commands

### Start Server
```bash
cd webapp
python manage.py runserver
```

### Create Superuser
```bash
cd webapp
python manage.py createsuperuser
```

### Make Migrations
```bash
cd webapp
python manage.py makemigrations
python manage.py migrate
```

### Collect Static Files
```bash
cd webapp
python manage.py collectstatic
```

### Run Tests
```bash
cd webapp
python manage.py test
```

### Django Shell
```bash
cd webapp
python manage.py shell
```

## 🔑 Default Credentials

### Admin Account
- **URL:** http://127.0.0.1:8000/admin/
- **Email:** admin@example.com
- **Password:** admin123

## 📍 Important URLs

| Page | URL |
|------|-----|
| Home | http://127.0.0.1:8000/ |
| Login | http://127.0.0.1:8000/login/ |
| Register | http://127.0.0.1:8000/register/ |
| Dashboard | http://127.0.0.1:8000/dashboard/ |
| Profile | http://127.0.0.1:8000/profile/ |
| Settings | http://127.0.0.1:8000/settings/ |
| Admin | http://127.0.0.1:8000/admin/ |
| API | http://127.0.0.1:8000/api/detect/ |

## 🗂️ File Locations

| Item | Path |
|------|------|
| Settings | `webapp/food_detection/settings.py` |
| URLs | `webapp/detector/urls.py` |
| Models | `webapp/detector/models.py` |
| Views | `webapp/detector/views.py` |
| Forms | `webapp/detector/forms.py` |
| Templates | `webapp/detector/templates/` |
| Static Files | `webapp/static/` |
| Media Files | `webapp/media/` |
| Database | `webapp/db.sqlite3` |
| Logs | `webapp/logs/debug.log` |

## 🔧 Environment Variables (.env)

```env
DJANGO_SECRET_KEY=your-secret-key
DJANGO_DEBUG=True
DJANGO_ALLOWED_HOSTS=localhost,127.0.0.1

EMAIL_HOST=smtp.gmail.com
EMAIL_PORT=587
EMAIL_HOST_USER=your-email@gmail.com
EMAIL_HOST_PASSWORD=your-app-password

GOOGLE_OAUTH2_CLIENT_ID=your-client-id
GOOGLE_OAUTH2_CLIENT_SECRET=your-client-secret
```

## 📦 Key Models

### CustomUser
```python
- email (login field)
- first_name, last_name
- phone_number
- profile_picture
- is_email_verified
```

### FoodProduct
```python
- user
- brand_name
- final_prediction (Real/Fake)
- overall_confidence
- risk_level
```

### FoodImage
```python
- product
- image
- view_type (front/back/side/barcode)
- prediction, confidence
```

## 🎯 Testing Checklist

- [ ] Server starts without errors
- [ ] Can access home page
- [ ] Can register new user
- [ ] Can login with credentials
- [ ] Can view dashboard
- [ ] Can edit profile
- [ ] Can upload product images
- [ ] Can access admin panel
- [ ] Google OAuth configured (optional)

## 🐛 Quick Fixes

### Clear Database
```bash
cd webapp
del db.sqlite3
python manage.py migrate
python manage.py createsuperuser
```

### Reset Migrations
```bash
cd webapp\detector\migrations
del 0*.py
cd ..\..
python manage.py makemigrations
python manage.py migrate
```

### Install Missing Package
```bash
pip install package-name
```

### Update Requirements
```bash
pip freeze > requirements.txt
```

## 📊 Database Queries (Django Shell)

```python
# Get all users
from detector.models import CustomUser
users = CustomUser.objects.all()

# Get user by email
user = CustomUser.objects.get(email='admin@example.com')

# Get user's products
products = user.food_products.all()

# Create new user
user = CustomUser.objects.create_user(
    email='test@example.com',
    first_name='Test',
    last_name='User',
    password='password123'
)

# Get recent activities
from detector.models import UserActivity
activities = UserActivity.objects.filter(user=user).order_by('-timestamp')[:10]
```

## 🔐 Security Checklist

- [x] CSRF protection enabled
- [x] Password hashing enabled
- [x] Session management configured
- [x] User authentication required for protected pages
- [x] Admin panel protected
- [x] Activity logging enabled
- [ ] Email verification (requires SMTP setup)
- [ ] Two-factor authentication (future)
- [ ] Rate limiting (future)

## 📝 Git Commands

```bash
# Initialize git
git init

# Add files
git add .

# Commit
git commit -m "Initial commit with authentication system"

# Add remote
git remote add origin <your-repo-url>

# Push
git push -u origin main
```

## 🎨 Frontend Stack

- **CSS Framework:** TailwindCSS (CDN)
- **Icons:** Font Awesome
- **JavaScript:** Vanilla JS
- **Templates:** Django Templates

## 🔄 Workflow

1. **Development:**
   - Make changes to code
   - Test locally with `python manage.py runserver`
   - Check for errors in terminal

2. **Database Changes:**
   - Modify models in `models.py`
   - Run `python manage.py makemigrations`
   - Run `python manage.py migrate`

3. **Static Files:**
   - Add files to `static/` directory
   - Run `python manage.py collectstatic`
   - Reference in templates with `{% static 'path/to/file' %}`

4. **Templates:**
   - Create/edit in `templates/` directory
   - Extend `base.html` for consistent layout
   - Use Django template tags

## 💡 Tips

- Always activate virtual environment before running commands
- Check `logs/debug.log` for detailed error information
- Use Django admin for quick data management
- Test authentication flows thoroughly
- Keep `.env` file secure and never commit it
- Use meaningful commit messages
- Document any custom changes

## 📞 Need Help?

1. Check error logs: `webapp/logs/debug.log`
2. Review Django docs: https://docs.djangoproject.com/
3. Check django-allauth docs: https://django-allauth.readthedocs.io/
4. Search Stack Overflow
5. Review this project's SETUP_GUIDE.md

---

**Last Updated:** November 5, 2025
**Django Version:** 5.2.6
**Python Version:** 3.13.3

# Google OAuth Setup Guide

## ✅ Issue Fixed!

The login/register pages now work without errors. Google OAuth buttons are temporarily disabled until you configure them properly.

---

## 🔧 How to Enable Google OAuth (Optional)

### Step 1: Get Google OAuth Credentials

1. Go to [Google Cloud Console](https://console.cloud.google.com/)
2. Create a new project or select an existing one
3. Enable the **Google+ API**
4. Go to **Credentials** → **Create Credentials** → **OAuth 2.0 Client ID**
5. Set Application type to **Web application**
6. Add these Authorized redirect URIs:
   ```
   http://127.0.0.1:8000/accounts/google/login/callback/
   http://localhost:8000/accounts/google/login/callback/
   ```
7. Copy your **Client ID** and **Client Secret**

### Step 2: Configure in Django Admin

1. Start your server:
   ```bash
   cd webapp
   python manage.py runserver
   ```

2. Login to admin panel: http://127.0.0.1:8000/admin/
   - Email: admin@example.com
   - Password: admin123

3. Go to **Sites** → Click on **example.com**
   - Change Domain name to: `127.0.0.1:8000`
   - Change Display name to: `Fake Product Detector`
   - Click **Save**

4. Go to **Social applications** → Click **Add social application**
   - Provider: **Google**
   - Name: `Google OAuth`
   - Client id: (paste your Client ID from Google)
   - Secret key: (paste your Client Secret from Google)
   - Sites: Select `127.0.0.1:8000` and move it to "Chosen sites"
   - Click **Save**

### Step 3: Update .env File

Edit `webapp/.env`:
```env
GOOGLE_OAUTH2_CLIENT_ID=your-actual-client-id-here
GOOGLE_OAUTH2_CLIENT_SECRET=your-actual-client-secret-here
```

### Step 4: Enable Google Buttons in Templates

#### In `login.html`:
Find this section (around line 77):
```html
<!-- Google Sign In (Configure in Admin Panel) -->
{% comment %}
```

Remove the `{% comment %}` and `{% endcomment %}` tags to enable the button.

#### In `register.html`:
Find the same section and remove the comment tags.

### Step 5: Test Google OAuth

1. Restart your server
2. Go to http://127.0.0.1:8000/login/
3. Click "Sign in with Google"
4. You should be redirected to Google login
5. After authentication, you'll be redirected back to your app

---

## 🎯 Current Status

✅ **Login page working** - http://127.0.0.1:8000/login/  
✅ **Register page working** - http://127.0.0.1:8000/register/  
✅ **Email/Password authentication working**  
⏸️ **Google OAuth ready** (needs configuration)

---

## 🚀 Quick Test

### Test Regular Login:
1. Go to http://127.0.0.1:8000/register/
2. Register a new account:
   - First Name: Test
   - Last Name: User
   - Email: test@example.com
   - Password: testpass123
3. Go to http://127.0.0.1:8000/login/
4. Login with your credentials
5. You should see the dashboard!

### Test Admin Login:
1. Go to http://127.0.0.1:8000/admin/
2. Login with:
   - Email: admin@example.com
   - Password: admin123
3. You can manage users, products, and configure Google OAuth here

---

## 📝 Notes

- Google OAuth is **optional** - your app works perfectly without it
- Email/Password authentication is fully functional
- You can enable Google OAuth anytime by following the steps above
- The infrastructure is already in place, just needs credentials

---

## 🐛 Troubleshooting

### If Google OAuth doesn't work after setup:

1. **Check redirect URIs** - Must match exactly in Google Console
2. **Check Site domain** - Must be `127.0.0.1:8000` in Django admin
3. **Check credentials** - Client ID and Secret must be correct
4. **Restart server** - After making changes
5. **Clear browser cache** - Sometimes helps with OAuth issues

---

## ✨ Your System is Ready!

You can now:
- ✅ Register new users
- ✅ Login with email/password
- ✅ Access dashboard
- ✅ Manage profiles
- ✅ Upload products
- ✅ Use admin panel
- ⏸️ Enable Google OAuth (optional)

**Everything is working perfectly! 🎉**

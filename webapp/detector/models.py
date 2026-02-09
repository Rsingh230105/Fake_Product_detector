from django.db import models
from django.contrib.auth.models import AbstractUser, BaseUserManager
from django.utils import timezone
from phonenumber_field.modelfields import PhoneNumberField

class CustomUserManager(BaseUserManager):
    """Custom user manager"""
    def create_user(self, email, first_name, last_name, password=None, **extra_fields):
        if not email:
            raise ValueError('The Email field must be set')
        email = self.normalize_email(email)
        user = self.model(email=email, first_name=first_name, last_name=last_name, username=email, **extra_fields)
        user.set_password(password)
        user.save(using=self._db)
        return user
    
    def create_superuser(self, email, first_name, last_name, password=None, **extra_fields):
        extra_fields.setdefault('is_staff', True)
        extra_fields.setdefault('is_superuser', True)
        
        if extra_fields.get('is_staff') is not True:
            raise ValueError('Superuser must have is_staff=True.')
        if extra_fields.get('is_superuser') is not True:
            raise ValueError('Superuser must have is_superuser=True.')
        
        return self.create_user(email, first_name, last_name, password, **extra_fields)

class CustomUser(AbstractUser):
    """Custom user model with additional fields"""
    email = models.EmailField(unique=True)
    first_name = models.CharField(max_length=30)
    last_name = models.CharField(max_length=30)
    phone_number = PhoneNumberField(blank=True)
    profile_picture = models.ImageField(upload_to='profiles/', blank=True, null=True)
    is_email_verified = models.BooleanField(default=False)
    created_at = models.DateTimeField(auto_now_add=True)
    updated_at = models.DateTimeField(auto_now=True)
    
    objects = CustomUserManager()
    
    USERNAME_FIELD = 'email'
    REQUIRED_FIELDS = ['first_name', 'last_name']
    
    def __str__(self):
        return self.email
    
    @property
    def full_name(self):
        return f"{self.first_name} {self.last_name}".strip()

class UserProfile(models.Model):
    """Extended user profile information"""
    user = models.OneToOneField(CustomUser, on_delete=models.CASCADE, related_name='profile')
    bio = models.TextField(max_length=500, blank=True)
    location = models.CharField(max_length=100, blank=True)
    birth_date = models.DateField(null=True, blank=True)
    website = models.URLField(blank=True)
    
    # Privacy settings
    profile_visibility = models.CharField(
        max_length=10,
        choices=[('public', 'Public'), ('private', 'Private')],
        default='public'
    )
    
    # Notification preferences
    email_notifications = models.BooleanField(default=True)
    sms_notifications = models.BooleanField(default=False)
    
    def __str__(self):
        return f"{self.user.email} Profile"

class FoodProduct(models.Model):
    """Model for storing food product analysis with ML and OCR results"""
    user = models.ForeignKey(CustomUser, on_delete=models.CASCADE, related_name='food_products', null=True, blank=True)
    brand_name = models.CharField(max_length=100)
    created_at = models.DateTimeField(auto_now_add=True)
    final_prediction = models.CharField(max_length=10, blank=True)  # 'REAL' or 'FAKE'
    overall_confidence = models.FloatField(null=True, blank=True)
    processing_time = models.FloatField(null=True, blank=True)  # Processing time in seconds
    brand_match = models.BooleanField(default=False)  # Whether OCR found matching brand
    ocr_results = models.JSONField(default=dict, blank=True)  # Structured OCR results
    
    # Component-wise scores for detailed analysis
    barcode_score = models.FloatField(null=True, blank=True)
    logo_score = models.FloatField(null=True, blank=True)
    ocr_score = models.FloatField(null=True, blank=True)
    packaging_score = models.FloatField(null=True, blank=True)
    final_score = models.FloatField(null=True, blank=True)
    failure_reasons = models.JSONField(default=list, blank=True)  # List of failure reasons
    detailed_analysis = models.JSONField(default=dict, blank=True)  # Structured component analysis
    user_report = models.JSONField(default=dict, blank=True)  # User-friendly report
    
    # Additional analysis fields
    risk_level = models.CharField(
        max_length=10,
        choices=[('low', 'Low'), ('medium', 'Medium'), ('high', 'High')],
        default='low'
    )
    analysis_notes = models.TextField(blank=True)
    
    class Meta:
        ordering = ['-created_at']
    
    def __str__(self):
        return f"{self.brand_name} Analysis - {self.final_prediction}"

class FoodImage(models.Model):
    """Model for storing multiple views of food product images"""
    VIEWS = [
        ('front', 'Front View'),
        ('back', 'Back View'),
        ('side', 'Side View'),
        ('barcode', 'Barcode/QR'),
        ('other', 'Other View')
    ]

    product = models.ForeignKey(FoodProduct, on_delete=models.CASCADE, related_name='images', null=True)
    image = models.ImageField(upload_to='food_images/')
    view_type = models.CharField(max_length=10, choices=VIEWS, default='front')
    uploaded_at = models.DateTimeField(auto_now_add=True)
    prediction = models.CharField(max_length=10, blank=True)  # Individual prediction for this view
    confidence = models.FloatField(null=True, blank=True)
    detected_text = models.TextField(blank=True)  # Text extracted from this view
    
    # Image metadata
    file_size = models.IntegerField(null=True, blank=True)  # Size in bytes
    image_width = models.IntegerField(null=True, blank=True)
    image_height = models.IntegerField(null=True, blank=True)

    class Meta:
        ordering = ['view_type']

    def __str__(self):
        return f"{self.product.brand_name} - {self.get_view_type_display()}"

class Advertisement(models.Model):
    """Model for storing promotional content and awareness campaigns"""
    CONTENT_TYPES = [
        ('banner', 'Promotional Banner'),
        ('campaign', 'Awareness Campaign'),
        ('comparison', 'Product Comparison')
    ]

    title = models.CharField(max_length=200)
    description = models.TextField()
    file = models.FileField(upload_to='advertisements/')
    content_type = models.CharField(max_length=20, choices=CONTENT_TYPES)
    duration_days = models.IntegerField(default=30)
    is_active = models.BooleanField(default=True)
    created_at = models.DateTimeField(auto_now_add=True)
    uploaded_by = models.ForeignKey(CustomUser, on_delete=models.CASCADE)

    def __str__(self):
        return self.title

    class Meta:
        ordering = ['-created_at']

class GalleryItem(models.Model):
    """Model for product comparison gallery items"""
    CATEGORIES = [
        ('comparison', 'Real vs Fake Comparison'),
        ('packaging', 'Packaging Verification'),
        ('safety', 'Safety Guidelines')
    ]
    
    STATUS_CHOICES = [
        ('pending', 'Pending Approval'),
        ('approved', 'Approved'),
        ('rejected', 'Rejected')
    ]

    title = models.CharField(max_length=200)
    description = models.TextField()
    category = models.CharField(max_length=20, choices=CATEGORIES)
    image = models.ImageField(upload_to='gallery/')
    status = models.CharField(max_length=10, choices=STATUS_CHOICES, default='pending')
    created_at = models.DateTimeField(auto_now_add=True)
    approved_at = models.DateTimeField(null=True, blank=True)
    is_featured = models.BooleanField(default=False)
    uploaded_by = models.ForeignKey(CustomUser, on_delete=models.CASCADE, null=True, blank=True)
    approved_by = models.ForeignKey(CustomUser, on_delete=models.SET_NULL, null=True, blank=True, related_name='approved_gallery_items')
    rejection_reason = models.TextField(blank=True)

    def __str__(self):
        return f"{self.get_category_display()} - {self.title}"

    class Meta:
        ordering = ['-created_at']

class MediaItem(models.Model):
    """Model for general media library items"""
    MEDIA_TYPES = [
        ('image', 'Image'),
        ('video', 'Video'),
        ('document', 'Document')
    ]
    
    STATUS_CHOICES = [
        ('pending', 'Pending Approval'),
        ('approved', 'Approved'),
        ('rejected', 'Rejected')
    ]

    title = models.CharField(max_length=200)
    description = models.TextField()
    file = models.FileField(upload_to='library/')
    media_type = models.CharField(max_length=20, choices=MEDIA_TYPES)
    status = models.CharField(max_length=10, choices=STATUS_CHOICES, default='pending')
    tags = models.JSONField(default=list)
    created_at = models.DateTimeField(auto_now_add=True)
    updated_at = models.DateTimeField(auto_now=True)
    approved_at = models.DateTimeField(null=True, blank=True)
    uploaded_by = models.ForeignKey(CustomUser, on_delete=models.CASCADE, null=True, blank=True)
    approved_by = models.ForeignKey(CustomUser, on_delete=models.SET_NULL, null=True, blank=True, related_name='approved_media_items')
    rejection_reason = models.TextField(blank=True)

    def __str__(self):
        return f"{self.get_media_type_display()} - {self.title}"

    class Meta:
        ordering = ['-created_at']

class UserActivity(models.Model):
    """Model to track user activities"""
    ACTIVITY_TYPES = [
        ('login', 'User Login'),
        ('logout', 'User Logout'),
        ('upload', 'Image Upload'),
        ('analysis', 'Product Analysis'),
        ('profile_update', 'Profile Update'),
    ]
    
    user = models.ForeignKey(CustomUser, on_delete=models.CASCADE, related_name='activities')
    activity_type = models.CharField(max_length=20, choices=ACTIVITY_TYPES)
    description = models.TextField(blank=True)
    ip_address = models.GenericIPAddressField(null=True, blank=True)
    user_agent = models.TextField(blank=True)
    timestamp = models.DateTimeField(auto_now_add=True)
    
    class Meta:
        ordering = ['-timestamp']
    
    def __str__(self):
        return f"{self.user.email} - {self.get_activity_type_display()}"

class UserFeedback(models.Model):
    """Model for storing user feedback and issue reports"""
    user = models.ForeignKey(CustomUser, on_delete=models.CASCADE, related_name='feedback')
    product = models.ForeignKey(FoodProduct, on_delete=models.CASCADE, related_name='feedback')
    reason = models.TextField()
    created_at = models.DateTimeField(auto_now_add=True)
    resolved = models.BooleanField(default=False)
    admin_response = models.TextField(blank=True)
    resolved_at = models.DateTimeField(null=True, blank=True)
    resolved_by = models.ForeignKey(CustomUser, on_delete=models.SET_NULL, null=True, blank=True, related_name='resolved_feedback')
    
    class Meta:
        ordering = ['-created_at']
    
    def __str__(self):
        return f"Feedback from {self.user.email} - {self.product.brand_name}"

class PasswordResetToken(models.Model):
    """Model for password reset tokens"""
    user = models.ForeignKey(CustomUser, on_delete=models.CASCADE)
    token = models.CharField(max_length=100, unique=True)
    created_at = models.DateTimeField(auto_now_add=True)
    expires_at = models.DateTimeField()
    used = models.BooleanField(default=False)

    def __str__(self):
        return f"Reset token for {self.user.email}"

    @property
    def is_expired(self):
        return timezone.now() > self.expires_at

    class Meta:
        ordering = ['-created_at']
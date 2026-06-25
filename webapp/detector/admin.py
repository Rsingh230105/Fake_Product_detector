from django.contrib import admin
from django.contrib.auth.admin import UserAdmin
from django.utils.html import format_html
from .models import (CustomUser, UserProfile, FoodProduct, FoodImage, 
                    Advertisement, GalleryItem, MediaItem, UserActivity)

@admin.register(CustomUser)
class CustomUserAdmin(UserAdmin):
    """Custom user admin"""
    list_display = ('email', 'first_name', 'last_name', 'is_staff', 'is_active', 'created_at')
    list_filter = ('is_staff', 'is_active', 'is_email_verified', 'created_at')
    search_fields = ('email', 'first_name', 'last_name')
    ordering = ('-created_at',)
    
    fieldsets = (
        (None, {'fields': ('email', 'password')}),
        ('Personal info', {'fields': ('first_name', 'last_name', 'phone_number', 'profile_picture')}),
        ('Permissions', {'fields': ('is_active', 'is_staff', 'is_superuser', 'groups', 'user_permissions')}),
        ('Important dates', {'fields': ('last_login', 'date_joined')}),
        ('Email verification', {'fields': ('is_email_verified',)}),
    )
    
    add_fieldsets = (
        (None, {
            'classes': ('wide',),
            'fields': ('email', 'first_name', 'last_name', 'password1', 'password2'),
        }),
    )

@admin.register(UserProfile)
class UserProfileAdmin(admin.ModelAdmin):
    """User profile admin"""
    list_display = ('user', 'location', 'profile_visibility', 'email_notifications')
    list_filter = ('profile_visibility', 'email_notifications', 'sms_notifications')
    search_fields = ('user__email', 'user__first_name', 'user__last_name', 'location')

class FoodImageInline(admin.TabularInline):
    """Inline admin for food images"""
    model = FoodImage
    extra = 0
    readonly_fields = ('uploaded_at', 'file_size', 'image_width', 'image_height')

@admin.register(FoodProduct)
class FoodProductAdmin(admin.ModelAdmin):
    """Food product admin"""
    list_display = ('brand_name', 'user', 'final_prediction', 'overall_confidence', 'risk_level', 'created_at')
    list_filter = ('final_prediction', 'risk_level', 'brand_match', 'created_at')
    search_fields = ('brand_name', 'user__email')
    readonly_fields = ('created_at', 'processing_time', 'ocr_results')
    inlines = [FoodImageInline]
    
    fieldsets = (
        ('Basic Information', {
            'fields': ('user', 'brand_name', 'created_at')
        }),
        ('Analysis Results', {
            'fields': ('final_prediction', 'overall_confidence', 'risk_level', 'brand_match', 'processing_time')
        }),
        ('Additional Information', {
            'fields': ('analysis_notes', 'ocr_results'),
            'classes': ('collapse',)
        }),
    )

@admin.register(FoodImage)
class FoodImageAdmin(admin.ModelAdmin):
    """Food image admin"""
    list_display = ('product', 'view_type', 'prediction', 'confidence', 'uploaded_at')
    list_filter = ('view_type', 'prediction', 'uploaded_at')
    search_fields = ('product__brand_name', 'detected_text')
    readonly_fields = ('uploaded_at', 'file_size', 'image_width', 'image_height')
    
    def get_queryset(self, request):
        return super().get_queryset(request).select_related('product')

@admin.register(Advertisement)
class AdvertisementAdmin(admin.ModelAdmin):
    """Advertisement admin"""
    list_display = ('title', 'content_type', 'is_active', 'duration_days', 'uploaded_by', 'created_at')
    list_filter = ('content_type', 'is_active', 'created_at')
    search_fields = ('title', 'description')
    readonly_fields = ('created_at',)
    
    def save_model(self, request, obj, form, change):
        if not change:  # If creating new object
            obj.uploaded_by = request.user
        super().save_model(request, obj, form, change)

@admin.register(GalleryItem)
class GalleryItemAdmin(admin.ModelAdmin):
    """Gallery item admin"""
    list_display = ('title', 'category', 'status', 'is_featured', 'uploaded_by', 'created_at')
    list_filter = ('category', 'status', 'is_featured', 'created_at')
    search_fields = ('title', 'description')
    readonly_fields = ('created_at',)
    actions = ['approve_items', 'reject_items']
    
    def approve_items(self, request, queryset):
        from django.utils import timezone
        queryset.update(status='approved', approved_by=request.user, approved_at=timezone.now())
        self.message_user(request, f"{queryset.count()} items approved.")
    approve_items.short_description = "Approve selected items"
    
    def reject_items(self, request, queryset):
        queryset.update(status='rejected')
        self.message_user(request, f"{queryset.count()} items rejected.")
    reject_items.short_description = "Reject selected items"
    
    def save_model(self, request, obj, form, change):
        if not change:
            obj.uploaded_by = request.user
            obj.status = 'approved'
            obj.approved_by = request.user
            from django.utils import timezone
            obj.approved_at = timezone.now()
        super().save_model(request, obj, form, change)

@admin.register(MediaItem)
class MediaItemAdmin(admin.ModelAdmin):
    """Media item admin"""
    list_display = ('title', 'media_type', 'status', 'uploaded_by', 'created_at')
    list_filter = ('media_type', 'status', 'created_at')
    search_fields = ('title', 'description', 'tags')
    readonly_fields = ('created_at', 'updated_at')
    actions = ['approve_items', 'reject_items']
    
    def approve_items(self, request, queryset):
        from django.utils import timezone
        queryset.update(status='approved', approved_by=request.user, approved_at=timezone.now())
        self.message_user(request, f"{queryset.count()} items approved.")
    approve_items.short_description = "Approve selected items"
    
    def reject_items(self, request, queryset):
        queryset.update(status='rejected')
        self.message_user(request, f"{queryset.count()} items rejected.")
    reject_items.short_description = "Reject selected items"
    
    def save_model(self, request, obj, form, change):
        if not change:
            obj.uploaded_by = request.user
            obj.status = 'approved'
            obj.approved_by = request.user
            from django.utils import timezone
            obj.approved_at = timezone.now()
        super().save_model(request, obj, form, change)

@admin.register(UserActivity)
class UserActivityAdmin(admin.ModelAdmin):
    """User activity admin"""
    list_display = ('user', 'activity_type', 'ip_address', 'timestamp')
    list_filter = ('activity_type', 'timestamp')
    search_fields = ('user__email', 'description', 'ip_address')
    readonly_fields = ('timestamp',)
    
    def has_add_permission(self, request):
        return False  # Prevent manual creation of activities
    
    def has_change_permission(self, request, obj=None):
        return False  # Prevent editing of activities

# Customize admin site
admin.site.site_header = "AI Product Verification System Admin"
admin.site.site_title = "FPD Admin"
admin.site.index_title = "Welcome to AI Product Verification System Administration"
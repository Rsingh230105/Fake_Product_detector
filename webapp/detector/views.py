"""Detector app views with complete authentication system."""

from rest_framework.views import APIView
from rest_framework.response import Response
from rest_framework import status
from rest_framework.parsers import MultiPartParser, FormParser
from django.views.generic import TemplateView, View
from django.shortcuts import render, redirect, get_object_or_404
from django.contrib.auth import login, logout, authenticate
from django.contrib.auth.mixins import LoginRequiredMixin, UserPassesTestMixin
from django.contrib.auth.decorators import login_required
from django.contrib import messages
from django.http import JsonResponse
from django.utils.decorators import method_decorator
from django.views.decorators.csrf import csrf_exempt
from django.db.models import Q
from django.core.paginator import Paginator
from django.utils import timezone
from django.db import transaction
import logging

from .models import (CustomUser, UserProfile, FoodProduct, FoodImage, 
                    Advertisement, GalleryItem, MediaItem, UserActivity, PasswordResetToken)
from .forms import CustomUserRegistrationForm, CustomUserLoginForm, UserProfileForm, CustomUserUpdateForm, MediaItemForm, AdvertisementForm, GalleryItemForm
from .serializers import FoodProductSerializer, FoodImageSerializer

logger = logging.getLogger(__name__)

def log_user_activity(user, activity_type, description="", request=None):
    """Helper function to log user activities"""
    if user and user.is_authenticated:
        ip_address = None
        user_agent = ""
        if request:
            ip_address = request.META.get('REMOTE_ADDR')
            user_agent = request.META.get('HTTP_USER_AGENT', '')
        
        UserActivity.objects.create(
            user=user,
            activity_type=activity_type,
            description=description,
            ip_address=ip_address,
            user_agent=user_agent
        )

class UploadView(TemplateView):
    """View for the image upload page"""
    template_name = 'detector/upload.html'
    
    def get_context_data(self, **kwargs):
        context = super().get_context_data(**kwargs)
        # Get active advertisements
        context['advertisements'] = Advertisement.objects.filter(is_active=True).order_by('-created_at')[:5]
        return context

class AboutView(TemplateView):
    template_name = 'detector/about.html'

class HomeView(TemplateView):
    template_name = 'detector/home.html'

class ContactView(TemplateView):
    template_name = 'detector/contact.html'

    def post(self, request, *args, **kwargs):
        # Handle contact form submission
        return redirect('detector:upload')

class RegisterView(View):
    """User registration view"""
    template_name = 'auth/register.html'
    form_class = CustomUserRegistrationForm

    def get(self, request):
        if request.user.is_authenticated:
            return redirect('detector:dashboard')
        form = self.form_class()
        return render(request, self.template_name, {'form': form})

    def post(self, request):
        if request.user.is_authenticated:
            return redirect('detector:dashboard')
        
        form = self.form_class(request.POST)
        if form.is_valid():
            try:
                with transaction.atomic():
                    user = form.save()
                    # Log registration activity
                    log_user_activity(user, 'register', 'User registered successfully', request)
                    
                    messages.success(request, 'Registration successful! Please check your email to verify your account.')
                    return redirect('detector:login')
            except Exception as e:
                logger.error(f"Registration error: {str(e)}")
                messages.error(request, 'Registration failed. Please try again.')
        
        return render(request, self.template_name, {'form': form})

class LoginView(View):
    """User login view"""
    template_name = 'auth/login.html'
    form_class = CustomUserLoginForm

    def get(self, request):
        if request.user.is_authenticated:
            return redirect('detector:dashboard')
        form = self.form_class()
        return render(request, self.template_name, {'form': form})

    def post(self, request):
        if request.user.is_authenticated:
            return redirect('detector:dashboard')
        
        form = self.form_class(request, data=request.POST)
        if form.is_valid():
            user = form.get_user()
            login(request, user)
            
            # Log login activity
            log_user_activity(user, 'login', 'User logged in successfully', request)
            
            messages.success(request, f'Welcome back, {user.first_name}!')
            
            # Redirect to next page or dashboard
            next_page = request.GET.get('next', 'detector:dashboard')
            return redirect(next_page)
        
        return render(request, self.template_name, {'form': form})

class ForgotPasswordView(View):
    """Forgot password view"""
    template_name = 'auth/forgot_password.html'

    def get(self, request):
        if request.user.is_authenticated:
            return redirect('detector:dashboard')
        return render(request, self.template_name)

    def post(self, request):
        if request.user.is_authenticated:
            return redirect('detector:dashboard')
        
        email = request.POST.get('email', '').strip()
        if not email:
            messages.error(request, 'Please enter your email address.')
            return render(request, self.template_name)
        
        try:
            user = CustomUser.objects.get(email=email)
        except CustomUser.DoesNotExist:
            # Don't reveal if email exists or not for security
            messages.success(request, 'If an account with this email exists, a password reset link has been sent.')
            return render(request, self.template_name)
        
        # Generate token
        import secrets
        token = secrets.token_urlsafe(32)
        expires_at = timezone.now() + timezone.timedelta(minutes=15)
        
        # Save token
        PasswordResetToken.objects.create(
            user=user,
            token=token,
            expires_at=expires_at
        )
        
        # Send email
        from django.core.mail import send_mail
        from django.conf import settings
        
        reset_url = request.build_absolute_uri(f'/reset-password/{token}/')
        subject = 'Password Reset Request'
        message = f'''
        Hi {user.first_name},

        You requested a password reset for your account.

        Click the link below to reset your password:
        {reset_url}

        This link will expire in 15 minutes.

        If you didn't request this, please ignore this email.

        Best regards,
        Food Detection Team
        '''
        
        try:
            send_mail(
                subject,
                message,
                settings.DEFAULT_FROM_EMAIL,
                [email],
                fail_silently=False,
            )
            messages.success(request, 'If an account with this email exists, a password reset link has been sent.')
        except Exception as e:
            logger.error(f"Failed to send password reset email: {str(e)}")
            messages.error(request, 'Failed to send reset email. Please try again later.')
        
        return render(request, self.template_name)

class ResetPasswordView(View):
    """Reset password view"""
    template_name = 'auth/reset_password.html'

    def get(self, request, token):
        if request.user.is_authenticated:
            return redirect('detector:dashboard')
        
        try:
            reset_token = PasswordResetToken.objects.get(token=token, used=False)
            if reset_token.is_expired:
                messages.error(request, 'This reset link has expired. Please request a new one.')
                return redirect('detector:forgot_password')
        except PasswordResetToken.DoesNotExist:
            messages.error(request, 'Invalid reset link.')
            return redirect('detector:forgot_password')
        
        return render(request, self.template_name, {'token': token})

    def post(self, request, token):
        if request.user.is_authenticated:
            return redirect('detector:dashboard')
        
        password = request.POST.get('password', '')
        confirm_password = request.POST.get('confirm_password', '')
        
        try:
            reset_token = PasswordResetToken.objects.get(token=token, used=False)
            if reset_token.is_expired:
                messages.error(request, 'This reset link has expired. Please request a new one.')
                return redirect('detector:forgot_password')
        except PasswordResetToken.DoesNotExist:
            messages.error(request, 'Invalid reset link.')
            return redirect('detector:forgot_password')
        
        if not password:
            messages.error(request, 'Please enter a new password.')
            return render(request, self.template_name, {'token': token})
        
        if password != confirm_password:
            messages.error(request, 'Passwords do not match.')
            return render(request, self.template_name, {'token': token})
        
        if len(password) < 8:
            messages.error(request, 'Password must be at least 8 characters long.')
            return render(request, self.template_name, {'token': token})
        
        # Update password
        user = reset_token.user
        user.set_password(password)
        user.save()
        
        # Mark token as used
        reset_token.used = True
        reset_token.save()
        
        # Log activity
        log_user_activity(user, 'password_reset', 'Password reset successfully', request)
        
        messages.success(request, 'Password reset successfully! You can now log in with your new password.')
        return redirect('detector:login')

class DashboardView(LoginRequiredMixin, TemplateView):
    """User dashboard view"""
    template_name = 'detector/dashboard.html'
    login_url = '/login/'

    def get_context_data(self, **kwargs):
        context = super().get_context_data(**kwargs)
        user = self.request.user
        
        # Get user's recent food products
        recent_products = FoodProduct.objects.filter(user=user).order_by('-created_at')[:5]
        
        # Convert confidence to percentage for display
        for product in recent_products:
            if product.overall_confidence is not None:
                product.display_confidence = round(product.overall_confidence * 100, 1)
        
        # Get user statistics
        total_analyses = FoodProduct.objects.filter(user=user).count()
        fake_products = FoodProduct.objects.filter(user=user, final_prediction='Fake').count()
        real_products = FoodProduct.objects.filter(user=user, final_prediction='Real').count()
        
        # Get recent activities
        recent_activities = UserActivity.objects.filter(user=user).order_by('-timestamp')[:10]
        
        context.update({
            'recent_products': recent_products,
            'total_analyses': total_analyses,
            'fake_products': fake_products,
            'real_products': real_products,
            'recent_activities': recent_activities,
        })
        return context

class ProfileView(LoginRequiredMixin, View):
    """User profile view"""
    template_name = 'detector/profile.html'
    login_url = '/login/'

    def get(self, request):
        user = request.user
        profile, created = UserProfile.objects.get_or_create(user=user)
        
        user_form = CustomUserUpdateForm(instance=user)
        profile_form = UserProfileForm(instance=profile)
        
        context = {
            'user_form': user_form,
            'profile_form': profile_form,
            'profile': profile,
        }
        return render(request, self.template_name, context)

    def post(self, request):
        user = request.user
        profile, created = UserProfile.objects.get_or_create(user=user)
        
        user_form = CustomUserUpdateForm(request.POST, request.FILES, instance=user)
        profile_form = UserProfileForm(request.POST, instance=profile)
        
        if user_form.is_valid() and profile_form.is_valid():
            try:
                with transaction.atomic():
                    user_form.save()
                    profile_form.save()
                    
                    # Log profile update activity
                    log_user_activity(user, 'profile_update', 'Profile updated successfully', request)
                    
                    messages.success(request, 'Profile updated successfully!')
                    return redirect('detector:profile')
            except Exception as e:
                logger.error(f"Profile update error: {str(e)}")
                messages.error(request, 'Profile update failed. Please try again.')
        
        context = {
            'user_form': user_form,
            'profile_form': profile_form,
            'profile': profile,
        }
        return render(request, self.template_name, context)

class AnalysesView(LoginRequiredMixin, TemplateView):
    """View for displaying user's analysis history with search and pagination"""
    template_name = 'detector/analyses.html'
    login_url = '/login/'

    def get_context_data(self, **kwargs):
        context = super().get_context_data(**kwargs)
        user = self.request.user
        
        # Get search query
        search_query = self.request.GET.get('search', '')
        
        # Get all user's products
        products = FoodProduct.objects.filter(user=user).order_by('-created_at')
        
        if search_query:
            products = products.filter(
                Q(brand_name__icontains=search_query) |
                Q(final_prediction__icontains=search_query) |
                Q(analysis_notes__icontains=search_query)
            )
        
        # Pagination
        paginator = Paginator(products, 10)  # 10 items per page
        page_number = self.request.GET.get('page')
        page_obj = paginator.get_page(page_number)
        
        # Convert confidence to percentage for display
        for product in page_obj:
            if product.overall_confidence is not None:
                product.display_confidence = round(product.overall_confidence * 100, 1)
        
        context.update({
            'products': page_obj,
            'search_query': search_query,
            'total_analyses': products.count(),
        })
        return context

class SettingsView(LoginRequiredMixin, TemplateView):
    """User settings view"""
    template_name = 'detector/settings.html'
    login_url = '/login/'

    def get_context_data(self, **kwargs):
        context = super().get_context_data(**kwargs)
        user = self.request.user
        profile, created = UserProfile.objects.get_or_create(user=user)
        context['profile'] = profile
        return context

@login_required
def logout_view(request):
    """User logout view"""
    user = request.user
    log_user_activity(user, 'logout', 'User logged out', request)
    logout(request)
    messages.success(request, 'You have been logged out successfully.')
    return redirect('detector:upload')

# Media Management Views (Admin only)
class AdsUploadView(UserPassesTestMixin, TemplateView):
    """View for uploading advertisements and awareness content"""
    template_name = 'detector/media/ads_upload.html'

    def test_func(self):
        return self.request.user.is_staff

    def get_context_data(self, **kwargs):
        context = super().get_context_data(**kwargs)
        context['advertisements'] = Advertisement.objects.all().order_by('-created_at')
        return context

    def post(self, request, *args, **kwargs):
        if not request.user.is_staff:
            return redirect('detector:upload')

        try:
            files = request.FILES.getlist('files')
            content_type = request.POST.get('content_type')
            title = request.POST.get('title')
            description = request.POST.get('description')
            duration = request.POST.get('duration')

            for file in files:
                Advertisement.objects.create(
                    title=title,
                    description=description,
                    file=file,
                    content_type=content_type,
                    duration_days=int(duration) if duration else 30,
                    uploaded_by=request.user
                )
            messages.success(request, "Advertisement uploaded successfully")
            return redirect('detector:upload_ads')
        except Exception as e:
            messages.error(request, f"Error uploading advertisement: {str(e)}")
            return self.get(request, *args, **kwargs)

class ProductGalleryView(UserPassesTestMixin, TemplateView):
    """View for managing product comparison gallery"""
    template_name = 'detector/media/gallery.html'

    def test_func(self):
        return self.request.user.is_staff

    def get_context_data(self, **kwargs):
        context = super().get_context_data(**kwargs)
        context['categories'] = [
            'Real vs Fake Comparisons',
            'Packaging Verification',
            'Safety Guidelines'
        ]
        # Show only approved items to non-staff, all items to staff
        if self.request.user.is_staff:
            context['gallery_items'] = GalleryItem.objects.all().order_by('-created_at')
        else:
            context['gallery_items'] = GalleryItem.objects.filter(status='approved').order_by('-created_at')
        return context

    def post(self, request, *args, **kwargs):
        try:
            files = request.FILES.getlist('images')
            title = request.POST.get('title')
            category = request.POST.get('category')
            description = request.POST.get('description', '')

            for file in files:
                # Admin uploads are auto-approved, user uploads need approval
                status = 'approved' if request.user.is_staff else 'pending'
                GalleryItem.objects.create(
                    title=title,
                    description=description,
                    category=category,
                    image=file,
                    status=status,
                    uploaded_by=request.user,
                    approved_by=request.user if request.user.is_staff else None,
                    approved_at=timezone.now() if request.user.is_staff else None
                )
            
            if request.user.is_staff:
                messages.success(request, "Gallery items uploaded and approved successfully")
            else:
                messages.success(request, "Gallery items uploaded! Waiting for admin approval.")
            return redirect('detector:gallery')
        except Exception as e:
            messages.error(request, f"Error uploading gallery items: {str(e)}")
            return self.get(request, *args, **kwargs)

class MediaLibraryView(TemplateView):
    """View for browsing all media content"""
    template_name = 'detector/media/library.html'

    def get_context_data(self, **kwargs):
        context = super().get_context_data(**kwargs)
        
        # Show only approved items to non-staff
        if self.request.user.is_authenticated and self.request.user.is_staff:
            media_items = MediaItem.objects.all().order_by('-created_at')
        else:
            media_items = MediaItem.objects.filter(status='approved').order_by('-created_at')
        
        # Pagination
        paginator = Paginator(media_items, 12)
        page_number = self.request.GET.get('page')
        page_obj = paginator.get_page(page_number)
        
        context['media_items'] = page_obj
        return context

class DeleteMediaView(UserPassesTestMixin, View):
    """View for deleting media items"""

    def test_func(self):
        return self.request.user.is_staff

    def delete(self, request, pk):
        if not request.user.is_staff:
            return JsonResponse({'error': 'Unauthorized'}, status=403)

        for model in (Advertisement, GalleryItem, MediaItem):
            try:
                obj = model.objects.get(pk=pk)
                obj.delete()
                return JsonResponse({'status': 'success'})
            except model.DoesNotExist:
                continue

        return JsonResponse({'error': 'Not found'}, status=404)

# API Views
class FoodDetectorView(APIView):
    """API endpoint for food detection"""
    parser_classes = (MultiPartParser, FormParser)

    def get(self, request, *args, **kwargs):
        return Response({
            'message': 'Welcome to the Food Detection API',
            'endpoints': {
                'POST /api/detect/': {
                    'description': 'Detect fake/real food from multiple images',
                    'parameters': {
                        'brand_name': 'Name of the food product',
                        'images': 'Multiple image files (JPEG/PNG)',
                        'view_types': 'Type of view for each image (front, back, side, barcode, other)'
                    },
                    'returns': {
                        'final_prediction': 'Overall Real/Fake classification',
                        'overall_confidence': 'Combined confidence score (0-1)',
                        'images': 'Analysis results for each uploaded image'
                    }
                }
            },
            'supported_formats': ['image/jpeg', 'image/png'],
            'max_file_size': '5MB per image',
            'required_views': ['front', 'back']
        }, status=status.HTTP_200_OK)

    def post(self, request, *args, **kwargs):
        try:
            logger.info(f"Received request data: {request.data}")
            logger.info(f"Files: {request.FILES}")

            brand_name = request.POST.get('brand_name')
            images = request.FILES.getlist('images[]')
            view_types = request.POST.getlist('view_types[]')

            if not brand_name:
                return Response({'error': 'Brand name is required'}, status=status.HTTP_400_BAD_REQUEST)

            if not images:
                return Response({'error': 'At least one image is required'}, status=status.HTTP_400_BAD_REQUEST)

            if len(images) != len(view_types):
                return Response({'error': 'Number of images and view types must match'}, status=status.HTTP_400_BAD_REQUEST)

            # Create product with user association if authenticated
            user = request.user if request.user.is_authenticated else None
            
            serializer = FoodProductSerializer(data={
                'brand_name': brand_name,
                'uploaded_images': images,
                'view_types': view_types,
                'user': user.id if user else None
            })

            if serializer.is_valid():
                product = serializer.save()
                
                # Log analysis activity
                if user:
                    log_user_activity(user, 'analysis', f'Analyzed product: {brand_name}', request)

                # Simulate ML processing (replace with actual ML logic)
                from random import random, choice
                predictions = ['Real', 'Fake']

                overall_confidence = 0
                predictions_count = {'Real': 0, 'Fake': 0}

                for image in product.images.all():
                    pred = choice(predictions)
                    conf = random() * 0.5 + 0.5
                    image.prediction = pred
                    image.confidence = conf
                    image.detected_text = f"Sample text for {image.get_view_type_display()}"
                    image.save()

                    predictions_count[pred] += 1
                    overall_confidence += conf

                total_images = len(product.images.all())
                final_pred = 'Real' if predictions_count['Real'] > predictions_count['Fake'] else 'Fake'
                overall_conf = overall_confidence / total_images if total_images > 0 else 0

                product.final_prediction = final_pred
                product.overall_confidence = overall_conf
                product.save()

                return Response(FoodProductSerializer(product).data, status=status.HTTP_201_CREATED)

            return Response(serializer.errors, status=status.HTTP_400_BAD_REQUEST)

        except Exception as e:
            logger.error(f"Error processing food detection request: {str(e)}")
            return Response({'error': 'Internal server error occurred'}, status=status.HTTP_500_INTERNAL_SERVER_ERROR)


# Custom Admin Interface Views

class AdminRequiredMixin(UserPassesTestMixin):
    """Mixin to ensure user is admin/staff"""
    def test_func(self):
        return self.request.user.is_authenticated and self.request.user.is_staff

class MediaAdminDashboard(AdminRequiredMixin, TemplateView):
    """Custom admin dashboard for media management"""
    template_name = 'detector/admin/dashboard.html'

    def get_context_data(self, **kwargs):
        context = super().get_context_data(**kwargs)
        context.update({
            'total_advertisements': Advertisement.objects.count(),
            'active_advertisements': Advertisement.objects.filter(is_active=True).count(),
            'total_media_items': MediaItem.objects.count(),
            'approved_media_items': MediaItem.objects.filter(status='approved').count(),
            'pending_media_items': MediaItem.objects.filter(status='pending').count(),
            'total_gallery_items': GalleryItem.objects.count(),
            'featured_gallery_items': GalleryItem.objects.filter(is_featured=True).count(),
            'recent_advertisements': Advertisement.objects.select_related('uploaded_by').order_by('-created_at')[:5],
            'recent_media': MediaItem.objects.select_related('uploaded_by').order_by('-created_at')[:5],
        })
        return context

class AdvertisementListView(AdminRequiredMixin, TemplateView):
    """List all advertisements with management options"""
    template_name = 'detector/admin/advertisements.html'

    def get_context_data(self, **kwargs):
        context = super().get_context_data(**kwargs)
        advertisements = Advertisement.objects.select_related('uploaded_by').order_by('-created_at')
        
        # Search functionality
        search_query = self.request.GET.get('search', '')
        if search_query:
            advertisements = advertisements.filter(
                Q(title__icontains=search_query) |
                Q(description__icontains=search_query) |
                Q(content_type__icontains=search_query)
            )
        
        # Pagination
        paginator = Paginator(advertisements, 10)
        page_number = self.request.GET.get('page')
        page_obj = paginator.get_page(page_number)
        
        context.update({
            'advertisements': page_obj,
            'search_query': search_query,
        })
        return context

class AdvertisementCreateView(AdminRequiredMixin, View):
    """Create new advertisement"""
    
    def get(self, request):
        form = AdvertisementForm()
        return render(request, 'detector/admin/advertisement_form.html', {'form': form, 'action': 'Create'})
    
    def post(self, request):
        form = AdvertisementForm(request.POST, request.FILES)
        if form.is_valid():
            advertisement = form.save(commit=False)
            advertisement.uploaded_by = request.user
            advertisement.save()
            messages.success(request, 'Advertisement created successfully!')
            return redirect('detector:admin_advertisements')
        
        return render(request, 'detector/admin/advertisement_form.html', {'form': form, 'action': 'Create'})

class AdvertisementUpdateView(AdminRequiredMixin, View):
    """Update existing advertisement"""
    
    def get(self, request, pk):
        advertisement = get_object_or_404(Advertisement, pk=pk)
        form = AdvertisementForm(instance=advertisement)
        return render(request, 'detector/admin/advertisement_form.html', {'form': form, 'action': 'Update', 'advertisement': advertisement})
    
    def post(self, request, pk):
        advertisement = get_object_or_404(Advertisement, pk=pk)
        form = AdvertisementForm(request.POST, request.FILES, instance=advertisement)
        if form.is_valid():
            form.save()
            messages.success(request, 'Advertisement updated successfully!')
            return redirect('detector:admin_advertisements')
        
        return render(request, 'detector/admin/advertisement_form.html', {'form': form, 'action': 'Update', 'advertisement': advertisement})

class AdvertisementDeleteView(AdminRequiredMixin, View):
    """Delete advertisement"""
    
    def post(self, request, pk):
        advertisement = get_object_or_404(Advertisement, pk=pk)
        advertisement.delete()
        messages.success(request, 'Advertisement deleted successfully!')
        return redirect('detector:admin_advertisements')

class MediaItemListView(AdminRequiredMixin, TemplateView):
    """List all media items with management options"""
    template_name = 'detector/admin/media_items.html'

    def get_context_data(self, **kwargs):
        context = super().get_context_data(**kwargs)
        media_items = MediaItem.objects.select_related('uploaded_by', 'approved_by').order_by('-created_at')
        
        # Search functionality
        search_query = self.request.GET.get('search', '')
        if search_query:
            media_items = media_items.filter(
                Q(title__icontains=search_query) |
                Q(description__icontains=search_query) |
                Q(media_type__icontains=search_query)
            )
        
        # Filter by status
        status_filter = self.request.GET.get('status', '')
        if status_filter:
            media_items = media_items.filter(status=status_filter)
        
        # Pagination
        paginator = Paginator(media_items, 12)
        page_number = self.request.GET.get('page')
        page_obj = paginator.get_page(page_number)
        
        context.update({
            'media_items': page_obj,
            'search_query': search_query,
            'status_filter': status_filter,
        })
        return context

class MediaItemCreateView(AdminRequiredMixin, View):
    """Create new media item"""
    
    def get(self, request):
        form = MediaItemForm()
        return render(request, 'detector/admin/media_form.html', {'form': form, 'action': 'Create'})
    
    def post(self, request):
        form = MediaItemForm(request.POST, request.FILES)
        if form.is_valid():
            media_item = form.save(commit=False)
            media_item.uploaded_by = request.user
            media_item.save()
            messages.success(request, 'Media item created successfully!')
            return redirect('detector:admin_media')
        
        return render(request, 'detector/admin/media_form.html', {'form': form, 'action': 'Create'})

class MediaItemUpdateView(AdminRequiredMixin, View):
    """Update existing media item"""
    
    def get(self, request, pk):
        media_item = get_object_or_404(MediaItem, pk=pk)
        form = MediaItemForm(instance=media_item)
        return render(request, 'detector/admin/media_form.html', {'form': form, 'action': 'Update', 'media_item': media_item})
    
    def post(self, request, pk):
        media_item = get_object_or_404(MediaItem, pk=pk)
        form = MediaItemForm(request.POST, request.FILES, instance=media_item)
        if form.is_valid():
            form.save()
            messages.success(request, 'Media item updated successfully!')
            return redirect('detector:admin_media')
        
        return render(request, 'detector/admin/media_form.html', {'form': form, 'action': 'Update', 'media_item': media_item})

class MediaItemApproveView(AdminRequiredMixin, View):
    """Approve or reject media item"""
    
    def post(self, request, pk):
        media_item = get_object_or_404(MediaItem, pk=pk)
        action = request.POST.get('action')
        
        if action == 'approve':
            media_item.status = 'approved'
            media_item.approved_by = request.user
            media_item.approved_at = timezone.now()
            media_item.rejection_reason = ''
            messages.success(request, 'Media item approved successfully!')
        elif action == 'reject':
            media_item.status = 'rejected'
            media_item.rejection_reason = request.POST.get('rejection_reason', '')
            messages.success(request, 'Media item rejected!')
        
        media_item.save()
        return redirect('detector:admin_media')

class MediaItemDeleteView(AdminRequiredMixin, View):
    """Delete media item"""
    
    def post(self, request, pk):
        media_item = get_object_or_404(MediaItem, pk=pk)
        media_item.delete()
        messages.success(request, 'Media item deleted successfully!')
        return redirect('detector:admin_media')

class GalleryItemListView(AdminRequiredMixin, TemplateView):
    """List all gallery items"""
    template_name = 'detector/admin/gallery.html'

    def get_context_data(self, **kwargs):
        context = super().get_context_data(**kwargs)
        gallery_items = GalleryItem.objects.select_related('uploaded_by', 'approved_by').order_by('-created_at')
        
        # Search functionality
        search_query = self.request.GET.get('search', '')
        if search_query:
            gallery_items = gallery_items.filter(
                Q(title__icontains=search_query) |
                Q(description__icontains=search_query) |
                Q(category__icontains=search_query)
            )
        
        # Pagination
        paginator = Paginator(gallery_items, 12)
        page_number = self.request.GET.get('page')
        page_obj = paginator.get_page(page_number)
        
        context.update({
            'gallery_items': page_obj,
            'search_query': search_query,
        })
        return context

class GalleryItemCreateView(AdminRequiredMixin, View):
    """Create new gallery item"""
    
    def get(self, request):
        form = GalleryItemForm()
        return render(request, 'detector/admin/gallery_form.html', {'form': form, 'action': 'Create'})
    
    def post(self, request):
        form = GalleryItemForm(request.POST, request.FILES)
        if form.is_valid():
            gallery_item = form.save(commit=False)
            gallery_item.uploaded_by = request.user
            gallery_item.save()
            messages.success(request, 'Gallery item created successfully!')
            return redirect('detector:admin_gallery')
        
        return render(request, 'detector/admin/gallery_form.html', {'form': form, 'action': 'Create'})

class GalleryItemUpdateView(AdminRequiredMixin, View):
    """Update existing gallery item"""
    
    def get(self, request, pk):
        gallery_item = get_object_or_404(GalleryItem, pk=pk)
        form = GalleryItemForm(instance=gallery_item)
        return render(request, 'detector/admin/gallery_form.html', {'form': form, 'action': 'Update', 'gallery_item': gallery_item})
    
    def post(self, request, pk):
        gallery_item = get_object_or_404(GalleryItem, pk=pk)
        form = GalleryItemForm(request.POST, request.FILES, instance=gallery_item)
        if form.is_valid():
            form.save()
            messages.success(request, 'Gallery item updated successfully!')
            return redirect('detector:admin_gallery')
        
        return render(request, 'detector/admin/gallery_form.html', {'form': form, 'action': 'Update', 'gallery_item': gallery_item})

class GalleryItemApproveView(AdminRequiredMixin, View):
    """Approve gallery item"""
    
    def post(self, request, pk):
        gallery_item = get_object_or_404(GalleryItem, pk=pk)
        action = request.POST.get('action')
        
        if action == 'approve':
            gallery_item.status = 'approved'
            gallery_item.approved_by = request.user
            gallery_item.approved_at = timezone.now()
            gallery_item.rejection_reason = ''
            messages.success(request, 'Gallery item approved successfully!')
        elif action == 'reject':
            gallery_item.status = 'rejected'
            gallery_item.rejection_reason = request.POST.get('rejection_reason', '')
            messages.success(request, 'Gallery item rejected!')
        elif action == 'feature':
            gallery_item.is_featured = not gallery_item.is_featured
            gallery_item.save()
            status_msg = 'featured' if gallery_item.is_featured else 'unfeatured'
            messages.success(request, f'Gallery item {status_msg} successfully!')
        
        gallery_item.save()
        return redirect('detector:admin_gallery')

class GalleryItemDeleteView(AdminRequiredMixin, View):
    """Delete gallery item"""
    
    def post(self, request, pk):
        gallery_item = get_object_or_404(GalleryItem, pk=pk)
        gallery_item.delete()
        messages.success(request, 'Gallery item deleted successfully!')
        return redirect('detector:admin_gallery')


# Public Media Display Views

class MediaLibraryView(TemplateView):
    """Public view for media library - shows approved media items"""
    template_name = 'detector/media_library.html'

    def get_context_data(self, **kwargs):
        context = super().get_context_data(**kwargs)
        
        # Only show approved media items
        media_items = MediaItem.objects.filter(status='approved').order_by('-created_at')
        
        # Filter by media type
        media_type = self.request.GET.get('type', '')
        if media_type:
            media_items = media_items.filter(media_type=media_type)
        
        # Search functionality
        search_query = self.request.GET.get('search', '')
        if search_query:
            media_items = media_items.filter(
                Q(title__icontains=search_query) |
                Q(description__icontains=search_query) |
                Q(tags__icontains=search_query)
            )
        
        # Pagination
        paginator = Paginator(media_items, 12)
        page_number = self.request.GET.get('page')
        page_obj = paginator.get_page(page_number)
        
        context.update({
            'media_items': page_obj,
            'search_query': search_query,
            'media_type': media_type,
            'media_types': MediaItem.MEDIA_TYPES,
        })
        return context

class AwarenessCampaignView(TemplateView):
    """Public view for awareness campaigns and advertisements"""
    template_name = 'detector/awareness.html'

    def get_context_data(self, **kwargs):
        context = super().get_context_data(**kwargs)
        
        # Get active advertisements
        advertisements = Advertisement.objects.filter(is_active=True).order_by('-created_at')
        
        # Get approved gallery items
        gallery_items = GalleryItem.objects.filter(status='approved').order_by('-created_at')
        
        # Featured items
        featured_gallery = gallery_items.filter(is_featured=True)[:6]
        
        context.update({
            'advertisements': advertisements,
            'gallery_items': gallery_items,
            'featured_gallery': featured_gallery,
        })
        return context
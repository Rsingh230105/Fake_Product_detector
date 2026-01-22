from django.urls import path, include
from . import views

app_name = 'detector'

urlpatterns = [
    # Main pages
    path('', views.HomeView.as_view(), name='home'),
    path('upload/', views.UploadView.as_view(), name='upload'),
    path('about/', views.AboutView.as_view(), name='about'),
    path('contact/', views.ContactView.as_view(), name='contact'),
    
    # Authentication URLs
    path('login/', views.LoginView.as_view(), name='login'),
    path('register/', views.RegisterView.as_view(), name='register'),
    path('logout/', views.logout_view, name='logout'),
    path('forgot-password/', views.ForgotPasswordView.as_view(), name='forgot_password'),
    path('reset-password/<str:token>/', views.ResetPasswordView.as_view(), name='reset_password'),
    
    # User dashboard and profile
    path('dashboard/', views.DashboardView.as_view(), name='dashboard'),
    path('analyses/', views.AnalysesView.as_view(), name='analyses'),
    path('profile/', views.ProfileView.as_view(), name='profile'),
    path('settings/', views.SettingsView.as_view(), name='settings'),
    
    # API endpoints
    path('api/detect/', views.FoodDetectorView.as_view(), name='detect_food'),
    
    # Custom Admin Interface (Staff only)
    path('media-admin/dashboard/', views.MediaAdminDashboard.as_view(), name='admin_dashboard'),
    path('media-admin/advertisements/', views.AdvertisementListView.as_view(), name='admin_advertisements'),
    path('media-admin/advertisements/create/', views.AdvertisementCreateView.as_view(), name='admin_advertisement_create'),
    path('media-admin/advertisements/<int:pk>/update/', views.AdvertisementUpdateView.as_view(), name='admin_advertisement_update'),
    path('media-admin/advertisements/<int:pk>/delete/', views.AdvertisementDeleteView.as_view(), name='admin_advertisement_delete'),
    
    path('media-admin/media/', views.MediaItemListView.as_view(), name='admin_media'),
    path('media-admin/media/create/', views.MediaItemCreateView.as_view(), name='admin_media_create'),
    path('media-admin/media/<int:pk>/update/', views.MediaItemUpdateView.as_view(), name='admin_media_update'),
    path('media-admin/media/<int:pk>/approve/', views.MediaItemApproveView.as_view(), name='admin_media_approve'),
    path('media-admin/media/<int:pk>/delete/', views.MediaItemDeleteView.as_view(), name='admin_media_delete'),
    
    path('media-admin/gallery/', views.GalleryItemListView.as_view(), name='admin_gallery'),
    path('media-admin/gallery/create/', views.GalleryItemCreateView.as_view(), name='admin_gallery_create'),
    path('media-admin/gallery/<int:pk>/update/', views.GalleryItemUpdateView.as_view(), name='admin_gallery_update'),
    path('media-admin/gallery/<int:pk>/approve/', views.GalleryItemApproveView.as_view(), name='admin_gallery_approve'),
    path('media-admin/gallery/<int:pk>/delete/', views.GalleryItemDeleteView.as_view(), name='admin_gallery_delete'),
    
    # Public Media Display
    path('media/library/', views.MediaLibraryView.as_view(), name='media_library'),
    path('awareness/', views.AwarenessCampaignView.as_view(), name='awareness'),
    
    # Django-allauth URLs
    # path('accounts/', include('allauth.urls')),
]

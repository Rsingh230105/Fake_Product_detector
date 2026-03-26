# """File upload security validators"""
# import magic
# from django.core.exceptions import ValidationError
# from PIL import Image
# import io

# MAX_FILE_SIZE = 5 * 1024 * 1024  # 5MB
# MAX_IMAGE_PIXELS = 89478485  # ~8K resolution
# ALLOWED_IMAGE_TYPES = {'image/jpeg', 'image/png', 'image/webp'}
# ALLOWED_IMAGE_EXTS = {'.jpg', '.jpeg', '.png', '.webp'}

# def validate_image_upload(file):
#     """Validate uploaded image file"""
#     # Size check
#     if file.size > MAX_FILE_SIZE:
#         raise ValidationError(f'File too large. Max size: 5MB')
    
#     # Read file header for magic byte check
#     file.seek(0)
#     header = file.read(2048)
#     file.seek(0)
    
#     # MIME type check using magic bytes
#     mime = magic.from_buffer(header, mime=True)
#     if mime not in ALLOWED_IMAGE_TYPES:
#         raise ValidationError(f'Invalid file type: {mime}')
    
#     # Extension check
#     ext = file.name.lower()[file.name.rfind('.'):]
#     if ext not in ALLOWED_IMAGE_EXTS:
#         raise ValidationError(f'Invalid extension: {ext}')
    
#     # PIL validation (decompression bomb protection)
#     try:
#         img = Image.open(io.BytesIO(header + file.read()))
#         img.verify()
#         file.seek(0)
        
#         # Pixel limit check
#         if img.size[0] * img.size[1] > MAX_IMAGE_PIXELS:
#             raise ValidationError('Image resolution too high')
#     except Exception as e:
#         raise ValidationError(f'Invalid image: {str(e)}')
    
#     return True
"""
File upload security validators
Cross-platform secure image validation (No libmagic dependency)
"""

from django.core.exceptions import ValidationError
from PIL import Image, UnidentifiedImageError
import os

MAX_FILE_SIZE = 5 * 1024 * 1024  # 5MB
MAX_IMAGE_PIXELS = 50_000_000  # Safe upper limit (~50MP)
ALLOWED_IMAGE_EXTS = {'.jpg', '.jpeg', '.png', '.webp'}
ALLOWED_IMAGE_FORMATS = {'JPEG', 'PNG', 'WEBP'}


def validate_image_upload(file):
    """
    Validate uploaded image securely:
    - File size limit
    - Extension check
    - Pillow format validation
    - Pixel limit protection
    """

    # -------------------------
    # 1. File size check
    # -------------------------
    if file.size > MAX_FILE_SIZE:
        raise ValidationError("File too large. Maximum size is 5MB.")

    # -------------------------
    # 2. Extension check
    # -------------------------
    ext = os.path.splitext(file.name)[1].lower()
    if ext not in ALLOWED_IMAGE_EXTS:
        raise ValidationError(f"Invalid file extension: {ext}")

    # -------------------------
    # 3. Pillow validation
    # -------------------------
    try:
        file.seek(0)
        img = Image.open(file)

        # Verify actual image integrity
        img.verify()

        # Reopen image after verify (Pillow requirement)
        file.seek(0)
        img = Image.open(file)

        # -------------------------
        # 4. Format check
        # -------------------------
        if img.format not in ALLOWED_IMAGE_FORMATS:
            raise ValidationError(f"Invalid image format: {img.format}")

        # -------------------------
        # 5. Pixel size check
        # -------------------------
        width, height = img.size
        if width * height > MAX_IMAGE_PIXELS:
            raise ValidationError("Image resolution too high.")

    except UnidentifiedImageError:
        raise ValidationError("Invalid image file.")
    except Exception:
        raise ValidationError("Corrupted or unsupported image.")

    finally:
        file.seek(0)

    return True

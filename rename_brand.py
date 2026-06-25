import os
from pathlib import Path

root = Path(__file__).resolve().parent
old_pkg = root / 'webapp' / 'ai_product_verification_system'
new_pkg = root / 'webapp' / 'ai_product_verification_system'

if old_pkg.exists() and not new_pkg.exists():
    os.rename(old_pkg, new_pkg)
    print(f'Renamed package folder: {old_pkg} -> {new_pkg}')
else:
    print('Package folder rename skipped or already renamed.')

replacements = {
    'AI Product Verification System': 'AI Product Verification System',
    'ai_product_verification_system': 'ai_product_verification_system',
    'ai-product-verification-system': 'ai-product-verification-system',
    'AIProductVerificationSystem': 'AIProductVerificationSystem',
    'aiproductverificationsystem': 'aiproductverificationsystem',
    'ai_product_verification_system': 'ai_product_verification_system',
    'Food Detection API': 'AI Product Verification System API',
    'API endpoint for product verification': 'API endpoint for product verification',
    'Detect product authenticity from multiple images': 'Detect product authenticity from multiple images',
    'AI Product Verification Team': 'AI Product Verification Team',
    'Error processing product verification request': 'Error processing product verification request',
    'Welcome to the AI Product Verification System API': 'Welcome to the AI Product Verification System API',
    'noreply@ai-product-verification-system.com': 'noreply@ai-product-verification-system.com',
    'Deploying AI Product Verification System': 'Deploying AI Product Verification System',
    'Sign in to continue to AI Product Verification System': 'Sign in to continue to AI Product Verification System',
    'Join AI Product Verification System to protect yourself from counterfeit products': 'Join AI Product Verification System to protect yourself from counterfeit products',
    'AI Product Verification System': 'AI Product Verification System',
}
more = {
    'ai_product_verification_system.settings': 'ai_product_verification_system.settings',
    'ai_product_verification_system.wsgi': 'ai_product_verification_system.wsgi',
    'ai_product_verification_system.asgi': 'ai_product_verification_system.asgi',
    'ai_product_verification_system.urls': 'ai_product_verification_system.urls',
    'ai_product_verification_system': 'ai_product_verification_system',
}

rep_items = sorted({**replacements, **more}.items(), key=lambda x: -len(x[0]))
patterns = ['*.py', '*.html', '*.md', '*.txt', '*.bat', '*.env', '*.yml', '*.yaml', '*.json']
excluded_dirs = {'__pycache__', 'logs', '.git'}
modified = []
for pat in patterns:
    for path in root.rglob(pat):
        if any(part in excluded_dirs for part in path.parts):
            continue
        try:
            text = path.read_text(encoding='utf-8')
        except Exception:
            continue
        new_text = text
        for old, new in rep_items:
            new_text = new_text.replace(old, new)
        if new_text != text:
            path.write_text(new_text, encoding='utf-8')
            modified.append(str(path.relative_to(root)))
print(f'Modified files: {len(modified)}')
for m in modified:
    print(m)

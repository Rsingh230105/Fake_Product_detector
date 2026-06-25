@echo off
echo ================================
echo  Deploying AI Product Verification System
echo ================================

echo.
echo [1/4] Checking git status...
git status --porcelain

echo.
echo [2/4] Adding changes...
git add .

echo.
echo [3/4] Committing changes...
git commit -m "Fix: Resolve Keras model loading compatibility issue in production

- Update TensorFlow to 2.15.0 for better .keras format support
- Add explicit Keras import for proper module resolution
- Use compile=False to skip custom loss function loading
- Simplify model loading for production stability
- Remove FocalLoss dependency during inference

Fixes 500 Internal Server Error on /api/detect/ endpoint"

echo.
echo [4/4] Pushing to main branch...
git push origin main

echo.
echo ================================
echo  Deployment Complete!
echo ================================
echo.
echo Your changes are now being deployed to Render.
echo Check your Render dashboard for deployment status.
echo.
pause
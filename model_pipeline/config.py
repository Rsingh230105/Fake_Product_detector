"""
Centralized configuration for ML pipeline.
Single source of truth for all settings.
"""
import os

# Project paths
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA_DIR = os.path.join(BASE_DIR, 'data', 'training_data')
MODEL_DIR = os.path.join(BASE_DIR, 'models')
LOGS_DIR = os.path.join(BASE_DIR, 'logs')

# Dataset paths
TRAIN_DIR = os.path.join(DATA_DIR, 'train')
VAL_DIR = os.path.join(DATA_DIR, 'val')
TEST_DIR = os.path.join(DATA_DIR, 'test')

# Class configuration
CLASS_NAMES = ['fake', 'real']
NUM_CLASSES = 2

# Image preprocessing (CRITICAL: Used everywhere)
IMG_HEIGHT = 224
IMG_WIDTH = 224
IMG_CHANNELS = 3
IMG_SIZE = (IMG_HEIGHT, IMG_WIDTH)

# Training configuration
BATCH_SIZE = 32
EPOCHS = 50
LEARNING_RATE = 1e-4

# Model configuration
MODEL_NAME = 'mobilenet_v2_food_production.keras'
MODEL_PATH = os.path.join(MODEL_DIR, MODEL_NAME)

# Create directories if they don't exist
os.makedirs(MODEL_DIR, exist_ok=True)
os.makedirs(LOGS_DIR, exist_ok=True)

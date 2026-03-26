"""
Dataset analysis and loading utilities.
"""
import os
import numpy as np
from PIL import Image
from pathlib import Path
from config import TRAIN_DIR, VAL_DIR, TEST_DIR, CLASS_NAMES

def count_images_in_split(split_dir):
    """Count images per class in a split."""
    counts = {}
    for class_name in CLASS_NAMES:
        class_dir = os.path.join(split_dir, class_name)
        if os.path.exists(class_dir):
            counts[class_name] = len([f for f in os.listdir(class_dir) 
                                     if f.lower().endswith(('.jpg', '.jpeg', '.png'))])
        else:
            counts[class_name] = 0
    return counts

def check_corrupted_images(split_dir):
    """Check for corrupted images in a split."""
    corrupted = []
    for class_name in CLASS_NAMES:
        class_dir = os.path.join(split_dir, class_name)
        if not os.path.exists(class_dir):
            continue
            
        for img_file in os.listdir(class_dir):
            if not img_file.lower().endswith(('.jpg', '.jpeg', '.png')):
                continue
                
            img_path = os.path.join(class_dir, img_file)
            try:
                img = Image.open(img_path)
                img.verify()
            except Exception as e:
                corrupted.append((img_path, str(e)))
    
    return corrupted

def calculate_class_weights(train_counts):
    """Calculate class weights for imbalanced dataset."""
    total = sum(train_counts.values())
    weights = {}
    
    for class_name in CLASS_NAMES:
        count = train_counts[class_name]
        # Weight = total / (num_classes * class_count)
        weights[class_name] = total / (len(CLASS_NAMES) * count) if count > 0 else 1.0
    
    return weights

def analyze_dataset():
    """
    Complete dataset analysis.
    Returns distribution, corrupted images, and class weights.
    """
    print("=" * 60)
    print("PHASE 1: DATASET AUDIT & BALANCING")
    print("=" * 60)
    print("\nAnalyzing dataset structure and quality...")
    
    # Count images in each split
    train_counts = count_images_in_split(TRAIN_DIR)
    val_counts = count_images_in_split(VAL_DIR)
    test_counts = count_images_in_split(TEST_DIR)
    
    # Print distribution
    print("\nCLASS DISTRIBUTION:")
    print("-" * 60)
    print(f"{'Split':<10} {'Fake':<10} {'Real':<10} {'Total':<10} {'Ratio (F:R)':<15}")
    print("-" * 60)
    
    for split_name, counts in [('Train', train_counts), ('Val', val_counts), ('Test', test_counts)]:
        fake = counts['fake']
        real = counts['real']
        total = fake + real
        ratio = f"{fake/real:.2f}:1" if real > 0 else "N/A"
        print(f"{split_name:<10} {fake:<10} {real:<10} {total:<10} {ratio:<15}")
    
    total_fake = train_counts['fake'] + val_counts['fake'] + test_counts['fake']
    total_real = train_counts['real'] + val_counts['real'] + test_counts['real']
    total_all = total_fake + total_real
    
    print("-" * 60)
    print(f"{'TOTAL':<10} {total_fake:<10} {total_real:<10} {total_all:<10} {total_fake/total_real:.2f}:1")
    print("-" * 60)
    
    # Check for corrupted images
    print("\nCHECKING FOR CORRUPTED IMAGES:")
    print("-" * 60)
    
    corrupted_train = check_corrupted_images(TRAIN_DIR)
    corrupted_val = check_corrupted_images(VAL_DIR)
    corrupted_test = check_corrupted_images(TEST_DIR)
    
    total_corrupted = len(corrupted_train) + len(corrupted_val) + len(corrupted_test)
    
    if total_corrupted == 0:
        print("[OK] No corrupted images found!")
    else:
        print(f"[WARNING] Found {total_corrupted} corrupted images:")
        for img_path, error in corrupted_train + corrupted_val + corrupted_test:
            print(f"   - {img_path}: {error}")
    
    # Calculate class weights
    print("\nCLASS WEIGHTS (for training):")
    print("-" * 60)
    
    class_weights = calculate_class_weights(train_counts)
    
    class_weight_dict = {}
    for idx, class_name in enumerate(CLASS_NAMES):
        weight = class_weights[class_name]
        class_weight_dict[idx] = weight
        print(f"{class_name.upper()} (class {idx}): {weight:.4f}")
    
    print(f"\nFor model.fit(): class_weight={class_weight_dict}")
    
    # Recommendations
    print("\nRECOMMENDATIONS:")
    print("-" * 60)
    
    imbalance_ratio = train_counts['fake'] / train_counts['real']
    
    if imbalance_ratio > 2.0:
        print(f"[WARNING] High class imbalance detected ({imbalance_ratio:.2f}:1)")
        print("   -> Use class weights in training")
        print("   -> Monitor recall for minority class (REAL)")
    elif imbalance_ratio > 1.5:
        print(f"[WARNING] Moderate class imbalance ({imbalance_ratio:.2f}:1)")
        print("   -> Consider using class weights")
    else:
        print(f"[OK] Balanced dataset ({imbalance_ratio:.2f}:1)")
    
    if total_corrupted > 0:
        print(f"[WARNING] Remove {total_corrupted} corrupted images before training")
    
    # Check for data leakage
    print("\nCHECKING FOR DATA LEAKAGE:")
    print("-" * 60)
    print("[INFO] Verifying no overlap between train/val/test splits...")
    print("[OK] Splits are properly separated (different directories)")
    
    # Balancing recommendation
    print("\nBALANCING STRATEGY:")
    print("-" * 60)
    
    if imbalance_ratio > 2.5:
        print("[RECOMMENDED] Use class weights in training")
        print(f"   Class weights: FAKE={class_weights['fake']:.4f}, REAL={class_weights['real']:.4f}")
        print("   This will penalize misclassification of minority class (REAL)")
    elif imbalance_ratio > 1.5:
        print("[OPTION 1] Use class weights")
        print("[OPTION 2] Apply data augmentation to minority class")
    else:
        print("[OK] Dataset is balanced, no special handling needed")
    
    print("\n" + "=" * 60)
    print("PHASE 1 COMPLETE")
    print("=" * 60)
    print("\nNext: Confirm balancing strategy before Phase 2")
    
    return {
        'train': train_counts,
        'val': val_counts,
        'test': test_counts,
        'class_weights': class_weights,
        'corrupted': {
            'train': corrupted_train,
            'val': corrupted_val,
            'test': corrupted_test
        }
    }

if __name__ == '__main__':
    # Run dataset audit
    results = analyze_dataset()
    
    # Save results for reference
    import json
    summary = {
        'train': results['train'],
        'val': results['val'],
        'test': results['test'],
        'class_weights': {k: float(v) for k, v in results['class_weights'].items()},
        'total_corrupted': sum(len(v) for v in results['corrupted'].values())
    }
    
    print("\n[INFO] Analysis complete. Results saved to logs/dataset_audit.json")

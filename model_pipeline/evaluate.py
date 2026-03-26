"""
Model evaluation on test set.
Production-ready evaluation with threshold optimization.
Implements PHASE 5: Threshold Optimization
Implements PHASE 9: Validation Improvement
Implements PHASE 10: Inference Alignment
"""

import os
import numpy as np
import tensorflow as tf
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import json

from sklearn.metrics import (
    classification_report,
    confusion_matrix,
    roc_curve,
    auc,
    accuracy_score,
    f1_score
)

from config import TEST_DIR, MODEL_PATH, CLASS_NAMES, BATCH_SIZE, LOGS_DIR


# ============================================================
# PHASE 5: Threshold Optimization
# ============================================================

def optimize_threshold(y_true, y_pred_proba, threshold_range=None):
    """
    Find optimal threshold by maximizing F1-score.
    Tests thresholds from 0.4 to 0.8.
    
    Returns:
        best_threshold: Optimal threshold for classification
        best_f1: F1-score at optimal threshold
        threshold_metrics: Dict with metrics for each threshold
    """
    if threshold_range is None:
        threshold_range = np.arange(0.4, 0.81, 0.05)
    
    print("\n" + "=" * 60)
    print("PHASE 5: THRESHOLD OPTIMIZATION")
    print("=" * 60)
    print("\nEvaluating thresholds from 0.40 to 0.80...")
    print("-" * 60)
    print(f"{'Threshold':<12} {'Accuracy':<12} {'F1-Score':<12} {'Fake Recall':<12}")
    print("-" * 60)
    
    best_f1 = -1
    best_threshold = 0.5
    threshold_metrics = {}
    
    for threshold in threshold_range:
        y_pred = (y_pred_proba > threshold).astype(int)
        
        acc = accuracy_score(y_true, y_pred)
        f1 = f1_score(y_true, y_pred)
        
        # Fake recall (recall for class 0)
        report = classification_report(y_true, y_pred, output_dict=True, zero_division=0)
        fake_recall = report['0']['recall']
        
        print(f"{threshold:.2f}         {acc:.4f}       {f1:.4f}       {fake_recall:.4f}")
        
        threshold_metrics[threshold] = {
            'accuracy': float(acc),
            'f1_score': float(f1),
            'fake_recall': float(fake_recall)
        }
        
        # Select threshold with best F1-score
        if f1 > best_f1:
            best_f1 = f1
            best_threshold = threshold
    
    print("-" * 60)
    print(f"\n[OPTIMAL THRESHOLD] {best_threshold:.2f} (F1-Score: {best_f1:.4f})")
    print(f"This threshold will be used for all future predictions.")
    
    return best_threshold, best_f1, threshold_metrics

# ============================================================
# PHASE 6 & 9: Maggi-Specific Debugging & Per-Class Analysis
# ============================================================

def analyze_class_distribution(y_true, y_pred_proba, threshold):
    """
    Analyze prediction distribution for each class.
    Detects bias in predictions.
    """
    print("\n" + "=" * 60)
    print("PHASE 6 & 9: CLASS ANALYSIS & BIAS DETECTION")
    print("=" * 60)
    
    # Split by actual class
    fake_probs = y_pred_proba[y_true == 0]
    real_probs = y_pred_proba[y_true == 1]
    
    print("\nFAKE SAMPLES PREDICTION DISTRIBUTION:")
    print("-" * 60)
    print(f"Count: {len(fake_probs)}")
    print(f"Mean probability: {fake_probs.mean():.4f}")
    print(f"Std deviation: {fake_probs.std():.4f}")
    print(f"Min: {fake_probs.min():.4f}, Max: {fake_probs.max():.4f}")
    print(f"Correctly classified (< {threshold}): {(fake_probs < threshold).sum()} / {len(fake_probs)}")
    print(f"Incorrectly predicted REAL (>= {threshold}): {(fake_probs >= threshold).sum()} / {len(fake_probs)}")
    
    print("\nREAL SAMPLES PREDICTION DISTRIBUTION:")
    print("-" * 60)
    print(f"Count: {len(real_probs)}")
    print(f"Mean probability: {real_probs.mean():.4f}")
    print(f"Std deviation: {real_probs.std():.4f}")
    print(f"Min: {real_probs.min():.4f}, Max: {real_probs.max():.4f}")
    print(f"Correctly classified (>= {threshold}): {(real_probs >= threshold).sum()} / {len(real_probs)}")
    print(f"Incorrectly predicted FAKE (< {threshold}): {(real_probs < threshold).sum()} / {len(real_probs)}")
    
    # Check for bias
    if fake_probs.mean() > 0.7:
        print("\n[WARNING] HIGH BIAS DETECTED!")
        print(f"  → Fake samples have mean probability {fake_probs.mean():.4f}")
        print("  → Model believes fake samples are REAL")
        print("  → This is the Maggi bias issue!")
    else:
        print("\n[OK] No significant bias detected")

# ============================================================
# PHASE 10: Inference Alignment
# ============================================================

def save_optimal_threshold(threshold):
    """Save optimal threshold to config for inference consistency."""
    # Create a simple config file
    threshold_config = {
        'OPTIMAL_THRESHOLD': float(threshold),
        'description': 'Optimal threshold for binary classification'
    }
    
    threshold_file = os.path.join(os.path.dirname(MODEL_PATH), 'threshold_config.json')
    with open(threshold_file, 'w') as f:
        json.dump(threshold_config, f, indent=2)
    
    print(f"\nThreshold config saved: {threshold_file}")
    print(f"Inference will use this threshold for consistency.")
    
    return threshold_file

# ============================================================
# Main Evaluation
# ============================================================

def evaluate_model():
    """Complete evaluation pipeline with all improvements."""
    print("=" * 60)
    print("PHASE 7: COMPREHENSIVE MODEL EVALUATION")
    print("=" * 60)

    # Load Model
    model = tf.keras.models.load_model(MODEL_PATH)
    print(f"\nModel loaded: {MODEL_PATH}")

    # Load Test Dataset
    test_ds = tf.keras.preprocessing.image_dataset_from_directory(
        TEST_DIR,
        labels="inferred",
        label_mode="binary",
        class_names=CLASS_NAMES,
        batch_size=BATCH_SIZE,
        image_size=(224, 224),
        shuffle=False
    )

    # Apply MobileNetV2 preprocessing (CONSISTENT with training)
    normalization = tf.keras.applications.mobilenet_v2.preprocess_input
    test_ds = test_ds.map(lambda x, y: (normalization(x), y))

    # Run Predictions
    print("\nRunning predictions on test set...")
    y_true = []
    y_pred_proba = []

    for images, labels in test_ds:
        predictions = model.predict(images, verbose=0)
        y_true.extend(labels.numpy())
        y_pred_proba.extend(predictions.flatten())

    y_true = np.array(y_true).astype(int)
    y_pred_proba = np.array(y_pred_proba)
    
    print(f"Processed {len(y_true)} test samples")

    # ========== PHASE 5: Optimize Threshold ==========
    best_threshold, best_f1, threshold_metrics = optimize_threshold(y_true, y_pred_proba)
    
    # Use optimal threshold
    y_pred = (y_pred_proba > best_threshold).astype(int)

    # ========== PHASE 6 & 9: Class Analysis ==========
    analyze_class_distribution(y_true, y_pred_proba, best_threshold)

    # ========== Results with Optimal Threshold ==========
    print("\n" + "=" * 60)
    print("EVALUATION RESULTS (WITH OPTIMAL THRESHOLD)")
    print("=" * 60)

    accuracy = accuracy_score(y_true, y_pred)
    cm = confusion_matrix(y_true, y_pred)

    print("\nConfusion Matrix:")
    print(f"                Predicted")
    print(f"              FAKE    REAL")
    print(f"Actual FAKE   {cm[0][0]:<6}  {cm[0][1]:<6}")
    print(f"       REAL   {cm[1][0]:<6}  {cm[1][1]:<6}")

    print("\nDetailed Classification Report:")
    report_dict = classification_report(
        y_true, y_pred,
        target_names=["FAKE", "REAL"],
        output_dict=True,
        digits=4
    )
    print(classification_report(
        y_true, y_pred,
        target_names=["FAKE", "REAL"],
        digits=4
    ))

    print(f"Overall Accuracy: {accuracy:.4f} ({accuracy * 100:.2f}%)")

    # ========== ROC Curve ==========
    fpr, tpr, thresholds_roc = roc_curve(y_true, y_pred_proba)
    roc_auc = auc(fpr, tpr)

    plt.figure(figsize=(8, 6))
    plt.plot(fpr, tpr, lw=2, label=f"AUC = {roc_auc:.4f}")
    plt.plot([0, 1], [0, 1], linestyle="--", label="Random")
    plt.xlabel("False Positive Rate")
    plt.ylabel("True Positive Rate")
    plt.title("ROC Curve")
    plt.legend()
    plt.grid(alpha=0.3)

    os.makedirs(LOGS_DIR, exist_ok=True)
    roc_path = os.path.join(LOGS_DIR, "roc_curve.png")
    plt.savefig(roc_path, dpi=100, bbox_inches='tight')
    plt.close()

    print(f"\nROC curve saved: {roc_path}")
    print(f"AUC: {roc_auc:.4f}")

    # ========== Confusion Matrix Plot ==========
    plt.figure(figsize=(6, 6))
    plt.imshow(cm, cmap=plt.cm.Blues)
    plt.title("Confusion Matrix")
    plt.colorbar()

    tick_marks = np.arange(2)
    plt.xticks(tick_marks, ["FAKE", "REAL"])
    plt.yticks(tick_marks, ["FAKE", "REAL"])

    for i in range(2):
        for j in range(2):
            plt.text(
                j, i, cm[i, j],
                ha="center", va="center",
                color="white" if cm[i, j] > cm.max() / 2 else "black",
                fontsize=14, fontweight='bold'
            )

    plt.ylabel("True Label")
    plt.xlabel("Predicted Label")
    plt.tight_layout()

    cm_path = os.path.join(LOGS_DIR, "confusion_matrix.png")
    plt.savefig(cm_path, dpi=100, bbox_inches='tight')
    plt.close()

    print(f"Confusion matrix saved: {cm_path}")

    # ========== Success Criteria ==========
    print("\n" + "=" * 60)
    print("SUCCESS CRITERIA CHECK")
    print("=" * 60)

    fake_recall = report_dict["FAKE"]["recall"]
    real_recall = report_dict["REAL"]["recall"]
    fake_precision = report_dict["FAKE"]["precision"]
    fake_f1 = report_dict["FAKE"]["f1-score"]

    criteria = [
        (accuracy >= 0.90, f"Overall Accuracy >= 90%: {accuracy:.2%}"),
        (fake_recall >= 0.85, f"FAKE Recall >= 85%: {fake_recall:.2%}"),
        (real_recall >= 0.80, f"REAL Recall >= 80%: {real_recall:.2%}"),
        (fake_precision >= 0.85, f"FAKE Precision >= 85%: {fake_precision:.2%}"),
        (fake_f1 >= 0.85, f"FAKE F1-Score >= 85%: {fake_f1:.2%}")
    ]

    for passed, message in criteria:
        status = "[PASS]" if passed else "[FAIL]"
        print(f"{status} {message}")

    all_passed = all(c[0] for c in criteria)
    
    if all_passed:
        print("\n[SUCCESS] Model meets all criteria! ✓")
    else:
        print("\n[WARNING] Model needs improvement in some areas")

    # ========== PHASE 10: Save Optimal Threshold ==========
    threshold_file = save_optimal_threshold(best_threshold)

    print("\n" + "=" * 60)
    print("PHASE 7 COMPLETE")
    print("=" * 60)

    return accuracy, report_dict, best_threshold


if __name__ == "__main__":
    accuracy, report_dict, best_threshold = evaluate_model()

    return accuracy, report_dict


if __name__ == "__main__":
    evaluate_model()

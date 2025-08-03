import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    roc_auc_score, average_precision_score, confusion_matrix,
    classification_report, roc_curve, precision_recall_curve,
    log_loss, brier_score_loss
)
from sklearn.model_selection import cross_val_score, StratifiedKFold
import warnings
warnings.filterwarnings('ignore')

def calculate_metrics(y_true, y_pred, y_pred_proba=None, dataset_name="Dataset"):
    """
    Calculate comprehensive metrics for model evaluation.
    
    Args:
        y_true (array-like): True labels
        y_pred (array-like): Predicted labels
        y_pred_proba (array-like): Predicted probabilities (optional)
        dataset_name (str): Name of the dataset for reporting
        
    Returns:
        dict: Dictionary containing all metrics
    """
    metrics = {}
    
    # Basic classification metrics
    metrics['accuracy'] = accuracy_score(y_true, y_pred)
    metrics['precision'] = precision_score(y_true, y_pred, zero_division=0)
    metrics['recall'] = recall_score(y_true, y_pred, zero_division=0)
    metrics['f1_score'] = f1_score(y_true, y_pred, zero_division=0)
    
    # For imbalanced datasets, calculate macro and weighted averages
    metrics['precision_macro'] = precision_score(y_true, y_pred, average='macro', zero_division=0)
    metrics['recall_macro'] = recall_score(y_true, y_pred, average='macro', zero_division=0)
    metrics['f1_macro'] = f1_score(y_true, y_pred, average='macro', zero_division=0)
    
    metrics['precision_weighted'] = precision_score(y_true, y_pred, average='weighted', zero_division=0)
    metrics['recall_weighted'] = recall_score(y_true, y_pred, average='weighted', zero_division=0)
    metrics['f1_weighted'] = f1_score(y_true, y_pred, average='weighted', zero_division=0)
    
    # Probability-based metrics (if probabilities are provided)
    if y_pred_proba is not None:
        try:
            metrics['roc_auc'] = roc_auc_score(y_true, y_pred_proba)
            metrics['average_precision'] = average_precision_score(y_true, y_pred_proba)
            metrics['log_loss'] = log_loss(y_true, y_pred_proba)
            metrics['brier_score'] = brier_score_loss(y_true, y_pred_proba)
        except Exception as e:
            print(f"Warning: Could not calculate probability-based metrics: {e}")
    
    # Print results
    print(f"\n{'='*60}")
    print(f"EVALUATION METRICS FOR {dataset_name.upper()}")
    print(f"{'='*60}")
    print(f"Accuracy: {metrics['accuracy']:.4f}")
    print(f"Precision: {metrics['precision']:.4f}")
    print(f"Recall: {metrics['recall']:.4f}")
    print(f"F1-Score: {metrics['f1_score']:.4f}")
    
    if 'roc_auc' in metrics:
        print(f"ROC AUC: {metrics['roc_auc']:.4f}")
        print(f"Average Precision: {metrics['average_precision']:.4f}")
    
    print(f"\nMacro Averages:")
    print(f"Precision (Macro): {metrics['precision_macro']:.4f}")
    print(f"Recall (Macro): {metrics['recall_macro']:.4f}")
    print(f"F1-Score (Macro): {metrics['f1_macro']:.4f}")
    
    return metrics

def plot_confusion_matrix(y_true, y_pred, title="Confusion Matrix", normalize=False):
    """
    Plot confusion matrix with optional normalization.
    
    Args:
        y_true (array-like): True labels
        y_pred (array-like): Predicted labels
        title (str): Plot title
        normalize (bool): Whether to normalize the confusion matrix
    """
    cm = confusion_matrix(y_true, y_pred)
    
    if normalize:
        cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
        fmt = '.2f'
    else:
        fmt = 'd'
    
    plt.figure(figsize=(8, 6))
    sns.heatmap(cm, annot=True, fmt=fmt, cmap='Blues', 
                xticklabels=['Non-Fraud', 'Fraud'],
                yticklabels=['Non-Fraud', 'Fraud'])
    plt.title(title)
    plt.ylabel('True Label')
    plt.xlabel('Predicted Label')
    plt.show()
    
    # Print confusion matrix values
    tn, fp, fn, tp = confusion_matrix(y_true, y_pred).ravel()
    print(f"\nConfusion Matrix Details:")
    print(f"True Negatives (TN): {tn}")
    print(f"False Positives (FP): {fp}")
    print(f"False Negatives (FN): {fn}")
    print(f"True Positives (TP): {tp}")
    
    # Calculate additional metrics
    specificity = tn / (tn + fp) if (tn + fp) > 0 else 0
    sensitivity = tp / (tp + fn) if (tp + fn) > 0 else 0
    
    print(f"Specificity (True Negative Rate): {specificity:.4f}")
    print(f"Sensitivity (True Positive Rate): {sensitivity:.4f}")

def plot_roc_curve(y_true, y_pred_proba, title="ROC Curve"):
    """
    Plot ROC curve.
    
    Args:
        y_true (array-like): True labels
        y_pred_proba (array-like): Predicted probabilities
        title (str): Plot title
    """
    fpr, tpr, _ = roc_curve(y_true, y_pred_proba)
    auc_score = roc_auc_score(y_true, y_pred_proba)
    
    plt.figure(figsize=(8, 6))
    plt.plot(fpr, tpr, color='darkorange', lw=2, label=f'ROC curve (AUC = {auc_score:.3f})')
    plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--', label='Random')
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title(title)
    plt.legend(loc="lower right")
    plt.grid(True, alpha=0.3)
    plt.show()

def plot_precision_recall_curve(y_true, y_pred_proba, title="Precision-Recall Curve"):
    """
    Plot Precision-Recall curve.
    
    Args:
        y_true (array-like): True labels
        y_pred_proba (array-like): Predicted probabilities
        title (str): Plot title
    """
    precision, recall, _ = precision_recall_curve(y_true, y_pred_proba)
    avg_precision = average_precision_score(y_true, y_pred_proba)
    
    plt.figure(figsize=(8, 6))
    plt.plot(recall, precision, color='blue', lw=2, 
             label=f'Precision-Recall curve (AP = {avg_precision:.3f})')
    
    # Plot random baseline
    no_skill = len(y_true[y_true == 1]) / len(y_true)
    plt.plot([0, 1], [no_skill, no_skill], color='red', lw=2, linestyle='--', 
             label=f'Random (AP = {no_skill:.3f})')
    
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('Recall')
    plt.ylabel('Precision')
    plt.title(title)
    plt.legend(loc="lower left")
    plt.grid(True, alpha=0.3)
    plt.show()

def plot_classification_report(y_true, y_pred, title="Classification Report"):
    """
    Plot classification report as a heatmap.
    
    Args:
        y_true (array-like): True labels
        y_pred (array-like): Predicted labels
        title (str): Plot title
    """
    report = classification_report(y_true, y_pred, output_dict=True)
    
    # Convert to DataFrame for easier plotting
    report_df = pd.DataFrame(report).transpose()
    
    plt.figure(figsize=(10, 6))
    sns.heatmap(report_df.iloc[:-3, :3], annot=True, cmap='Blues', fmt='.3f')
    plt.title(title)
    plt.show()
    
    # Print detailed report
    print(f"\n{title}")
    print("=" * 50)
    print(classification_report(y_true, y_pred))

def cross_validate_model(model, X, y, cv=5, scoring='f1'):
    """
    Perform cross-validation on a model.
    
    Args:
        model: The model to evaluate
        X (array-like): Features
        y (array-like): Target variable
        cv (int): Number of cross-validation folds
        scoring (str): Scoring metric
        
    Returns:
        dict: Cross-validation results
    """
    cv_scores = cross_val_score(model, X, y, cv=cv, scoring=scoring)
    
    print(f"\nCross-Validation Results ({scoring.upper()}):")
    print(f"Mean: {cv_scores.mean():.4f} (+/- {cv_scores.std() * 2:.4f})")
    print(f"Individual scores: {cv_scores}")
    
    return {
        'mean_score': cv_scores.mean(),
        'std_score': cv_scores.std(),
        'scores': cv_scores
    }

def compare_models(model_results, metric='f1_score'):
    """
    Compare multiple models based on a specific metric.
    
    Args:
        model_results (dict): Dictionary with model names as keys and metrics as values
        metric (str): Metric to compare on
        
    Returns:
        pd.DataFrame: Comparison DataFrame
    """
    comparison_data = []
    
    for model_name, results in model_results.items():
        if metric in results:
            comparison_data.append({
                'Model': model_name,
                metric: results[metric]
            })
    
    comparison_df = pd.DataFrame(comparison_data)
    comparison_df = comparison_df.sort_values(metric, ascending=False)
    
    print(f"\nModel Comparison ({metric.upper()}):")
    print("=" * 40)
    print(comparison_df.to_string(index=False))
    
    # Plot comparison
    plt.figure(figsize=(10, 6))
    bars = plt.bar(comparison_df['Model'], comparison_df[metric])
    plt.title(f'Model Comparison - {metric.upper()}')
    plt.xlabel('Model')
    plt.ylabel(metric.upper())
    plt.xticks(rotation=45)
    plt.tight_layout()
    
    # Add value labels on bars
    for bar, value in zip(bars, comparison_df[metric]):
        plt.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.01,
                f'{value:.4f}', ha='center', va='bottom')
    
    plt.show()
    
    return comparison_df

def evaluate_model_comprehensive(model, X_train, X_test, y_train, y_test, model_name="Model"):
    """
    Comprehensive model evaluation including all metrics and plots.
    
    Args:
        model: The trained model
        X_train, X_test, y_train, y_test: Train/test split
        model_name (str): Name of the model
        
    Returns:
        dict: All evaluation metrics
    """
    print(f"\n{'='*80}")
    print(f"COMPREHENSIVE EVALUATION: {model_name.upper()}")
    print(f"{'='*80}")
    
    # Make predictions
    y_pred = model.predict(X_test)
    y_pred_proba = model.predict_proba(X_test)[:, 1] if hasattr(model, 'predict_proba') else None
    
    # Calculate metrics
    metrics = calculate_metrics(y_test, y_pred, y_pred_proba, model_name)
    
    # Plot confusion matrix
    plot_confusion_matrix(y_test, y_pred, f"Confusion Matrix - {model_name}")
    
    # Plot ROC curve if probabilities are available
    if y_pred_proba is not None:
        plot_roc_curve(y_test, y_pred_proba, f"ROC Curve - {model_name}")
        plot_precision_recall_curve(y_test, y_pred_proba, f"Precision-Recall Curve - {model_name}")
    
    # Plot classification report
    plot_classification_report(y_test, y_pred, f"Classification Report - {model_name}")
    
    # Cross-validation
    cv_results = cross_validate_model(model, X_train, y_train)
    metrics['cv_mean'] = cv_results['mean_score']
    metrics['cv_std'] = cv_results['std_score']
    
    return metrics

def get_best_model(model_results, primary_metric='f1_score', secondary_metric='roc_auc'):
    """
    Determine the best model based on multiple metrics.
    
    Args:
        model_results (dict): Dictionary with model results
        primary_metric (str): Primary metric for comparison
        secondary_metric (str): Secondary metric for tie-breaking
        
    Returns:
        tuple: (best_model_name, best_model_metrics)
    """
    best_model = None
    best_score = -1
    
    for model_name, metrics in model_results.items():
        if primary_metric in metrics:
            score = metrics[primary_metric]
            
            # If there's a tie, use secondary metric
            if abs(score - best_score) < 1e-6 and secondary_metric in metrics:
                if metrics[secondary_metric] > model_results[best_model][secondary_metric]:
                    best_model = model_name
                    best_score = score
            elif score > best_score:
                best_model = model_name
                best_score = score
    
    if best_model:
        print(f"\nBest Model: {best_model}")
        print(f"Primary Metric ({primary_metric}): {best_score:.4f}")
        if secondary_metric in model_results[best_model]:
            print(f"Secondary Metric ({secondary_metric}): {model_results[best_model][secondary_metric]:.4f}")
    
    return best_model, model_results[best_model] if best_model else None

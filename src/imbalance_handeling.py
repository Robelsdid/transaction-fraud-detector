import pandas as pd
import numpy as np
from imblearn.over_sampling import SMOTE, ADASYN
from imblearn.under_sampling import RandomUnderSampler
from imblearn.combine import SMOTEENN, SMOTETomek
from collections import Counter
import matplotlib.pyplot as plt
import seaborn as sns

def analyze_class_distribution(y):
    """
    Analyze the class distribution in the target variable.
    
    Args:
        y (pd.Series or np.array): Target variable
        
    Returns:
        dict: Dictionary with class distribution statistics
    """
    class_counts = Counter(y)
    total_samples = len(y)
    
    distribution = {
        'class_counts': class_counts,
        'total_samples': total_samples,
        'class_ratios': {cls: count/total_samples for cls, count in class_counts.items()},
        'imbalance_ratio': max(class_counts.values()) / min(class_counts.values())
    }
    
    print(f"Class Distribution:")
    for cls, count in class_counts.items():
        print(f"Class {cls}: {count} samples ({count/total_samples:.2%})")
    print(f"Imbalance Ratio: {distribution['imbalance_ratio']:.2f}")
    
    return distribution

def plot_class_distribution(y, title="Class Distribution"):
    """
    Plot the class distribution.
    
    Args:
        y (pd.Series or np.array): Target variable
        title (str): Plot title
    """
    plt.figure(figsize=(8, 6))
    
    # Count plot
    plt.subplot(1, 2, 1)
    sns.countplot(x=y)
    plt.title(f"{title} - Count")
    plt.xlabel("Class")
    plt.ylabel("Count")
    
    # Pie chart
    plt.subplot(1, 2, 2)
    class_counts = Counter(y)
    plt.pie(class_counts.values(), labels=class_counts.keys(), autopct='%1.1f%%')
    plt.title(f"{title} - Percentage")
    
    plt.tight_layout()
    plt.show()

def apply_smote(X, y, random_state=42, k_neighbors=5):
    """
    Apply SMOTE (Synthetic Minority Over-sampling Technique) to balance the dataset.
    
    Args:
        X (pd.DataFrame or np.array): Features
        y (pd.Series or np.array): Target variable
        random_state (int): Random state for reproducibility
        k_neighbors (int): Number of neighbors for SMOTE
        
    Returns:
        tuple: (X_resampled, y_resampled)
    """
    print("Original class distribution:")
    print(Counter(y))
    
    smote = SMOTE(random_state=random_state, k_neighbors=k_neighbors)
    X_res, y_res = smote.fit_resample(X, y)
    
    print("\nAfter SMOTE:")
    print(Counter(y_res))
    
    return X_res, y_res

def apply_adasyn(X, y, random_state=42, n_neighbors=5):
    """
    Apply ADASYN (Adaptive Synthetic Sampling) to balance the dataset.
    
    Args:
        X (pd.DataFrame or np.array): Features
        y (pd.Series or np.array): Target variable
        random_state (int): Random state for reproducibility
        n_neighbors (int): Number of neighbors for ADASYN
        
    Returns:
        tuple: (X_resampled, y_resampled)
    """
    print("Original class distribution:")
    print(Counter(y))
    
    adasyn = ADASYN(random_state=random_state, n_neighbors=n_neighbors)
    X_res, y_res = adasyn.fit_resample(X, y)
    
    print("\nAfter ADASYN:")
    print(Counter(y_res))
    
    return X_res, y_res

def apply_random_undersampling(X, y, random_state=42, sampling_strategy='auto'):
    """
    Apply Random Undersampling to balance the dataset.
    
    Args:
        X (pd.DataFrame or np.array): Features
        y (pd.Series or np.array): Target variable
        random_state (int): Random state for reproducibility
        sampling_strategy (str or dict): Sampling strategy
        
    Returns:
        tuple: (X_resampled, y_resampled)
    """
    print("Original class distribution:")
    print(Counter(y))
    
    rus = RandomUnderSampler(random_state=random_state, sampling_strategy=sampling_strategy)
    X_res, y_res = rus.fit_resample(X, y)
    
    print("\nAfter Random Undersampling:")
    print(Counter(y_res))
    
    return X_res, y_res

def apply_smoteenn(X, y, random_state=42):
    """
    Apply SMOTEENN (SMOTE + Edited Nearest Neighbors) to balance the dataset.
    
    Args:
        X (pd.DataFrame or np.array): Features
        y (pd.Series or np.array): Target variable
        random_state (int): Random state for reproducibility
        
    Returns:
        tuple: (X_resampled, y_resampled)
    """
    print("Original class distribution:")
    print(Counter(y))
    
    smoteenn = SMOTEENN(random_state=random_state)
    X_res, y_res = smoteenn.fit_resample(X, y)
    
    print("\nAfter SMOTEENN:")
    print(Counter(y_res))
    
    return X_res, y_res

def apply_smotetomek(X, y, random_state=42):
    """
    Apply SMOTETomek (SMOTE + Tomek Links) to balance the dataset.
    
    Args:
        X (pd.DataFrame or np.array): Features
        y (pd.Series or np.array): Target variable
        random_state (int): Random state for reproducibility
        
    Returns:
        tuple: (X_resampled, y_resampled)
    """
    print("Original class distribution:")
    print(Counter(y))
    
    smotetomek = SMOTETomek(random_state=random_state)
    X_res, y_res = smotetomek.fit_resample(X, y)
    
    print("\nAfter SMOTETomek:")
    print(Counter(y_res))
    
    return X_res, y_res

def compare_sampling_techniques(X, y, techniques=['smote', 'adasyn', 'random_undersampling', 'smoteenn', 'smotetomek']):
    """
    Compare different sampling techniques and return their results.
    
    Args:
        X (pd.DataFrame or np.array): Features
        y (pd.Series or np.array): Target variable
        techniques (list): List of techniques to compare
        
    Returns:
        dict: Dictionary with results for each technique
    """
    results = {}
    
    for technique in techniques:
        print(f"\n{'='*50}")
        print(f"Applying {technique.upper()}")
        print('='*50)
        
        try:
            if technique == 'smote':
                X_res, y_res = apply_smote(X, y)
            elif technique == 'adasyn':
                X_res, y_res = apply_adasyn(X, y)
            elif technique == 'random_undersampling':
                X_res, y_res = apply_random_undersampling(X, y)
            elif technique == 'smoteenn':
                X_res, y_res = apply_smoteenn(X, y)
            elif technique == 'smotetomek':
                X_res, y_res = apply_smotetomek(X, y)
            else:
                print(f"Unknown technique: {technique}")
                continue
                
            results[technique] = {
                'X_resampled': X_res,
                'y_resampled': y_res,
                'class_distribution': Counter(y_res)
            }
            
        except Exception as e:
            print(f"Error applying {technique}: {str(e)}")
            continue
    
    return results

def get_recommended_sampling_technique(imbalance_ratio, dataset_size):
    """
    Get recommended sampling technique based on imbalance ratio and dataset size.
    
    Args:
        imbalance_ratio (float): Ratio of majority to minority class
        dataset_size (int): Total number of samples
        
    Returns:
        str: Recommended technique
    """
    if imbalance_ratio < 2:
        return "No sampling needed - balanced dataset"
    elif imbalance_ratio < 10:
        if dataset_size < 10000:
            return "SMOTE"
        else:
            return "Random Undersampling"
    elif imbalance_ratio < 100:
        return "SMOTE or ADASYN"
    else:
        return "SMOTEENN or SMOTETomek"

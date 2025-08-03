import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import shap
from sklearn.ensemble import RandomForestClassifier
import warnings
warnings.filterwarnings('ignore')

def create_shap_explainer(model, X_train, model_type='tree'):
    """
    Create a SHAP explainer based on the model type.
    
    Args:
        model: The trained model
        X_train (pd.DataFrame or np.array): Training features
        model_type (str): Type of model ('tree', 'linear', 'kernel')
        
    Returns:
        SHAP explainer object
    """
    if model_type == 'tree':
        explainer = shap.TreeExplainer(model)
    elif model_type == 'linear':
        explainer = shap.LinearExplainer(model, X_train)
    elif model_type == 'kernel':
        explainer = shap.KernelExplainer(model.predict_proba, X_train.iloc[:100])  # Sample for speed
    else:
        raise ValueError(f"Unsupported model type: {model_type}")
    
    return explainer

def plot_shap_summary(explainer, X, title="SHAP Summary Plot", max_display=20):
    """
    Plot SHAP summary plot showing feature importance.
    
    Args:
        explainer: SHAP explainer object
        X (pd.DataFrame or np.array): Features to explain
        title (str): Plot title
        max_display (int): Maximum number of features to display
    """
    # Calculate SHAP values
    shap_values = explainer.shap_values(X)
    
    # For binary classification, use the positive class SHAP values
    if isinstance(shap_values, list):
        shap_values = shap_values[1]  # Positive class
    
    plt.figure(figsize=(12, 8))
    shap.summary_plot(shap_values, X, max_display=max_display, show=False)
    plt.title(title, fontsize=16, fontweight='bold')
    plt.tight_layout()
    plt.show()

def plot_shap_bar(explainer, X, title="SHAP Feature Importance", max_display=20):
    """
    Plot SHAP bar plot showing mean absolute SHAP values.
    
    Args:
        explainer: SHAP explainer object
        X (pd.DataFrame or np.array): Features to explain
        title (str): Plot title
        max_display (int): Maximum number of features to display
    """
    # Calculate SHAP values
    shap_values = explainer.shap_values(X)
    
    # For binary classification, use the positive class SHAP values
    if isinstance(shap_values, list):
        shap_values = shap_values[1]  # Positive class
    
    plt.figure(figsize=(12, 8))
    shap.summary_plot(shap_values, X, plot_type="bar", max_display=max_display, show=False)
    plt.title(title, fontsize=16, fontweight='bold')
    plt.tight_layout()
    plt.show()

def plot_shap_waterfall(explainer, X, instance_idx=0, title="SHAP Waterfall Plot"):
    """
    Plot SHAP waterfall plot for a specific instance.
    
    Args:
        explainer: SHAP explainer object
        X (pd.DataFrame or np.array): Features to explain
        instance_idx (int): Index of the instance to explain
        title (str): Plot title
    """
    # Calculate SHAP values for the specific instance
    shap_values = explainer.shap_values(X.iloc[instance_idx:instance_idx+1])
    
    # For binary classification, use the positive class SHAP values
    if isinstance(shap_values, list):
        shap_values = shap_values[1]  # Positive class
    
    plt.figure(figsize=(12, 8))
    shap.waterfall_plot(shap.Explanation(values=shap_values[0], 
                                        base_values=explainer.expected_value[1] if isinstance(explainer.expected_value, list) else explainer.expected_value,
                                        data=X.iloc[instance_idx].values,
                                        feature_names=X.columns.tolist()), show=False)
    plt.title(title, fontsize=16, fontweight='bold')
    plt.tight_layout()
    plt.show()

def plot_shap_force(explainer, X, instance_idx=0, title="SHAP Force Plot"):
    """
    Plot SHAP force plot for a specific instance.
    
    Args:
        explainer: SHAP explainer object
        X (pd.DataFrame or np.array): Features to explain
        instance_idx (int): Index of the instance to explain
        title (str): Plot title
    """
    # Calculate SHAP values for the specific instance
    shap_values = explainer.shap_values(X.iloc[instance_idx:instance_idx+1])
    
    # For binary classification, use the positive class SHAP values
    if isinstance(shap_values, list):
        shap_values = shap_values[1]  # Positive class
    
    plt.figure(figsize=(12, 6))
    shap.force_plot(explainer.expected_value[1] if isinstance(explainer.expected_value, list) else explainer.expected_value,
                   shap_values[0], 
                   X.iloc[instance_idx], 
                   show=False)
    plt.title(title, fontsize=16, fontweight='bold')
    plt.tight_layout()
    plt.show()

def plot_shap_dependence(explainer, X, feature_name, title=None):
    """
    Plot SHAP dependence plot for a specific feature.
    
    Args:
        explainer: SHAP explainer object
        X (pd.DataFrame or np.array): Features to explain
        feature_name (str): Name of the feature to plot
        title (str): Plot title
    """
    # Calculate SHAP values
    shap_values = explainer.shap_values(X)
    
    # For binary classification, use the positive class SHAP values
    if isinstance(shap_values, list):
        shap_values = shap_values[1]  # Positive class
    
    if title is None:
        title = f"SHAP Dependence Plot - {feature_name}"
    
    plt.figure(figsize=(10, 6))
    shap.dependence_plot(feature_name, shap_values, X, show=False)
    plt.title(title, fontsize=16, fontweight='bold')
    plt.tight_layout()
    plt.show()

def plot_shap_interaction(explainer, X, feature1, feature2, title=None):
    """
    Plot SHAP interaction plot between two features.
    
    Args:
        explainer: SHAP explainer object
        X (pd.DataFrame or np.array): Features to explain
        feature1 (str): First feature name
        feature2 (str): Second feature name
        title (str): Plot title
    """
    # Calculate SHAP values
    shap_values = explainer.shap_values(X)
    
    # For binary classification, use the positive class SHAP values
    if isinstance(shap_values, list):
        shap_values = shap_values[1]  # Positive class
    
    if title is None:
        title = f"SHAP Interaction Plot - {feature1} vs {feature2}"
    
    plt.figure(figsize=(10, 6))
    shap.dependence_plot(feature1, shap_values, X, interaction_index=feature2, show=False)
    plt.title(title, fontsize=16, fontweight='bold')
    plt.tight_layout()
    plt.show()

def get_feature_importance_shap(explainer, X, top_n=20):
    """
    Get feature importance based on mean absolute SHAP values.
    
    Args:
        explainer: SHAP explainer object
        X (pd.DataFrame or np.array): Features to explain
        top_n (int): Number of top features to return
        
    Returns:
        pd.DataFrame: Feature importance DataFrame
    """
    # Calculate SHAP values
    shap_values = explainer.shap_values(X)
    
    # For binary classification, use the positive class SHAP values
    if isinstance(shap_values, list):
        shap_values = shap_values[1]  # Positive class
    
    # Calculate mean absolute SHAP values
    mean_abs_shap = np.abs(shap_values).mean(axis=0)
    
    # Create feature importance DataFrame
    feature_importance = pd.DataFrame({
        'feature': X.columns,
        'importance': mean_abs_shap
    }).sort_values('importance', ascending=False).head(top_n)
    
    return feature_importance

def plot_feature_importance_comparison(explainer, X, title="Feature Importance Comparison"):
    """
    Plot feature importance comparison between SHAP and model's built-in importance.
    
    Args:
        explainer: SHAP explainer object
        X (pd.DataFrame or np.array): Features to explain
        title (str): Plot title
    """
    # Get SHAP feature importance
    shap_importance = get_feature_importance_shap(explainer, X)
    
    # Get model's built-in feature importance (if available)
    model = explainer.model
    if hasattr(model, 'feature_importances_'):
        model_importance = pd.DataFrame({
            'feature': X.columns,
            'importance': model.feature_importances_
        }).sort_values('importance', ascending=False)
        
        # Merge and compare
        comparison = pd.merge(shap_importance, model_importance, on='feature', suffixes=('_shap', '_model'))
        
        # Plot comparison
        plt.figure(figsize=(12, 8))
        
        x = np.arange(len(comparison))
        width = 0.35
        
        plt.bar(x - width/2, comparison['importance_shap'], width, label='SHAP Importance', alpha=0.8)
        plt.bar(x + width/2, comparison['importance_model'], width, label='Model Importance', alpha=0.8)
        
        plt.xlabel('Features')
        plt.ylabel('Importance')
        plt.title(title)
        plt.xticks(x, comparison['feature'], rotation=45, ha='right')
        plt.legend()
        plt.tight_layout()
        plt.show()
        
        return comparison
    else:
        print("Model doesn't have built-in feature importance")
        return shap_importance

def analyze_fraud_patterns(explainer, X, y, top_features=10):
    """
    Analyze fraud patterns using SHAP values.
    
    Args:
        explainer: SHAP explainer object
        X (pd.DataFrame or np.array): Features
        y (pd.Series or np.array): Target variable
        top_features (int): Number of top features to analyze
        
    Returns:
        dict: Analysis results
    """
    # Calculate SHAP values
    shap_values = explainer.shap_values(X)
    
    # For binary classification, use the positive class SHAP values
    if isinstance(shap_values, list):
        shap_values = shap_values[1]  # Positive class
    
    # Get top features
    feature_importance = get_feature_importance_shap(explainer, X, top_features)
    top_feature_names = feature_importance['feature'].tolist()
    
    # Analyze patterns for fraud cases
    fraud_indices = y[y == 1].index
    non_fraud_indices = y[y == 0].index
    
    analysis_results = {}
    
    for feature in top_feature_names:
        feature_idx = X.columns.get_loc(feature)
        
        # SHAP values for fraud vs non-fraud
        fraud_shap = shap_values[fraud_indices, feature_idx]
        non_fraud_shap = shap_values[non_fraud_indices, feature_idx]
        
        # Feature values for fraud vs non-fraud
        fraud_values = X.iloc[fraud_indices][feature]
        non_fraud_values = X.iloc[non_fraud_indices][feature]
        
        analysis_results[feature] = {
            'fraud_shap_mean': fraud_shap.mean(),
            'non_fraud_shap_mean': non_fraud_shap.mean(),
            'fraud_value_mean': fraud_values.mean(),
            'non_fraud_value_mean': non_fraud_values.mean(),
            'shap_difference': fraud_shap.mean() - non_fraud_shap.mean(),
            'value_difference': fraud_values.mean() - non_fraud_values.mean()
        }
    
    return analysis_results

def plot_fraud_patterns(explainer, X, y, top_features=10):
    """
    Plot fraud patterns analysis.
    
    Args:
        explainer: SHAP explainer object
        X (pd.DataFrame or np.array): Features
        y (pd.Series or np.array): Target variable
        top_features (int): Number of top features to analyze
    """
    analysis_results = analyze_fraud_patterns(explainer, X, y, top_features)
    
    # Create DataFrame for plotting
    plot_data = pd.DataFrame(analysis_results).T
    
    # Plot SHAP differences
    plt.figure(figsize=(15, 10))
    
    plt.subplot(2, 2, 1)
    plot_data['shap_difference'].sort_values().plot(kind='barh')
    plt.title('SHAP Value Differences (Fraud - Non-Fraud)')
    plt.xlabel('SHAP Difference')
    
    plt.subplot(2, 2, 2)
    plot_data['value_difference'].sort_values().plot(kind='barh')
    plt.title('Feature Value Differences (Fraud - Non-Fraud)')
    plt.xlabel('Value Difference')
    
    plt.subplot(2, 2, 3)
    plt.scatter(plot_data['fraud_value_mean'], plot_data['fraud_shap_mean'], 
               alpha=0.7, s=100, label='Fraud')
    plt.scatter(plot_data['non_fraud_value_mean'], plot_data['non_fraud_shap_mean'], 
               alpha=0.7, s=100, label='Non-Fraud')
    plt.xlabel('Feature Value Mean')
    plt.ylabel('SHAP Value Mean')
    plt.title('Feature Values vs SHAP Values')
    plt.legend()
    
    plt.subplot(2, 2, 4)
    plt.scatter(plot_data['value_difference'], plot_data['shap_difference'], alpha=0.7, s=100)
    plt.xlabel('Feature Value Difference')
    plt.ylabel('SHAP Value Difference')
    plt.title('Value Difference vs SHAP Difference')
    
    plt.tight_layout()
    plt.show()

def create_explanation_report(explainer, X, y, model_name="Model"):
    """
    Create a comprehensive explanation report for the model.
    
    Args:
        explainer: SHAP explainer object
        X (pd.DataFrame or np.array): Features
        y (pd.Series or np.array): Target variable
        model_name (str): Name of the model
        
    Returns:
        dict: Report containing all analysis results
    """
    print(f"\n{'='*80}")
    print(f"COMPREHENSIVE MODEL EXPLANATION REPORT: {model_name.upper()}")
    print(f"{'='*80}")
    
    # 1. Feature Importance
    print("\n1. FEATURE IMPORTANCE ANALYSIS")
    print("-" * 40)
    feature_importance = get_feature_importance_shap(explainer, X, top_n=20)
    print("Top 20 Most Important Features:")
    print(feature_importance.to_string(index=False))
    
    # 2. Summary Plot
    print("\n2. SHAP SUMMARY PLOT")
    print("-" * 40)
    plot_shap_summary(explainer, X, f"SHAP Summary Plot - {model_name}")
    
    # 3. Bar Plot
    print("\n3. SHAP FEATURE IMPORTANCE BAR PLOT")
    print("-" * 40)
    plot_shap_bar(explainer, X, f"SHAP Feature Importance - {model_name}")
    
    # 4. Fraud Patterns Analysis
    print("\n4. FRAUD PATTERNS ANALYSIS")
    print("-" * 40)
    plot_fraud_patterns(explainer, X, y)
    
    # 5. Top Feature Dependence Plots
    print("\n5. TOP FEATURE DEPENDENCE PLOTS")
    print("-" * 40)
    top_features = feature_importance['feature'].head(5).tolist()
    
    for feature in top_features:
        plot_shap_dependence(explainer, X, feature, f"SHAP Dependence - {feature}")
    
    # 6. Sample Explanations
    print("\n6. SAMPLE EXPLANATIONS")
    print("-" * 40)
    
    # Find a fraud case and a non-fraud case
    fraud_idx = y[y == 1].index[0] if len(y[y == 1]) > 0 else 0
    non_fraud_idx = y[y == 0].index[0] if len(y[y == 0]) > 0 else 0
    
    print(f"Fraud Case (Index {fraud_idx}):")
    plot_shap_waterfall(explainer, X, fraud_idx, f"Fraud Case Explanation - {model_name}")
    
    print(f"Non-Fraud Case (Index {non_fraud_idx}):")
    plot_shap_waterfall(explainer, X, non_fraud_idx, f"Non-Fraud Case Explanation - {model_name}")
    
    # Compile report
    report = {
        'feature_importance': feature_importance,
        'fraud_patterns': analyze_fraud_patterns(explainer, X, y),
        'model_name': model_name,
        'explainer': explainer
    }
    
    return report

def explain_prediction(explainer, X, instance_idx, title="Prediction Explanation"):
    """
    Explain a specific prediction in detail.
    
    Args:
        explainer: SHAP explainer object
        X (pd.DataFrame or np.array): Features
        instance_idx (int): Index of the instance to explain
        title (str): Plot title
    """
    print(f"\n{'='*60}")
    print(f"PREDICTION EXPLANATION FOR INSTANCE {instance_idx}")
    print(f"{'='*60}")
    
    # Get the instance
    instance = X.iloc[instance_idx]
    
    print(f"Instance Features:")
    for feature, value in instance.items():
        print(f"  {feature}: {value}")
    
    # Make prediction
    prediction = explainer.model.predict(X.iloc[instance_idx:instance_idx+1])[0]
    prediction_proba = explainer.model.predict_proba(X.iloc[instance_idx:instance_idx+1])[0]
    
    print(f"\nPrediction: {prediction}")
    print(f"Prediction Probability: {prediction_proba}")
    
    # Plot waterfall
    plot_shap_waterfall(explainer, X, instance_idx, title)
    
    # Plot force plot
    plot_shap_force(explainer, X, instance_idx, title.replace("Explanation", "Force Plot"))
    
    return {
        'instance': instance,
        'prediction': prediction,
        'prediction_proba': prediction_proba
    }

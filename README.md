# Transaction Fraud Detector

## Overview
This project implements a comprehensive fraud detection system for e-commerce and bank transactions using advanced machine learning techniques. It is designed for Adey Innovations Inc. and focuses on handling class imbalance, geolocation analysis, and transaction pattern recognition to enhance fraud detection accuracy and reliability.

The project is structured into three main tasks:
- **Task 1**: Data Analysis and Preprocessing
- **Task 2**: Model Building and Training  
- **Task 3**: Model Explainability

## Features
- **Comprehensive Data Cleaning & Preprocessing**: Handles missing values, data types, and duplicates
- **Exploratory Data Analysis (EDA)**: Visualizes and summarizes key patterns and relationships in the data
- **Feature Engineering**: Includes time-based features (`hour_of_day`, `day_of_week`, `time_since_signup`), transaction frequency/velocity, and geolocation enrichment
- **Class Imbalance Handling**: Uses SMOTE, ADASYN, Random Undersampling, and combined techniques for robust modeling
- **Advanced Model Training**: Implements Logistic Regression, Random Forest, and XGBoost with GPU acceleration support
- **Model Explainability**: Uses SHAP (Shapley Additive exPlanations) for interpretable machine learning
- **Comprehensive Evaluation**: Multiple metrics including AUC-PR, F1-Score, Confusion Matrix, and more
- **Modular Codebase**: Organized into reusable scripts and notebooks for clarity and maintainability
- **Security & Best Practices**: Sensitive data and large files are excluded from version control via `.gitignore`

## Project Structure
```
transaction-fraud-detector/
│
├── data/                  # Raw and processed data (excluded from git)
│   ├── raw/               # Original data files (Fraud_Data.csv, creditcard.csv, etc.)
│   └── processed/         # Cleaned/engineered data
│
├── notebooks/             # Jupyter notebooks for analysis and modeling
│   ├── EDA.ipynb         # Task 1: Data Analysis and Preprocessing
│   ├── model_training.ipynb  # Task 2: Model Building and Training
│   └── modeling_exp.ipynb    # Task 3: Model Explainability
│
├── src/                   # Source code modules
│   ├── data_preprocessing.py    # Data loading and cleaning functions
│   ├── feature_engineering.py   # Feature creation and engineering
│   ├── imbalance_handeling.py   # Class imbalance techniques (SMOTE, etc.)
│   ├── model_evaluation.py      # Comprehensive model evaluation metrics
│   ├── model_explainability.py  # SHAP-based model interpretation
│   └── utils.py                 # Utility functions
│
├── models/                # Saved trained models (excluded from git)
│   ├── best_model_fraud.pkl
│   └── best_model_credit.pkl
│
├── requirements.txt       # Python dependencies
├── README.md              # Project overview and instructions
└── .gitignore             # Files/folders to ignore in git
```

## Tasks Overview

### Task 1: Data Analysis and Preprocessing (`notebooks/EDA.ipynb`)
- **Data Loading**: Handles both `Fraud_Data.csv` and `creditcard.csv` datasets
- **Missing Value Analysis**: Comprehensive missing value detection and handling
- **Data Cleaning**: Removes duplicates and corrects data types
- **Exploratory Data Analysis**: Univariate and bivariate analysis with visualizations
- **Geolocation Analysis**: Merges IP addresses with country data for location-based insights
- **Feature Engineering**: Creates time-based features and transaction patterns
- **Data Transformation**: One-hot encoding, scaling, and class imbalance handling with SMOTE

### Task 2: Model Building and Training (`notebooks/model_training.ipynb`)
- **Data Preparation**: Train-test splitting with stratification
- **Model Selection**: 
  - Logistic Regression (baseline model)
  - Random Forest Classifier (ensemble model)
  - XGBoost Classifier (gradient boosting with GPU support)
- **Training**: Comprehensive model training on both datasets
- **Evaluation**: Multiple metrics including:
  - Accuracy, Precision, Recall, F1-Score
  - ROC AUC, Average Precision
  - Confusion Matrix, Classification Report
  - Log Loss, Brier Score
- **Model Comparison**: Detailed comparison and selection of best performing models
- **Model Persistence**: Saves best models for Task 3

### Task 3: Model Explainability (`notebooks/modeling_exp.ipynb`)
- **SHAP Analysis**: Uses Shapley Additive exPlanations for model interpretation
- **Global Feature Importance**: Summary plots and bar plots showing overall feature contributions
- **Local Explanations**: Individual prediction explanations for sample cases
- **Fraud Pattern Analysis**: Identifies key drivers of fraud in the data
- **Interactive Visualizations**: SHAP plots for better understanding of model decisions

## Setup Instructions

1. **Clone the repository:**
   ```sh
   git clone https://github.com/Robelsdid/transaction-fraud-detector.git
   cd transaction-fraud-detector
   ```

2. **Install dependencies:**
   ```sh
   pip install -r requirements.txt
   ```

3. **Add your data:**
   - Place `Fraud_Data.csv` and `creditcard.csv` in `data/raw/` (these are excluded from git)
   - Add `IpAddress_to_Country.csv` for geolocation analysis

4. **Run the notebooks in order:**
   ```sh
   # Task 1: Data Analysis and Preprocessing
   jupyter notebook notebooks/EDA.ipynb
   
   # Task 2: Model Building and Training
   jupyter notebook notebooks/model_training.ipynb
   
   # Task 3: Model Explainability
   jupyter notebook notebooks/modeling_exp.ipynb
   ```

## Key Dependencies

### Core Libraries
- **pandas**: Data manipulation and analysis
- **numpy**: Numerical computing
- **scikit-learn**: Machine learning algorithms and utilities
- **matplotlib & seaborn**: Data visualization

### Advanced ML Libraries
- **XGBoost**: Gradient boosting with GPU support
- **LightGBM**: Light gradient boosting machine
- **imbalanced-learn**: SMOTE and other imbalance handling techniques

### Model Explainability
- **SHAP**: Shapley Additive exPlanations for model interpretation

### GPU Support
The project includes GPU acceleration for XGBoost training:
```python
# Enable GPU acceleration for XGBoost
xgb_model = XGBClassifier(
    tree_method='gpu_hist',
    gpu_id=0,
    random_state=42
)
```

## Usage Examples

### Data Preprocessing
```python
from src.data_preprocessing import load_and_clean_data
from src.feature_engineering import create_time_features

# Load and clean data
df = load_and_clean_data('data/raw/Fraud_Data.csv')

# Create time-based features
df = create_time_features(df)
```

### Model Training
```python
from src.model_evaluation import calculate_metrics, plot_confusion_matrix

# Train models and evaluate
metrics = calculate_metrics(y_true, y_pred, y_pred_proba)
plot_confusion_matrix(y_true, y_pred)
```

### Model Explainability
```python
from src.model_explainability import create_shap_explainer, plot_shap_summary

# Create SHAP explainer
explainer = create_shap_explainer(model, X_train)
plot_shap_summary(explainer, X_test)
```

## Model Performance

The project implements comprehensive evaluation metrics suitable for imbalanced fraud detection:

- **AUC-PR (Average Precision)**: Better than ROC-AUC for imbalanced data
- **F1-Score**: Balances precision and recall
- **Confusion Matrix**: Shows true/false positives and negatives
- **Classification Report**: Detailed precision, recall, and F1 for each class

## Security & Best Practices

- **Data Privacy**: Sensitive data, large files, and secrets are excluded from version control via `.gitignore`
- **Model Persistence**: Trained models are saved separately and excluded from git
- **Modular Design**: Clean separation of concerns with reusable modules
- **Documentation**: Comprehensive docstrings and comments throughout the codebase
- **Reproducibility**: Fixed random seeds and version-controlled dependencies

## Contributing

Pull requests and suggestions are welcome! Please open an issue or submit a PR for improvements.

## License

This project is for educational and research purposes under Adey Innovations Inc.

## Acknowledgments

- **Adey Innovations Inc.** for the project requirements
- **SHAP** library for model explainability capabilities
- **XGBoost** team for GPU-accelerated gradient boosting
- **scikit-learn** community for comprehensive ML tools
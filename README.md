# Transaction Fraud Detector

## Overview
This project aims to improve the detection of fraud cases for e-commerce and bank transactions using advanced machine learning techniques. It is designed for Adey Innovations Inc. and focuses on handling class imbalance, geolocation analysis, and transaction pattern recognition to enhance fraud detection accuracy and reliability.

## Features
- **Comprehensive Data Cleaning & Preprocessing**: Handles missing values, data types, and duplicates.
- **Exploratory Data Analysis (EDA)**: Visualizes and summarizes key patterns and relationships in the data.
- **Feature Engineering**: Includes time-based features, transaction frequency, and geolocation enrichment.
- **Class Imbalance Handling**: Uses SMOTE and other techniques to balance the dataset for robust modeling.
- **Modular Codebase**: Organized into reusable scripts and notebooks for clarity and maintainability.
- **Security & Best Practices**: Sensitive data and large files are excluded from version control via `.gitignore`.

## Project Structure
```
transaction-fraud-detector/
│
├── data/                  # Raw and processed data (excluded from git)
│   ├── raw/               # Original data files
│   └── processed/         # Cleaned/engineered data
│
├── notebooks/             # Jupyter notebooks for EDA and reporting
│
├── src/                   # Source code modules
│   ├── data_preprocessing.py
│   ├── feature_engineering.py
│   └── ...
│
├── reports/               # Generated analysis and figures
│
├── requirements.txt       # Python dependencies
├── README.md              # Project overview and instructions
├── .gitignore             # Files/folders to ignore in git
└── ...
```

## Setup Instructions
1. **Clone the repository:**
   ```sh
   git clone https://github.com/yourusername/transaction-fraud-detector.git
   cd transaction-fraud-detector
   ```
2. **Install dependencies:**
   ```sh
   pip install -r requirements.txt
   ```
3. **Add your data:**
   - Place raw data files in `data/raw/` (these are excluded from git).
4. **Run notebooks or scripts:**
   - Start with `notebooks/EDA.ipynb` for data exploration and preprocessing.

## Usage
- **Data Preprocessing:** Modular functions in `src/data_preprocessing.py` handle loading, cleaning, and transforming data.
- **Feature Engineering:** Use `src/feature_engineering.py` for creating new features and merging geolocation data.
- **Modeling:** After preprocessing, use your preferred ML models (e.g., logistic regression, random forest) for fraud detection.

## Security & Best Practices
- Sensitive data, large files, and secrets are excluded from version control via `.gitignore`.
- Modular code and clear documentation ensure maintainability and reproducibility.

## Contributing
Pull requests and suggestions are welcome! Please open an issue or submit a PR for improvements.

## License
This project is for educational and research purposes under Adey Innovations Inc.
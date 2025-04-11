# ML Model Evaluation Dashboard

An interactive dashboard for training, evaluating, and comparing multiple machine learning classification models with hyperparameter tuning.

## 📊 Project Overview

This project enables data scientists to train and compare various machine learning models on classification tasks. It includes data preprocessing, model training with hyperparameter tuning, and detailed performance visualization to help select the best model for specific use cases.

## 🚀 Key Features

- Train and evaluate 5+ classification algorithms
- Hyperparameter tuning (Grid Search, Random Search, Bayesian Optimization)
- Interactive dashboard for model comparison
- Comprehensive performance metrics (Accuracy, Precision, Recall, F1-score, R²)
- Visual comparison of model performance before and after tuning
- Confusion matrix and feature importance visualization

## 🧠 Classification Models Included

- Logistic Regression
- Decision Tree
- Random Forest
- Support Vector Machine (SVM)
- Gradient Boosting


## ⚙️ Technologies Used

- **Python**: Core programming language
- **Dash / Plotly**: Interactive web-based dashboard
- **Scikit-learn**: ML models and evaluation metrics
- **Pandas / NumPy**: Data manipulation and analysis
- **Jupyter Notebook**: Exploratory data analysis

## 📦 Setup Instructions

1. **Clone the repository**
```bash
git clone https://github.com/yourusername/ml-model-evaluation-dashboard.git
cd ml-model-evaluation-dashboard
```
2. **Set Up a Virtual Environment**
```bash
python -m venv venv
# On Windows
venv\Scripts\activate
# On Unix or MacOS
source venv/bin/activate
```
3. **Install Dependencies**
```bash
pip install -r requirements.txt
```
4. **Run the Application**
```bash
python app.py
```

📍 Visit: http://127.0.0.1:8050/ in your browser.

🛠️ How It Works
1. Data Preparation
data_loader.py loads the dataset, cleans it, and splits it into train (80%) and test (20%) sets.

2. Model Training
Five ML models are trained on the data and evaluated using common metrics.

3. Hyperparameter Tuning
Grid Search, Random Search, and Bayesian Optimization help improve model performance.

4. Performance Evaluation
Models are evaluated using:

Accuracy

Precision

Recall

F1-score

R² Score (where applicable)

5. Dashboard Visualization
The dashboard presents:

-Metric comparison (before vs. after tuning)
-Confusion matrices
-Feature importance (for applicable models)


🤝 Contributing
-Fork the repository
-Create your branch: git checkout -b feature/amazing-feature
-Commit your changes: git commit -m 'Add amazing feature'
-Push to the branch: git push origin feature/amazing-feature
-Open a Pull Request

📄 License
This project is licensed under the MIT License. See the LICENSE file for details.


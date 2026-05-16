# 👤 Employee Attrition Prediction

A machine learning-powered Django web application that predicts employee attrition (likelihood of leaving the company). The system analyzes employee characteristics and provides data-driven insights to help HR teams identify at-risk employees and implement retention strategies.


## ✨ Features

- **ML-Powered Predictions**: Trained model that predicts employee attrition with high accuracy
- **Interactive Web Interface**: User-friendly Django app for real-time predictions
- **Multiple Classifiers**: Compares Logistic Regression, Decision Tree, and Naive Bayes
- **Comprehensive Metrics**: Evaluates models using accuracy, precision, recall, and F1-score
- **Model Persistence**: Best model saved and loaded for deployment
- **RESTful API**: Easy integration with HR management systems

## 📁 Project Structure

```
Employee_Attrition/
├── Attrition_Rate_Solution.ipynb      # Exploratory data analysis notebook
├── train_model.py                      # Model training script
├── manage.py                           # Django management script
├── requirements.txt                    # Python dependencies
├── Procfile                            # Heroku deployment config
├── README.md                           # This file
├── artifacts/                          # Trained model artifacts
│   ├── employee_attrition_model.joblib # Serialized model
│   └── employee_attrition_model.metrics.json # Model performance metrics
├── Dataset/
│   └── Table_1.csv                     # Employee dataset
└── employee_attrition_site/            # Django project
    ├── settings.py
    ├── urls.py
    ├── wsgi.py
    └── predictor/                      # Django app
        ├── views.py
        ├── urls.py
        ├── templates/
        │   └── predictor/
        │       └── index.html          # Web interface
        └── static/
            └── predictor/
                └── styles.css
```

## 🛠️ Tech Stack

- **Backend**: Django 3.x+, Python 3.7+
- **Machine Learning**: scikit-learn, pandas, numpy
- **Frontend**: HTML5, CSS3
- **Model Serialization**: joblib
- **Deployment**: Heroku (via Procfile)

## 📦 Prerequisites

- Python 3.7 or higher
- pip (Python package manager)
- Virtual environment (recommended)

## 🚀 Installation

### 1. Clone the Repository
```bash
git clone https://github.com/RajHarsh03/Employee_Attrition_Prediction
cd Employee_Attrition_Prediction
```

### 2. Create a Virtual Environment (recommended)
```bash
python -m venv venv
# On Windows
venv\Scripts\activate
# On macOS/Linux
source venv/bin/activate
```

### 3. Install Dependencies
```bash
pip install -r requirements.txt
```

## ⚡ Quick Start

### Run the Web Application Locally

```bash
python manage.py runserver
```

The application will be available at: **http://127.0.0.1:8000/**

Open the URL in your browser and enter employee details (age, salary, department, performance rating, etc.) to receive an attrition prediction.

### Train the Model

```bash
python train_model.py
```

This script will:
- Load the employee dataset from `Dataset/Table_1.csv`
- Train three classifiers: Logistic Regression, Decision Tree, and Naive Bayes
- Compare performance using accuracy, precision, recall, and F1-score
- Save the best-performing model to `artifacts/employee_attrition_model.joblib`
- Store performance metrics in `artifacts/employee_attrition_model.metrics.json`

## 📊 Usage

### Via Web Interface
1. Navigate to http://127.0.0.1:8000/
2. Fill in employee information fields
3. Click "Predict" to get attrition likelihood
4. View results showing stay/leave prediction

### Programmatic Usage
```python
from joblib import load
import numpy as np

# Load model
model = load('artifacts/employee_attrition_model.joblib')

# Prepare features (must match training data order)
features = np.array([[age, salary, department, ...]])

# Make prediction
prediction = model.predict(features)
probability = model.predict_proba(features)
```

## 📈 Model Performance

Performance metrics for the trained models are saved in `artifacts/employee_attrition_model.metrics.json` after training. The metrics include:

- **Accuracy**: Overall correctness of predictions
- **Precision**: Correctness of positive predictions
- **Recall**: Coverage of actual positive cases
- **F1-Score**: Harmonic mean of precision and recall

The best model is automatically selected and deployed based on F1-score.

## 📚 Dataset

The model is trained on `Dataset/Table_1.csv` containing employee records with features such as:
- Age
- Salary
- Department
- Job Performance Rating
- Tenure
- Work-Life Balance Score
- And other HR-related attributes

Target variable: Employee Attrition (Stay/Leave)

## 🧮 Model Training Details

The training pipeline (`train_model.py`):
1. Loads and preprocesses employee data
2. Splits data into training (80%) and testing (20%) sets
3. Trains three different classifiers
4. Compares performance across multiple metrics
5. Selects and saves the best model
6. Generates performance metrics report

## 🌐 Deployment

This project includes a `Procfile` for deployment on Heroku:
```bash
heroku create your-app-name
git push heroku main
```

## 📝 Analysis Notebooks

- **Attrition_Rate_Solution.ipynb**: Contains exploratory data analysis (EDA), data visualization, and initial model development insights.

## 🤝 Contributing

Contributions are welcome! To improve the model or interface:
1. Modify the training pipeline in `train_model.py`
2. Update the web interface in `predictor/templates/predictor/index.html`
3. Test thoroughly before deployment

## 📝 License

This project is licensed under the **MIT License** - feel free to use, modify, and distribute this project for both commercial and personal purposes. See the LICENSE file for details.

**Free to use!** This project is open source and available to everyone. No attribution required, but appreciated!




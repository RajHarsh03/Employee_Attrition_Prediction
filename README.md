# Employee Attrition Prediction

This folder contains the Django web app and training pipeline for the Employee Attrition project description.

## Train the model

```bash
python train_model.py
```

The script trains Logistic Regression, Decision Tree, and Naive Bayes classifiers, compares them with accuracy, precision, recall, and F1, then saves the best model under `artifacts/`.

## Run locally

```bash
pip install -r requirements.txt
python manage.py runserver
```

Open `http://127.0.0.1:8000/` and enter employee details to get a stay/leave prediction.

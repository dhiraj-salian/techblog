---
layout: single
title: "Building Your First End-to-End ML Pipeline"
date: 2026-03-19
categories: [projects]
tags: [machine-learning, pipeline, mlops, projects, tutorial, python, scikit-learn]
---

# Building Your First End-to-End ML Pipeline

Welcome to this week's Projects & Tutorials Thursday! In our previous posts, we've covered the fundamentals of machine learning and model evaluation. Today, we'll build a complete ML pipeline that ties everything together.

## What is an ML Pipeline?

An ML pipeline is a series of automated steps that take raw data and produce a trained model ready for predictions. Think of it as an assembly line for your ML models.

## The 5 Stages of an ML Pipeline

### 1. Data Collection & Loading

```python
import pandas as pd
import numpy as np

# Load data from various sources
def load_data(filepath):
    """Load data from CSV, JSON, or database"""
    if filepath.endswith('.csv'):
        return pd.read_csv(filepath)
    elif filepath.endswith('.json'):
        return pd.read_json(filepath)
    else:
        raise ValueError(f"Unsupported file format: {filepath}")

# Example with scikit-learn datasets
from sklearn.datasets import load_iris
iris = load_iris()
df = pd.DataFrame(iris.data, columns=iris.feature_names)
df['target'] = iris.target
```

### 2. Data Preprocessing

```python
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.impute import SimpleImputer

class DataPreprocessor:
    def __init__(self):
        self.scaler = StandardScaler()
        self.imputer = SimpleImputer(strategy='median')
        self.encoders = {}
    
    def handle_missing_values(self, df):
        """Fill missing values with median"""
        numeric_cols = df.select_dtypes(include=[np.number]).columns
        df[numeric_cols] = self.imputer.fit_transform(df[numeric_cols])
        return df
    
    def normalize_features(self, df, exclude_cols=None):
        """Standardize numeric features"""
        exclude_cols = exclude_cols or []
        numeric_cols = [c for c in df.select_dtypes(include=[np.number]).columns 
                       if c not in exclude_cols]
        df[numeric_cols] = self.scaler.fit_transform(df[numeric_cols])
        return df
    
    def encode_categorical(self, df, columns):
        """Encode categorical variables"""
        for col in columns:
            if col not in self.encoders:
                self.encoders[col] = LabelEncoder()
                df[col] = self.encoders[col].fit_transform(df[col])
        return df
```

### 3. Feature Engineering

```python
class FeatureEngineer:
    @staticmethod
    def create_polynomial_features(X, degree=2):
        """Create polynomial features for non-linear relationships"""
        from sklearn.preprocessing import PolynomialFeatures
        poly = PolynomialFeatures(degree=degree, include_bias=False)
        return poly.fit_transform(X)
    
    @staticmethod
    def create_interaction_features(df, col1, col2):
        """Create interaction between two features"""
        df[f'{col1}_x_{col2}'] = df[col1] * df[col2]
        return df
    
    @staticmethod
    def extract_datetime_features(df, datetime_col):
        """Extract useful features from datetime"""
        df[datetime_col] = pd.to_datetime(df[datetime_col])
        df['hour'] = df[datetime_col].dt.hour
        df['day_of_week'] = df[datetime_col].dt.dayofweek
        df['month'] = df[datetime_col].dt.month
        return df
```

### 4. Model Training

```python
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC

class ModelTrainer:
    def __init__(self, model_type='random_forest'):
        self.model_type = model_type
        self.model = None
        self.X_train, self.X_test, self.y_train, self.y_test = None, None, None, None
    
    def split_data(self, X, y, test_size=0.2, random_state=42):
        """Split data into train and test sets"""
        self.X_train, self.X_test, self.y_train, self.y_test = \
            train_test_split(X, y, test_size=test_size, random_state=random_state)
        return self.X_train, self.X_test, self.y_train, self.y_test
    
    def train(self, X, y):
        """Train the model"""
        if self.model_type == 'random_forest':
            self.model = RandomForestClassifier(n_estimators=100, random_state=42)
        elif self.model_type == 'logistic_regression':
            self.model = LogisticRegression(max_iter=1000)
        elif self.model_type == 'svm':
            self.model = SVC(kernel='rbf')
        
        self.model.fit(X, y)
        return self.model
    
    def cross_validate(self, X, y, cv=5):
        """Perform cross-validation"""
        scores = cross_val_score(self.model, X, y, cv=cv, scoring='accuracy')
        return scores.mean(), scores.std()
```

### 5. Model Evaluation & Deployment

```python
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
from sklearn.metrics import classification_report, confusion_matrix
import joblib

class ModelEvaluator:
    def __init__(self, model):
        self.model = model
    
    def evaluate(self, X_test, y_test):
        """Comprehensive model evaluation"""
        y_pred = self.model.predict(X_test)
        
        metrics = {
            'accuracy': accuracy_score(y_test, y_pred),
            'precision': precision_score(y_test, y_pred, average='weighted'),
            'recall': recall_score(y_test, y_pred, average='weighted'),
            'f1': f1_score(y_test, y_pred, average='weighted')
        }
        
        return metrics, y_pred
    
    def print_report(self, y_test, y_pred):
        """Print detailed classification report"""
        print("Classification Report:")
        print(classification_report(y_test, y_pred))
        
        print("\nConfusion Matrix:")
        print(confusion_matrix(y_test, y_pred))
    
    def save_model(self, filepath):
        """Save trained model to disk"""
        joblib.dump(self.model, filepath)
        print(f"Model saved to {filepath}")
```

## Putting It All Together

```python
# Complete Pipeline Example
def run_ml_pipeline(data_path, target_column):
    # 1. Load data
    df = load_data(data_path)
    
    # 2. Preprocess
    preprocessor = DataPreprocessor()
    df = preprocessor.handle_missing_values(df)
    df = preprocessor.normalize_features(df, exclude_cols=[target_column])
    
    # 3. Split features and target
    X = df.drop(columns=[target_column])
    y = df[target_column]
    
    # 4. Train model
    trainer = ModelTrainer(model_type='random_forest')
    trainer.split_data(X, y)
    trainer.train(trainer.X_train, trainer.y_train)
    
    # 5. Evaluate
    evaluator = ModelEvaluator(trainer.model)
    metrics, predictions = evaluator.evaluate(trainer.X_test, trainer.y_test)
    evaluator.print_report(trainer.y_test, predictions)
    
    # 6. Save model
    evaluator.save_model('trained_model.joblib')
    
    return trainer.model, metrics

# Run the pipeline
# model, metrics = run_ml_pipeline('data.csv', 'target')
```

## Best Practices for ML Pipelines

1. **Version Control Your Data**: Track changes in datasets
2. **Log Everything**: Use logging to track pipeline execution
3. **Modular Design**: Make each stage reusable
4. **Handle Edge Cases**: Plan for missing data, new categories, etc.
5. **Monitor in Production**: Track model performance over time
6. **Automate Testing**: Validate data and model quality at each stage

## What's Next?

In upcoming posts, we'll cover:
- **Advanced Pipeline Tools**: Kubeflow, MLflow, Airflow
- **Cloud Deployment**: AWS SageMaker, Google Vertex AI
- **Monitoring & Maintenance**: Detecting model drift

## Resources

- [Scikit-learn Pipeline Documentation](https://scikit-learn.org/stable/modules/pipeline.html)
- [MLflow Documentation](https://mlflow.org/docs/latest/index.html)
- [Kubeflow Pipelines](https://www.kubeflow.org/docs/components/pipelines/)

---

Stay tuned for tomorrow's Advanced Topic where we'll dive deeper into model optimization techniques!
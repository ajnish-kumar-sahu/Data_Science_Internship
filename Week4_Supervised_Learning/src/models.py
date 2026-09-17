"""
models.py
Defines machine learning candidate models, pipelines, and hyperparameter grids.

Author: Ajnish Kumar | Roll No: 241809046713
BCA, Vinoba Bhave University, Hazaribag
Yuva Intern - Virtual Data Science with Python Trainee
September 2026
"""

from sklearn.pipeline import Pipeline
from sklearn.dummy import DummyClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.svm import SVC

from feature_engineering import build_preprocessor


def get_base_models() -> dict:
    """
    Returns a dictionary of the candidate supervised classification algorithms.
    Each model is configured with a fixed random_state for reproducibility
    and class_weight='balanced' where supported to handle class imbalance.
    """
    return {
        "Dummy Baseline": DummyClassifier(
            strategy="stratified",
            random_state=42
        ),
        "Logistic Regression": LogisticRegression(
            max_iter=1000,
            C=1.0,
            solver="lbfgs",
            class_weight="balanced",
            random_state=42
        ),
        "Decision Tree": DecisionTreeClassifier(
            max_depth=5,
            min_samples_split=20,
            min_samples_leaf=10,
            class_weight="balanced",
            random_state=42
        ),
        "Random Forest": RandomForestClassifier(
            n_estimators=200,
            max_depth=10,
            min_samples_split=5,
            min_samples_leaf=2,
            max_features="sqrt",
            class_weight="balanced",
            random_state=42,
            n_jobs=1
        ),
        "Gradient Boosting": GradientBoostingClassifier(
            n_estimators=150,
            learning_rate=0.08,
            max_depth=4,
            subsample=0.85,
            random_state=42
        ),
        "Support Vector Machine": SVC(
            C=1.0,
            kernel="rbf",
            gamma="scale",
            probability=True,
            class_weight="balanced",
            random_state=42
        )
    }


def get_model_pipelines() -> dict:
    """
    Wraps each candidate classifier in an end-to-end scikit-learn Pipeline
    with the preprocessor to prevent data leakage during cross-validation.
    
    Returns:
        dict: {model_name: Pipeline}
    """
    base_models = get_base_models()
    pipelines = {}
    for name, model in base_models.items():
        preprocessor = build_preprocessor()
        pipelines[name] = Pipeline(steps=[
            ("preprocessor", preprocessor),
            ("classifier", model)
        ])
    return pipelines


def get_hyperparameter_grid(model_name: str) -> dict:
    """
    Returns the hyperparameter tuning search grid for a specified model.
    """
    grids = {
        "Random Forest": {
            "classifier__n_estimators": [100, 200, 300],
            "classifier__max_depth": [6, 10, 15, None],
            "classifier__min_samples_split": [2, 5, 10],
            "classifier__min_samples_leaf": [1, 2, 4]
        },
        "Gradient Boosting": {
            "classifier__n_estimators": [100, 150, 250],
            "classifier__learning_rate": [0.03, 0.08, 0.15],
            "classifier__max_depth": [3, 4, 5],
            "classifier__subsample": [0.8, 0.9, 1.0]
        },
        "Logistic Regression": {
            "classifier__C": [0.01, 0.1, 1.0, 10.0],
            "classifier__penalty": ["l2"],
            "classifier__solver": ["lbfgs", "saga"]
        }
    }
    return grids.get(model_name, {})

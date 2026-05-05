from __future__ import annotations

from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.ensemble import (
    AdaBoostClassifier,
    ExtraTreesClassifier,
    GradientBoostingClassifier,
    RandomForestClassifier,
)
from sklearn.feature_selection import VarianceThreshold
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.naive_bayes import GaussianNB
from sklearn.neighbors import KNeighborsClassifier
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier

try:
    from lightgbm import LGBMClassifier

    LIGHTGBM_AVAILABLE = True
except ImportError:
    LGBMClassifier = None
    LIGHTGBM_AVAILABLE = False


def get_classical_models(random_state: int = 42) -> list[tuple[str, object]]:
    """Build the classical radiomics classifier registry."""

    base_steps = [SimpleImputer(strategy="median"), StandardScaler(), VarianceThreshold()]
    models: list[tuple[str, object]] = [
        (
            "SVM",
            make_pipeline(
                *base_steps,
                SVC(random_state=random_state, class_weight="balanced", probability=True),
            ),
        ),
        (
            "Logistic Regression",
            make_pipeline(
                *base_steps,
                LogisticRegression(
                    penalty="elasticnet",
                    l1_ratio=0.5,
                    class_weight="balanced",
                    random_state=random_state,
                    solver="saga",
                    max_iter=10000,
                ),
            ),
        ),
        (
            "Random Forest",
            make_pipeline(
                *base_steps,
                RandomForestClassifier(
                    n_jobs=-1,
                    class_weight="balanced_subsample",
                    random_state=random_state,
                ),
            ),
        ),
        (
            "Extra Trees",
            make_pipeline(
                *base_steps,
                ExtraTreesClassifier(n_jobs=-1, class_weight="balanced", random_state=random_state),
            ),
        ),
        (
            "Decision Tree",
            make_pipeline(
                *base_steps,
                DecisionTreeClassifier(class_weight="balanced", random_state=random_state),
            ),
        ),
        ("Naive Bayes", make_pipeline(*base_steps, GaussianNB())),
        ("KNN", make_pipeline(*base_steps, KNeighborsClassifier(n_jobs=-1))),
        ("Gradient Boosting", make_pipeline(*base_steps, GradientBoostingClassifier(random_state=random_state))),
        ("AdaBoost", make_pipeline(*base_steps, AdaBoostClassifier(random_state=random_state))),
        (
            "LASSO Logistic Regression",
            make_pipeline(
                *base_steps,
                LogisticRegression(
                    penalty="l1",
                    class_weight="balanced",
                    random_state=random_state,
                    solver="saga",
                    max_iter=10000,
                ),
            ),
        ),
        ("LDA", make_pipeline(*base_steps, LinearDiscriminantAnalysis())),
    ]
    if LIGHTGBM_AVAILABLE and LGBMClassifier is not None:
        models.append(
            (
                "LightGBM",
                make_pipeline(
                    *base_steps,
                    LGBMClassifier(
                        random_state=random_state,
                        class_weight="balanced",
                        n_jobs=-1,
                        verbose=-1,
                    ),
                ),
            )
        )
    return models


def get_param_distributions() -> dict[str, dict]:
    """Return randomized-search spaces keyed by human-readable model name."""

    distributions = {
        "SVM": {
            "svc__C": [0.1, 1.0, 10.0, 100.0],
            "svc__gamma": ["scale", 0.001, 0.01, 0.1],
            "svc__kernel": ["rbf"],
        },
        "Logistic Regression": {
            "logisticregression__C": [0.01, 0.1, 1.0, 10.0, 100.0],
            "logisticregression__l1_ratio": [0.1, 0.3, 0.5, 0.7, 0.9],
        },
        "LASSO Logistic Regression": {
            "logisticregression__C": [0.01, 0.05, 0.1, 0.5, 1.0, 10.0],
        },
        "Random Forest": {
            "randomforestclassifier__n_estimators": [200, 500],
            "randomforestclassifier__max_depth": [None, 5, 10, 20],
            "randomforestclassifier__min_samples_leaf": [1, 2, 5],
            "randomforestclassifier__max_features": ["sqrt", "log2"],
        },
        "Extra Trees": {
            "extratreesclassifier__n_estimators": [200, 500],
            "extratreesclassifier__max_depth": [None, 5, 10, 20],
            "extratreesclassifier__min_samples_leaf": [1, 2, 5],
            "extratreesclassifier__max_features": ["sqrt", "log2"],
        },
        "Decision Tree": {
            "decisiontreeclassifier__max_depth": [3, 5, 10, None],
            "decisiontreeclassifier__min_samples_leaf": [1, 5, 10],
            "decisiontreeclassifier__criterion": ["gini", "entropy"],
        },
        "KNN": {
            "kneighborsclassifier__n_neighbors": [3, 5, 7, 11, 15, 21],
            "kneighborsclassifier__weights": ["uniform", "distance"],
            "kneighborsclassifier__p": [1, 2],
        },
        "Gradient Boosting": {
            "gradientboostingclassifier__n_estimators": [100, 200],
            "gradientboostingclassifier__learning_rate": [0.01, 0.05, 0.1],
            "gradientboostingclassifier__max_depth": [2, 3, 5],
        },
        "AdaBoost": {
            "adaboostclassifier__n_estimators": [50, 100, 200],
            "adaboostclassifier__learning_rate": [0.5, 1.0],
        },
        "LDA": {
            "lineardiscriminantanalysis__solver": ["lsqr", "eigen"],
            "lineardiscriminantanalysis__shrinkage": [None, "auto", 0.1, 0.5],
        },
    }
    if LIGHTGBM_AVAILABLE:
        distributions["LightGBM"] = {
            "lgbmclassifier__n_estimators": [200, 500],
            "lgbmclassifier__learning_rate": [0.01, 0.05, 0.1],
            "lgbmclassifier__num_leaves": [15, 31, 63],
            "lgbmclassifier__min_child_samples": [5, 10, 20],
        }
    return distributions

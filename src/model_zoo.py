from __future__ import annotations
import warnings
from typing import Dict, Any, Optional, List

import numpy as np
import pandas as pd
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder, StandardScaler, OrdinalEncoder, KBinsDiscretizer
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC, LinearSVC
from sklearn.calibration import CalibratedClassifierCV
from sklearn.naive_bayes import GaussianNB, CategoricalNB
from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import (
    RandomForestClassifier, ExtraTreesClassifier, GradientBoostingClassifier,
    HistGradientBoostingClassifier, AdaBoostClassifier, StackingClassifier, VotingClassifier
)

from .tan_bayes import TanBayesClassifier

def get_feature_schema():
    CATEGORICAL_FEATURES = ['sex','cp','fbs','restecg','exang','slope','ca','thal']
    NUMERIC_FEATURES = ['age','trestbps','chol','thalach','oldpeak']
    return NUMERIC_FEATURES, CATEGORICAL_FEATURES

def linear_preprocessor():
    num, cat = get_feature_schema()
    return ColumnTransformer(
        transformers=[
            ('num', Pipeline([('imp', SimpleImputer(strategy='median')),
                              ('sc', StandardScaler())]), num),
            ('cat', Pipeline([('imp', SimpleImputer(strategy='most_frequent')),
                              ('ohe', OneHotEncoder(handle_unknown='ignore'))]), cat)
        ],
        remainder='drop'
    )

def tree_preprocessor():
    num, cat = get_feature_schema()
    return ColumnTransformer(
        transformers=[
            ('num', SimpleImputer(strategy='median'), num),
            ('cat', Pipeline([('imp', SimpleImputer(strategy='most_frequent')),
                              ('ohe', OneHotEncoder(handle_unknown='ignore'))]), cat)
        ],
        remainder='drop'
    )

def discretized_preprocessor(n_bins:int=5, strategy:str="quantile"):
    # For CategoricalNB: make all features discrete integers
    num, cat = get_feature_schema()
    return ColumnTransformer(
        transformers=[
            ('num', Pipeline([('imp', SimpleImputer(strategy='median')),
                              ('kb', KBinsDiscretizer(n_bins=n_bins, encode='ordinal', strategy=strategy))]), num),
            ('cat', Pipeline([('imp', SimpleImputer(strategy='most_frequent')),
                              ('ord', OrdinalEncoder(handle_unknown='use_encoded_value', unknown_value=-1))]), cat),
        ],
        remainder='drop'
    )

def build_model_zoo(task: str = "binary", random_state: int = 42) -> Dict[str, Any]:
    models = {}
    # Logistic Regression
    models["logreg_L2"] = Pipeline([
        ('prep', linear_preprocessor()),
        ('clf', LogisticRegression(max_iter=5000, class_weight='balanced', solver='liblinear'))
    ])
    models["logreg_L1"] = Pipeline([
        ('prep', linear_preprocessor()),
        ('clf', LogisticRegression(max_iter=5000, class_weight='balanced', solver='saga', penalty='l1'))
    ])
    # SVM
    models["svm_linear_calibrated"] = Pipeline([
        ('prep', linear_preprocessor()),
        ('clf', CalibratedClassifierCV(LinearSVC(class_weight='balanced', max_iter=10000), cv=3, method='sigmoid'))
    ])
    models["svm_rbf"] = Pipeline([
        ('prep', linear_preprocessor()),
        ('clf', SVC(kernel='rbf', probability=True, class_weight='balanced', C=2.0, gamma='scale', max_iter=10000))
    ])
    # Naive Bayes
    models["gaussian_nb_num_only"] = Pipeline([
        ('prep', ColumnTransformer([
            ('num', Pipeline([('imp', SimpleImputer(strategy='median')),
                              ('sc', StandardScaler())]), get_feature_schema()[0])
        ], remainder='drop')),
        ('clf', GaussianNB())
    ])
    models["categorical_nb_discretized"] = Pipeline([
        ('prep', discretized_preprocessor(n_bins=6)),
        ('clf', CategoricalNB())
    ])
    # kNN
    models["knn_k7"] = Pipeline([
        ('prep', linear_preprocessor()),
        ('clf', KNeighborsClassifier(n_neighbors=7, weights='distance'))
    ])
    # Trees & ensembles
    models["decision_tree"] = Pipeline([
        ('prep', tree_preprocessor()),
        ('clf', DecisionTreeClassifier(max_depth=None, min_samples_leaf=5, class_weight='balanced', random_state=random_state))
    ])
    models["random_forest"] = Pipeline([
        ('prep', tree_preprocessor()),
        ('clf', RandomForestClassifier(n_estimators=500, max_depth=None, min_samples_leaf=3,
                                       class_weight='balanced_subsample', random_state=random_state, n_jobs=-1))
    ])
    models["extra_trees"] = Pipeline([
        ('prep', tree_preprocessor()),
        ('clf', ExtraTreesClassifier(n_estimators=600, max_depth=None, min_samples_leaf=2,
                                     class_weight='balanced', random_state=random_state, n_jobs=-1))
    ])
    models["grad_boost"] = Pipeline([
        ('prep', tree_preprocessor()),
        ('clf', GradientBoostingClassifier(random_state=random_state))
    ])
    models["hist_grad_boost"] = Pipeline([
        ('prep', tree_preprocessor()),
        ('clf', HistGradientBoostingClassifier(max_depth=None, learning_rate=0.08, max_iter=400,
                                               l2_regularization=0.0, random_state=random_state))
    ])
    models["adaboost"] = Pipeline([
        ('prep', tree_preprocessor()),
        ('clf', AdaBoostClassifier(n_estimators=400, learning_rate=0.5, random_state=random_state))
    ])

    # TAN Bayes (our BN)
    models["tan_bayes"] = TanBayesClassifier(
        categorical_features=['sex','cp','fbs','restecg','exang','slope','ca','thal'],
        numeric_features=['age','trestbps','chol','thalach','oldpeak'],
        n_bins=6, discretizer_strategy="quantile", smoothing=1.0, root_feature=None, random_state=random_state
    )

    # Optional: XGBoost / LightGBM if installed
    try:
        from xgboost import XGBClassifier
        models["xgboost"] = Pipeline([
            ('prep', tree_preprocessor()),
            ('clf', XGBClassifier(
                n_estimators=800, max_depth=4, learning_rate=0.05, subsample=0.9, colsample_bytree=0.8,
                eval_metric='logloss', reg_lambda=1.0, random_state=random_state, n_jobs=-1,
                tree_method='hist'
            ))
        ])
    except Exception:
        pass
    try:
        import lightgbm as lgb
        models["lightgbm"] = Pipeline([
            ('prep', tree_preprocessor()),
            ('clf', lgb.LGBMClassifier(
                n_estimators=1200, num_leaves=31, learning_rate=0.03, subsample=0.9, colsample_bytree=0.8,
                objective='multiclass' if task=='multiclass' else 'binary', random_state=random_state, n_jobs=-1
            ))
        ])
    except Exception:
        pass

    # Voting / Stacking (wrap only if base models exist)
    base_voters = []
    for key in ["logreg_L2", "random_forest", "grad_boost"]:
        if key in models:
            base_voters.append((key, models[key]))
    if base_voters:
        try:
            models["voting_soft"] = VotingClassifier(estimators=base_voters, voting='soft', n_jobs=-1)
        except TypeError:
            # sklearn<1.4 doesn't support n_jobs in VotingClassifier
            models["voting_soft"] = VotingClassifier(estimators=base_voters, voting='soft')
    base_stack = []
    for key in ["logreg_L2", "svm_rbf", "random_forest", "grad_boost"]:
        if key in models:
            base_stack.append((key, models[key]))
    if base_stack:
        models["stacking"] = StackingClassifier(
            estimators=base_stack,
            final_estimator=LogisticRegression(max_iter=5000, class_weight='balanced'),
            stack_method='predict_proba',
            passthrough=False, n_jobs=-1
        )
    return models

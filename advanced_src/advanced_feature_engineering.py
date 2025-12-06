from __future__ import annotations
import numpy as np
import pandas as pd
from typing import List, Dict, Tuple, Optional, Any
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.preprocessing import StandardScaler, RobustScaler, QuantileTransformer
from sklearn.feature_selection import SelectKBest, f_classif, mutual_info_classif
from sklearn.ensemble import RandomForestClassifier
import warnings
warnings.filterwarnings('ignore')

class MedicalFeatureEngineer(BaseEstimator, TransformerMixin):
    """
    Domain-specific feature engineering for cardiovascular risk assessment.
    
    Creates clinically-meaningful derived features based on medical knowledge:
    - Cardiovascular risk scores (Framingham-like features)
    - Metabolic syndrome indicators
    - Exercise capacity metrics
    - Composite risk indices
    """
    
    def __init__(self):
        self.feature_names_ = None
        self.original_features_ = None
    
    def fit(self, X, y=None):
        if isinstance(X, pd.DataFrame):
            self.original_features_ = X.columns.tolist()
        return self
    
    def transform(self, X):
        if isinstance(X, np.ndarray):
            if self.original_features_ is None:
                raise ValueError("Cannot transform numpy array without original column names")
            X = pd.DataFrame(X, columns=self.original_features_)
        
        X_new = X.copy()
        
        # 1. Cardiovascular Risk Indicators
        # Age-adjusted maximum heart rate (based on 220-age formula)
        X_new['age_adjusted_max_hr'] = 220 - X_new['age']
        X_new['hr_reserve'] = X_new['age_adjusted_max_hr'] - X_new['thalach']
        X_new['hr_utilization'] = X_new['thalach'] / X_new['age_adjusted_max_hr']
        
        # 2. Metabolic Risk Features
        # High cholesterol flag (>240 mg/dL is high risk)
        X_new['high_cholesterol'] = (X_new['chol'] > 240).astype(int)
        X_new['very_high_cholesterol'] = (X_new['chol'] > 300).astype(int)
        
        # Blood pressure categories (based on AHA guidelines)
        # Systolic BP categories: <120 (normal), 120-129 (elevated), 130-139 (stage 1), >=140 (stage 2)
        X_new['bp_normal'] = (X_new['trestbps'] < 120).astype(int)
        X_new['bp_elevated'] = ((X_new['trestbps'] >= 120) & (X_new['trestbps'] < 130)).astype(int)
        X_new['bp_stage1'] = ((X_new['trestbps'] >= 130) & (X_new['trestbps'] < 140)).astype(int)
        X_new['bp_stage2'] = (X_new['trestbps'] >= 140).astype(int)
        
        # 3. Exercise & Functional Capacity
        # Exercise-induced symptoms
        X_new['exercise_symptoms'] = X_new['exang']  # Already binary
        X_new['st_depression_severe'] = (X_new['oldpeak'] >= 2.0).astype(int)
        X_new['st_depression_moderate'] = ((X_new['oldpeak'] >= 1.0) & (X_new['oldpeak'] < 2.0)).astype(int)
        
        # 4. Age-Gender Risk Interaction
        # Men have higher risk at younger ages, women's risk increases post-menopause
        X_new['male_young'] = ((X_new['sex'] == 1) & (X_new['age'] < 50)).astype(int)
        X_new['male_middle'] = ((X_new['sex'] == 1) & (X_new['age'] >= 50) & (X_new['age'] < 65)).astype(int)
        X_new['male_elderly'] = ((X_new['sex'] == 1) & (X_new['age'] >= 65)).astype(int)
        X_new['female_premenopausal'] = ((X_new['sex'] == 0) & (X_new['age'] < 55)).astype(int)
        X_new['female_postmenopausal'] = ((X_new['sex'] == 0) & (X_new['age'] >= 55)).astype(int)
        
        # 5. Composite Risk Scores
        # Simple risk score based on major risk factors
        risk_score = 0
        risk_score += (X_new['age'] > 55).astype(int) * 2  # Age
        risk_score += X_new['sex'] * 1  # Male gender
        risk_score += (X_new['chol'] > 240).astype(int) * 2  # High cholesterol
        risk_score += (X_new['trestbps'] > 140).astype(int) * 2  # High BP
        risk_score += X_new['fbs'] * 1  # Diabetes
        risk_score += X_new['exang'] * 2  # Exercise angina
        X_new['composite_risk_score'] = risk_score
        
        # 6. Chest Pain Risk Stratification
        # Typical angina (cp=1) is highest risk, atypical (cp=2), non-anginal (cp=3), asymptomatic (cp=4)
        X_new['typical_angina'] = (X_new['cp'] == 1).astype(int)
        X_new['atypical_angina'] = (X_new['cp'] == 2).astype(int)
        X_new['non_anginal'] = (X_new['cp'] == 3).astype(int)
        X_new['asymptomatic'] = (X_new['cp'] == 4).astype(int)
        
        # 7. Thallium Stress Test Risk
        X_new['thal_normal'] = (X_new['thal'] == 3).astype(int)
        X_new['thal_fixed_defect'] = (X_new['thal'] == 6).astype(int)
        X_new['thal_reversible'] = (X_new['thal'] == 7).astype(int)
        
        # 8. ECG Risk Features
        X_new['ecg_normal'] = (X_new['restecg'] == 0).astype(int)
        X_new['ecg_abnormal'] = (X_new['restecg'] == 1).astype(int)
        X_new['ecg_hypertrophy'] = (X_new['restecg'] == 2).astype(int)
        
        # 9. Advanced Ratios and Interactions
        # Heart rate to age ratio (fitness indicator)
        X_new['hr_age_ratio'] = X_new['thalach'] / X_new['age']
        
        # Cholesterol to age ratio
        X_new['chol_age_ratio'] = X_new['chol'] / X_new['age']
        
        # Multiple risk factors interaction
        X_new['multiple_risks'] = (
            (X_new['chol'] > 240).astype(int) + 
            (X_new['trestbps'] > 140).astype(int) + 
            X_new['fbs'] + 
            X_new['exang']
        )
        
        self.feature_names_ = X_new.columns.tolist()
        return X_new
    
    def get_feature_names_out(self, input_features=None):
        return self.feature_names_

class PolynomialInteractionFeatures(BaseEstimator, TransformerMixin):
    """
    Intelligent polynomial and interaction feature generation with automatic selection.
    
    Instead of generating all possible interactions, this uses feature importance
    and correlation analysis to select the most promising feature pairs.
    """
    
    def __init__(self, max_interactions=20, min_importance=0.01, random_state=42):
        self.max_interactions = max_interactions
        self.min_importance = min_importance
        self.random_state = random_state
        self.selected_interactions_ = None
        self.feature_names_ = None
    
    def fit(self, X, y):
        if isinstance(X, pd.DataFrame):
            feature_names = X.columns.tolist()
            X_array = X.values
        else:
            feature_names = [f"feature_{i}" for i in range(X.shape[1])]
            X_array = X
        
        # Use Random Forest to get feature importances
        rf = RandomForestClassifier(n_estimators=100, random_state=self.random_state, n_jobs=-1)
        rf.fit(X_array, y)
        importances = rf.feature_importances_
        
        # Select important features for interactions
        important_features = [i for i, imp in enumerate(importances) if imp >= self.min_importance]
        
        if len(important_features) < 2:
            important_features = np.argsort(importances)[-10:].tolist()  # Take top 10
        
        # Generate interaction candidates
        from itertools import combinations
        interaction_candidates = list(combinations(important_features, 2))
        
        # Score interactions using mutual information
        interaction_scores = []
        for i, j in interaction_candidates:
            interaction_feature = X_array[:, i] * X_array[:, j]
            score = mutual_info_classif(interaction_feature.reshape(-1, 1), y, random_state=self.random_state)[0]
            interaction_scores.append((score, i, j, f"{feature_names[i]}_x_{feature_names[j]}"))
        
        # Select top interactions
        interaction_scores.sort(reverse=True)
        self.selected_interactions_ = interaction_scores[:self.max_interactions]
        
        return self
    
    def transform(self, X):
        if isinstance(X, pd.DataFrame):
            X_array = X.values
            original_columns = X.columns.tolist()
            X_new = X.copy()
        else:
            X_array = X
            original_columns = [f"feature_{i}" for i in range(X.shape[1])]
            X_new = pd.DataFrame(X_array, columns=original_columns)
        
        # Add selected interaction features
        for score, i, j, name in self.selected_interactions_:
            X_new[name] = X_array[:, i] * X_array[:, j]
        
        # Add polynomial features for most important features
        if hasattr(self, 'selected_interactions_') and self.selected_interactions_:
            # Get unique important feature indices
            important_indices = set()
            for _, i, j, _ in self.selected_interactions_[:5]:  # Top 5 interactions
                important_indices.add(i)
                important_indices.add(j)
            
            for idx in list(important_indices)[:5]:  # Limit to avoid explosion
                col_name = original_columns[idx]
                X_new[f"{col_name}_squared"] = X_array[:, idx] ** 2
                X_new[f"{col_name}_sqrt"] = np.sqrt(np.abs(X_array[:, idx]))
        
        self.feature_names_ = X_new.columns.tolist()
        return X_new
    
    def get_feature_names_out(self, input_features=None):
        return self.feature_names_

class AdvancedFeatureSelector(BaseEstimator, TransformerMixin):
    """
    Multi-algorithm feature selection with ensemble voting.
    
    Combines multiple feature selection algorithms:
    1. Univariate statistical tests (f_classif)
    2. Mutual information
    3. Random Forest feature importance
    4. Boruta-like algorithm (if available)
    """
    
    def __init__(self, k_features=50, selection_methods=['statistical', 'mutual_info', 'rf_importance']):
        self.k_features = k_features
        self.selection_methods = selection_methods
        self.selected_features_ = None
        self.feature_scores_ = None
    
    def fit(self, X, y):
        if isinstance(X, pd.DataFrame):
            feature_names = X.columns.tolist()
            X_array = X.values
        else:
            feature_names = [f"feature_{i}" for i in range(X.shape[1])]
            X_array = X
        
        n_features = X_array.shape[1]
        feature_scores = np.zeros(n_features)
        
        # 1. Statistical test
        if 'statistical' in self.selection_methods:
            selector = SelectKBest(f_classif, k='all')
            selector.fit(X_array, y)
            scores = selector.scores_
            scores = (scores - scores.min()) / (scores.max() - scores.min())  # Normalize
            feature_scores += scores
        
        # 2. Mutual Information
        if 'mutual_info' in self.selection_methods:
            mi_scores = mutual_info_classif(X_array, y, random_state=42)
            mi_scores = (mi_scores - mi_scores.min()) / (mi_scores.max() - mi_scores.min())  # Normalize
            feature_scores += mi_scores
        
        # 3. Random Forest Importance
        if 'rf_importance' in self.selection_methods:
            rf = RandomForestClassifier(n_estimators=100, random_state=42, n_jobs=-1)
            rf.fit(X_array, y)
            rf_scores = rf.feature_importances_
            rf_scores = (rf_scores - rf_scores.min()) / (rf_scores.max() - rf_scores.min())  # Normalize
            feature_scores += rf_scores
        
        # Select top k features
        self.feature_scores_ = feature_scores
        top_indices = np.argsort(feature_scores)[-self.k_features:]
        self.selected_features_ = [(idx, feature_names[idx], feature_scores[idx]) for idx in top_indices]
        self.selected_features_.sort(key=lambda x: x[2], reverse=True)  # Sort by score
        
        return self
    
    def transform(self, X):
        if isinstance(X, pd.DataFrame):
            feature_names = X.columns.tolist()
            selected_cols = [feature_names[idx] for idx, _, _ in self.selected_features_]
            return X[selected_cols]
        else:
            selected_indices = [idx for idx, _, _ in self.selected_features_]
            return X[:, selected_indices]
    
    def get_feature_names_out(self, input_features=None):
        return [name for _, name, _ in self.selected_features_]

class AdvancedPreprocessor(BaseEstimator, TransformerMixin):
    """
    Advanced preprocessing pipeline that combines all feature engineering techniques.
    """
    
    def __init__(self, 
                 medical_features=True,
                 polynomial_interactions=True,
                 feature_selection=True,
                 scaling_method='robust',
                 max_interactions=15,
                 k_features=40):
        self.medical_features = medical_features
        self.polynomial_interactions = polynomial_interactions
        self.feature_selection = feature_selection
        self.scaling_method = scaling_method
        self.max_interactions = max_interactions
        self.k_features = k_features
        
        # Pipeline components
        self.medical_engineer_ = None
        self.poly_engineer_ = None
        self.feature_selector_ = None
        self.scaler_ = None
        self.final_feature_names_ = None
    
    def fit(self, X, y):
        X_current = X.copy()
        
        # 1. Medical feature engineering
        if self.medical_features:
            self.medical_engineer_ = MedicalFeatureEngineer()
            X_current = self.medical_engineer_.fit_transform(X_current, y)
            print(f"After medical features: {X_current.shape[1]} features")
        
        # 2. Polynomial and interaction features
        if self.polynomial_interactions:
            self.poly_engineer_ = PolynomialInteractionFeatures(
                max_interactions=self.max_interactions,
                random_state=42
            )
            X_current = self.poly_engineer_.fit_transform(X_current, y)
            print(f"After polynomial features: {X_current.shape[1]} features")
        
        # 3. Feature selection
        if self.feature_selection:
            self.feature_selector_ = AdvancedFeatureSelector(k_features=self.k_features)
            self.feature_selector_.fit(X_current, y)
            X_current = self.feature_selector_.transform(X_current)
            print(f"After feature selection: {X_current.shape[1]} features")
        
        # 4. Scaling
        if self.scaling_method == 'standard':
            self.scaler_ = StandardScaler()
        elif self.scaling_method == 'robust':
            self.scaler_ = RobustScaler()
        elif self.scaling_method == 'quantile':
            self.scaler_ = QuantileTransformer(random_state=42)
        
        if self.scaler_:
            self.scaler_.fit(X_current)
        
        if isinstance(X_current, pd.DataFrame):
            self.final_feature_names_ = X_current.columns.tolist()
        else:
            self.final_feature_names_ = [f"feature_{i}" for i in range(X_current.shape[1])]
        
        return self
    
    def transform(self, X):
        X_current = X.copy()
        
        # Apply same transformations in order
        if self.medical_engineer_:
            X_current = self.medical_engineer_.transform(X_current)
        
        if self.poly_engineer_:
            X_current = self.poly_engineer_.transform(X_current)
        
        if self.feature_selector_:
            X_current = self.feature_selector_.transform(X_current)
        
        if self.scaler_:
            if isinstance(X_current, pd.DataFrame):
                X_scaled = self.scaler_.transform(X_current)
                X_current = pd.DataFrame(X_scaled, columns=X_current.columns, index=X_current.index)
            else:
                X_current = self.scaler_.transform(X_current)
        
        return X_current
    
    def get_feature_names_out(self, input_features=None):
        return self.final_feature_names_

def demo_advanced_features():
    """Show the feature engineering on the heart disease dataset."""
    # This would be called from a notebook or script to show the features in action
    pass

if __name__ == "__main__":
    demo_advanced_features()
from __future__ import annotations
import numpy as np
import pandas as pd
from typing import List, Dict, Tuple, Optional, Any, Union, Callable
import warnings
warnings.filterwarnings('ignore')

try:
    import shap
    HAS_SHAP = True
except ImportError:
    HAS_SHAP = False
    print("SHAP not available. Install with: pip install shap")

try:
    import lime
    from lime.lime_tabular import LimeTabularExplainer
    HAS_LIME = True
except ImportError:
    HAS_LIME = False
    print("LIME not available. Install with: pip install lime")

try:
    import matplotlib.pyplot as plt
    import seaborn as sns
    HAS_PLOTTING = True
except ImportError:
    HAS_PLOTTING = False

from sklearn.metrics import accuracy_score, roc_auc_score
from sklearn.model_selection import train_test_split
from sklearn.inspection import permutation_importance
import itertools

class MedicalExplainer:
    """
    Explainability framework for medical ML models.
    
    Provides multiple explanation methods and medical-specific interpretations
    to support clinical decision-making.
    """
    
    def __init__(self, 
                 model, 
                 X_train, 
                 y_train, 
                 feature_names=None,
                 categorical_features=None,
                 medical_feature_groups=None,
                 class_names=None):
        """
        Initialize the medical explainer.
        
        Args:
            model: Fitted sklearn-compatible model
            X_train: Training features (for SHAP background)
            y_train: Training labels
            feature_names: List of feature names
            categorical_features: Indices of categorical features
            medical_feature_groups: Dict grouping features by medical category
            class_names: Names of classes for classification
        """
        self.model = model
        self.X_train = X_train
        self.y_train = y_train
        
        # Handle feature names
        if isinstance(X_train, pd.DataFrame):
            self.feature_names = list(X_train.columns)
            self.X_train_array = X_train.values
        else:
            self.X_train_array = X_train
            self.feature_names = feature_names or [f"feature_{i}" for i in range(X_train.shape[1])]
        
        self.categorical_features = categorical_features or []
        self.medical_feature_groups = medical_feature_groups or self._default_medical_groups()
        self.class_names = class_names or ['No Disease', 'Disease']
        
        # Initialize explainers
        self.shap_explainer_ = None
        self.lime_explainer_ = None
        self.feature_importances_ = None
        
    def _default_medical_groups(self):
        """Default medical feature groupings for heart disease."""
        groups = {}
        
        # Map features to medical categories
        for i, name in enumerate(self.feature_names):
            name_lower = name.lower()
            if any(term in name_lower for term in ['age', 'sex', 'male', 'female']):
                groups.setdefault('demographics', []).append(i)
            elif any(term in name_lower for term in ['bp', 'trestbps', 'pressure']):
                groups.setdefault('blood_pressure', []).append(i)
            elif any(term in name_lower for term in ['chol', 'cholesterol']):
                groups.setdefault('cholesterol', []).append(i)
            elif any(term in name_lower for term in ['hr', 'thalach', 'heart_rate']):
                groups.setdefault('heart_rate', []).append(i)
            elif any(term in name_lower for term in ['cp', 'angina', 'chest', 'pain']):
                groups.setdefault('chest_symptoms', []).append(i)
            elif any(term in name_lower for term in ['ecg', 'restecg']):
                groups.setdefault('ecg', []).append(i)
            elif any(term in name_lower for term in ['exercise', 'exang', 'thal']):
                groups.setdefault('exercise_stress', []).append(i)
            elif any(term in name_lower for term in ['oldpeak', 'slope', 'st']):
                groups.setdefault('st_changes', []).append(i)
            elif any(term in name_lower for term in ['ca', 'vessels']):
                groups.setdefault('vessels', []).append(i)
            elif any(term in name_lower for term in ['fbs', 'diabetes', 'blood_sugar']):
                groups.setdefault('diabetes', []).append(i)
            else:
                groups.setdefault('other', []).append(i)
        
        return groups
    
    def setup_shap_explainer(self, explainer_type='auto', background_size=100):
        """
        Set up SHAP explainer based on model type.
        
        Args:
            explainer_type: 'tree', 'linear', 'kernel', 'deep', or 'auto'
            background_size: Size of background dataset for kernel explainer
        """
        if not HAS_SHAP:
            print("SHAP not available. Skipping SHAP explainer setup.")
            return
        
        # Select background data
        if len(self.X_train_array) > background_size:
            background_indices = np.random.choice(len(self.X_train_array), 
                                                background_size, replace=False)
            background_data = self.X_train_array[background_indices]
        else:
            background_data = self.X_train_array
        
        # Auto-detect explainer type if needed
        if explainer_type == 'auto':
            model_name = type(self.model).__name__.lower()
            if 'tree' in model_name or 'forest' in model_name or 'boost' in model_name:
                explainer_type = 'tree'
            elif 'linear' in model_name or 'logistic' in model_name:
                explainer_type = 'linear'
            elif 'svm' in model_name or 'svc' in model_name:
                explainer_type = 'kernel'
            else:
                explainer_type = 'kernel'  # Default fallback
        
        # Create appropriate explainer
        try:
            if explainer_type == 'tree':
                self.shap_explainer_ = shap.TreeExplainer(self.model)
            elif explainer_type == 'linear':
                self.shap_explainer_ = shap.LinearExplainer(self.model, background_data)
            elif explainer_type == 'kernel':
                self.shap_explainer_ = shap.KernelExplainer(self.model.predict_proba, background_data)
            elif explainer_type == 'deep':
                self.shap_explainer_ = shap.DeepExplainer(self.model, background_data)
            
            print(f"SHAP {explainer_type} explainer initialized")
            
        except Exception as e:
            print(f"Failed to initialize SHAP {explainer_type} explainer: {e}")
            # Fallback to kernel explainer
            try:
                self.shap_explainer_ = shap.KernelExplainer(self.model.predict_proba, background_data)
                print("Fallback: SHAP Kernel explainer initialized")
            except Exception as e2:
                print(f"Failed to initialize fallback SHAP explainer: {e2}")
    
    def setup_lime_explainer(self, mode='classification'):
        """Set up LIME explainer."""
        if not HAS_LIME:
            print("LIME not available. Skipping LIME explainer setup.")
            return
        
        try:
            self.lime_explainer_ = LimeTabularExplainer(
                self.X_train_array,
                feature_names=self.feature_names,
                class_names=self.class_names,
                categorical_features=self.categorical_features,
                mode=mode,
                discretize_continuous=True,
                random_state=42
            )
            print("LIME explainer initialized")
            
        except Exception as e:
            print(f"Failed to initialize LIME explainer: {e}")
    
    def get_global_shap_importance(self, X_test, max_samples=500):
        """
        Get global SHAP feature importance.
        
        Args:
            X_test: Test dataset for explanation
            max_samples: Maximum samples to explain (for computational efficiency)
        """
        if not self.shap_explainer_:
            print("SHAP explainer not initialized. Call setup_shap_explainer() first.")
            return None
        
        # Prepare test data
        if isinstance(X_test, pd.DataFrame):
            X_test_array = X_test.values
        else:
            X_test_array = X_test
        
        # Limit samples for computational efficiency
        if len(X_test_array) > max_samples:
            sample_indices = np.random.choice(len(X_test_array), max_samples, replace=False)
            X_test_sample = X_test_array[sample_indices]
        else:
            X_test_sample = X_test_array
        
        try:
            # Get SHAP values
            shap_values = self.shap_explainer_.shap_values(X_test_sample)
            
            # Handle multi-class case
            if isinstance(shap_values, list):
                # For binary classification, typically use positive class
                shap_values_combined = shap_values[-1]
            else:
                shap_values_combined = shap_values
            
            # Calculate global importance
            global_importance = np.abs(shap_values_combined).mean(axis=0)
            
            # Create importance dataframe
            importance_df = pd.DataFrame({
                'feature': self.feature_names,
                'importance': global_importance,
                'medical_group': [self._get_medical_group(i) for i in range(len(self.feature_names))]
            }).sort_values('importance', ascending=False)
            
            return importance_df, shap_values_combined
            
        except Exception as e:
            print(f"Error computing SHAP values: {e}")
            return None, None
    
    def explain_instance_lime(self, instance, num_features=10):
        """
        Explain a single instance using LIME.
        
        Args:
            instance: Single instance to explain (1D array or Series)
            num_features: Number of top features to include in explanation
        """
        if not self.lime_explainer_:
            print("LIME explainer not initialized. Call setup_lime_explainer() first.")
            return None
        
        # Ensure instance is 1D array
        if isinstance(instance, pd.Series):
            instance = instance.values
        if instance.ndim > 1:
            instance = instance.flatten()
        
        try:
            # Get LIME explanation
            explanation = self.lime_explainer_.explain_instance(
                instance, 
                self.model.predict_proba,
                num_features=num_features
            )
            
            return explanation
            
        except Exception as e:
            print(f"Error generating LIME explanation: {e}")
            return None
    
    def _get_medical_group(self, feature_idx):
        """Get medical group for a feature index."""
        for group, indices in self.medical_feature_groups.items():
            if feature_idx in indices:
                return group
        return 'other'
    
    def analyze_feature_interactions(self, X_test, max_interactions=10, method='shap'):
        """
        Analyze feature interactions using SHAP interaction values.
        
        Args:
            X_test: Test dataset
            max_interactions: Maximum number of interactions to analyze
            method: Method to use ('shap' or 'permutation')
        """
        if method == 'shap' and self.shap_explainer_:
            return self._analyze_shap_interactions(X_test, max_interactions)
        else:
            return self._analyze_permutation_interactions(X_test, max_interactions)
    
    def _analyze_shap_interactions(self, X_test, max_interactions):
        """Analyze interactions using SHAP interaction values."""
        if not HAS_SHAP:
            return None
        
        try:
            # Limit samples for computational efficiency
            X_test_array = X_test.values if isinstance(X_test, pd.DataFrame) else X_test
            sample_size = min(100, len(X_test_array))  # Small sample for interaction analysis
            sample_indices = np.random.choice(len(X_test_array), sample_size, replace=False)
            X_sample = X_test_array[sample_indices]
            
            # Get interaction values (if supported by explainer)
            if hasattr(self.shap_explainer_, 'shap_interaction_values'):
                interaction_values = self.shap_explainer_.shap_interaction_values(X_sample)
                
                # Average interaction strengths
                if isinstance(interaction_values, list):
                    interaction_values = interaction_values[-1]  # Use positive class for binary
                
                interaction_matrix = np.abs(interaction_values).mean(axis=0)
                
                # Find top interactions (excluding diagonal)
                interactions = []
                n_features = len(self.feature_names)
                
                for i in range(n_features):
                    for j in range(i+1, n_features):
                        strength = interaction_matrix[i, j]
                        interactions.append({
                            'feature_1': self.feature_names[i],
                            'feature_2': self.feature_names[j],
                            'interaction_strength': strength,
                            'medical_interpretation': self._interpret_medical_interaction(i, j)
                        })
                
                # Sort by interaction strength
                interactions.sort(key=lambda x: x['interaction_strength'], reverse=True)
                
                return interactions[:max_interactions]
            
        except Exception as e:
            print(f"Error analyzing SHAP interactions: {e}")
        
        return None
    
    def _analyze_permutation_interactions(self, X_test, max_interactions):
        """Analyze interactions using permutation-based method."""
        # This is a simplified interaction analysis
        if isinstance(X_test, pd.DataFrame):
            X_test_array = X_test.values
        else:
            X_test_array = X_test
        
        y_test_pred = self.model.predict_proba(X_test_array)[:, 1]  # Positive class probabilities
        
        interactions = []
        n_features = X_test_array.shape[1]
        
        # Sample feature pairs to avoid computational explosion
        feature_pairs = list(itertools.combinations(range(min(20, n_features)), 2))
        if len(feature_pairs) > max_interactions * 3:
            feature_pairs = np.random.choice(len(feature_pairs), max_interactions * 3, replace=False)
            feature_pairs = [list(itertools.combinations(range(min(20, n_features)), 2))[i] 
                           for i in feature_pairs]
        
        for i, j in feature_pairs[:max_interactions]:
            # Permute both features together
            X_permuted = X_test_array.copy()
            X_permuted[:, [i, j]] = X_permuted[np.random.permutation(len(X_permuted)), [i, j]]
            
            y_permuted = self.model.predict_proba(X_permuted)[:, 1]
            interaction_strength = np.abs(y_test_pred - y_permuted).mean()
            
            interactions.append({
                'feature_1': self.feature_names[i],
                'feature_2': self.feature_names[j], 
                'interaction_strength': interaction_strength,
                'medical_interpretation': self._interpret_medical_interaction(i, j)
            })
        
        interactions.sort(key=lambda x: x['interaction_strength'], reverse=True)
        return interactions
    
    def _interpret_medical_interaction(self, idx1, idx2):
        """Provide medical interpretation for feature interactions."""
        name1 = self.feature_names[idx1].lower()
        name2 = self.feature_names[idx2].lower()
        
        # Define some common medical interactions
        interactions_map = {
            ('age', 'sex'): 'Age-gender interaction: Risk profiles differ by age and gender',
            ('bp', 'chol'): 'Cardiovascular risk: Both blood pressure and cholesterol affect heart disease risk',
            ('exercise', 'age'): 'Exercise capacity typically decreases with age',
            ('chest_pain', 'exercise'): 'Exercise-induced chest pain is a key diagnostic indicator',
            ('ecg', 'exercise'): 'ECG changes during exercise reveal cardiac stress',
            ('hr', 'exercise'): 'Heart rate response to exercise indicates cardiovascular fitness'
        }
        
        # Check for known patterns
        for key_pair, interpretation in interactions_map.items():
            if (any(k in name1 for k in key_pair[0].split('_')) and 
                any(k in name2 for k in key_pair[1].split('_'))) or \
               (any(k in name2 for k in key_pair[0].split('_')) and 
                any(k in name1 for k in key_pair[1].split('_'))):
                return interpretation
        
        return f"Interaction between {self.feature_names[idx1]} and {self.feature_names[idx2]}"
    
    def generate_counterfactuals(self, instance, target_class=None, n_counterfactuals=5):
        """
        Generate counterfactual explanations for clinical decision support.
        
        Shows what minimal changes would flip the model's prediction.
        """
        if isinstance(instance, pd.Series):
            instance = instance.values
        if instance.ndim > 1:
            instance = instance.flatten()
        
        original_pred = self.model.predict([instance])[0]
        original_proba = self.model.predict_proba([instance])[0]
        
        if target_class is None:
            target_class = 1 - original_pred  # Flip to opposite class
        
        counterfactuals = []
        
        # Simple counterfactual generation by feature perturbation
        for _ in range(n_counterfactuals * 10):  # Generate extra to find valid ones
            # Create modified instance
            modified = instance.copy()
            
            # Randomly select features to modify
            n_changes = np.random.randint(1, min(4, len(instance)))  # 1-3 changes
            feature_indices = np.random.choice(len(instance), n_changes, replace=False)
            
            for idx in feature_indices:
                # Use training data distribution to generate realistic changes
                feature_values = self.X_train_array[:, idx]
                
                if idx in self.categorical_features:
                    # For categorical features, sample from observed values
                    modified[idx] = np.random.choice(np.unique(feature_values))
                else:
                    # For continuous features, add reasonable noise
                    std = np.std(feature_values)
                    modified[idx] = instance[idx] + np.random.normal(0, std * 0.5)
                    # Clip to observed range
                    modified[idx] = np.clip(modified[idx], 
                                          np.min(feature_values), 
                                          np.max(feature_values))
            
            # Check if this generates the target class
            new_pred = self.model.predict([modified])[0]
            if new_pred == target_class:
                new_proba = self.model.predict_proba([modified])[0]
                
                # Calculate changes
                changes = []
                for idx in feature_indices:
                    if instance[idx] != modified[idx]:
                        change_desc = f"{self.feature_names[idx]}: {instance[idx]:.3f} → {modified[idx]:.3f}"
                        changes.append({
                            'feature': self.feature_names[idx],
                            'original': instance[idx],
                            'modified': modified[idx],
                            'change': modified[idx] - instance[idx],
                            'description': change_desc
                        })
                
                counterfactual = {
                    'modified_instance': modified,
                    'changes': changes,
                    'original_prediction': original_pred,
                    'original_probability': original_proba,
                    'new_prediction': new_pred,
                    'new_probability': new_proba,
                    'n_changes': len(changes),
                    'total_distance': np.linalg.norm(modified - instance)
                }
                
                counterfactuals.append(counterfactual)
                
                if len(counterfactuals) >= n_counterfactuals:
                    break
        
        # Sort by number of changes and total distance
        counterfactuals.sort(key=lambda x: (x['n_changes'], x['total_distance']))
        
        return counterfactuals[:n_counterfactuals]
    
    def generate_medical_report(self, instance, explanation_methods=['lime', 'shap'], 
                               include_counterfactuals=True):
        """
        Generate a comprehensive medical explanation report for a single instance.
        
        This provides a clinically-relevant summary suitable for healthcare professionals.
        """
        if isinstance(instance, pd.Series):
            instance_values = instance.values
            instance_series = instance
        else:
            instance_values = instance
            instance_series = pd.Series(instance, index=self.feature_names)
        
        # Get prediction
        prediction = self.model.predict([instance_values])[0]
        probabilities = self.model.predict_proba([instance_values])[0]
        
        report = {
            'patient_id': 'Unknown',
            'prediction': {
                'class': prediction,
                'probability': probabilities,
                'risk_level': self._categorize_risk(probabilities[-1])
            },
            'explanations': {},
            'clinical_summary': {},
            'recommendations': []
        }
        
        # LIME explanation
        if 'lime' in explanation_methods and self.lime_explainer_:
            lime_exp = self.explain_instance_lime(instance_values)
            if lime_exp:
                lime_features = lime_exp.as_list()
                report['explanations']['lime'] = {
                    'top_features': lime_features[:5],
                    'supporting_features': [f for f in lime_features if f[1] > 0],
                    'opposing_features': [f for f in lime_features if f[1] < 0]
                }
        
        # SHAP explanation
        if 'shap' in explanation_methods and self.shap_explainer_:
            try:
                shap_values = self.shap_explainer_.shap_values([instance_values])
                if isinstance(shap_values, list):
                    shap_values = shap_values[-1][0]  # Positive class, single instance
                else:
                    shap_values = shap_values[0]
                
                # Sort features by SHAP value magnitude
                shap_importance = [(self.feature_names[i], shap_values[i]) 
                                 for i in range(len(shap_values))]
                shap_importance.sort(key=lambda x: abs(x[1]), reverse=True)
                
                report['explanations']['shap'] = {
                    'top_features': shap_importance[:5],
                    'supporting_features': [f for f in shap_importance if f[1] > 0],
                    'opposing_features': [f for f in shap_importance if f[1] < 0]
                }
            except Exception as e:
                print(f"Error computing SHAP for report: {e}")
        
        # Clinical summary by medical category
        for group, feature_indices in self.medical_feature_groups.items():
            group_features = []
            for idx in feature_indices:
                if idx < len(instance_values):
                    feature_name = self.feature_names[idx]
                    feature_value = instance_values[idx]
                    group_features.append((feature_name, feature_value))
            
            if group_features:
                report['clinical_summary'][group] = group_features
        
        # Counterfactuals for actionable insights
        if include_counterfactuals:
            counterfactuals = self.generate_counterfactuals(instance_values, n_counterfactuals=3)
            if counterfactuals:
                report['counterfactuals'] = counterfactuals[:2]  # Top 2 counterfactuals
                
                # Generate recommendations based on counterfactuals
                for cf in counterfactuals[:2]:
                    for change in cf['changes']:
                        feature = change['feature']
                        direction = "increase" if change['change'] > 0 else "decrease"
                        recommendation = f"Consider strategies to {direction} {feature}"
                        if recommendation not in report['recommendations']:
                            report['recommendations'].append(recommendation)
        
        return report
    
    def _categorize_risk(self, probability):
        """Categorize risk level based on probability."""
        if probability < 0.3:
            return 'Low Risk'
        elif probability < 0.7:
            return 'Moderate Risk'
        else:
            return 'High Risk'
    
    def compare_explanations_consistency(self, X_test, sample_size=50):
        """
        Compare consistency between different explanation methods.
        
        This is important for building trust in explanations.
        """
        if not (self.shap_explainer_ and self.lime_explainer_):
            print("Both SHAP and LIME explainers needed for consistency comparison")
            return None
        
        # Sample instances
        if isinstance(X_test, pd.DataFrame):
            X_test_array = X_test.values
        else:
            X_test_array = X_test
        
        sample_indices = np.random.choice(len(X_test_array), 
                                        min(sample_size, len(X_test_array)), 
                                        replace=False)
        X_sample = X_test_array[sample_indices]
        
        consistency_scores = []
        
        for instance in X_sample:
            try:
                # Get SHAP explanation
                shap_values = self.shap_explainer_.shap_values([instance])
                if isinstance(shap_values, list):
                    shap_values = shap_values[-1][0]
                else:
                    shap_values = shap_values[0]
                
                # Get LIME explanation
                lime_exp = self.explain_instance_lime(instance, num_features=len(self.feature_names))
                lime_values = np.zeros(len(self.feature_names))
                
                if lime_exp:
                    lime_dict = dict(lime_exp.as_list())
                    for i, feature_name in enumerate(self.feature_names):
                        if feature_name in lime_dict:
                            lime_values[i] = lime_dict[feature_name]
                
                # Calculate correlation between explanations
                correlation = np.corrcoef(shap_values, lime_values)[0, 1]
                if not np.isnan(correlation):
                    consistency_scores.append(correlation)
                
            except Exception as e:
                print(f"Error in consistency comparison: {e}")
                continue
        
        if consistency_scores:
            avg_consistency = np.mean(consistency_scores)
            return {
                'average_consistency': avg_consistency,
                'consistency_scores': consistency_scores,
                'interpretation': self._interpret_consistency(avg_consistency)
            }
        
        return None
    
    def _interpret_consistency(self, consistency_score):
        """Interpret consistency score."""
        if consistency_score > 0.7:
            return "High consistency - explanations agree well"
        elif consistency_score > 0.4:
            return "Moderate consistency - some agreement between explanations"
        else:
            return "Low consistency - explanations may capture different aspects"

def demonstrate_interpretability():
    """Print information about interpretability methods."""
    print("Interpretability Framework")
    print("\nAvailable explanation methods:")
    print("1. SHAP - Global and local explanations with Shapley values")
    print("2. LIME - Local interpretable model-agnostic explanations")
    print("3. Feature Interactions - Understanding how features work together")
    print("4. Counterfactual Explanations - What-if scenarios for decision support")
    print("5. Medical Reports - Clinical summaries with actionable insights")
    print("6. Explanation Consistency - Comparing different explanation methods")
    print("\nDesigned for medical ML applications.")

if __name__ == "__main__":
    demonstrate_interpretability()
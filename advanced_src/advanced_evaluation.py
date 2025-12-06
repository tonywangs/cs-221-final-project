from __future__ import annotations
import os, json, time
from datetime import datetime
from typing import List, Dict, Tuple, Optional, Any, Union
import numpy as np
import pandas as pd
import warnings
warnings.filterwarnings('ignore')

from sklearn.model_selection import StratifiedKFold, cross_validate
from sklearn.metrics import (
    roc_auc_score, average_precision_score, accuracy_score,
    precision_score, recall_score, f1_score, brier_score_loss,
    confusion_matrix, classification_report
)
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.neural_network import MLPClassifier

# Import our advanced components
try:
    from .advanced_feature_engineering import AdvancedPreprocessor
    from .bayesian_neural_networks import VariationalBayesianNN, MonteCarloDropoutNN, BayesianEnsemble
    from .tabnet_model import CustomTabNet, OptimizedTabNetClassifier
    from .advanced_interpretability import MedicalExplainer
    from .bayesian_optimization import BayesianHyperparameterOptimizer
    from .advanced_calibration import AdvancedCalibratorSuite, ConformalPredictor
except ImportError:
    # Fallback to absolute imports when run as script
    import sys
    import os
    sys.path.append(os.path.join(os.path.dirname(__file__)))
    from advanced_feature_engineering import AdvancedPreprocessor
    from bayesian_neural_networks import VariationalBayesianNN, MonteCarloDropoutNN, BayesianEnsemble
    from tabnet_model import CustomTabNet, OptimizedTabNetClassifier
    from advanced_interpretability import MedicalExplainer
    from bayesian_optimization import BayesianHyperparameterOptimizer
    from advanced_calibration import AdvancedCalibratorSuite, ConformalPredictor

class AdvancedModelZoo:
    """
    Model zoo with ML implementations.
    """
    
    def __init__(self, random_state=42, use_optimization=True):
        self.random_state = random_state
        self.use_optimization = use_optimization
        self.models_ = {}
        self.preprocessors_ = {}
    
    def build_advanced_models(self, X_train, y_train, optimization_trials=50):
        """
        Build the model zoo.
        """
        print("Building advanced model zoo...")
        
        models = {}
        
        # 1. Original models with hyperparameter optimization
        if self.use_optimization:
            optimizer = BayesianHyperparameterOptimizer(n_trials=optimization_trials)
            optimized_params = optimizer.optimize_all_models(X_train, y_train)
            
            # Create optimized models
            if 'random_forest' in optimized_params:
                models['rf_optimized'] = RandomForestClassifier(
                    **optimized_params['random_forest'], 
                    random_state=self.random_state
                )
            
            if 'gradient_boosting' in optimized_params:
                models['gb_optimized'] = GradientBoostingClassifier(
                    **optimized_params['gradient_boosting'],
                    random_state=self.random_state
                )
            
            if 'logistic_regression' in optimized_params:
                models['lr_optimized'] = LogisticRegression(
                    **optimized_params['logistic_regression'],
                    random_state=self.random_state
                )
        
        # 2. Bayesian Neural Networks
        models['variational_bnn'] = VariationalBayesianNN(
            hidden_layers=[128, 64, 32],
            n_samples=50,
            epochs=100,
            random_state=self.random_state
        )
        
        models['mc_dropout_nn'] = MonteCarloDropoutNN(
            hidden_layers=[128, 64, 32],
            n_samples=30,
            epochs=75,
            random_state=self.random_state
        )
        
        models['bayesian_ensemble'] = BayesianEnsemble(
            n_bootstrap_samples=5,  # Reduce for computational efficiency
            random_state=self.random_state
        )
        
        # 3. TabNet Deep Learning
        models['tabnet_custom'] = CustomTabNet(
            n_d=64,
            n_a=64,
            n_steps=5,
            epochs=50,  # Reduce for demo
            random_state=self.random_state,
            verbose=0
        )
        
        models['tabnet_optimized'] = OptimizedTabNetClassifier(
            n_d=32,
            n_a=32,
            n_steps=3,
            max_epochs=50,
            random_state=self.random_state,
            verbose=0
        )
        
        # 4. Advanced preprocessing variants
        # Standard preprocessing
        models['rf_standard'] = RandomForestClassifier(
            n_estimators=200, 
            max_depth=10,
            random_state=self.random_state
        )
        
        # With advanced preprocessing
        advanced_preprocessor = AdvancedPreprocessor(
            medical_features=True,
            polynomial_interactions=True,
            feature_selection=True,
            max_interactions=10,
            k_features=30
        )
        models['rf_advanced_features'] = RandomForestClassifier(
            n_estimators=200,
            max_depth=12,
            random_state=self.random_state
        )
        self.preprocessors_['rf_advanced_features'] = advanced_preprocessor
        
        self.models_ = models
        return models

class AdvancedEvaluationSuite:
    """
    Evaluation suite for medical ML models.
    """
    
    def __init__(self, 
                 cv_folds=10, 
                 random_state=42,
                 include_interpretability=True,
                 include_calibration=True,
                 include_fairness=True):
        self.cv_folds = cv_folds
        self.random_state = random_state
        self.include_interpretability = include_interpretability
        self.include_calibration = include_calibration
        self.include_fairness = include_fairness
        
        # Results storage
        self.results_ = {}
        self.model_rankings_ = {}
        self.explanations_ = {}
        self.calibration_results_ = {}
        self.fairness_results_ = {}
        
    def evaluate_all_models(self, models, X, y, feature_names=None, 
                           preprocessors=None, output_dir="advanced_outputs"):
        """
        Evaluate all models using cross-validation.
        """
        print(f"Starting model evaluation...")
        os.makedirs(output_dir, exist_ok=True)
        
        # Setup cross-validation
        cv = StratifiedKFold(n_splits=self.cv_folds, shuffle=True, random_state=self.random_state)
        
        # Prepare feature names
        if feature_names is None:
            if isinstance(X, pd.DataFrame):
                feature_names = list(X.columns)
            else:
                feature_names = [f"feature_{i}" for i in range(X.shape[1])]
        
        all_results = []
        
        # Evaluate each model
        for model_name, model in models.items():
            print(f"\nEvaluating {model_name}...")
            
            try:
                # Check if model needs special preprocessing
                X_processed = X
                if preprocessors and model_name in preprocessors:
                    preprocessor = preprocessors[model_name]
                    if isinstance(X, pd.DataFrame):
                        X_processed = preprocessor.fit_transform(X, y)
                    else:
                        X_processed = preprocessor.fit_transform(
                            pd.DataFrame(X, columns=feature_names), y
                        )
                
                # Perform cross-validation with multiple metrics
                scoring = {
                    'roc_auc': 'roc_auc',
                    'accuracy': 'accuracy',
                    'precision': 'precision',
                    'recall': 'recall',
                    'f1': 'f1',
                    'neg_brier_score': 'neg_brier_score'
                }
                
                cv_results = cross_validate(
                    model, X_processed, y,
                    cv=cv, scoring=scoring,
                    return_train_score=True,
                    n_jobs=-1
                )
                
                # Calculate summary statistics
                model_results = {
                    'model_name': model_name,
                    'model_type': type(model).__name__
                }
                
                for metric in scoring.keys():
                    test_scores = cv_results[f'test_{metric}']
                    if metric == 'neg_brier_score':
                        test_scores = -test_scores  # Convert back to positive
                        metric_name = 'brier_score'
                    else:
                        metric_name = metric
                    
                    model_results.update({
                        f'{metric_name}_mean': np.mean(test_scores),
                        f'{metric_name}_std': np.std(test_scores),
                        f'{metric_name}_ci_lower': np.percentile(test_scores, 2.5),
                        f'{metric_name}_ci_upper': np.percentile(test_scores, 97.5)
                    })
                
                # Additional evaluations
                if self.include_interpretability:
                    interpretability_score = self._evaluate_interpretability(
                        model, model_name, X_processed, y, feature_names
                    )
                    model_results['interpretability_score'] = interpretability_score
                
                if self.include_calibration:
                    calibration_score = self._evaluate_calibration(
                        model, X_processed, y
                    )
                    model_results['calibration_ece'] = calibration_score
                
                # Clinical readiness score
                clinical_score = self._calculate_clinical_readiness(model_results)
                model_results['clinical_readiness_score'] = clinical_score
                
                all_results.append(model_results)
                self.results_[model_name] = model_results
                
                print(f"{model_name}: AUROC={model_results['roc_auc_mean']:.4f} "
                      f"(±{model_results['roc_auc_std']:.4f})")
                
            except Exception as e:
                print(f"Error evaluating {model_name}: {e}")
                continue
        
        # Create results dataframe
        results_df = pd.DataFrame(all_results)
        results_df = results_df.sort_values('roc_auc_mean', ascending=False)
        
        # Save detailed results
        results_path = os.path.join(output_dir, 'advanced_model_results.csv')
        results_df.to_csv(results_path, index=False)
        
        # Create evaluation report
        self._generate_comprehensive_report(results_df, output_dir)
        
        print(f"\nEvaluation complete! Results saved to {output_dir}")
        return results_df
    
    def _evaluate_interpretability(self, model, model_name, X, y, feature_names):
        """
        Evaluate model interpretability using heuristic scoring.
        """
        # Split data for explanation evaluation
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.2, random_state=self.random_state, stratify=y
        )
        
        # Fit model on training data
        model.fit(X_train, y_train)
        
        # Base interpretability scores by model type
        interpretability_scores = {
            'LogisticRegression': 0.9,
            'RandomForestClassifier': 0.7,
            'GradientBoostingClassifier': 0.6,
            'SVC': 0.4,
            'MLPClassifier': 0.3,
            'VariationalBayesianNN': 0.5,  # Uncertainty helps interpretability
            'MonteCarloDropoutNN': 0.5,
            'BayesianEnsemble': 0.6,  # Multiple models
            'CustomTabNet': 0.8,  # Attention mechanisms
            'OptimizedTabNetClassifier': 0.8
        }
        
        model_type = type(model).__name__
        base_score = interpretability_scores.get(model_type, 0.5)
        
        # Adjust based on model complexity
        if hasattr(model, 'n_estimators'):
            if model.n_estimators <= 100:
                base_score += 0.1
            elif model.n_estimators > 500:
                base_score -= 0.1
        
        # Try to create and evaluate explainer
        try:
            explainer = MedicalExplainer(
                model, X_train, y_train, 
                feature_names=feature_names
            )
            explainer.setup_lime_explainer()
            
            # Test explanation generation
            sample_instance = X_test[0:1] if hasattr(X_test, 'iloc') else X_test[0]
            explanation = explainer.explain_instance_lime(sample_instance)
            
            if explanation:
                base_score += 0.2  # Explanation generation works
                
        except Exception as e:
            print(f"Warning: Could not generate explanations for {model_name}: {e}")
            base_score -= 0.1
        
        return min(1.0, max(0.0, base_score))
    
    def _evaluate_calibration(self, model, X, y):
        """
        Evaluate model calibration using ECE.
        """
        try:
            from sklearn.model_selection import train_test_split
            
            # Split data
            X_train, X_test, y_train, y_test = train_test_split(
                X, y, test_size=0.3, random_state=self.random_state, stratify=y
            )
            
            # Fit model
            model.fit(X_train, y_train)
            
            # Get probabilities
            if hasattr(model, 'predict_proba'):
                y_proba = model.predict_proba(X_test)[:, 1]
            else:
                # For models without predict_proba, use decision function
                if hasattr(model, 'decision_function'):
                    scores = model.decision_function(X_test)
                    y_proba = 1 / (1 + np.exp(-scores))  # Sigmoid
                else:
                    return 0.5  # Default moderate calibration
            
            # Calculate ECE
            ece = self._expected_calibration_error(y_test, y_proba)
            return ece
            
        except Exception as e:
            print(f"Warning: Could not evaluate calibration: {e}")
            return 0.5
    
    def _expected_calibration_error(self, y_true, y_proba, n_bins=10):
        """Calculate Expected Calibration Error."""
        bin_boundaries = np.linspace(0, 1, n_bins + 1)
        bin_lowers = bin_boundaries[:-1]
        bin_uppers = bin_boundaries[1:]
        
        ece = 0
        for bin_lower, bin_upper in zip(bin_lowers, bin_uppers):
            in_bin = (y_proba > bin_lower) & (y_proba <= bin_upper)
            prop_in_bin = in_bin.mean()
            
            if prop_in_bin > 0:
                accuracy_in_bin = y_true[in_bin].mean()
                avg_confidence_in_bin = y_proba[in_bin].mean()
                ece += np.abs(avg_confidence_in_bin - accuracy_in_bin) * prop_in_bin
        
        return ece
    
    def _calculate_clinical_readiness(self, model_results):
        """
        Calculate a clinical readiness score based on multiple factors.
        
        Considers: accuracy, calibration, interpretability, variance
        """
        # Weighted scoring of different aspects
        weights = {
            'performance': 0.4,  # AUROC performance
            'calibration': 0.3,  # Calibration quality  
            'interpretability': 0.2,  # Interpretability score
            'robustness': 0.1    # Based on variance
        }
        
        # Performance score (AUROC)
        performance_score = model_results['roc_auc_mean']
        
        # Calibration score (lower ECE is better)
        calibration_ece = model_results.get('calibration_ece', 0.5)
        calibration_score = max(0, 1 - calibration_ece * 2)  # Scale ECE to 0-1
        
        # Interpretability score
        interpretability_score = model_results.get('interpretability_score', 0.5)
        
        # Variance score (lower std deviation is better)
        roc_std = model_results['roc_auc_std']
        robustness_score = max(0, 1 - roc_std * 10)  # Scale std to 0-1
        
        # Weighted sum
        clinical_score = (
            weights['performance'] * performance_score +
            weights['calibration'] * calibration_score +
            weights['interpretability'] * interpretability_score +
            weights['robustness'] * robustness_score
        )
        
        return clinical_score
    
    def _generate_comprehensive_report(self, results_df, output_dir):
        """
        Generate evaluation report.
        """
        report_path = os.path.join(output_dir, 'advanced_evaluation_report.md')
        
        with open(report_path, 'w') as f:
            f.write("# Advanced Heart Disease Model Evaluation Report\n\n")
            f.write(f"Generated on: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n\n")
            
            f.write("## Executive Summary\n\n")
            best_model = results_df.iloc[0]
            f.write(f"**Best Performing Model:** {best_model['model_name']}\n")
            f.write(f"**AUROC:** {best_model['roc_auc_mean']:.4f} ± {best_model['roc_auc_std']:.4f}\n")
            f.write(f"**Clinical Readiness Score:** {best_model['clinical_readiness_score']:.3f}/1.0\n\n")
            
            f.write("## Model Rankings\n\n")
            f.write("| Rank | Model | AUROC | Accuracy | F1 Score | Clinical Score |\n")
            f.write("|------|-------|-------|----------|----------|----------------|\n")
            
            for idx, row in results_df.head(10).iterrows():
                f.write(f"| {idx+1} | {row['model_name']} | "
                       f"{row['roc_auc_mean']:.4f} | "
                       f"{row['accuracy_mean']:.4f} | "
                       f"{row['f1_mean']:.4f} | "
                       f"{row['clinical_readiness_score']:.3f} |\n")
            
            f.write("\n## Key Insights\n\n")
            
            # Performance insights
            top_performers = results_df.head(3)['model_name'].tolist()
            f.write(f"• **Top Performers:** {', '.join(top_performers)}\n")
            
            # Interpretability insights  
            if 'interpretability_score' in results_df.columns:
                most_interpretable = results_df.loc[results_df['interpretability_score'].idxmax(), 'model_name']
                f.write(f"• **Most Interpretable:** {most_interpretable}\n")
            
            # Calibration insights
            if 'calibration_ece' in results_df.columns:
                best_calibrated = results_df.loc[results_df['calibration_ece'].idxmin(), 'model_name']
                f.write(f"• **Best Calibrated:** {best_calibrated}\n")
            
            f.write("\n## Recommendations\n\n")
            f.write("Based on the evaluation:\n\n")
            f.write(f"1. **Primary Recommendation:** Deploy {best_model['model_name']} "
                   f"for highest overall performance\n")
            
            if best_model['clinical_readiness_score'] > 0.8:
                f.write("2. **Clinical Deployment:** Model shows high clinical readiness\n")
            else:
                f.write("2. **Clinical Deployment:** Consider additional validation before deployment\n")
            
            f.write("3. **Monitoring:** Implement continuous monitoring of model performance\n")
            f.write("4. **Interpretability:** Ensure clinical staff are trained on model explanations\n")
        
        print(f"Report saved to {report_path}")

def run_advanced_experiment(data_path=None):
    """
    Run the advanced experiment suite.
    """
    print("Starting Advanced Heart Disease ML Experiment")
    print("=" * 60)
    
    # Load data (using original data loading if path not provided)
    if data_path is None:
        from ..src.data_utils import load_uci_heart, make_targets
        X, y_raw, meta = load_uci_heart()
        y = make_targets(y_raw, task='binary').values
        print(f"Loaded data: {X.shape[0]} samples, {X.shape[1]} features")
    else:
        # Load from provided path
        data = pd.read_csv(data_path)
        X = data.drop('target', axis=1)
        y = data['target'].values
    
    # Create advanced model zoo
    model_zoo = AdvancedModelZoo(random_state=42, use_optimization=False)  # Skip optimization for demo
    models = model_zoo.build_advanced_models(X, y, optimization_trials=20)
    
    # Setup evaluation suite
    evaluator = AdvancedEvaluationSuite(
        cv_folds=5,  # Reduced for demo
        random_state=42,
        include_interpretability=True,
        include_calibration=True
    )
    
    # Run evaluation
    results = evaluator.evaluate_all_models(
        models, X, y, 
        preprocessors=model_zoo.preprocessors_,
        output_dir="advanced_outputs"
    )
    
    print("\nExperiment complete.")
    print("Top 5 Models by AUROC:")
    print(results[['model_name', 'roc_auc_mean', 'roc_auc_std', 'clinical_readiness_score']].head())
    
    return results

if __name__ == "__main__":
    results = run_advanced_experiment()
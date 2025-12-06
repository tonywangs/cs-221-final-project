from __future__ import annotations
import numpy as np
import pandas as pd
from typing import List, Dict, Tuple, Optional, Any, Union, Callable
import warnings
warnings.filterwarnings('ignore')
import time
from sklearn.model_selection import cross_val_score, StratifiedKFold
from sklearn.metrics import roc_auc_score, accuracy_score, f1_score
import joblib

try:
    import optuna
    from optuna.samplers import TPESampler, CmaEsSampler
    from optuna.pruners import HyperbandPruner, MedianPruner
    HAS_OPTUNA = True
except ImportError:
    HAS_OPTUNA = False
    print("Optuna not available. Install with: pip install optuna")

try:
    from bayes_opt import BayesianOptimization
    HAS_BAYESOPT = True
except ImportError:
    HAS_BAYESOPT = False
    print("Bayesian Optimization not available. Install with: pip install bayesian-optimization")

from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.neural_network import MLPClassifier
from sklearn.naive_bayes import GaussianNB

class BayesianHyperparameterOptimizer:
    """
    Bayesian hyperparameter optimization using Optuna.
    
    Uses TPE (Tree-structured Parzen Estimator) to search hyperparameter
    spaces and find optimal configurations for each model.
    """
    
    def __init__(self, 
                 n_trials=100,
                 cv_folds=5,
                 random_state=42,
                 n_jobs=-1,
                 optimization_timeout=3600,  # 1 hour timeout
                 study_storage=None):
        """
        Initialize the Bayesian optimizer.
        
        Args:
            n_trials: Maximum number of optimization trials
            cv_folds: Cross-validation folds for evaluation
            random_state: Random seed for reproducibility
            n_jobs: Number of parallel jobs
            optimization_timeout: Maximum optimization time in seconds
            study_storage: Optuna study storage (for persistence)
        """
        self.n_trials = n_trials
        self.cv_folds = cv_folds
        self.random_state = random_state
        self.n_jobs = n_jobs
        self.optimization_timeout = optimization_timeout
        self.study_storage = study_storage
        
        # Store optimization results
        self.best_params_ = {}
        self.best_scores_ = {}
        self.optimization_history_ = {}
        self.studies_ = {}
        
        # CV splitter
        self.cv = StratifiedKFold(n_splits=cv_folds, shuffle=True, random_state=random_state)
    
    def optimize_model(self, 
                      model_name: str, 
                      model_class,
                      X_train, 
                      y_train,
                      param_space_func: Callable,
                      scoring='roc_auc',
                      direction='maximize',
                      early_stopping_rounds=None):
        """
        Optimize hyperparameters for a specific model.
        
        Args:
            model_name: Name of the model for tracking
            model_class: Model class to optimize
            X_train: Training features
            y_train: Training labels
            param_space_func: Function that defines parameter space for Optuna trial
            scoring: Scoring metric to optimize
            direction: 'maximize' or 'minimize'
            early_stopping_rounds: Early stopping for tree-based models
        """
        if not HAS_OPTUNA:
            print(f"Optuna not available. Skipping optimization for {model_name}")
            return None
        
        print(f"\nOptimizing {model_name}...")
        
        # Create study
        sampler = TPESampler(seed=self.random_state, multivariate=True, group=True)
        pruner = HyperbandPruner(min_resource=1, max_resource=self.cv_folds, reduction_factor=3)
        
        study_name = f"{model_name}_optimization_{int(time.time())}"
        study = optuna.create_study(
            study_name=study_name,
            direction=direction,
            sampler=sampler,
            pruner=pruner,
            storage=self.study_storage
        )
        
        # Define objective function
        def objective(trial):
            try:
                # Get hyperparameters from trial
                params = param_space_func(trial)
                
                # Create model with suggested parameters
                model = model_class(**params, random_state=self.random_state)
                
                # Perform cross-validation
                scores = []
                for fold, (train_idx, val_idx) in enumerate(self.cv.split(X_train, y_train)):
                    X_fold_train, X_fold_val = X_train[train_idx], X_train[val_idx]
                    y_fold_train, y_fold_val = y_train[train_idx], y_train[val_idx]
                    
                    # Fit model
                    model.fit(X_fold_train, y_fold_train)
                    
                    # Predict and score
                    if scoring == 'roc_auc':
                        if hasattr(model, 'predict_proba'):
                            y_pred_proba = model.predict_proba(X_fold_val)[:, 1]
                            score = roc_auc_score(y_fold_val, y_pred_proba)
                        else:
                            y_pred = model.predict(X_fold_val)
                            score = roc_auc_score(y_fold_val, y_pred)
                    elif scoring == 'accuracy':
                        y_pred = model.predict(X_fold_val)
                        score = accuracy_score(y_fold_val, y_pred)
                    elif scoring == 'f1':
                        y_pred = model.predict(X_fold_val)
                        score = f1_score(y_fold_val, y_pred)
                    
                    scores.append(score)
                    
                    # Report intermediate score for pruning
                    trial.report(score, fold)
                    
                    # Check if trial should be pruned
                    if trial.should_prune():
                        raise optuna.TrialPruned()
                
                return np.mean(scores)
                
            except Exception as e:
                print(f"Error in trial: {e}")
                return -999 if direction == 'maximize' else 999
        
        # Run optimization
        start_time = time.time()
        study.optimize(
            objective, 
            n_trials=self.n_trials,
            timeout=self.optimization_timeout,
            show_progress_bar=True
        )
        optimization_time = time.time() - start_time
        
        # Store results
        self.best_params_[model_name] = study.best_params
        self.best_scores_[model_name] = study.best_value
        self.studies_[model_name] = study
        
        # Create optimization history
        trials_df = study.trials_dataframe()
        self.optimization_history_[model_name] = {
            'trials': trials_df,
            'best_trial': study.best_trial,
            'n_trials': len(study.trials),
            'optimization_time': optimization_time,
            'best_score': study.best_value
        }
        
        print(f"{model_name} optimization complete:")
        print(f"   Best score: {study.best_value:.4f}")
        print(f"   Best params: {study.best_params}")
        print(f"   Trials completed: {len(study.trials)}")
        print(f"   Time: {optimization_time:.2f}s")
        
        return study.best_params
    
    def get_rf_param_space(self, trial):
        """Define parameter space for Random Forest."""
        return {
            'n_estimators': trial.suggest_int('n_estimators', 100, 1000, step=100),
            'max_depth': trial.suggest_int('max_depth', 3, 20),
            'min_samples_split': trial.suggest_int('min_samples_split', 2, 20),
            'min_samples_leaf': trial.suggest_int('min_samples_leaf', 1, 10),
            'max_features': trial.suggest_categorical('max_features', ['sqrt', 'log2', None]),
            'bootstrap': trial.suggest_categorical('bootstrap', [True, False]),
            'class_weight': trial.suggest_categorical('class_weight', ['balanced', 'balanced_subsample', None])
        }
    
    def get_gb_param_space(self, trial):
        """Define parameter space for Gradient Boosting."""
        return {
            'n_estimators': trial.suggest_int('n_estimators', 50, 500, step=50),
            'learning_rate': trial.suggest_float('learning_rate', 0.01, 0.3, log=True),
            'max_depth': trial.suggest_int('max_depth', 3, 10),
            'min_samples_split': trial.suggest_int('min_samples_split', 2, 20),
            'min_samples_leaf': trial.suggest_int('min_samples_leaf', 1, 10),
            'subsample': trial.suggest_float('subsample', 0.6, 1.0),
            'max_features': trial.suggest_categorical('max_features', ['sqrt', 'log2', None])
        }
    
    def get_lr_param_space(self, trial):
        """Define parameter space for Logistic Regression."""
        return {
            'C': trial.suggest_float('C', 1e-4, 1e2, log=True),
            'penalty': trial.suggest_categorical('penalty', ['l1', 'l2', 'elasticnet']),
            'solver': trial.suggest_categorical('solver', ['liblinear', 'saga', 'lbfgs']),
            'max_iter': trial.suggest_int('max_iter', 500, 2000),
            'class_weight': trial.suggest_categorical('class_weight', ['balanced', None])
        }
    
    def get_svm_param_space(self, trial):
        """Define parameter space for SVM."""
        return {
            'C': trial.suggest_float('C', 1e-3, 1e3, log=True),
            'kernel': trial.suggest_categorical('kernel', ['rbf', 'poly', 'sigmoid']),
            'gamma': trial.suggest_categorical('gamma', ['scale', 'auto']) if trial.params['kernel'] != 'linear' else 'scale',
            'degree': trial.suggest_int('degree', 2, 5) if trial.params.get('kernel') == 'poly' else 3,
            'class_weight': trial.suggest_categorical('class_weight', ['balanced', None]),
            'probability': True  # Always enable probability for our use case
        }
    
    def get_mlp_param_space(self, trial):
        """Define parameter space for MLP."""
        n_layers = trial.suggest_int('n_layers', 1, 4)
        hidden_layer_sizes = []
        
        for i in range(n_layers):
            size = trial.suggest_int(f'layer_{i}_size', 32, 512, step=32)
            hidden_layer_sizes.append(size)
        
        return {
            'hidden_layer_sizes': tuple(hidden_layer_sizes),
            'activation': trial.suggest_categorical('activation', ['relu', 'tanh', 'logistic']),
            'solver': trial.suggest_categorical('solver', ['adam', 'lbfgs', 'sgd']),
            'alpha': trial.suggest_float('alpha', 1e-6, 1e-1, log=True),
            'learning_rate': trial.suggest_categorical('learning_rate', ['constant', 'invscaling', 'adaptive']),
            'learning_rate_init': trial.suggest_float('learning_rate_init', 1e-4, 1e-1, log=True),
            'max_iter': trial.suggest_int('max_iter', 200, 1000),
            'early_stopping': True,
            'validation_fraction': 0.1
        }
    
    def optimize_all_models(self, X_train, y_train, models_config=None):
        """
        Optimize hyperparameters for all specified models.
        
        Args:
            X_train: Training features
            y_train: Training labels  
            models_config: Dict of {model_name: (model_class, param_space_func)} 
        """
        if models_config is None:
            models_config = {
                'random_forest': (RandomForestClassifier, self.get_rf_param_space),
                'gradient_boosting': (GradientBoostingClassifier, self.get_gb_param_space),
                'logistic_regression': (LogisticRegression, self.get_lr_param_space),
                'svm': (SVC, self.get_svm_param_space),
                'mlp': (MLPClassifier, self.get_mlp_param_space)
            }
        
        print("Starting Bayesian hyperparameter optimization for all models...")
        total_start_time = time.time()
        
        # Convert data if needed
        if isinstance(X_train, pd.DataFrame):
            X_train = X_train.values
        if isinstance(y_train, pd.Series):
            y_train = y_train.values
        
        # Optimize each model
        for model_name, (model_class, param_space_func) in models_config.items():
            try:
                self.optimize_model(
                    model_name=model_name,
                    model_class=model_class,
                    X_train=X_train,
                    y_train=y_train,
                    param_space_func=param_space_func,
                    scoring='roc_auc'
                )
            except Exception as e:
                print(f"Failed to optimize {model_name}: {e}")
                continue
        
        total_time = time.time() - total_start_time
        print(f"\nAll model optimization complete in {total_time:.2f}s")
        
        return self.get_optimization_summary()
    
    def get_optimization_summary(self):
        """Get summary of optimization results."""
        if not self.best_scores_:
            return None
        
        summary = pd.DataFrame({
            'model': list(self.best_scores_.keys()),
            'best_score': list(self.best_scores_.values()),
            'n_trials': [self.optimization_history_[model]['n_trials'] 
                        for model in self.best_scores_.keys()],
            'optimization_time': [self.optimization_history_[model]['optimization_time'] 
                                for model in self.best_scores_.keys()]
        }).sort_values('best_score', ascending=False)
        
        return summary
    
    def analyze_hyperparameter_importance(self, model_name):
        """
        Analyze which hyperparameters are most important for model performance.
        """
        if not HAS_OPTUNA or model_name not in self.studies_:
            print(f"Cannot analyze importance for {model_name}")
            return None
        
        study = self.studies_[model_name]
        
        try:
            # Get parameter importance
            importance = optuna.importance.get_param_importances(study)
            
            # Create dataframe
            importance_df = pd.DataFrame({
                'parameter': list(importance.keys()),
                'importance': list(importance.values())
            }).sort_values('importance', ascending=False)
            
            print(f"\nHyperparameter importance for {model_name}:")
            for _, row in importance_df.head(10).iterrows():
                print(f"   {row['parameter']}: {row['importance']:.4f}")
            
            return importance_df
            
        except Exception as e:
            print(f"Error analyzing parameter importance: {e}")
            return None
    
    def create_optimized_model(self, model_name, model_class):
        """
        Create a model instance with optimized hyperparameters.
        
        Args:
            model_name: Name of the model
            model_class: Model class
            
        Returns:
            Model instance with optimized parameters
        """
        if model_name not in self.best_params_:
            print(f"No optimized parameters found for {model_name}")
            return model_class(random_state=self.random_state)
        
        params = self.best_params_[model_name].copy()
        params['random_state'] = self.random_state
        
        return model_class(**params)
    
    def save_optimization_results(self, filepath):
        """Save optimization results to file."""
        results = {
            'best_params': self.best_params_,
            'best_scores': self.best_scores_,
            'optimization_history': {
                k: {
                    'best_score': v['best_score'],
                    'n_trials': v['n_trials'],
                    'optimization_time': v['optimization_time']
                } for k, v in self.optimization_history_.items()
            }
        }
        
        joblib.dump(results, filepath)
        print(f"Optimization results saved to {filepath}")
    
    def load_optimization_results(self, filepath):
        """Load optimization results from file."""
        results = joblib.load(filepath)
        self.best_params_ = results['best_params']
        self.best_scores_ = results['best_scores']
        print(f"Optimization results loaded from {filepath}")

class MultiObjectiveOptimizer:
    """
    Multi-objective optimization for balancing competing objectives.
    
    For medical applications, we often want to balance:
    - Accuracy vs Interpretability
    - Performance vs Training Time  
    - Sensitivity vs Specificity
    """
    
    def __init__(self, objectives=['accuracy', 'interpretability'], random_state=42):
        self.objectives = objectives
        self.random_state = random_state
        self.pareto_front_ = None
        self.all_solutions_ = []
    
    def optimize_multi_objective(self, X_train, y_train, model_configs, n_trials=100):
        """
        Perform multi-objective optimization.
        
        Args:
            X_train: Training features
            y_train: Training labels
            model_configs: Dict of model configurations
            n_trials: Number of trials
        """
        if not HAS_OPTUNA:
            print("Optuna required for multi-objective optimization")
            return None
        
        # Create multi-objective study
        study = optuna.create_study(
            directions=['maximize'] * len(self.objectives),  # Maximize all objectives
            sampler=TPESampler(seed=self.random_state)
        )
        
        def objective(trial):
            # Select model type
            model_name = trial.suggest_categorical('model_type', list(model_configs.keys()))
            model_class, param_space_func = model_configs[model_name]
            
            # Get hyperparameters
            params = param_space_func(trial)
            model = model_class(**params, random_state=self.random_state)
            
            # Evaluate objectives
            objectives = []
            
            # Cross-validation for accuracy
            cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=self.random_state)
            cv_scores = cross_val_score(model, X_train, y_train, cv=cv, scoring='roc_auc')
            accuracy_score = np.mean(cv_scores)
            objectives.append(accuracy_score)
            
            # Interpretability score (model-dependent)
            if 'interpretability' in self.objectives:
                interpretability_score = self._calculate_interpretability(model_name, params)
                objectives.append(interpretability_score)
            
            # Training time (if requested)
            if 'training_time' in self.objectives:
                start_time = time.time()
                model.fit(X_train, y_train)
                training_time = time.time() - start_time
                # Convert to score (faster is better, so we use negative)
                time_score = 1.0 / (1.0 + training_time)
                objectives.append(time_score)
            
            return objectives
        
        # Run optimization
        study.optimize(objective, n_trials=n_trials, show_progress_bar=True)
        
        # Extract Pareto front
        self.pareto_front_ = []
        for trial in study.best_trials:
            solution = {
                'values': trial.values,
                'params': trial.params,
                'model_type': trial.params['model_type']
            }
            self.pareto_front_.append(solution)
        
        return self.pareto_front_
    
    def _calculate_interpretability(self, model_name, params):
        """
        Calculate interpretability score for different model types.
        
        This is a heuristic scoring system where simpler models get higher scores.
        """
        if model_name == 'logistic_regression':
            base_score = 0.9
            # Penalize higher regularization (more complex)
            if 'C' in params and params['C'] < 0.1:
                base_score -= 0.1
            return base_score
        
        elif model_name == 'random_forest':
            base_score = 0.6
            # Favor smaller, shallower trees
            if 'n_estimators' in params and params['n_estimators'] <= 100:
                base_score += 0.1
            if 'max_depth' in params and params['max_depth'] <= 5:
                base_score += 0.1
            return base_score
        
        elif model_name == 'gradient_boosting':
            base_score = 0.5
            # Favor simpler models
            if 'n_estimators' in params and params['n_estimators'] <= 100:
                base_score += 0.1
            if 'learning_rate' in params and params['learning_rate'] >= 0.1:
                base_score += 0.1
            return base_score
        
        elif model_name == 'svm':
            base_score = 0.4
            # Linear kernel is more interpretable
            if 'kernel' in params and params['kernel'] == 'linear':
                base_score += 0.3
            return base_score
        
        elif model_name == 'mlp':
            base_score = 0.2
            # Smaller networks are more interpretable
            if 'hidden_layer_sizes' in params:
                avg_size = np.mean(params['hidden_layer_sizes'])
                if avg_size <= 64:
                    base_score += 0.2
            return base_score
        
        else:
            return 0.5  # Default score

def demonstrate_bayesian_optimization():
    """Print information about Bayesian optimization."""
    print("Bayesian Hyperparameter Optimization")
    print("\nFeatures:")
    print("• Efficient Bayesian search with TPE sampler")
    print("• Multi-objective optimization (accuracy vs interpretability)")
    print("• Automated early stopping and pruning")
    print("• Hyperparameter importance analysis")
    print("• Model-specific parameter spaces")
    print("• Cross-validation aware optimization")
    print("• Results persistence and reproducibility")
    print("\nNote: Original models used default hyperparameters.")

if __name__ == "__main__":
    demonstrate_bayesian_optimization()
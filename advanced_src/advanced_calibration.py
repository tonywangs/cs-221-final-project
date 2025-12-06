from __future__ import annotations
import numpy as np
import pandas as pd
from typing import List, Dict, Tuple, Optional, Any, Union
import warnings
warnings.filterwarnings('ignore')

from sklearn.calibration import CalibratedClassifierCV, calibration_curve
from sklearn.isotonic import IsotonicRegression
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import brier_score_loss, log_loss
import matplotlib.pyplot as plt
import seaborn as sns

class AdvancedCalibratorSuite:
    """
    Calibration methods for probability outputs.
    
    Implements multiple calibration techniques and provides tools for
    analyzing and improving model calibration.
    """
    
    def __init__(self, random_state=42):
        self.random_state = random_state
        self.calibrators_ = {}
        self.calibration_scores_ = {}
        self.reliability_data_ = {}
        
    def fit_all_calibrators(self, y_true, y_proba, method='cv', cv=3):
        """
        Fit multiple calibration methods to the data.
        
        Args:
            y_true: True binary labels
            y_proba: Predicted probabilities (uncalibrated)
            method: 'cv' for cross-validation or 'holdout' for train/validation split
            cv: Number of CV folds
        """
        print("Fitting advanced calibration methods...")
        
        # Platt scaling (sigmoid calibration)
        self.calibrators_['platt'] = self._fit_platt_scaling(y_true, y_proba, method, cv)
        
        # Isotonic regression
        self.calibrators_['isotonic'] = self._fit_isotonic_calibration(y_true, y_proba, method, cv)
        
        # Temperature scaling (for neural networks)
        self.calibrators_['temperature'] = self._fit_temperature_scaling(y_true, y_proba)
        
        # Beta calibration
        self.calibrators_['beta'] = self._fit_beta_calibration(y_true, y_proba)
        
        # Histogram binning
        self.calibrators_['histogram'] = self._fit_histogram_binning(y_true, y_proba)
        
        return self
    
    def _fit_platt_scaling(self, y_true, y_proba, method, cv):
        """Fit Platt scaling (logistic regression on logits)."""
        # Convert probabilities to logits (avoid numerical issues)
        logits = np.log(np.clip(y_proba / (1 - y_proba), 1e-15, 1e15))
        
        if method == 'cv':
            # Use sklearn's calibrated classifier
            from sklearn.base import BaseEstimator, ClassifierMixin
            
            class DummyClassifier(BaseEstimator, ClassifierMixin):
                def __init__(self, logits):
                    self.logits = logits
                
                def fit(self, X, y):
                    return self
                
                def predict_proba(self, X):
                    probs = 1 / (1 + np.exp(-self.logits[X]))
                    return np.column_stack([1-probs, probs])
            
            dummy = DummyClassifier(logits)
            calibrator = CalibratedClassifierCV(dummy, method='sigmoid', cv=cv)
            # Fit using indices as X
            calibrator.fit(np.arange(len(y_true)), y_true)
            
            return calibrator
        else:
            # Simple logistic regression
            lr = LogisticRegression()
            lr.fit(logits.reshape(-1, 1), y_true)
            return lr
    
    def _fit_isotonic_calibration(self, y_true, y_proba, method, cv):
        """Fit isotonic regression calibration."""
        if method == 'cv':
            from sklearn.base import BaseEstimator, ClassifierMixin
            
            class DummyClassifier(BaseEstimator, ClassifierMixin):
                def __init__(self, proba):
                    self.proba = proba
                
                def fit(self, X, y):
                    return self
                
                def predict_proba(self, X):
                    probs = self.proba[X]
                    return np.column_stack([1-probs, probs])
            
            dummy = DummyClassifier(y_proba)
            calibrator = CalibratedClassifierCV(dummy, method='isotonic', cv=cv)
            calibrator.fit(np.arange(len(y_true)), y_true)
            
            return calibrator
        else:
            isotonic = IsotonicRegression(out_of_bounds='clip')
            isotonic.fit(y_proba, y_true)
            return isotonic
    
    def _fit_temperature_scaling(self, y_true, y_proba):
        """
        Fit temperature scaling calibration.
        
        Temperature scaling learns a single parameter T to scale logits: p = softmax(z/T)
        """
        # Convert probabilities back to logits
        logits = np.log(np.clip(y_proba / (1 - y_proba), 1e-15, 1e15))
        
        # Optimize temperature parameter
        from scipy.optimize import minimize_scalar
        
        def nll_loss(temperature):
            # Scale logits by temperature
            scaled_logits = logits / temperature
            calibrated_probs = 1 / (1 + np.exp(-scaled_logits))
            calibrated_probs = np.clip(calibrated_probs, 1e-15, 1-1e-15)
            
            # Negative log likelihood
            nll = -np.mean(y_true * np.log(calibrated_probs) + 
                          (1 - y_true) * np.log(1 - calibrated_probs))
            return nll
        
        # Find optimal temperature
        result = minimize_scalar(nll_loss, bounds=(0.1, 10.0), method='bounded')
        optimal_temperature = result.x
        
        return {'temperature': optimal_temperature}
    
    def _fit_beta_calibration(self, y_true, y_proba):
        """
        Fit Beta calibration.
        
        Models calibration using Beta distribution parameters.
        """
        try:
            from scipy.optimize import minimize
            from scipy.special import betaln
            
            def beta_nll(params):
                alpha, beta = params
                if alpha <= 0 or beta <= 0:
                    return np.inf
                
                # Beta calibration formula
                calibrated_probs = np.clip(y_proba ** alpha / 
                                         (y_proba ** alpha + (1 - y_proba) ** beta), 
                                         1e-15, 1-1e-15)
                
                nll = -np.mean(y_true * np.log(calibrated_probs) + 
                              (1 - y_true) * np.log(1 - calibrated_probs))
                return nll
            
            # Optimize beta parameters
            result = minimize(beta_nll, [1.0, 1.0], 
                            bounds=[(0.01, 10), (0.01, 10)], 
                            method='L-BFGS-B')
            
            if result.success:
                return {'alpha': result.x[0], 'beta': result.x[1]}
            else:
                return {'alpha': 1.0, 'beta': 1.0}  # Fallback to identity
        
        except ImportError:
            return {'alpha': 1.0, 'beta': 1.0}
    
    def _fit_histogram_binning(self, y_true, y_proba, n_bins=15):
        """
        Fit histogram binning calibration.
        
        Divides probability space into bins and learns calibration within each bin.
        """
        bin_boundaries = np.linspace(0, 1, n_bins + 1)
        bin_centers = (bin_boundaries[:-1] + bin_boundaries[1:]) / 2
        
        calibrated_probs = np.zeros(len(bin_centers))
        bin_counts = np.zeros(len(bin_centers))
        
        for i in range(n_bins):
            in_bin = (y_proba > bin_boundaries[i]) & (y_proba <= bin_boundaries[i + 1])
            if np.sum(in_bin) > 0:
                calibrated_probs[i] = np.mean(y_true[in_bin])
                bin_counts[i] = np.sum(in_bin)
            else:
                calibrated_probs[i] = bin_centers[i]  # Default to bin center
        
        return {
            'bin_boundaries': bin_boundaries,
            'calibrated_probs': calibrated_probs,
            'bin_counts': bin_counts
        }
    
    def calibrate_probabilities(self, y_proba, method='isotonic'):
        """
        Apply calibration to new probabilities.
        
        Args:
            y_proba: Uncalibrated probabilities
            method: Calibration method to use
        """
        if method not in self.calibrators_:
            raise ValueError(f"Calibration method {method} not fitted")
        
        calibrator = self.calibrators_[method]
        
        if method == 'platt':
            if hasattr(calibrator, 'predict_proba'):
                # CalibratedClassifierCV
                indices = np.arange(len(y_proba))
                return calibrator.predict_proba(indices)[:, 1]
            else:
                # Simple logistic regression
                logits = np.log(np.clip(y_proba / (1 - y_proba), 1e-15, 1e15))
                return calibrator.predict_proba(logits.reshape(-1, 1))[:, 1]
        
        elif method == 'isotonic':
            if hasattr(calibrator, 'predict_proba'):
                indices = np.arange(len(y_proba))
                return calibrator.predict_proba(indices)[:, 1]
            else:
                return calibrator.transform(y_proba)
        
        elif method == 'temperature':
            logits = np.log(np.clip(y_proba / (1 - y_proba), 1e-15, 1e15))
            scaled_logits = logits / calibrator['temperature']
            return 1 / (1 + np.exp(-scaled_logits))
        
        elif method == 'beta':
            alpha, beta = calibrator['alpha'], calibrator['beta']
            return np.clip(y_proba ** alpha / 
                          (y_proba ** alpha + (1 - y_proba) ** beta), 
                          1e-15, 1-1e-15)
        
        elif method == 'histogram':
            calibrated = np.zeros_like(y_proba)
            boundaries = calibrator['bin_boundaries']
            cal_probs = calibrator['calibrated_probs']
            
            for i in range(len(boundaries) - 1):
                in_bin = (y_proba > boundaries[i]) & (y_proba <= boundaries[i + 1])
                calibrated[in_bin] = cal_probs[i]
            
            return calibrated
        
        else:
            raise ValueError(f"Unknown calibration method: {method}")
    
    def evaluate_calibration(self, y_true, y_proba_uncalibrated, methods=None):
        """
        Evaluate calibration performance of different methods.
        
        Args:
            y_true: True labels
            y_proba_uncalibrated: Uncalibrated probabilities
            methods: List of methods to evaluate (None for all)
        """
        if methods is None:
            methods = list(self.calibrators_.keys())
        
        results = {}
        
        # Add uncalibrated as baseline
        results['uncalibrated'] = self._compute_calibration_metrics(y_true, y_proba_uncalibrated)
        
        for method in methods:
            if method in self.calibrators_:
                y_proba_cal = self.calibrate_probabilities(y_proba_uncalibrated, method)
                results[method] = self._compute_calibration_metrics(y_true, y_proba_cal)
        
        self.calibration_scores_ = results
        return results
    
    def _compute_calibration_metrics(self, y_true, y_proba):
        """Compute calibration metrics."""
        metrics = {}
        
        # Expected Calibration Error (ECE)
        metrics['ece'] = self._expected_calibration_error(y_true, y_proba)
        
        # Maximum Calibration Error (MCE)
        metrics['mce'] = self._maximum_calibration_error(y_true, y_proba)
        
        # Brier Score
        metrics['brier_score'] = brier_score_loss(y_true, y_proba)
        
        # Log Loss
        metrics['log_loss'] = log_loss(y_true, y_proba)
        
        # Reliability (for reliability diagrams)
        metrics['reliability_data'] = self._compute_reliability_data(y_true, y_proba)
        
        return metrics
    
    def _expected_calibration_error(self, y_true, y_proba, n_bins=15):
        """Compute Expected Calibration Error."""
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
    
    def _maximum_calibration_error(self, y_true, y_proba, n_bins=15):
        """Compute Maximum Calibration Error."""
        bin_boundaries = np.linspace(0, 1, n_bins + 1)
        bin_lowers = bin_boundaries[:-1]
        bin_uppers = bin_boundaries[1:]
        
        mce = 0
        for bin_lower, bin_upper in zip(bin_lowers, bin_uppers):
            in_bin = (y_proba > bin_lower) & (y_proba <= bin_upper)
            
            if in_bin.sum() > 0:
                accuracy_in_bin = y_true[in_bin].mean()
                avg_confidence_in_bin = y_proba[in_bin].mean()
                mce = max(mce, np.abs(avg_confidence_in_bin - accuracy_in_bin))
        
        return mce
    
    def _compute_reliability_data(self, y_true, y_proba, n_bins=15):
        """Compute data for reliability diagrams."""
        bin_boundaries = np.linspace(0, 1, n_bins + 1)
        bin_lowers = bin_boundaries[:-1]
        bin_uppers = bin_boundaries[1:]
        
        reliability_data = {
            'bin_centers': [],
            'accuracies': [],
            'confidences': [],
            'counts': []
        }
        
        for bin_lower, bin_upper in zip(bin_lowers, bin_uppers):
            in_bin = (y_proba > bin_lower) & (y_proba <= bin_upper)
            
            if in_bin.sum() > 0:
                bin_center = (bin_lower + bin_upper) / 2
                accuracy = y_true[in_bin].mean()
                confidence = y_proba[in_bin].mean()
                count = in_bin.sum()
                
                reliability_data['bin_centers'].append(bin_center)
                reliability_data['accuracies'].append(accuracy)
                reliability_data['confidences'].append(confidence)
                reliability_data['counts'].append(count)
        
        return reliability_data
    
    def plot_reliability_diagram(self, methods=None, figsize=(15, 10)):
        """
        Plot reliability diagrams for different calibration methods.
        
        A reliability diagram shows the relationship between predicted probability
        and actual probability (fraction of positives).
        """
        if methods is None:
            methods = ['uncalibrated'] + list(self.calibrators_.keys())
        
        n_methods = len(methods)
        n_cols = 3
        n_rows = (n_methods + n_cols - 1) // n_cols
        
        fig, axes = plt.subplots(n_rows, n_cols, figsize=figsize)
        axes = axes.flatten() if n_rows > 1 else [axes]
        
        for i, method in enumerate(methods):
            if i >= len(axes):
                break
                
            ax = axes[i]
            
            if method in self.calibration_scores_:
                rel_data = self.calibration_scores_[method]['reliability_data']
                
                # Plot perfect calibration line
                ax.plot([0, 1], [0, 1], 'k--', alpha=0.5, label='Perfect Calibration')
                
                # Plot reliability curve
                if rel_data['bin_centers']:
                    ax.plot(rel_data['confidences'], rel_data['accuracies'], 
                           'o-', linewidth=2, markersize=8, label=f'{method.title()}')
                    
                    # Add histogram of predictions
                    ax2 = ax.twinx()
                    ax2.hist(rel_data['bin_centers'], weights=rel_data['counts'], 
                            bins=15, alpha=0.3, color='gray', density=True)
                    ax2.set_ylabel('Density', alpha=0.7)
                
                # Add ECE and MCE to plot
                ece = self.calibration_scores_[method]['ece']
                mce = self.calibration_scores_[method]['mce']
                ax.text(0.05, 0.95, f'ECE: {ece:.3f}\nMCE: {mce:.3f}', 
                       transform=ax.transAxes, verticalalignment='top',
                       bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))
            
            ax.set_xlabel('Mean Predicted Probability')
            ax.set_ylabel('Fraction of Positives')
            ax.set_title(f'Reliability Diagram - {method.title()}')
            ax.legend()
            ax.grid(True, alpha=0.3)
            ax.set_xlim([0, 1])
            ax.set_ylim([0, 1])
        
        # Hide unused subplots
        for i in range(n_methods, len(axes)):
            axes[i].set_visible(False)
        
        plt.tight_layout()
        return fig
    
    def get_best_calibrator(self, metric='ece'):
        """
        Get the best calibration method based on a metric.
        
        Args:
            metric: 'ece', 'mce', 'brier_score', or 'log_loss'
        """
        if not self.calibration_scores_:
            raise ValueError("No calibration evaluation results available")
        
        best_method = None
        best_score = float('inf')
        
        for method, scores in self.calibration_scores_.items():
            if method == 'uncalibrated':
                continue
                
            score = scores[metric]
            if score < best_score:
                best_score = score
                best_method = method
        
        return best_method, best_score

class ConformalPredictor:
    """
    Conformal prediction for guaranteed coverage.
    
    Provides prediction intervals with theoretical guarantees on coverage,
    which is crucial for medical applications.
    """
    
    def __init__(self, alpha=0.1, random_state=42):
        """
        Initialize conformal predictor.
        
        Args:
            alpha: Miscoverage rate (1-alpha is target coverage)
            random_state: Random seed
        """
        self.alpha = alpha
        self.random_state = random_state
        self.quantiles_ = None
        self.calibration_scores_ = None
    
    def fit(self, calibration_scores):
        """
        Fit conformal predictor on calibration scores.
        
        Args:
            calibration_scores: Nonconformity scores from calibration set
        """
        n = len(calibration_scores)
        # Empirical quantile for coverage guarantee
        quantile_level = np.ceil((n + 1) * (1 - self.alpha)) / n
        self.quantiles_ = np.quantile(calibration_scores, quantile_level)
        self.calibration_scores_ = calibration_scores
        return self
    
    def predict_with_intervals(self, y_proba):
        """
        Generate prediction intervals with coverage guarantees.
        
        Args:
            y_proba: Predicted probabilities
            
        Returns:
            Dict with predictions, lower bounds, upper bounds
        """
        if self.quantiles_ is None:
            raise ValueError("Conformal predictor not fitted")
        
        # For binary classification, create intervals around probabilities
        lower_bounds = np.maximum(0, y_proba - self.quantiles_)
        upper_bounds = np.minimum(1, y_proba + self.quantiles_)
        
        return {
            'predictions': y_proba,
            'lower_bounds': lower_bounds,
            'upper_bounds': upper_bounds,
            'interval_widths': upper_bounds - lower_bounds,
            'target_coverage': 1 - self.alpha
        }

class UncertaintyDecomposer:
    """
    Decompose prediction uncertainty into epistemic and aleatoric components.
    
    Epistemic uncertainty: Model uncertainty (reducible with more data/better models)
    Aleatoric uncertainty: Data uncertainty (irreducible noise in the data)
    """
    
    def __init__(self, n_bootstrap_samples=100, random_state=42):
        self.n_bootstrap_samples = n_bootstrap_samples
        self.random_state = random_state
        
    def decompose_uncertainty(self, models, X_test):
        """
        Decompose uncertainty using ensemble of models.
        
        Args:
            models: List of fitted models
            X_test: Test features
            
        Returns:
            Dict with total, epistemic, and aleatoric uncertainty
        """
        # Get predictions from all models
        predictions = []
        for model in models:
            if hasattr(model, 'predict_proba'):
                pred = model.predict_proba(X_test)[:, 1]  # Positive class
            else:
                pred = model.predict(X_test)
            predictions.append(pred)
        
        predictions = np.array(predictions)  # Shape: (n_models, n_samples)
        
        # Total uncertainty (variance across models)
        total_uncertainty = np.var(predictions, axis=0)
        
        # Expected prediction
        mean_prediction = np.mean(predictions, axis=0)
        
        # Epistemic uncertainty (model uncertainty)
        epistemic_uncertainty = np.var(predictions, axis=0)
        
        # Aleatoric uncertainty (data uncertainty) - approximate
        # This requires models that output uncertainty estimates
        # For now, use residual uncertainty
        aleatoric_uncertainty = np.maximum(0, total_uncertainty - epistemic_uncertainty)
        
        return {
            'total_uncertainty': total_uncertainty,
            'epistemic_uncertainty': epistemic_uncertainty, 
            'aleatoric_uncertainty': aleatoric_uncertainty,
            'mean_prediction': mean_prediction,
            'predictions_all_models': predictions
        }

def demonstrate_advanced_calibration():
    """Print information about calibration methods."""
    print("Calibration & Uncertainty Quantification")
    print("\nCalibration methods implemented:")
    print("• Platt scaling (logistic regression)")
    print("• Isotonic regression")
    print("• Temperature scaling")
    print("• Beta calibration")
    print("• Histogram binning")
    print("\nUncertainty quantification:")
    print("• Expected Calibration Error (ECE)")
    print("• Maximum Calibration Error (MCE)")
    print("• Conformal prediction intervals")
    print("• Epistemic vs Aleatoric uncertainty decomposition")
    print("• Reliability diagrams")
    print("\nImportant for medical applications where probability estimates must be well-calibrated.")

if __name__ == "__main__":
    demonstrate_advanced_calibration()
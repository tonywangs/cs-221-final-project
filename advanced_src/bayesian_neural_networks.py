from __future__ import annotations
import numpy as np
import pandas as pd
from typing import List, Dict, Tuple, Optional, Any
from sklearn.base import BaseEstimator, ClassifierMixin
from sklearn.preprocessing import StandardScaler
import warnings
warnings.filterwarnings('ignore')

try:
    import torch
    import torch.nn as nn
    import torch.nn.functional as F
    import torch.optim as optim
    from torch.utils.data import DataLoader, TensorDataset
    HAS_TORCH = True
except ImportError:
    HAS_TORCH = False
    print("PyTorch not available. Bayesian Neural Networks will use simplified implementations.")

class VariationalBayesianNN(BaseEstimator, ClassifierMixin):
    """
    Variational Bayesian Neural Network using mean-field variational inference.
    
    Each weight has a prior distribution (typically Gaussian) and the posterior
    is approximated using a variational distribution. During inference, we
    sample from the weight distributions to get uncertainty estimates.
    """
    
    def __init__(self, 
                 hidden_layers=[128, 64, 32],
                 n_samples=100,
                 learning_rate=0.001,
                 epochs=200,
                 batch_size=32,
                 prior_sigma=1.0,
                 kl_weight=0.01,
                 device='cpu',
                 random_state=42):
        self.hidden_layers = hidden_layers
        self.n_samples = n_samples
        self.learning_rate = learning_rate
        self.epochs = epochs
        self.batch_size = batch_size
        self.prior_sigma = prior_sigma
        self.kl_weight = kl_weight
        self.device = device
        self.random_state = random_state
        
        self.model_ = None
        self.scaler_ = None
        self.classes_ = None
        self.n_features_in_ = None
        self.is_fitted_ = False
        
        if HAS_TORCH:
            torch.manual_seed(random_state)
            np.random.seed(random_state)
    
    def _create_model(self, n_features, n_classes):
        """Create the variational Bayesian neural network model."""
        if not HAS_TORCH:
            return None
            
        class BayesianLayer(nn.Module):
            """Bayesian linear layer with weight uncertainty."""
            
            def __init__(self, in_features, out_features, prior_sigma=1.0):
                super().__init__()
                self.in_features = in_features
                self.out_features = out_features
                self.prior_sigma = prior_sigma
                
                # Mean parameters
                self.weight_mu = nn.Parameter(torch.zeros(out_features, in_features))
                self.bias_mu = nn.Parameter(torch.zeros(out_features))
                
                # Std parameters (log scale for numerical stability)
                self.weight_log_sigma = nn.Parameter(torch.full((out_features, in_features), -3.0))
                self.bias_log_sigma = nn.Parameter(torch.full((out_features,), -3.0))
                
                self.reset_parameters()
            
            def reset_parameters(self):
                # Initialize similar to standard linear layer
                nn.init.xavier_uniform_(self.weight_mu)
                nn.init.zeros_(self.bias_mu)
            
            def forward(self, x):
                # Sample weights from variational distribution
                weight_sigma = torch.exp(self.weight_log_sigma)
                bias_sigma = torch.exp(self.bias_log_sigma)
                
                weight_eps = torch.randn_like(self.weight_mu)
                bias_eps = torch.randn_like(self.bias_mu)
                
                weight = self.weight_mu + weight_sigma * weight_eps
                bias = self.bias_mu + bias_sigma * bias_eps
                
                return F.linear(x, weight, bias)
            
            def kl_divergence(self):
                """Compute KL divergence between variational and prior distributions."""
                # KL(q(w) || p(w)) for Gaussian distributions
                weight_sigma = torch.exp(self.weight_log_sigma)
                bias_sigma = torch.exp(self.bias_log_sigma)
                
                # KL for weights
                weight_kl = -0.5 * torch.sum(
                    1 + 2 * self.weight_log_sigma - 
                    (self.weight_mu.pow(2) + weight_sigma.pow(2)) / (self.prior_sigma ** 2)
                )
                
                # KL for biases
                bias_kl = -0.5 * torch.sum(
                    1 + 2 * self.bias_log_sigma - 
                    (self.bias_mu.pow(2) + bias_sigma.pow(2)) / (self.prior_sigma ** 2)
                )
                
                return weight_kl + bias_kl
        
        class VariationalBNN(nn.Module):
            """Complete Variational Bayesian Neural Network."""
            
            def __init__(self, n_features, n_classes, hidden_layers, prior_sigma):
                super().__init__()
                self.layers = nn.ModuleList()
                
                # Input layer
                prev_size = n_features
                for hidden_size in hidden_layers:
                    self.layers.append(BayesianLayer(prev_size, hidden_size, prior_sigma))
                    prev_size = hidden_size
                
                # Output layer
                self.layers.append(BayesianLayer(prev_size, n_classes, prior_sigma))
            
            def forward(self, x):
                for i, layer in enumerate(self.layers[:-1]):
                    x = F.relu(layer(x))
                x = self.layers[-1](x)  # No activation on output layer
                return x
            
            def kl_divergence(self):
                """Total KL divergence of all layers."""
                return sum(layer.kl_divergence() for layer in self.layers)
        
        return VariationalBNN(n_features, n_classes, self.hidden_layers, self.prior_sigma)
    
    def fit(self, X, y):
        """Fit the Bayesian neural network."""
        if not HAS_TORCH:
            # Fallback to simple ensemble for demonstration
            return self._fit_ensemble_fallback(X, y)
        
        # Prepare data
        if isinstance(X, pd.DataFrame):
            X = X.values
        if isinstance(y, pd.Series):
            y = y.values
            
        self.classes_ = np.unique(y)
        n_classes = len(self.classes_)
        self.n_features_in_ = X.shape[1]
        
        # Scale features
        self.scaler_ = StandardScaler()
        X_scaled = self.scaler_.fit_transform(X)
        
        # Convert to torch tensors
        X_tensor = torch.FloatTensor(X_scaled).to(self.device)
        y_tensor = torch.LongTensor(y).to(self.device)
        
        # Create model
        self.model_ = self._create_model(self.n_features_in_, n_classes)
        self.model_.to(self.device)
        
        # Create data loader
        dataset = TensorDataset(X_tensor, y_tensor)
        dataloader = DataLoader(dataset, batch_size=self.batch_size, shuffle=True)
        
        # Optimizer
        optimizer = optim.Adam(self.model_.parameters(), lr=self.learning_rate)
        
        # Training loop
        self.model_.train()
        for epoch in range(self.epochs):
            total_loss = 0
            n_batches = 0
            
            for batch_X, batch_y in dataloader:
                optimizer.zero_grad()
                
                # Forward pass
                outputs = self.model_(batch_X)
                
                # Likelihood loss (cross-entropy)
                likelihood_loss = F.cross_entropy(outputs, batch_y)
                
                # KL divergence loss
                kl_loss = self.model_.kl_divergence()
                
                # Total loss (ELBO = likelihood - KL)
                total_loss_batch = likelihood_loss + self.kl_weight * kl_loss / len(dataset)
                
                # Backward pass
                total_loss_batch.backward()
                optimizer.step()
                
                total_loss += total_loss_batch.item()
                n_batches += 1
            
            if epoch % 50 == 0:
                avg_loss = total_loss / n_batches
                print(f"Epoch {epoch}, Average Loss: {avg_loss:.4f}")
        
        self.is_fitted_ = True
        return self
    
    def _fit_ensemble_fallback(self, X, y):
        """Fallback implementation using ensemble of standard networks."""
        from sklearn.neural_network import MLPClassifier
        from sklearn.ensemble import BaggingClassifier
        
        self.classes_ = np.unique(y)
        self.n_features_in_ = X.shape[1] if hasattr(X, 'shape') else len(X.iloc[0])
        
        # Scale features
        self.scaler_ = StandardScaler()
        if isinstance(X, pd.DataFrame):
            X_scaled = self.scaler_.fit_transform(X)
        else:
            X_scaled = self.scaler_.fit_transform(X)
        
        # Create ensemble of MLPs for uncertainty estimation
        base_estimator = MLPClassifier(
            hidden_layer_sizes=tuple(self.hidden_layers),
            max_iter=self.epochs,
            random_state=self.random_state
        )
        
        self.model_ = BaggingClassifier(
            estimator=base_estimator,
            n_estimators=min(10, self.n_samples // 10),  # Reduce for speed
            random_state=self.random_state,
            n_jobs=-1
        )
        
        self.model_.fit(X_scaled, y)
        self.is_fitted_ = True
        return self
    
    def predict_proba(self, X):
        """Predict class probabilities with uncertainty estimation."""
        if not self.is_fitted_:
            raise ValueError("Model must be fitted before prediction")
        
        if not HAS_TORCH:
            return self._predict_proba_fallback(X)
        
        # Prepare data
        if isinstance(X, pd.DataFrame):
            X = X.values
        
        X_scaled = self.scaler_.transform(X)
        X_tensor = torch.FloatTensor(X_scaled).to(self.device)
        
        # Monte Carlo sampling for uncertainty estimation
        self.model_.eval()
        predictions = []
        
        with torch.no_grad():
            for _ in range(self.n_samples):
                outputs = self.model_(X_tensor)
                probs = F.softmax(outputs, dim=1)
                predictions.append(probs.cpu().numpy())
        
        # Stack predictions
        predictions = np.stack(predictions, axis=0)  # (n_samples, n_instances, n_classes)
        
        # Mean prediction
        mean_predictions = np.mean(predictions, axis=0)
        
        # Store uncertainty information
        self.prediction_uncertainty_ = np.std(predictions, axis=0)  # Epistemic uncertainty
        self.prediction_samples_ = predictions
        
        return mean_predictions
    
    def _predict_proba_fallback(self, X):
        """Fallback probability prediction."""
        if isinstance(X, pd.DataFrame):
            X = X.values
        X_scaled = self.scaler_.transform(X)
        return self.model_.predict_proba(X_scaled)
    
    def predict(self, X):
        """Predict class labels."""
        proba = self.predict_proba(X)
        return self.classes_[np.argmax(proba, axis=1)]
    
    def predict_with_uncertainty(self, X):
        """
        Predict with uncertainty quantification.
        
        Returns:
            predictions: Most likely class
            probabilities: Mean probabilities
            epistemic_uncertainty: Model uncertainty
            confidence_intervals: 95% confidence intervals for probabilities
        """
        probabilities = self.predict_proba(X)
        predictions = self.classes_[np.argmax(probabilities, axis=1)]
        
        if hasattr(self, 'prediction_uncertainty_'):
            epistemic_uncertainty = self.prediction_uncertainty_
            
            # Calculate confidence intervals
            if hasattr(self, 'prediction_samples_'):
                ci_lower = np.percentile(self.prediction_samples_, 2.5, axis=0)
                ci_upper = np.percentile(self.prediction_samples_, 97.5, axis=0)
                confidence_intervals = (ci_lower, ci_upper)
            else:
                confidence_intervals = None
        else:
            epistemic_uncertainty = None
            confidence_intervals = None
        
        return {
            'predictions': predictions,
            'probabilities': probabilities,
            'epistemic_uncertainty': epistemic_uncertainty,
            'confidence_intervals': confidence_intervals
        }

class MonteCarloDropoutNN(BaseEstimator, ClassifierMixin):
    """
    Monte Carlo Dropout Neural Network for uncertainty estimation.
    
    Uses dropout at test time to approximate Bayesian inference.
    Simpler than full Bayesian networks but still provides uncertainty estimates.
    """
    
    def __init__(self, 
                 hidden_layers=[128, 64, 32],
                 dropout_rate=0.3,
                 n_samples=50,
                 learning_rate=0.001,
                 epochs=100,
                 batch_size=32,
                 device='cpu',
                 random_state=42):
        self.hidden_layers = hidden_layers
        self.dropout_rate = dropout_rate
        self.n_samples = n_samples
        self.learning_rate = learning_rate
        self.epochs = epochs
        self.batch_size = batch_size
        self.device = device
        self.random_state = random_state
        
        self.model_ = None
        self.scaler_ = None
        self.classes_ = None
        self.n_features_in_ = None
        self.is_fitted_ = False
        
        if HAS_TORCH:
            torch.manual_seed(random_state)
            np.random.seed(random_state)
    
    def _create_model(self, n_features, n_classes):
        """Create Monte Carlo Dropout model."""
        if not HAS_TORCH:
            return None
        
        class MCDropoutNN(nn.Module):
            def __init__(self, n_features, n_classes, hidden_layers, dropout_rate):
                super().__init__()
                self.layers = nn.ModuleList()
                self.dropouts = nn.ModuleList()
                
                # Input layer
                prev_size = n_features
                for hidden_size in hidden_layers:
                    self.layers.append(nn.Linear(prev_size, hidden_size))
                    self.dropouts.append(nn.Dropout(dropout_rate))
                    prev_size = hidden_size
                
                # Output layer
                self.layers.append(nn.Linear(prev_size, n_classes))
            
            def forward(self, x, mc_dropout=False):
                for i, (layer, dropout) in enumerate(zip(self.layers[:-1], self.dropouts)):
                    x = F.relu(layer(x))
                    if mc_dropout or self.training:
                        x = dropout(x)
                
                x = self.layers[-1](x)  # No dropout on output layer
                return x
        
        return MCDropoutNN(n_features, n_classes, self.hidden_layers, self.dropout_rate)
    
    def fit(self, X, y):
        """Fit the MC Dropout network."""
        if not HAS_TORCH:
            return self._fit_ensemble_fallback(X, y)
        
        # Prepare data
        if isinstance(X, pd.DataFrame):
            X = X.values
        if isinstance(y, pd.Series):
            y = y.values
            
        self.classes_ = np.unique(y)
        n_classes = len(self.classes_)
        self.n_features_in_ = X.shape[1]
        
        # Scale features
        self.scaler_ = StandardScaler()
        X_scaled = self.scaler_.fit_transform(X)
        
        # Convert to torch tensors
        X_tensor = torch.FloatTensor(X_scaled).to(self.device)
        y_tensor = torch.LongTensor(y).to(self.device)
        
        # Create model
        self.model_ = self._create_model(self.n_features_in_, n_classes)
        self.model_.to(self.device)
        
        # Create data loader
        dataset = TensorDataset(X_tensor, y_tensor)
        dataloader = DataLoader(dataset, batch_size=self.batch_size, shuffle=True)
        
        # Optimizer and loss
        optimizer = optim.Adam(self.model_.parameters(), lr=self.learning_rate)
        criterion = nn.CrossEntropyLoss()
        
        # Training loop
        self.model_.train()
        for epoch in range(self.epochs):
            total_loss = 0
            n_batches = 0
            
            for batch_X, batch_y in dataloader:
                optimizer.zero_grad()
                
                outputs = self.model_(batch_X)
                loss = criterion(outputs, batch_y)
                
                loss.backward()
                optimizer.step()
                
                total_loss += loss.item()
                n_batches += 1
            
            if epoch % 25 == 0:
                avg_loss = total_loss / n_batches
                print(f"MC Dropout Epoch {epoch}, Average Loss: {avg_loss:.4f}")
        
        self.is_fitted_ = True
        return self
    
    def _fit_ensemble_fallback(self, X, y):
        """Fallback using sklearn."""
        from sklearn.neural_network import MLPClassifier
        
        self.classes_ = np.unique(y)
        self.n_features_in_ = X.shape[1] if hasattr(X, 'shape') else len(X.iloc[0])
        
        self.scaler_ = StandardScaler()
        if isinstance(X, pd.DataFrame):
            X_scaled = self.scaler_.fit_transform(X)
        else:
            X_scaled = self.scaler_.fit_transform(X)
        
        self.model_ = MLPClassifier(
            hidden_layer_sizes=tuple(self.hidden_layers),
            max_iter=self.epochs,
            random_state=self.random_state
        )
        
        self.model_.fit(X_scaled, y)
        self.is_fitted_ = True
        return self
    
    def predict_proba(self, X):
        """Predict probabilities using Monte Carlo dropout."""
        if not self.is_fitted_:
            raise ValueError("Model must be fitted before prediction")
        
        if not HAS_TORCH:
            return self._predict_proba_fallback(X)
        
        # Prepare data
        if isinstance(X, pd.DataFrame):
            X = X.values
        
        X_scaled = self.scaler_.transform(X)
        X_tensor = torch.FloatTensor(X_scaled).to(self.device)
        
        # MC Dropout inference
        self.model_.eval()  # Set to eval mode but keep dropout active
        predictions = []
        
        with torch.no_grad():
            for _ in range(self.n_samples):
                # Forward pass with MC dropout
                outputs = self.model_(X_tensor, mc_dropout=True)
                probs = F.softmax(outputs, dim=1)
                predictions.append(probs.cpu().numpy())
        
        # Stack and average predictions
        predictions = np.stack(predictions, axis=0)
        mean_predictions = np.mean(predictions, axis=0)
        
        # Store uncertainty
        self.prediction_uncertainty_ = np.std(predictions, axis=0)
        self.prediction_samples_ = predictions
        
        return mean_predictions
    
    def _predict_proba_fallback(self, X):
        """Fallback probability prediction."""
        if isinstance(X, pd.DataFrame):
            X = X.values
        X_scaled = self.scaler_.transform(X)
        return self.model_.predict_proba(X_scaled)
    
    def predict(self, X):
        """Predict class labels."""
        proba = self.predict_proba(X)
        return self.classes_[np.argmax(proba, axis=1)]

class BayesianEnsemble(BaseEstimator, ClassifierMixin):
    """
    Ensemble of diverse models for Bayesian Model Averaging.
    
    Combines different model types and uses Bayesian averaging
    for final predictions with uncertainty quantification.
    """
    
    def __init__(self, 
                 base_models=None,
                 n_bootstrap_samples=10,
                 random_state=42):
        self.base_models = base_models
        self.n_bootstrap_samples = n_bootstrap_samples
        self.random_state = random_state
        
        self.models_ = []
        self.model_weights_ = None
        self.classes_ = None
        self.scaler_ = StandardScaler()
        self.is_fitted_ = False
    
    def _get_default_models(self):
        """Get default ensemble of diverse models."""
        from sklearn.ensemble import RandomForestClassifier
        from sklearn.linear_model import LogisticRegression
        from sklearn.svm import SVC
        from sklearn.naive_bayes import GaussianNB
        
        return [
            ('rf', RandomForestClassifier(n_estimators=100, random_state=self.random_state)),
            ('lr', LogisticRegression(random_state=self.random_state, max_iter=1000)),
            ('svm', SVC(probability=True, random_state=self.random_state)),
            ('nb', GaussianNB()),
            ('vbnn', VariationalBayesianNN(epochs=100, random_state=self.random_state)),
            ('mcdrop', MonteCarloDropoutNN(epochs=50, random_state=self.random_state))
        ]
    
    def fit(self, X, y):
        """Fit the Bayesian ensemble."""
        if self.base_models is None:
            self.base_models = self._get_default_models()
        
        self.classes_ = np.unique(y)
        
        # Scale features
        if isinstance(X, pd.DataFrame):
            X_scaled = self.scaler_.fit_transform(X)
            X_scaled = pd.DataFrame(X_scaled, columns=X.columns, index=X.index)
        else:
            X_scaled = self.scaler_.fit_transform(X)
        
        # Fit each model
        self.models_ = []
        for name, model in self.base_models:
            try:
                print(f"Fitting {name}...")
                fitted_model = model.fit(X_scaled, y)
                self.models_.append((name, fitted_model))
            except Exception as e:
                print(f"Failed to fit {name}: {e}")
                continue
        
        # Equal weights for now (could use validation-based weighting)
        self.model_weights_ = np.ones(len(self.models_)) / len(self.models_)
        
        self.is_fitted_ = True
        return self
    
    def predict_proba(self, X):
        """Predict probabilities using Bayesian model averaging."""
        if not self.is_fitted_:
            raise ValueError("Ensemble must be fitted before prediction")
        
        # Scale features
        if isinstance(X, pd.DataFrame):
            X_scaled = self.scaler_.transform(X)
            X_scaled = pd.DataFrame(X_scaled, columns=X.columns, index=X.index)
        else:
            X_scaled = self.scaler_.transform(X)
        
        # Get predictions from each model
        all_predictions = []
        for name, model in self.models_:
            try:
                proba = model.predict_proba(X_scaled)
                all_predictions.append(proba)
            except Exception as e:
                print(f"Failed to predict with {name}: {e}")
                continue
        
        if not all_predictions:
            raise ValueError("No models produced valid predictions")
        
        # Weighted average
        predictions_array = np.stack(all_predictions, axis=0)
        weights = self.model_weights_[:len(all_predictions)]
        weights = weights / weights.sum()  # Renormalize
        
        mean_predictions = np.average(predictions_array, axis=0, weights=weights)
        
        # Store prediction uncertainty (disagreement between models)
        self.prediction_uncertainty_ = np.std(predictions_array, axis=0)
        self.prediction_samples_ = predictions_array
        
        return mean_predictions
    
    def predict(self, X):
        """Predict class labels."""
        proba = self.predict_proba(X)
        return self.classes_[np.argmax(proba, axis=1)]

def demonstrate_bayesian_models():
    """Print information about Bayesian neural network models."""
    print("Bayesian Neural Networks implementation")
    print("Available models:")
    print("- VariationalBayesianNN: Full Bayesian inference with weight uncertainty")
    print("- MonteCarloDropoutNN: MC Dropout for uncertainty estimation")  
    print("- BayesianEnsemble: Bayesian Model Averaging across diverse models")
    print("All models provide uncertainty quantification for medical predictions.")

if __name__ == "__main__":
    demonstrate_bayesian_models()
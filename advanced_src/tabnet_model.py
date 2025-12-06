from __future__ import annotations
import numpy as np
import pandas as pd
from typing import List, Dict, Tuple, Optional, Any, Union
from sklearn.base import BaseEstimator, ClassifierMixin
from sklearn.preprocessing import StandardScaler, LabelEncoder
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

try:
    from pytorch_tabnet.tab_model import TabNetClassifier as PyTorchTabNet
    HAS_PYTORCH_TABNET = True
except ImportError:
    HAS_PYTORCH_TABNET = False
    print("pytorch-tabnet not available. Using custom TabNet implementation.")

class CustomTabNet(BaseEstimator, ClassifierMixin):
    """
    Custom implementation of TabNet for educational purposes and flexibility.
    """
    
    def __init__(self,
                 n_d=64,              # Width of decision prediction layer
                 n_a=64,              # Width of attention embedding for each mask
                 n_steps=5,           # Number of steps in the architecture
                 gamma=1.5,           # Coefficient for feature reusage in the masks  
                 n_independent=2,     # Number of independent Gated Linear Unit layers at each step
                 n_shared=2,          # Number of shared Gated Linear Unit layers at each step
                 epsilon=1e-15,       # Small number to avoid log(0)
                 momentum=0.98,       # Batch normalization momentum
                 mask_type='entmax',  # Type of mask function (sparsemax or entmax)
                 lambda_sparse=1e-3,  # Sparsity regularization coefficient
                 learning_rate=0.02,
                 epochs=100,
                 batch_size=256,
                 device='cpu',
                 random_state=42,
                 verbose=1):
        
        self.n_d = n_d
        self.n_a = n_a
        self.n_steps = n_steps
        self.gamma = gamma
        self.n_independent = n_independent
        self.n_shared = n_shared
        self.epsilon = epsilon
        self.momentum = momentum
        self.mask_type = mask_type
        self.lambda_sparse = lambda_sparse
        self.learning_rate = learning_rate
        self.epochs = epochs
        self.batch_size = batch_size
        self.device = device
        self.random_state = random_state
        self.verbose = verbose
        
        self.model_ = None
        self.scaler_ = None
        self.label_encoder_ = None
        self.classes_ = None
        self.n_features_in_ = None
        self.is_fitted_ = False
        self.feature_importances_ = None
        self.attention_masks_ = None
        
        if HAS_TORCH:
            torch.manual_seed(random_state)
            np.random.seed(random_state)
    
    def _glu_block(self, input_layer, n_units, shared_layers=None, n_shared=None):
        """Gated Linear Unit block."""
        if not HAS_TORCH:
            return None
            
        # Shared layers
        if shared_layers is None:
            shared_layers = []
        
        x = input_layer
        for shared_layer in shared_layers:
            x = shared_layer(x)
        
        # Split for gating
        # We'll create a simple GLU approximation
        linear = nn.Linear(x.shape[-1], n_units * 2)
        x = linear(x)
        gate, input_part = torch.chunk(x, 2, dim=-1)
        gate = torch.sigmoid(gate)
        
        return gate * input_part
    
    def _create_model(self, n_features, n_classes):
        """Create the TabNet model architecture."""
        if not HAS_TORCH:
            return None
        
        class TabNetModel(nn.Module):
            def __init__(self, n_features, n_classes, n_d, n_a, n_steps, gamma, 
                         n_independent, n_shared, momentum, lambda_sparse):
                super().__init__()
                self.n_features = n_features
                self.n_classes = n_classes
                self.n_d = n_d
                self.n_a = n_a
                self.n_steps = n_steps
                self.gamma = gamma
                self.lambda_sparse = lambda_sparse
                
                # Feature transformer (initial embedding)
                self.initial_bn = nn.BatchNorm1d(n_features, momentum=momentum)
                
                # Shared layers for all steps
                self.shared_layers = nn.ModuleList()
                for i in range(n_shared):
                    self.shared_layers.append(nn.Linear(n_features, n_features))
                    self.shared_layers.append(nn.BatchNorm1d(n_features, momentum=momentum))
                
                # Step-specific layers
                self.step_layers = nn.ModuleList()
                self.attention_layers = nn.ModuleList()
                
                for step in range(n_steps):
                    # Independent layers for this step
                    independent = nn.ModuleList()
                    for i in range(n_independent):
                        independent.append(nn.Linear(n_features, n_d))
                        independent.append(nn.BatchNorm1d(n_d, momentum=momentum))
                    self.step_layers.append(independent)
                    
                    # Attention layer
                    self.attention_layers.append(nn.Linear(n_features, n_features))
                
                # Final classifier
                self.final_projection = nn.Linear(n_d, n_classes)
                
                # Prior for attention (learnable)
                self.attention_prior = nn.Parameter(torch.ones(n_features))
            
            def forward(self, x):
                # Initial batch normalization
                x = self.initial_bn(x)
                prior = self.attention_prior.unsqueeze(0).expand(x.size(0), -1)
                
                step_outputs = []
                attention_masks = []
                
                for step in range(self.n_steps):
                    # Attention mechanism
                    attention_input = x if step == 0 else torch.cat([x, step_outputs[-1]], dim=-1)
                    if attention_input.size(-1) != self.n_features:
                        # Project back to feature space if needed
                        attention_input = x
                    
                    attention_weights = self.attention_layers[step](attention_input)
                    attention_weights = torch.sigmoid(attention_weights)
                    
                    # Apply prior (feature reusage penalty)
                    masked_features = attention_weights * prior * x
                    attention_masks.append(attention_weights)
                    
                    # Update prior for next step
                    prior = prior * (self.gamma - attention_weights)
                    
                    # Forward through step-specific layers
                    step_input = masked_features
                    
                    # Shared layers
                    for shared_layer in self.shared_layers:
                        if isinstance(shared_layer, nn.Linear):
                            step_input = F.relu(shared_layer(step_input))
                        else:  # BatchNorm
                            step_input = shared_layer(step_input)
                    
                    # Independent layers
                    for ind_layer in self.step_layers[step]:
                        if isinstance(ind_layer, nn.Linear):
                            step_input = F.relu(ind_layer(step_input))
                        else:  # BatchNorm
                            step_input = ind_layer(step_input)
                    
                    step_outputs.append(step_input)
                
                # Aggregate step outputs
                final_representation = torch.stack(step_outputs, dim=0).sum(dim=0)
                
                # Final prediction
                output = self.final_projection(final_representation)
                
                return output, attention_masks
            
            def get_feature_importance(self, attention_masks):
                """Calculate feature importance from attention masks."""
                # Stack all attention masks and average across steps and samples
                stacked_masks = torch.stack(attention_masks, dim=0)  # (n_steps, batch_size, n_features)
                importance = stacked_masks.mean(dim=(0, 1))  # Average across steps and samples
                return importance.detach().cpu().numpy()
        
        return TabNetModel(n_features, n_classes, self.n_d, self.n_a, self.n_steps,
                          self.gamma, self.n_independent, self.n_shared, 
                          self.momentum, self.lambda_sparse)
    
    def fit(self, X, y):
        """Fit the TabNet model."""
        if not HAS_TORCH:
            # Fallback to ensemble
            return self._fit_fallback(X, y)
        
        # Prepare data
        if isinstance(X, pd.DataFrame):
            X = X.values
        if isinstance(y, pd.Series):
            y = y.values
        
        # Encode labels
        self.label_encoder_ = LabelEncoder()
        y_encoded = self.label_encoder_.fit_transform(y)
        self.classes_ = self.label_encoder_.classes_
        n_classes = len(self.classes_)
        self.n_features_in_ = X.shape[1]
        
        # Scale features
        self.scaler_ = StandardScaler()
        X_scaled = self.scaler_.fit_transform(X)
        
        # Convert to torch tensors
        X_tensor = torch.FloatTensor(X_scaled).to(self.device)
        y_tensor = torch.LongTensor(y_encoded).to(self.device)
        
        # Create model
        self.model_ = self._create_model(self.n_features_in_, n_classes)
        if self.model_ is None:
            return self._fit_fallback(X, y)
        
        self.model_.to(self.device)
        
        # Create data loader
        dataset = TensorDataset(X_tensor, y_tensor)
        dataloader = DataLoader(dataset, batch_size=self.batch_size, shuffle=True)
        
        # Optimizer
        optimizer = optim.Adam(self.model_.parameters(), lr=self.learning_rate)
        criterion = nn.CrossEntropyLoss()
        
        # Training loop
        self.model_.train()
        for epoch in range(self.epochs):
            total_loss = 0
            total_sparsity_loss = 0
            all_attention_masks = []
            n_batches = 0
            
            for batch_X, batch_y in dataloader:
                optimizer.zero_grad()
                
                # Forward pass
                outputs, attention_masks = self.model_(batch_X)
                
                # Classification loss
                class_loss = criterion(outputs, batch_y)
                
                # Sparsity regularization
                sparsity_loss = 0
                for mask in attention_masks:
                    sparsity_loss += torch.sum(torch.abs(mask)) / mask.numel()
                sparsity_loss = self.lambda_sparse * sparsity_loss / len(attention_masks)
                
                # Total loss
                total_loss_batch = class_loss + sparsity_loss
                
                # Backward pass
                total_loss_batch.backward()
                optimizer.step()
                
                total_loss += class_loss.item()
                total_sparsity_loss += sparsity_loss.item()
                all_attention_masks.extend(attention_masks)
                n_batches += 1
            
            if self.verbose and epoch % 20 == 0:
                avg_loss = total_loss / n_batches
                avg_sparse_loss = total_sparsity_loss / n_batches
                print(f"TabNet Epoch {epoch}: Loss={avg_loss:.4f}, Sparsity={avg_sparse_loss:.6f}")
        
        # Calculate feature importance
        self.model_.eval()
        with torch.no_grad():
            outputs, attention_masks = self.model_(X_tensor)
            self.feature_importances_ = self.model_.get_feature_importance(attention_masks)
            self.attention_masks_ = attention_masks
        
        self.is_fitted_ = True
        return self
    
    def _fit_fallback(self, X, y):
        """Fallback to ensemble when PyTorch not available."""
        from sklearn.ensemble import GradientBoostingClassifier
        from sklearn.neural_network import MLPClassifier
        
        self.classes_ = np.unique(y)
        self.n_features_in_ = X.shape[1] if hasattr(X, 'shape') else len(X.iloc[0])
        
        # Scale features
        self.scaler_ = StandardScaler()
        if isinstance(X, pd.DataFrame):
            X_scaled = self.scaler_.fit_transform(X)
        else:
            X_scaled = self.scaler_.fit_transform(X)
        
        # Use gradient boosting as approximation
        self.model_ = GradientBoostingClassifier(
            n_estimators=200,
            learning_rate=0.1,
            max_depth=6,
            random_state=self.random_state
        )
        
        self.model_.fit(X_scaled, y)
        self.feature_importances_ = self.model_.feature_importances_
        self.is_fitted_ = True
        return self
    
    def predict_proba(self, X):
        """Predict class probabilities."""
        if not self.is_fitted_:
            raise ValueError("Model must be fitted before prediction")
        
        if not HAS_TORCH or not hasattr(self.model_, 'final_projection'):
            return self._predict_proba_fallback(X)
        
        # Prepare data
        if isinstance(X, pd.DataFrame):
            X = X.values
        
        X_scaled = self.scaler_.transform(X)
        X_tensor = torch.FloatTensor(X_scaled).to(self.device)
        
        self.model_.eval()
        with torch.no_grad():
            outputs, _ = self.model_(X_tensor)
            probabilities = F.softmax(outputs, dim=1)
            return probabilities.cpu().numpy()
    
    def _predict_proba_fallback(self, X):
        """Fallback probability prediction."""
        if isinstance(X, pd.DataFrame):
            X = X.values
        X_scaled = self.scaler_.transform(X)
        return self.model_.predict_proba(X_scaled)
    
    def predict(self, X):
        """Predict class labels."""
        proba = self.predict_proba(X)
        predicted_indices = np.argmax(proba, axis=1)
        if hasattr(self, 'label_encoder_') and self.label_encoder_:
            return self.label_encoder_.inverse_transform(predicted_indices)
        else:
            return self.classes_[predicted_indices]
    
    def get_feature_importance(self, normalize=True):
        """Get feature importance scores."""
        if not self.is_fitted_:
            raise ValueError("Model must be fitted to get feature importance")
        
        importance = self.feature_importances_.copy()
        if normalize and importance.sum() > 0:
            importance = importance / importance.sum()
        
        return importance

class OptimizedTabNetClassifier(BaseEstimator, ClassifierMixin):
    """
    Wrapper around pytorch-tabnet with optimized hyperparameters for medical data.
    """
    
    def __init__(self,
                 n_d=64,
                 n_a=64, 
                 n_steps=5,
                 gamma=1.3,
                 lambda_sparse=1e-3,
                 optimizer_fn=optim.Adam,
                 optimizer_params={'lr': 0.02},
                 max_epochs=100,
                 patience=15,
                 batch_size=256,
                 virtual_batch_size=128,
                 device_name='auto',
                 seed=42,
                 verbose=1):
        
        self.n_d = n_d
        self.n_a = n_a
        self.n_steps = n_steps
        self.gamma = gamma
        self.lambda_sparse = lambda_sparse
        self.optimizer_fn = optimizer_fn
        self.optimizer_params = optimizer_params
        self.max_epochs = max_epochs
        self.patience = patience
        self.batch_size = batch_size
        self.virtual_batch_size = virtual_batch_size
        self.device_name = device_name
        self.seed = seed
        self.verbose = verbose
        
        self.model_ = None
        self.classes_ = None
        self.n_features_in_ = None
        self.is_fitted_ = False
    
    def fit(self, X, y):
        """Fit the optimized TabNet model."""
        if not HAS_PYTORCH_TABNET:
            # Fallback to custom implementation
            fallback_model = CustomTabNet(
                n_d=self.n_d,
                n_a=self.n_a,
                n_steps=self.n_steps,
                gamma=self.gamma,
                lambda_sparse=self.lambda_sparse,
                epochs=self.max_epochs,
                batch_size=self.batch_size,
                random_state=self.seed,
                verbose=self.verbose
            )
            return fallback_model.fit(X, y)
        
        self.classes_ = np.unique(y)
        self.n_features_in_ = X.shape[1] if hasattr(X, 'shape') else len(X.iloc[0])
        
        # Convert data if needed
        if isinstance(X, pd.DataFrame):
            X_array = X.values
        else:
            X_array = X
        
        if isinstance(y, pd.Series):
            y_array = y.values
        else:
            y_array = y
        
        # Create and fit TabNet
        self.model_ = PyTorchTabNet(
            n_d=self.n_d,
            n_a=self.n_a,
            n_steps=self.n_steps,
            gamma=self.gamma,
            lambda_sparse=self.lambda_sparse,
            optimizer_fn=self.optimizer_fn,
            optimizer_params=self.optimizer_params,
            scheduler_params={"step_size": 10, "gamma": 0.9},
            mask_type='entmax',
            device_name=self.device_name,
            verbose=self.verbose,
            seed=self.seed
        )
        
        # Fit with early stopping
        self.model_.fit(
            X_array, y_array,
            eval_metric=['accuracy', 'logloss'],
            max_epochs=self.max_epochs,
            patience=self.patience,
            batch_size=self.batch_size,
            virtual_batch_size=self.virtual_batch_size,
        )
        
        self.is_fitted_ = True
        return self
    
    def predict_proba(self, X):
        """Predict class probabilities."""
        if not self.is_fitted_:
            raise ValueError("Model must be fitted before prediction")
        
        if isinstance(X, pd.DataFrame):
            X_array = X.values
        else:
            X_array = X
        
        return self.model_.predict_proba(X_array)
    
    def predict(self, X):
        """Predict class labels."""
        proba = self.predict_proba(X)
        return self.classes_[np.argmax(proba, axis=1)]
    
    def get_feature_importance(self):
        """Get global feature importance."""
        if not self.is_fitted_:
            raise ValueError("Model must be fitted to get feature importance")
        
        return self.model_.feature_importances_
    
    def explain(self, X):
        """Get local explanations for predictions."""
        if not self.is_fitted_:
            raise ValueError("Model must be fitted to generate explanations")
        
        if isinstance(X, pd.DataFrame):
            X_array = X.values
        else:
            X_array = X
        
        # Local explanations (attention masks for each sample)
        explanations = self.model_.explain(X_array)
        return explanations

def demonstrate_tabnet():
    """Print information about TabNet implementations."""
    print("TabNet implementations")
    print("\nAvailable TabNet models:")
    print("1. CustomTabNet - Educational implementation with full control")
    print("2. OptimizedTabNetClassifier - Production-ready with pytorch-tabnet backend")
    print("\nKey TabNet features:")
    print("- Learnable feature selection via attention")
    print("- Sequential multi-step reasoning")
    print("- Built-in interpretability") 
    print("- Sparsity regularization")
    print("- Superior performance on tabular data")

if __name__ == "__main__":
    demonstrate_tabnet()
import numpy as np
import pandas as pd
from typing import Dict, List, Tuple, Optional, Union
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE
from sklearn.preprocessing import StandardScaler
from sklearn.feature_selection import SelectKBest, f_regression, mutual_info_regression
import umap
import warnings

class DimensionalityReducer:
    """
    Comprehensive dimensionality reduction for financial features.
    Provides PCA, UMAP, t-SNE, and feature selection capabilities.
    """
    
    def __init__(self):
        self.fitted_models = {}
        self.scalers = {}
        self.feature_importance = {}
        
    def fit_pca(self, features: np.ndarray, n_components: Optional[int] = None, 
                variance_threshold: float = 0.95) -> Dict:
        """
        Fit PCA to the feature data.
        
        Args:
            features: Feature matrix of shape [n_samples, n_features]
            n_components: Number of components to keep (if None, determined by variance_threshold)
            variance_threshold: Minimum cumulative variance to explain
        
        Returns:
            Dictionary with PCA results and information
        """
        print(f"Fitting PCA to features with shape {features.shape}")
        
        # Scale features first
        scaler = StandardScaler()
        features_scaled = scaler.fit_transform(features)
        self.scalers['pca'] = scaler
        
        # Fit PCA with all components first to determine optimal number
        pca_full = PCA()
        pca_full.fit(features_scaled)
        
        # Determine number of components if not specified
        if n_components is None:
            cumsum_variance = np.cumsum(pca_full.explained_variance_ratio_)
            n_components = np.argmax(cumsum_variance >= variance_threshold) + 1
            print(f"Selected {n_components} components to explain {variance_threshold:.1%} of variance")
        
        # Fit final PCA
        pca = PCA(n_components=n_components)
        transformed_features = pca.fit_transform(features_scaled)
        
        self.fitted_models['pca'] = pca
        
        results = {
            'transformed_features': transformed_features,
            'explained_variance_ratio': pca.explained_variance_ratio_,
            'cumulative_variance': np.cumsum(pca.explained_variance_ratio_),
            'n_components': n_components,
            'total_variance_explained': np.sum(pca.explained_variance_ratio_),
            'components': pca.components_,
            'feature_importance': np.abs(pca.components_).mean(axis=0)
        }
        
        self.feature_importance['pca'] = results['feature_importance']
        
        print(f"PCA completed: {n_components} components explain {results['total_variance_explained']:.3f} of variance")
        return results
    
    def fit_umap(self, features: np.ndarray, n_components: int = 2, 
                 n_neighbors: int = 15, min_dist: float = 0.1, 
                 random_state: int = 42) -> Dict:
        """
        Fit UMAP to the feature data.
        
        Args:
            features: Feature matrix of shape [n_samples, n_features]
            n_components: Number of dimensions in the reduced space
            n_neighbors: Number of neighbors for UMAP
            min_dist: Minimum distance between points in reduced space
            random_state: Random seed for reproducibility
        
        Returns:
            Dictionary with UMAP results
        """
        print(f"Fitting UMAP to features with shape {features.shape}")
        
        # Scale features
        scaler = StandardScaler()
        features_scaled = scaler.fit_transform(features)
        self.scalers['umap'] = scaler
        
        # Fit UMAP
        umap_reducer = umap.UMAP(
            n_components=n_components,
            n_neighbors=n_neighbors,
            min_dist=min_dist,
            random_state=random_state
        )
        
        transformed_features = umap_reducer.fit_transform(features_scaled)
        self.fitted_models['umap'] = umap_reducer
        
        results = {
            'transformed_features': transformed_features,
            'n_components': n_components,
            'n_neighbors': n_neighbors,
            'min_dist': min_dist
        }
        
        print(f"UMAP completed: Reduced to {n_components} dimensions")
        return results
    
    def fit_tsne(self, features: np.ndarray, n_components: int = 2, 
                 perplexity: float = 30.0, learning_rate: float = 200.0,
                 n_iter: int = 1000, random_state: int = 42) -> Dict:
        """
        Fit t-SNE to the feature data.
        
        Args:
            features: Feature matrix of shape [n_samples, n_features]
            n_components: Number of dimensions in the reduced space
            perplexity: Perplexity parameter for t-SNE
            learning_rate: Learning rate for t-SNE
            n_iter: Number of iterations
            random_state: Random seed for reproducibility
        
        Returns:
            Dictionary with t-SNE results
        """
        print(f"Fitting t-SNE to features with shape {features.shape}")
        
        # Scale features
        scaler = StandardScaler()
        features_scaled = scaler.fit_transform(features)
        self.scalers['tsne'] = scaler
        
        # Apply PCA first if features are high-dimensional (t-SNE recommendation)
        if features_scaled.shape[1] > 50:
            print("Applying PCA preprocessing for t-SNE (>50 features)")
            pca_pre = PCA(n_components=50)
            features_scaled = pca_pre.fit_transform(features_scaled)
        
        # Fit t-SNE
        tsne = TSNE(
            n_components=n_components,
            perplexity=perplexity,
            learning_rate=learning_rate,
            n_iter=n_iter,
            random_state=random_state
        )
        
        transformed_features = tsne.fit_transform(features_scaled)
        
        results = {
            'transformed_features': transformed_features,
            'n_components': n_components,
            'perplexity': perplexity,
            'kl_divergence': tsne.kl_divergence_
        }
        
        print(f"t-SNE completed: Final KL divergence = {tsne.kl_divergence_:.3f}")
        return results
    
    def select_features_univariate(self, features: np.ndarray, targets: np.ndarray,
                                 k: int = 10, score_func='f_regression') -> Dict:
        """
        Select top k features using univariate statistical tests.
        
        Args:
            features: Feature matrix of shape [n_samples, n_features]
            targets: Target values of shape [n_samples,]
            k: Number of features to select
            score_func: Scoring function ('f_regression' or 'mutual_info')
        
        Returns:
            Dictionary with feature selection results
        """
        print(f"Selecting top {k} features using {score_func}")
        
        # Choose scoring function
        if score_func == 'f_regression':
            scoring_func = f_regression
        elif score_func == 'mutual_info':
            scoring_func = mutual_info_regression
        else:
            raise ValueError(f"Unknown scoring function: {score_func}")
        
        # Fit feature selector
        selector = SelectKBest(score_func=scoring_func, k=k)
        selected_features = selector.fit_transform(features, targets.flatten())
        
        # Get feature information
        selected_indices = selector.get_support(indices=True)
        feature_scores = selector.scores_
        
        self.fitted_models[f'selector_{score_func}'] = selector
        
        results = {
            'selected_features': selected_features,
            'selected_indices': selected_indices,
            'feature_scores': feature_scores,
            'selected_scores': feature_scores[selected_indices],
            'k': k,
            'score_func': score_func
        }
        
        self.feature_importance[f'univariate_{score_func}'] = feature_scores
        
        print(f"Feature selection completed: Selected {k} features")
        return results
    
    def select_features_correlation(self, features: np.ndarray, targets: np.ndarray,
                                  k: int = 10, method: str = 'pearson') -> Dict:
        """
        Select features based on correlation with target.
        
        Args:
            features: Feature matrix of shape [n_samples, n_features]
            targets: Target values of shape [n_samples,]
            k: Number of features to select
            method: Correlation method ('pearson', 'spearman', 'kendall')
        
        Returns:
            Dictionary with correlation-based feature selection results
        """
        print(f"Selecting top {k} features using {method} correlation")
        
        # Calculate correlations
        feature_df = pd.DataFrame(features)
        target_series = pd.Series(targets.flatten())
        
        correlations = feature_df.corrwith(target_series, method=method)
        abs_correlations = np.abs(correlations)
        
        # Select top k features
        selected_indices = abs_correlations.nlargest(k).index.values
        selected_features = features[:, selected_indices]
        
        results = {
            'selected_features': selected_features,
            'selected_indices': selected_indices,
            'correlations': correlations.values,
            'abs_correlations': abs_correlations.values,
            'selected_correlations': correlations.iloc[selected_indices].values,
            'k': k,
            'method': method
        }
        
        self.feature_importance[f'correlation_{method}'] = abs_correlations.values
        
        print(f"Correlation-based selection completed: Selected {k} features")
        return results
    
    def transform_new_data(self, features: np.ndarray, method: str) -> np.ndarray:
        """
        Transform new data using fitted models.
        
        Args:
            features: New feature matrix to transform
            method: Method to use ('pca', 'umap', 'selector_f_regression', etc.)
        
        Returns:
            Transformed features
        """
        if method not in self.fitted_models:
            raise ValueError(f"Model {method} has not been fitted")
        
        if method not in self.scalers:
            raise ValueError(f"Scaler for {method} not found")
        
        # Scale features
        features_scaled = self.scalers[method].transform(features)
        
        # Transform using fitted model
        if method == 'tsne':
            # t-SNE doesn't support transform, only fit_transform
            raise ValueError("t-SNE doesn't support transforming new data")
        
        transformed = self.fitted_models[method].transform(features_scaled)
        return transformed
    
    def get_feature_rankings(self, feature_names: list) -> dict:
        """
        Get feature rankings from different methods with robust error handling.
        
        Args:
            feature_names: List of feature names
            
        Returns:
            Dictionary of ranking DataFrames for each method
        """
        rankings = {}
        
        # Ensure we have feature names
        if not feature_names or len(feature_names) == 0:
            print("Warning: No feature names provided for ranking")
            return rankings
        
        print(f"Creating feature rankings for {len(feature_names)} features")
        
        # Helper function to safely create ranking DataFrame
        def create_safe_ranking_df(method_name: str, scores: np.ndarray, names: list) -> pd.DataFrame:
            """Create a ranking DataFrame with length validation."""
            try:
                # Convert scores to numpy array if it isn't already
                if not isinstance(scores, np.ndarray):
                    scores = np.array(scores)
                
                # Handle empty or invalid scores
                if scores is None or len(scores) == 0:
                    print(f"Warning: No scores available for {method_name}")
                    return pd.DataFrame()
                
                # Remove any NaN or infinite values
                valid_mask = np.isfinite(scores)
                if not np.any(valid_mask):
                    print(f"Warning: No valid scores for {method_name}")
                    return pd.DataFrame()
                
                valid_scores = scores[valid_mask]
                valid_names = [names[i] for i in range(len(names)) if i < len(valid_mask) and valid_mask[i]]
                
                # Ensure we have matching lengths
                min_length = min(len(valid_scores), len(valid_names))
                if min_length == 0:
                    print(f"Warning: No valid features for {method_name}")
                    return pd.DataFrame()
                
                final_scores = valid_scores[:min_length]
                final_names = valid_names[:min_length]
                
                # Create ranks (higher scores = better ranks, so we use negative for argsort)
                ranks = np.argsort(-final_scores) + 1  # +1 to make ranks start from 1
                
                # Verify all arrays have same length before creating DataFrame
                assert len(final_names) == len(final_scores) == len(ranks), \
                    f"Length mismatch in {method_name}: names={len(final_names)}, scores={len(final_scores)}, ranks={len(ranks)}"
                
                # Create DataFrame
                df = pd.DataFrame({
                    'feature_name': final_names,
                    'score': final_scores,
                    'rank': ranks
                })
                
                # Sort by rank
                df = df.sort_values('rank').reset_index(drop=True)
                
                print(f"✓ Created {method_name} ranking with {len(df)} features")
                return df
                
            except Exception as e:
                print(f"Error creating ranking for {method_name}: {e}")
                return pd.DataFrame()
        
        # Try different ranking methods if they exist
        ranking_methods = []
        
        # Check for PCA results
        if hasattr(self, 'pca_') and self.pca_ is not None:
            try:
                # Use explained variance ratio as importance scores
                if hasattr(self.pca_, 'components_') and self.pca_.components_ is not None:
                    # Sum of absolute values across all components
                    feature_importance = np.abs(self.pca_.components_).sum(axis=0)
                    ranking_methods.append(('PCA', feature_importance))
            except Exception as e:
                print(f"Could not create PCA ranking: {e}")
        
        # Check for feature selection results
        if hasattr(self, 'feature_selector_') and self.feature_selector_ is not None:
            try:
                if hasattr(self.feature_selector_, 'scores_'):
                    ranking_methods.append(('Feature_Selection', self.feature_selector_.scores_))
                elif hasattr(self.feature_selector_, 'ranking_'):
                    # For RFE, lower ranking is better, so we invert
                    inverted_ranks = 1.0 / (self.feature_selector_.ranking_ + 1e-8)
                    ranking_methods.append(('Feature_Selection', inverted_ranks))
            except Exception as e:
                print(f"Could not create feature selection ranking: {e}")
        
        # Check for variance threshold results
        if hasattr(self, 'variance_selector_') and self.variance_selector_ is not None:
            try:
                if hasattr(self.variance_selector_, 'variances_'):
                    ranking_methods.append(('Variance', self.variance_selector_.variances_))
            except Exception as e:
                print(f"Could not create variance ranking: {e}")
        
        # Create rankings for each method
        for method_name, scores in ranking_methods:
            ranking_df = create_safe_ranking_df(method_name, scores, feature_names)
            if not ranking_df.empty:
                rankings[method_name] = ranking_df
        
        # If no rankings were created, create a simple default ranking
        if not rankings:
            print("Warning: No valid rankings created, using default feature order")
            try:
                default_scores = np.arange(len(feature_names), 0, -1)  # Descending order
                rankings['Default'] = create_safe_ranking_df('Default', default_scores, feature_names)
            except Exception as e:
                print(f"Error creating default ranking: {e}")
        
        print(f"✓ Created {len(rankings)} feature ranking methods")
        return rankings
    
    def compare_dimensionality_methods(self, features: np.ndarray, targets: np.ndarray,
                                     feature_names: Optional[List[str]] = None) -> Dict:
        """
        Compare multiple dimensionality reduction methods on the same data.
        
        Args:
            features: Feature matrix of shape [n_samples, n_features]
            targets: Target values of shape [n_samples,]
            feature_names: List of feature names (optional)
        
        Returns:
            Dictionary with comparison results
        """
        print("Comparing dimensionality reduction methods...")
        
        results = {}
        
        # PCA
        try:
            pca_results = self.fit_pca(features, n_components=10)
            results['pca'] = pca_results
        except Exception as e:
            print(f"PCA failed: {e}")
            results['pca'] = None
        
        # UMAP
        try:
            umap_results = self.fit_umap(features, n_components=2)
            results['umap'] = umap_results
        except Exception as e:
            print(f"UMAP failed: {e}")
            results['umap'] = None
        
        # Feature selection
        try:
            k = min(10, features.shape[1])  # Don't select more features than available
            selection_results = self.select_features_univariate(features, targets, k=k)
            results['feature_selection'] = selection_results
        except Exception as e:
            print(f"Feature selection failed: {e}")
            results['feature_selection'] = None
        
        # Correlation-based selection
        try:
            k = min(10, features.shape[1])
            corr_results = self.select_features_correlation(features, targets, k=k)
            results['correlation_selection'] = corr_results
        except Exception as e:
            print(f"Correlation selection failed: {e}")
            results['correlation_selection'] = None
        
        # Get rankings
        rankings = self.get_feature_rankings(feature_names)
        results['feature_rankings'] = rankings
        
        print("Dimensionality reduction comparison completed")
        return results
    
    def get_reduction_summary(self) -> Dict:
        """Get a summary of all fitted reduction methods."""
        summary = {
            'fitted_methods': list(self.fitted_models.keys()),
            'available_scalers': list(self.scalers.keys()),
            'feature_importance_methods': list(self.feature_importance.keys())
        }
        
        # Add method-specific info
        for method, model in self.fitted_models.items():
            if method == 'pca':
                summary[f'{method}_info'] = {
                    'n_components': model.n_components_,
                    'explained_variance_ratio': model.explained_variance_ratio_.tolist(),
                    'total_variance_explained': np.sum(model.explained_variance_ratio_)
                }
            elif method == 'umap':
                summary[f'{method}_info'] = {
                    'n_components': model.n_components,
                    'n_neighbors': model.n_neighbors,
                    'min_dist': model.min_dist
                }
        
        return summary

class QuickDimensionalityReducer:
    """
    Simplified dimensionality reducer for quick analysis.
    Provides essential functionality without complexity.
    """
    
    def __init__(self):
        self.pca_model = None
        self.scaler = None
    
    def fit_transform_pca(self, features: np.ndarray, n_components: int = 5) -> Tuple[np.ndarray, Dict]:
        """
        Quick PCA fit and transform.
        
        Args:
            features: Feature matrix
            n_components: Number of components to keep
        
        Returns:
            Tuple of (transformed_features, pca_info)
        """
        # Scale features
        self.scaler = StandardScaler()
        features_scaled = self.scaler.fit_transform(features)
        
        # Fit PCA
        self.pca_model = PCA(n_components=n_components)
        transformed = self.pca_model.fit_transform(features_scaled)
        
        info = {
            'explained_variance_ratio': self.pca_model.explained_variance_ratio_,
            'cumulative_variance': np.cumsum(self.pca_model.explained_variance_ratio_),
            'n_components': n_components
        }
        
        return transformed, info
    
    def get_top_features(self, features: np.ndarray, targets: np.ndarray, 
                        feature_names: List[str], k: int = 5) -> pd.DataFrame:
        """
        Get top k features based on correlation with target.
        
        Args:
            features: Feature matrix
            targets: Target values
            feature_names: List of feature names
            k: Number of top features to return
        
        Returns:
            DataFrame with top features and their correlations
        """
        # Calculate correlations
        correlations = []
        for i in range(features.shape[1]):
            corr = np.corrcoef(features[:, i], targets.flatten())[0, 1]
            correlations.append(abs(corr))
        
        # Create results DataFrame
        results_df = pd.DataFrame({
            'feature_name': feature_names,
            'abs_correlation': correlations
        }).sort_values('abs_correlation', ascending=False).head(k)
        
        return results_df.reset_index(drop=True)

"""
Feature Selector Module for DataScout
Identifies important features, correlations, and performs feature selection for analysis.
"""

import pandas as pd
import numpy as np
from typing import Dict, List, Any, Optional, Union, Tuple
from sklearn.feature_selection import (
    SelectKBest, f_classif, f_regression, chi2, mutual_info_classif, 
    mutual_info_regression, RFE
)
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from sklearn.preprocessing import LabelEncoder
from scipy.stats import pearsonr, spearmanr, chi2_contingency
import logging

logger = logging.getLogger(__name__)

class FeatureSelector:
    """
    Feature selection class that identifies important features and relationships.
    
    Features:
    - Correlation analysis
    - Statistical feature selection
    - Tree-based feature importance
    - Multicollinearity detection
    - Feature ranking and scoring
    - Automated feature selection
    """
    
    def __init__(self):
        self.feature_scores = {}
        self.encoders = {}
        
    def analyze_feature_importance(self, df: pd.DataFrame, target_col: Optional[str] = None) -> Dict[str, Any]:
        """
        Comprehensive feature importance analysis.
        
        Args:
            df: Input DataFrame
            target_col: Target column name (if supervised analysis)
            
        Returns:
            Dict containing feature importance analysis
        """
        analysis = {
            'correlation_analysis': self._analyze_correlations(df, target_col),
            'statistical_tests': self._statistical_feature_tests(df, target_col),
            'multicollinearity_analysis': self._detect_multicollinearity(df),
            'feature_rankings': self._rank_features(df, target_col),
            'recommendations': self._generate_feature_recommendations(df, target_col)
        }
        
        if target_col:
            analysis['supervised_selection'] = self._supervised_feature_selection(df, target_col)
        else:
            analysis['unsupervised_selection'] = self._unsupervised_feature_selection(df)
            
        logger.info(f"Completed feature importance analysis for {len(df.columns)} features")
        return analysis
        
    def _analyze_correlations(self, df: pd.DataFrame, target_col: Optional[str] = None) -> Dict[str, Any]:
        """Analyze correlations between features."""
        numeric_df = df.select_dtypes(include=[np.number])
        
        if len(numeric_df.columns) < 2:
            return {'message': 'Need at least 2 numerical columns for correlation analysis'}
            
        correlation_matrix = numeric_df.corr()
        
        # Find strong correlations
        strong_correlations = []
        for i in range(len(correlation_matrix.columns)):
            for j in range(i+1, len(correlation_matrix.columns)):
                corr_value = correlation_matrix.iloc[i, j]
                if abs(corr_value) > 0.7:
                    strong_correlations.append({
                        'feature1': correlation_matrix.columns[i],
                        'feature2': correlation_matrix.columns[j],
                        'correlation': round(corr_value, 4),
                        'strength': self._get_correlation_strength(abs(corr_value))
                    })
                    
        # Target correlations if target specified
        target_correlations = {}
        if target_col and target_col in numeric_df.columns:
            target_corr = correlation_matrix[target_col].drop(target_col).sort_values(
                key=abs, ascending=False
            )
            target_correlations = {
                'correlations': target_corr.round(4).to_dict(),
                'top_positive': target_corr.head(5).to_dict(),
                'top_negative': target_corr.tail(5).to_dict()
            }
            
        return {
            'correlation_matrix': correlation_matrix.round(4).to_dict(),
            'strong_correlations': strong_correlations,
            'target_correlations': target_correlations,
            'average_correlation': round(
                correlation_matrix.values[np.triu_indices_from(correlation_matrix.values, k=1)].mean(), 4
            )
        }
        
    def _get_correlation_strength(self, abs_corr: float) -> str:
        """Classify correlation strength."""
        if abs_corr >= 0.9:
            return 'very strong'
        elif abs_corr >= 0.7:
            return 'strong'
        elif abs_corr >= 0.5:
            return 'moderate'
        elif abs_corr >= 0.3:
            return 'weak'
        else:
            return 'very weak'
            
    def _statistical_feature_tests(self, df: pd.DataFrame, target_col: Optional[str] = None) -> Dict[str, Any]:
        """Perform statistical tests for feature selection."""
        if not target_col or target_col not in df.columns:
            return {'message': 'Target column required for statistical tests'}
            
        results = {}
        numeric_features = df.select_dtypes(include=[np.number]).columns.tolist()
        categorical_features = df.select_dtypes(include=['object', 'category']).columns.tolist()
        
        # Remove target from feature lists
        if target_col in numeric_features:
            numeric_features.remove(target_col)
        if target_col in categorical_features:
            categorical_features.remove(target_col)
            
        target_data = df[target_col]
        is_target_numeric = pd.api.types.is_numeric_dtype(target_data)
        
        # Statistical tests for numeric features
        if numeric_features:
            numeric_scores = {}
            
            for feature in numeric_features:
                feature_data = df[feature].dropna()
                target_aligned = target_data[feature_data.index].dropna()
                
                if len(feature_data) < 3 or len(target_aligned) < 3:
                    continue
                    
                try:
                    if is_target_numeric:
                        # Pearson correlation for numeric target
                        corr, p_value = pearsonr(feature_data, target_aligned)
                        numeric_scores[feature] = {
                            'test': 'pearson_correlation',
                            'score': abs(corr),
                            'p_value': p_value,
                            'significant': p_value < 0.05
                        }
                    else:
                        # F-test for categorical target
                        f_stat, p_value = f_classif(feature_data.values.reshape(-1, 1), target_aligned)
                        numeric_scores[feature] = {
                            'test': 'f_classif',
                            'score': f_stat[0],
                            'p_value': p_value[0],
                            'significant': p_value[0] < 0.05
                        }
                except Exception as e:
                    logger.warning(f"Could not compute statistics for {feature}: {e}")
                    
            results['numeric_features'] = numeric_scores
            
        # Statistical tests for categorical features
        if categorical_features and not is_target_numeric:
            categorical_scores = {}
            
            for feature in categorical_features:
                try:
                    # Chi-square test for categorical-categorical relationship
                    contingency_table = pd.crosstab(df[feature], df[target_col])
                    chi2_stat, p_value, dof, expected = chi2_contingency(contingency_table)
                    
                    categorical_scores[feature] = {
                        'test': 'chi_square',
                        'score': chi2_stat,
                        'p_value': p_value,
                        'significant': p_value < 0.05,
                        'degrees_of_freedom': dof
                    }
                except Exception as e:
                    logger.warning(f"Could not compute chi-square for {feature}: {e}")
                    
            results['categorical_features'] = categorical_scores
            
        return results
        
    def _detect_multicollinearity(self, df: pd.DataFrame) -> Dict[str, Any]:
        """Detect multicollinearity among numeric features."""
        numeric_df = df.select_dtypes(include=[np.number])
        
        if len(numeric_df.columns) < 2:
            return {'message': 'Need at least 2 numerical columns for multicollinearity analysis'}
            
        correlation_matrix = numeric_df.corr()
        
        # Find highly correlated pairs
        high_correlations = []
        for i in range(len(correlation_matrix.columns)):
            for j in range(i+1, len(correlation_matrix.columns)):
                corr_value = correlation_matrix.iloc[i, j]
                if abs(corr_value) > 0.8:  # High correlation threshold
                    high_correlations.append({
                        'feature1': correlation_matrix.columns[i],
                        'feature2': correlation_matrix.columns[j],
                        'correlation': round(corr_value, 4),
                        'concern_level': 'high' if abs(corr_value) > 0.9 else 'moderate'
                    })
                    
        # Calculate VIF (Variance Inflation Factor) approximation
        vif_scores = {}
        for col in numeric_df.columns:
            try:
                # Simple VIF approximation using R-squared
                other_cols = [c for c in numeric_df.columns if c != col]
                if len(other_cols) > 0:
                    r_squared = numeric_df[other_cols].corrwith(numeric_df[col]).abs().max()
                    vif_approx = 1 / (1 - r_squared**2) if r_squared < 0.99 else float('inf')
                    vif_scores[col] = {
                        'vif_approximation': round(vif_approx, 2),
                        'concern_level': 'high' if vif_approx > 10 else 'moderate' if vif_approx > 5 else 'low'
                    }
            except Exception as e:
                logger.warning(f"Could not calculate VIF for {col}: {e}")
                
        return {
            'high_correlations': high_correlations,
            'vif_scores': vif_scores,
            'multicollinearity_detected': len(high_correlations) > 0,
            'recommendations': self._generate_multicollinearity_recommendations(high_correlations)
        }
        
    def _generate_multicollinearity_recommendations(self, high_correlations: List[Dict]) -> List[str]:
        """Generate recommendations for handling multicollinearity."""
        recommendations = []
        
        if not high_correlations:
            recommendations.append("No significant multicollinearity detected")
        else:
            recommendations.append(f"Found {len(high_correlations)} highly correlated feature pairs")
            recommendations.append("Consider removing one feature from each highly correlated pair")
            recommendations.append("Alternatively, use dimensionality reduction techniques like PCA")
            
            # Specific recommendations for each pair
            for pair in high_correlations[:3]:  # Limit to first 3
                recommendations.append(
                    f"Consider removing either '{pair['feature1']}' or '{pair['feature2']}' "
                    f"(correlation: {pair['correlation']})"
                )
                
        return recommendations
        
    def _rank_features(self, df: pd.DataFrame, target_col: Optional[str] = None) -> Dict[str, Any]:
        """Rank features by importance using multiple methods."""
        rankings = {}
        
        # Correlation-based ranking (for numeric features)
        numeric_features = df.select_dtypes(include=[np.number]).columns.tolist()
        if target_col and target_col in numeric_features:
            numeric_features.remove(target_col)
            
        if numeric_features and target_col and target_col in df.columns:
            if pd.api.types.is_numeric_dtype(df[target_col]):
                correlations = df[numeric_features + [target_col]].corr()[target_col].drop(target_col)
                rankings['correlation_based'] = correlations.abs().sort_values(ascending=False).to_dict()
                
        # Variance-based ranking (unsupervised)
        if numeric_features:
            variances = df[numeric_features].var().sort_values(ascending=False)
            rankings['variance_based'] = variances.to_dict()
            
        # Missing data ranking (features with less missing data ranked higher)
        missing_percentages = (df.isnull().sum() / len(df) * 100).sort_values()
        rankings['completeness_based'] = (100 - missing_percentages).to_dict()
        
        # Unique value ratio ranking
        uniqueness_ratios = {}
        for col in df.columns:
            if col != target_col:
                uniqueness_ratios[col] = df[col].nunique() / len(df)
        rankings['uniqueness_based'] = dict(
            sorted(uniqueness_ratios.items(), key=lambda x: x[1], reverse=True)
        )
        
        return rankings
        
    def _supervised_feature_selection(self, df: pd.DataFrame, target_col: str) -> Dict[str, Any]:
        """Perform supervised feature selection methods."""
        if target_col not in df.columns:
            return {'error': f'Target column {target_col} not found'}
            
        # Prepare data
        feature_cols = [col for col in df.columns if col != target_col]
        X = df[feature_cols]
        y = df[target_col]
        
        # Handle missing values and encode categorical variables
        X_processed = self._prepare_features_for_selection(X)
        
        if X_processed.empty:
            return {'error': 'No valid features for selection after preprocessing'}
            
        # Determine if target is categorical or numeric
        is_classification = not pd.api.types.is_numeric_dtype(y)
        
        results = {}
        
        try:
            # SelectKBest with appropriate scoring function
            if is_classification:
                selector = SelectKBest(score_func=f_classif, k='all')
            else:
                selector = SelectKBest(score_func=f_regression, k='all')
                
            X_selected = selector.fit_transform(X_processed, y)
            feature_scores = dict(zip(X_processed.columns, selector.scores_))
            
            results['selectkbest'] = {
                'method': 'f_classif' if is_classification else 'f_regression',
                'scores': {k: round(v, 4) for k, v in feature_scores.items()},
                'top_features': dict(sorted(feature_scores.items(), key=lambda x: x[1], reverse=True)[:10])
            }
            
        except Exception as e:
            logger.warning(f"SelectKBest failed: {e}")
            
        try:
            # Tree-based feature importance
            if is_classification:
                model = RandomForestClassifier(n_estimators=100, random_state=42)
            else:
                model = RandomForestRegressor(n_estimators=100, random_state=42)
                
            model.fit(X_processed, y)
            feature_importance = dict(zip(X_processed.columns, model.feature_importances_))
            
            results['tree_based'] = {
                'method': 'random_forest',
                'importances': {k: round(v, 4) for k, v in feature_importance.items()},
                'top_features': dict(sorted(feature_importance.items(), key=lambda x: x[1], reverse=True)[:10])
            }
            
        except Exception as e:
            logger.warning(f"Tree-based selection failed: {e}")
            
        return results
        
    def _unsupervised_feature_selection(self, df: pd.DataFrame) -> Dict[str, Any]:
        """Perform unsupervised feature selection methods."""
        results = {}
        
        # Variance threshold
        numeric_df = df.select_dtypes(include=[np.number])
        if not numeric_df.empty:
            variances = numeric_df.var()
            low_variance_features = variances[variances < 0.01].index.tolist()
            
            results['variance_threshold'] = {
                'low_variance_features': low_variance_features,
                'variances': variances.round(4).to_dict(),
                'recommendation': f"Consider removing {len(low_variance_features)} low-variance features"
            }
            
        # Correlation threshold
        if len(numeric_df.columns) > 1:
            correlation_matrix = numeric_df.corr()
            high_corr_pairs = []
            
            for i in range(len(correlation_matrix.columns)):
                for j in range(i+1, len(correlation_matrix.columns)):
                    if abs(correlation_matrix.iloc[i, j]) > 0.95:
                        high_corr_pairs.append({
                            'feature1': correlation_matrix.columns[i],
                            'feature2': correlation_matrix.columns[j],
                            'correlation': round(correlation_matrix.iloc[i, j], 4)
                        })
                        
            results['correlation_threshold'] = {
                'highly_correlated_pairs': high_corr_pairs,
                'recommendation': f"Consider removing one feature from each of {len(high_corr_pairs)} highly correlated pairs"
            }
            
        return results
        
    def _prepare_features_for_selection(self, X: pd.DataFrame) -> pd.DataFrame:
        """Prepare features for selection algorithms."""
        X_processed = X.copy()
        
        # Handle categorical variables
        categorical_cols = X_processed.select_dtypes(include=['object', 'category']).columns
        for col in categorical_cols:
            if X_processed[col].nunique() > 20:  # High cardinality
                # Label encode high cardinality categorical variables
                encoder = LabelEncoder()
                X_processed[col] = encoder.fit_transform(X_processed[col].astype(str))
                self.encoders[col] = encoder
            else:
                # One-hot encode low cardinality categorical variables
                dummies = pd.get_dummies(X_processed[col], prefix=col)
                X_processed = pd.concat([X_processed.drop(col, axis=1), dummies], axis=1)
                
        # Handle missing values
        numeric_cols = X_processed.select_dtypes(include=[np.number]).columns
        X_processed[numeric_cols] = X_processed[numeric_cols].fillna(X_processed[numeric_cols].median())
        
        return X_processed
        
    def _generate_feature_recommendations(self, df: pd.DataFrame, target_col: Optional[str] = None) -> List[str]:
        """Generate feature selection recommendations."""
        recommendations = []
        
        # Basic recommendations
        total_features = len(df.columns) - (1 if target_col else 0)
        missing_data_pct = (df.isnull().sum().sum() / (len(df) * len(df.columns))) * 100
        
        recommendations.append(f"Dataset has {total_features} features for analysis")
        
        if missing_data_pct > 20:
            recommendations.append("High missing data percentage - consider imputation or feature removal")
            
        # Feature type recommendations
        numeric_count = len(df.select_dtypes(include=[np.number]).columns)
        categorical_count = len(df.select_dtypes(include=['object', 'category']).columns)
        
        if numeric_count == 0:
            recommendations.append("No numeric features - limited statistical analysis possible")
        elif categorical_count == 0:
            recommendations.append("No categorical features - consider feature engineering")
        else:
            recommendations.append(f"Good mix of {numeric_count} numeric and {categorical_count} categorical features")
            
        # Dataset size recommendations
        if len(df) < 100:
            recommendations.append("Small dataset - feature selection results may not be reliable")
        elif total_features > len(df):
            recommendations.append("More features than samples - consider dimensionality reduction")
            
        # Feature selection method recommendations
        if target_col:
            recommendations.append("Supervised feature selection available - use correlation and statistical tests")
        else:
            recommendations.append("No target specified - use unsupervised methods like variance threshold")
            
        return recommendations
        
    def select_best_features(self, df: pd.DataFrame, target_col: Optional[str] = None, 
                           method: str = 'auto', k: int = 10) -> Dict[str, Any]:
        """Select the best k features using specified method."""
        if method == 'auto':
            method = 'correlation' if target_col else 'variance'
            
        feature_cols = [col for col in df.columns if col != target_col]
        
        if method == 'correlation' and target_col:
            # Correlation-based selection
            numeric_features = df.select_dtypes(include=[np.number]).columns.tolist()
            if target_col in numeric_features:
                numeric_features.remove(target_col)
                
            if numeric_features and pd.api.types.is_numeric_dtype(df[target_col]):
                correlations = df[numeric_features + [target_col]].corr()[target_col].drop(target_col)
                selected_features = correlations.abs().nlargest(k).index.tolist()
            else:
                selected_features = feature_cols[:k]  # Fallback
                
        elif method == 'variance':
            # Variance-based selection
            numeric_features = df.select_dtypes(include=[np.number]).columns.tolist()
            if target_col and target_col in numeric_features:
                numeric_features.remove(target_col)
                
            if numeric_features:
                variances = df[numeric_features].var()
                selected_features = variances.nlargest(k).index.tolist()
            else:
                selected_features = feature_cols[:k]  # Fallback
                
        else:
            # Default selection (first k features)
            selected_features = feature_cols[:k]
            
        return {
            'method': method,
            'selected_features': selected_features,
            'feature_count': len(selected_features),
            'selection_summary': f"Selected {len(selected_features)} features using {method} method"
        }


# Factory function
def create_feature_selector() -> FeatureSelector:
    """Create and return a FeatureSelector instance."""
    return FeatureSelector()


# Convenience functions
def quick_feature_analysis(df: pd.DataFrame, target_col: Optional[str] = None) -> Dict[str, Any]:
    """Quick feature importance analysis."""
    selector = FeatureSelector()
    return selector.analyze_feature_importance(df, target_col)


def select_top_features(df: pd.DataFrame, k: int = 10, target_col: Optional[str] = None) -> List[str]:
    """Select top k features."""
    selector = FeatureSelector()
    result = selector.select_best_features(df, target_col, k=k)
    return result['selected_features']
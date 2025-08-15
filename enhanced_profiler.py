#!/usr/bin/env python3
"""
Enhanced Data Profiler for DataScout
Addresses gaps in existing reporting by providing comprehensive, context-aware analysis
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import warnings
from scipy import stats
from scipy.stats import chi2_contingency, f_oneway
from sklearn.preprocessing import LabelEncoder
from sklearn.ensemble import RandomForestRegressor
from sklearn.feature_selection import mutual_info_regression
import base64
import io
from datetime import datetime
import json

warnings.filterwarnings('ignore')

class EnhancedDataProfiler:
    """
    Comprehensive data profiler that generates deep, context-aware insights
    """
    
    def __init__(self, df, target_column='SalePrice'):
        self.df = df.copy()
        self.target_column = target_column
        self.numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
        self.categorical_cols = df.select_dtypes(include=['object']).columns.tolist()
        self.report = {}
        
    def generate_comprehensive_profile(self):
        """Generate complete data profile with all sections"""
        print("🏠 Generating Enhanced Ames Housing Data Profile...")
        
        # A. Basic Data Overview
        self.report['basic_overview'] = self._analyze_basic_overview()
        
        # B. Data Quality Report
        self.report['data_quality'] = self._analyze_data_quality()
        
        # C. Descriptive Statistics
        self.report['descriptive_stats'] = self._analyze_descriptive_statistics()
        
        # D. Relationship & Correlation Insights
        self.report['relationships'] = self._analyze_relationships()
        
        # E. Visualization Section
        self.report['visualizations'] = self._generate_visualizations()
        
        # F. Automated Narrative Insights
        self.report['narrative_insights'] = self._generate_narrative_insights()
        
        return self.report
    
    def _analyze_basic_overview(self):
        """A. Basic Data Overview"""
        overview = {}
        
        # Dataset dimensions
        overview['dimensions'] = {
            'rows': len(self.df),
            'columns': len(self.df.columns),
            'numeric_columns': len(self.numeric_cols),
            'categorical_columns': len(self.categorical_cols)
        }
        
        # Missing values analysis
        missing_analysis = {}
        for col in self.df.columns:
            missing_count = self.df[col].isnull().sum()
            missing_pct = (missing_count / len(self.df)) * 100
            missing_analysis[col] = {
                'count': int(missing_count),
                'percentage': round(missing_pct, 2)
            }
        overview['missing_values'] = missing_analysis
        
        # Duplicate rows
        overview['duplicate_rows'] = int(self.df.duplicated().sum())
        
        # Unique values per column
        unique_analysis = {}
        for col in self.df.columns:
            unique_count = self.df[col].nunique()
            unique_analysis[col] = {
                'unique_count': int(unique_count),
                'uniqueness_ratio': round(unique_count / len(self.df), 4),
                'likely_type': self._classify_column_type(col, unique_count)
            }
        overview['unique_values'] = unique_analysis
        
        return overview
    
    def _classify_column_type(self, col, unique_count):
        """Classify column as ID, categorical, or numeric based on characteristics"""
        if unique_count == len(self.df):
            return 'ID/Unique_Identifier'
        elif unique_count < 10 and col in self.categorical_cols:
            return 'Low_Cardinality_Categorical'
        elif unique_count > len(self.df) * 0.5 and col in self.categorical_cols:
            return 'High_Cardinality_Categorical'
        elif col in self.numeric_cols and unique_count < 20:
            return 'Ordinal/Discrete_Numeric'
        elif col in self.numeric_cols:
            return 'Continuous_Numeric'
        else:
            return 'Mixed_Type'
    
    def _analyze_data_quality(self):
        """B. Data Quality Report"""
        quality = {}
        
        # Missing value patterns
        missing_matrix = self.df.isnull().sum().sort_values(ascending=False)
        high_missing = missing_matrix[missing_matrix > len(self.df) * 0.1]
        quality['high_missing_columns'] = {
            col: {
                'count': int(count),
                'percentage': round((count / len(self.df)) * 100, 2)
            } for col, count in high_missing.items()
        }
        
        # Constant/Near-constant values
        constant_cols = []
        near_constant_cols = []
        for col in self.df.columns:
            value_counts = self.df[col].value_counts()
            if len(value_counts) == 1:
                constant_cols.append(col)
            elif len(value_counts) > 1 and value_counts.iloc[0] / len(self.df) > 0.95:
                near_constant_cols.append({
                    'column': col,
                    'dominant_value': str(value_counts.index[0]),
                    'dominance_ratio': round(value_counts.iloc[0] / len(self.df), 4)
                })
        
        quality['constant_columns'] = constant_cols
        quality['near_constant_columns'] = near_constant_cols
        
        # Outlier detection for numeric columns
        outlier_analysis = {}
        for col in self.numeric_cols:
            if self.df[col].notna().sum() > 0:
                Q1 = self.df[col].quantile(0.25)
                Q3 = self.df[col].quantile(0.75)
                IQR = Q3 - Q1
                lower_bound = Q1 - 1.5 * IQR
                upper_bound = Q3 + 1.5 * IQR
                
                outliers = self.df[(self.df[col] < lower_bound) | (self.df[col] > upper_bound)][col]
                
                # Z-score outliers
                z_scores = np.abs(stats.zscore(self.df[col].dropna()))
                z_outliers = self.df[col][z_scores > 3]
                
                outlier_analysis[col] = {
                    'iqr_outliers': int(len(outliers)),
                    'iqr_outlier_percentage': round((len(outliers) / len(self.df)) * 100, 2),
                    'zscore_outliers': int(len(z_outliers)),
                    'bounds': {
                        'iqr_lower': float(lower_bound),
                        'iqr_upper': float(upper_bound)
                    }
                }
        
        quality['outlier_analysis'] = outlier_analysis
        
        # Inconsistent categorical labels
        categorical_consistency = {}
        for col in self.categorical_cols:
            if self.df[col].notna().sum() > 0:
                values = self.df[col].dropna().astype(str)
                # Check for potential case inconsistencies
                value_counts = values.value_counts()
                potential_duplicates = []
                
                for val in value_counts.index:
                    similar_vals = [v for v in value_counts.index 
                                  if v.lower() == val.lower() and v != val]
                    if similar_vals:
                        potential_duplicates.append({
                            'base_value': val,
                            'similar_values': similar_vals
                        })
                
                if potential_duplicates:
                    categorical_consistency[col] = potential_duplicates
        
        quality['categorical_consistency'] = categorical_consistency
        
        # Overall data quality score
        quality_score = self._calculate_quality_score()
        quality['overall_quality_score'] = quality_score
        
        return quality
    
    def _calculate_quality_score(self):
        """Calculate overall data quality score (0-100)"""
        scores = []
        
        # Missing data score (0-25 points)
        missing_ratio = self.df.isnull().sum().sum() / (len(self.df) * len(self.df.columns))
        missing_score = max(0, 25 - (missing_ratio * 100))
        scores.append(missing_score)
        
        # Duplicate data score (0-15 points)
        duplicate_ratio = self.df.duplicated().sum() / len(self.df)
        duplicate_score = max(0, 15 - (duplicate_ratio * 100))
        scores.append(duplicate_score)
        
        # Outlier score (0-20 points)
        total_outliers = 0
        total_numeric_values = 0
        for col in self.numeric_cols:
            if self.df[col].notna().sum() > 0:
                z_scores = np.abs(stats.zscore(self.df[col].dropna()))
                outliers = (z_scores > 3).sum()
                total_outliers += outliers
                total_numeric_values += len(self.df[col].dropna())
        
        outlier_ratio = total_outliers / max(total_numeric_values, 1)
        outlier_score = max(0, 20 - (outlier_ratio * 100))
        scores.append(outlier_score)
        
        # Consistency score (0-20 points)
        consistency_score = 20  # Start with full points
        for col in self.categorical_cols[:10]:  # Check first 10 categorical columns
            if self.df[col].notna().sum() > 0:
                values = self.df[col].dropna().astype(str)
                unique_lower = set(values.str.lower())
                if len(unique_lower) < len(values.unique()):
                    consistency_score -= 2
        
        scores.append(max(0, consistency_score))
        
        # Completeness score (0-20 points)
        completeness = (1 - missing_ratio) * 20
        scores.append(completeness)
        
        return round(sum(scores), 1)
    
    def _analyze_descriptive_statistics(self):
        """C. Descriptive Statistics"""
        stats_analysis = {}
        
        # Numeric features statistics
        numeric_stats = {}
        for col in self.numeric_cols:
            if self.df[col].notna().sum() > 0:
                data = self.df[col].dropna()
                numeric_stats[col] = {
                    'count': int(len(data)),
                    'mean': float(data.mean()),
                    'median': float(data.median()),
                    'min': float(data.min()),
                    'max': float(data.max()),
                    'std': float(data.std()),
                    'skewness': float(stats.skew(data)),
                    'kurtosis': float(stats.kurtosis(data)),
                    'q1': float(data.quantile(0.25)),
                    'q3': float(data.quantile(0.75)),
                    'iqr': float(data.quantile(0.75) - data.quantile(0.25))
                }
        
        stats_analysis['numeric_statistics'] = numeric_stats
        
        # Categorical features analysis
        categorical_stats = {}
        for col in self.categorical_cols:
            if self.df[col].notna().sum() > 0:
                value_counts = self.df[col].value_counts()
                total_count = len(self.df[col].dropna())
                
                # Top categories with percentages
                top_categories = []
                for val, count in value_counts.head(10).items():
                    top_categories.append({
                        'value': str(val),
                        'count': int(count),
                        'percentage': round((count / total_count) * 100, 2)
                    })
                
                categorical_stats[col] = {
                    'unique_count': int(len(value_counts)),
                    'most_frequent': str(value_counts.index[0]),
                    'most_frequent_count': int(value_counts.iloc[0]),
                    'most_frequent_percentage': round((value_counts.iloc[0] / total_count) * 100, 2),
                    'cardinality': len(value_counts),
                    'top_categories': top_categories
                }
        
        stats_analysis['categorical_statistics'] = categorical_stats
        
        return stats_analysis
    
    def _analyze_relationships(self):
        """D. Relationship & Correlation Insights"""
        relationships = {}
        
        if self.target_column not in self.df.columns:
            relationships['error'] = f"Target column {self.target_column} not found"
            return relationships
        
        target_data = self.df[self.target_column].dropna()
        
        # Numeric-Numeric correlations with target
        numeric_correlations = {}
        for col in self.numeric_cols:
            if col != self.target_column and self.df[col].notna().sum() > 10:
                # Pearson correlation
                pearson_corr, pearson_p = stats.pearsonr(
                    self.df[col].dropna(), 
                    self.df.loc[self.df[col].dropna().index, self.target_column]
                )
                
                # Spearman correlation
                spearman_corr, spearman_p = stats.spearmanr(
                    self.df[col].dropna(), 
                    self.df.loc[self.df[col].dropna().index, self.target_column]
                )
                
                numeric_correlations[col] = {
                    'pearson_correlation': float(pearson_corr),
                    'pearson_p_value': float(pearson_p),
                    'spearman_correlation': float(spearman_corr),
                    'spearman_p_value': float(spearman_p),
                    'strength': self._classify_correlation_strength(abs(pearson_corr))
                }
        
        relationships['numeric_correlations'] = numeric_correlations
        
        # Categorical-Numeric analysis (ANOVA)
        categorical_numeric = {}
        for col in self.categorical_cols:
            if self.df[col].notna().sum() > 10:
                # Group target by categorical variable
                groups = []
                categories = []
                valid_data = self.df[[col, self.target_column]].dropna()
                
                for category in valid_data[col].unique():
                    group_data = valid_data[valid_data[col] == category][self.target_column]
                    if len(group_data) >= 3:  # Need at least 3 observations
                        groups.append(group_data)
                        categories.append(category)
                
                if len(groups) >= 2:
                    # ANOVA F-test
                    f_stat, p_value = f_oneway(*groups)
                    
                    # Mean target value per category
                    category_means = []
                    for category in categories:
                        cat_data = valid_data[valid_data[col] == category][self.target_column]
                        category_means.append({
                            'category': str(category),
                            'mean_target': float(cat_data.mean()),
                            'count': int(len(cat_data)),
                            'std': float(cat_data.std())
                        })
                    
                    # Sort by mean target value
                    category_means.sort(key=lambda x: x['mean_target'], reverse=True)
                    
                    categorical_numeric[col] = {
                        'f_statistic': float(f_stat),
                        'p_value': float(p_value),
                        'significance': 'Significant' if p_value < 0.05 else 'Not Significant',
                        'category_analysis': category_means,
                        'effect_size': self._calculate_eta_squared(groups)
                    }
        
        relationships['categorical_numeric_analysis'] = categorical_numeric
        
        # Feature importance using Random Forest
        feature_importance = self._calculate_feature_importance()
        relationships['feature_importance'] = feature_importance
        
        # Correlation matrix for top numeric features
        top_numeric = self._get_top_correlated_features(numeric_correlations, n=15)
        correlation_matrix = self._create_correlation_matrix(top_numeric)
        relationships['correlation_matrix'] = correlation_matrix
        
        return relationships
    
    def _classify_correlation_strength(self, abs_corr):
        """Classify correlation strength"""
        if abs_corr >= 0.7:
            return 'Strong'
        elif abs_corr >= 0.5:
            return 'Moderate'
        elif abs_corr >= 0.3:
            return 'Weak'
        else:
            return 'Very Weak'
    
    def _calculate_eta_squared(self, groups):
        """Calculate eta-squared (effect size) for ANOVA"""
        # Simple eta-squared calculation
        all_values = np.concatenate(groups)
        grand_mean = np.mean(all_values)
        
        ss_between = sum(len(group) * (np.mean(group) - grand_mean)**2 for group in groups)
        ss_total = sum((val - grand_mean)**2 for val in all_values)
        
        eta_squared = ss_between / ss_total if ss_total > 0 else 0
        return float(eta_squared)
    
    def _calculate_feature_importance(self):
        """Calculate feature importance using Random Forest"""
        # Prepare data for Random Forest
        feature_data = pd.get_dummies(self.df, columns=self.categorical_cols, prefix_sep='_')
        feature_data = feature_data.select_dtypes(include=[np.number])
        
        if self.target_column not in feature_data.columns:
            return {'error': 'Target column not numeric'}
        
        # Remove target and ID columns
        X_cols = [col for col in feature_data.columns 
                  if col != self.target_column and 'Order' not in col and 'PID' not in col]
        
        X = feature_data[X_cols].fillna(feature_data[X_cols].median())
        y = feature_data[self.target_column].dropna()
        
        # Align X and y
        common_idx = X.index.intersection(y.index)
        X = X.loc[common_idx]
        y = y.loc[common_idx]
        
        if len(X) < 10:
            return {'error': 'Insufficient data for feature importance'}
        
        # Random Forest
        rf = RandomForestRegressor(n_estimators=100, random_state=42, n_jobs=-1)
        rf.fit(X, y)
        
        # Feature importance scores
        importance_scores = []
        for i, col in enumerate(X.columns):
            importance_scores.append({
                'feature': col,
                'importance': float(rf.feature_importances_[i]),
                'rank': i + 1
            })
        
        # Sort by importance
        importance_scores.sort(key=lambda x: x['importance'], reverse=True)
        
        # Update ranks
        for i, item in enumerate(importance_scores):
            item['rank'] = i + 1
        
        return {
            'top_features': importance_scores[:20],
            'model_score': float(rf.score(X, y))
        }
    
    def _get_top_correlated_features(self, correlations, n=15):
        """Get top N features correlated with target"""
        sorted_features = sorted(
            correlations.items(), 
            key=lambda x: abs(x[1]['pearson_correlation']), 
            reverse=True
        )
        return [feat[0] for feat in sorted_features[:n]]
    
    def _create_correlation_matrix(self, features):
        """Create correlation matrix for specified features"""
        if len(features) < 2:
            return {'error': 'Insufficient features for correlation matrix'}
        
        # Include target column
        all_features = features + [self.target_column]
        correlation_data = self.df[all_features].corr()
        
        # Convert to serializable format
        corr_matrix = {}
        for i, col1 in enumerate(correlation_data.columns):
            corr_matrix[col1] = {}
            for j, col2 in enumerate(correlation_data.columns):
                corr_matrix[col1][col2] = float(correlation_data.iloc[i, j])
        
        return corr_matrix
    
    def _generate_visualizations(self):
        """E. Visualization Section"""
        visualizations = {}
        
        # Set style
        plt.style.use('seaborn-v0_8')
        
        # 1. Target variable distribution
        target_viz = self._create_target_distribution()
        visualizations['target_distribution'] = target_viz
        
        # 2. Missing values heatmap
        missing_viz = self._create_missing_values_heatmap()
        visualizations['missing_values_heatmap'] = missing_viz
        
        # 3. Correlation heatmap
        correlation_viz = self._create_correlation_heatmap()
        visualizations['correlation_heatmap'] = correlation_viz
        
        # 4. Top features vs target
        feature_scatter = self._create_feature_scatter_plots()
        visualizations['feature_scatter_plots'] = feature_scatter
        
        # 5. Categorical analysis plots
        categorical_viz = self._create_categorical_analysis()
        visualizations['categorical_analysis'] = categorical_viz
        
        return visualizations
    
    def _create_target_distribution(self):
        """Create target variable distribution plot"""
        fig, axes = plt.subplots(2, 2, figsize=(15, 10))
        
        # Histogram
        axes[0, 0].hist(self.df[self.target_column].dropna(), bins=50, alpha=0.7, color='steelblue')
        axes[0, 0].set_title(f'{self.target_column} Distribution')
        axes[0, 0].set_xlabel('Sale Price ($)')
        axes[0, 0].set_ylabel('Frequency')
        
        # Box plot
        axes[0, 1].boxplot(self.df[self.target_column].dropna())
        axes[0, 1].set_title(f'{self.target_column} Box Plot')
        axes[0, 1].set_ylabel('Sale Price ($)')
        
        # Q-Q plot
        stats.probplot(self.df[self.target_column].dropna(), dist="norm", plot=axes[1, 0])
        axes[1, 0].set_title('Q-Q Plot (Normal Distribution)')
        
        # Log-transformed histogram
        log_prices = np.log(self.df[self.target_column].dropna())
        axes[1, 1].hist(log_prices, bins=50, alpha=0.7, color='lightcoral')
        axes[1, 1].set_title('Log(Sale Price) Distribution')
        axes[1, 1].set_xlabel('Log(Sale Price)')
        axes[1, 1].set_ylabel('Frequency')
        
        plt.tight_layout()
        
        # Save as base64
        buffer = io.BytesIO()
        plt.savefig(buffer, format='png', dpi=300, bbox_inches='tight')
        buffer.seek(0)
        image_base64 = base64.b64encode(buffer.getvalue()).decode()
        plt.close()
        
        # Statistics
        target_stats = {
            'mean': float(self.df[self.target_column].mean()),
            'median': float(self.df[self.target_column].median()),
            'std': float(self.df[self.target_column].std()),
            'skewness': float(stats.skew(self.df[self.target_column].dropna())),
            'min': float(self.df[self.target_column].min()),
            'max': float(self.df[self.target_column].max())
        }
        
        return {
            'image': image_base64,
            'statistics': target_stats,
            'interpretation': self._interpret_distribution(target_stats)
        }
    
    def _interpret_distribution(self, stats):
        """Interpret distribution characteristics"""
        interpretation = []
        
        if stats['skewness'] > 1:
            interpretation.append("Distribution is highly right-skewed")
        elif stats['skewness'] > 0.5:
            interpretation.append("Distribution is moderately right-skewed")
        elif stats['skewness'] < -1:
            interpretation.append("Distribution is highly left-skewed")
        elif stats['skewness'] < -0.5:
            interpretation.append("Distribution is moderately left-skewed")
        else:
            interpretation.append("Distribution is approximately symmetric")
        
        if stats['mean'] > stats['median']:
            interpretation.append("Mean > Median suggests right tail influence")
        elif stats['mean'] < stats['median']:
            interpretation.append("Mean < Median suggests left tail influence")
        
        cv = stats['std'] / stats['mean']
        if cv > 0.3:
            interpretation.append(f"High variability (CV = {cv:.2f})")
        elif cv < 0.1:
            interpretation.append(f"Low variability (CV = {cv:.2f})")
        
        return interpretation
    
    def _create_missing_values_heatmap(self):
        """Create missing values heatmap"""
        # Calculate missing percentages
        missing_data = self.df.isnull().sum().sort_values(ascending=False)
        missing_data = missing_data[missing_data > 0]
        
        if len(missing_data) == 0:
            return {'message': 'No missing values found in dataset'}
        
        missing_pct = (missing_data / len(self.df)) * 100
        
        fig, ax = plt.subplots(figsize=(12, max(6, len(missing_data) * 0.3)))
        
        # Create horizontal bar plot
        bars = ax.barh(range(len(missing_data)), missing_pct.values, color='coral')
        ax.set_yticks(range(len(missing_data)))
        ax.set_yticklabels(missing_data.index)
        ax.set_xlabel('Missing Percentage (%)')
        ax.set_title('Missing Values Analysis')
        
        # Add percentage labels
        for i, (bar, pct) in enumerate(zip(bars, missing_pct.values)):
            ax.text(bar.get_width() + 0.5, bar.get_y() + bar.get_height()/2, 
                   f'{pct:.1f}%', va='center')
        
        plt.tight_layout()
        
        # Save as base64
        buffer = io.BytesIO()
        plt.savefig(buffer, format='png', dpi=300, bbox_inches='tight')
        buffer.seek(0)
        image_base64 = base64.b64encode(buffer.getvalue()).decode()
        plt.close()
        
        return {
            'image': image_base64,
            'missing_summary': {
                col: {'count': int(count), 'percentage': float(pct)} 
                for col, count, pct in zip(missing_data.index, missing_data.values, missing_pct.values)
            }
        }
    
    def _create_correlation_heatmap(self):
        """Create correlation heatmap for top numeric features"""
        # Get numeric correlations with target
        correlations = {}
        for col in self.numeric_cols:
            if col != self.target_column and self.df[col].notna().sum() > 10:
                corr, _ = stats.pearsonr(
                    self.df[col].dropna(), 
                    self.df.loc[self.df[col].dropna().index, self.target_column]
                )
                correlations[col] = abs(corr)
        
        # Get top correlated features
        top_features = sorted(correlations.items(), key=lambda x: x[1], reverse=True)[:15]
        feature_names = [feat[0] for feat in top_features] + [self.target_column]
        
        # Create correlation matrix
        corr_matrix = self.df[feature_names].corr()
        
        # Create heatmap
        fig, ax = plt.subplots(figsize=(14, 12))
        
        mask = np.triu(np.ones_like(corr_matrix, dtype=bool))
        sns.heatmap(corr_matrix, mask=mask, annot=True, cmap='RdBu_r', center=0,
                   square=True, linewidths=0.5, cbar_kws={"shrink": 0.8}, fmt='.2f')
        
        ax.set_title('Correlation Matrix: Top Features vs Target')
        plt.xticks(rotation=45, ha='right')
        plt.yticks(rotation=0)
        plt.tight_layout()
        
        # Save as base64
        buffer = io.BytesIO()
        plt.savefig(buffer, format='png', dpi=300, bbox_inches='tight')
        buffer.seek(0)
        image_base64 = base64.b64encode(buffer.getvalue()).decode()
        plt.close()
        
        return {
            'image': image_base64,
            'top_correlations': [
                {'feature': feat, 'correlation': float(corr)} 
                for feat, corr in top_features[:10]
            ]
        }
    
    def _create_feature_scatter_plots(self):
        """Create scatter plots for top features vs target"""
        # Get top 6 correlated features
        correlations = {}
        for col in self.numeric_cols:
            if col != self.target_column and self.df[col].notna().sum() > 10:
                corr, _ = stats.pearsonr(
                    self.df[col].dropna(), 
                    self.df.loc[self.df[col].dropna().index, self.target_column]
                )
                correlations[col] = (corr, abs(corr))
        
        top_features = sorted(correlations.items(), key=lambda x: x[1][1], reverse=True)[:6]
        
        fig, axes = plt.subplots(2, 3, figsize=(18, 12))
        axes = axes.ravel()
        
        for i, (feature, (corr, _)) in enumerate(top_features):
            valid_data = self.df[[feature, self.target_column]].dropna()
            
            axes[i].scatter(valid_data[feature], valid_data[self.target_column], 
                          alpha=0.6, s=20, color='steelblue')
            axes[i].set_xlabel(feature)
            axes[i].set_ylabel(self.target_column)
            axes[i].set_title(f'{feature} vs {self.target_column}\n(r = {corr:.3f})')
            
            # Add trend line
            z = np.polyfit(valid_data[feature], valid_data[self.target_column], 1)
            p = np.poly1d(z)
            axes[i].plot(valid_data[feature].sort_values(), 
                        p(valid_data[feature].sort_values()), 
                        "r--", alpha=0.8, linewidth=2)
        
        plt.tight_layout()
        
        # Save as base64
        buffer = io.BytesIO()
        plt.savefig(buffer, format='png', dpi=300, bbox_inches='tight')
        buffer.seek(0)
        image_base64 = base64.b64encode(buffer.getvalue()).decode()
        plt.close()
        
        return {
            'image': image_base64,
            'featured_relationships': [
                {
                    'feature': feature,
                    'correlation': float(corr),
                    'strength': self._classify_correlation_strength(abs(corr))
                } for feature, (corr, _) in top_features
            ]
        }
    
    def _create_categorical_analysis(self):
        """Create categorical analysis plots"""
        # Get categorical columns with reasonable cardinality
        cat_cols = [col for col in self.categorical_cols 
                   if 2 <= self.df[col].nunique() <= 20 and self.df[col].notna().sum() > 10][:4]
        
        if len(cat_cols) < 2:
            return {'message': 'Insufficient categorical variables for analysis'}
        
        fig, axes = plt.subplots(2, 2, figsize=(16, 12))
        axes = axes.ravel()
        
        for i, col in enumerate(cat_cols):
            if i >= 4:
                break
                
            # Calculate mean target by category
            cat_analysis = self.df.groupby(col)[self.target_column].agg(['mean', 'count']).reset_index()
            cat_analysis = cat_analysis.sort_values('mean', ascending=False)
            
            # Create bar plot
            bars = axes[i].bar(range(len(cat_analysis)), cat_analysis['mean'], 
                             color='lightblue', edgecolor='navy', alpha=0.7)
            
            axes[i].set_xlabel(col)
            axes[i].set_ylabel(f'Mean {self.target_column}')
            axes[i].set_title(f'Mean {self.target_column} by {col}')
            axes[i].set_xticks(range(len(cat_analysis)))
            axes[i].set_xticklabels(cat_analysis[col], rotation=45, ha='right')
            
            # Add count labels
            for j, (bar, count) in enumerate(zip(bars, cat_analysis['count'])):
                axes[i].text(bar.get_x() + bar.get_width()/2, bar.get_height() + 1000, 
                           f'n={int(count)}', ha='center', va='bottom', fontsize=8)
        
        plt.tight_layout()
        
        # Save as base64
        buffer = io.BytesIO()
        plt.savefig(buffer, format='png', dpi=300, bbox_inches='tight')
        buffer.seek(0)
        image_base64 = base64.b64encode(buffer.getvalue()).decode()
        plt.close()
        
        return {
            'image': image_base64,
            'analyzed_categories': cat_cols
        }
    
    def _generate_narrative_insights(self):
        """F. Automated Narrative Insights - Context-aware analysis"""
        insights = {
            'key_findings': [],
            'data_quality_insights': [],
            'predictive_insights': [],
            'business_recommendations': [],
            'data_issues': [],
            'domain_specific_insights': []
        }
        
        # Key Findings
        target_mean = self.df[self.target_column].mean()
        target_median = self.df[self.target_column].median()
        
        insights['key_findings'].append(
            f"Average home price is ${target_mean:,.0f}, with median at ${target_median:,.0f}"
        )
        
        # Skewness analysis
        skewness = stats.skew(self.df[self.target_column].dropna())
        if skewness > 0.5:
            insights['key_findings'].append(
                f"Housing prices show right-skewed distribution (skewness: {skewness:.2f}), "
                f"indicating presence of luxury homes driving up the average"
            )
        
        # Missing data insights
        missing_cols = self.df.isnull().sum()
        high_missing = missing_cols[missing_cols > len(self.df) * 0.2]
        
        if len(high_missing) > 0:
            insights['data_quality_insights'].append(
                f"Critical data quality issue: {len(high_missing)} features have >20% missing values, "
                f"including {', '.join(high_missing.head(3).index.tolist())}"
            )
        
        # Feature correlation insights
        correlations = {}
        for col in self.numeric_cols:
            if col != self.target_column and self.df[col].notna().sum() > 10:
                corr, _ = stats.pearsonr(
                    self.df[col].dropna(), 
                    self.df.loc[self.df[col].dropna().index, self.target_column]
                )
                correlations[col] = corr
        
        # Top positive correlations
        top_positive = sorted([(k, v) for k, v in correlations.items() if v > 0.5], 
                            key=lambda x: x[1], reverse=True)[:3]
        
        for feature, corr in top_positive:
            insights['predictive_insights'].append(
                f"Strong positive relationship: {feature} shows {corr:.2f} correlation with sale price - "
                f"key predictor for home valuation"
            )
        
        # Domain-specific housing insights
        if 'Overall Qual' in self.df.columns:
            qual_corr = correlations.get('Overall Qual', 0)
            insights['domain_specific_insights'].append(
                f"Overall Quality rating is {'strongly' if abs(qual_corr) > 0.7 else 'moderately'} "
                f"correlated ({qual_corr:.2f}) with sale price, confirming importance of build quality"
            )
        
        if 'Gr Liv Area' in self.df.columns:
            area_corr = correlations.get('Gr Liv Area', 0)
            insights['domain_specific_insights'].append(
                f"Living area shows {area_corr:.2f} correlation with price - "
                f"each additional square foot significantly impacts valuation"
            )
        
        # Neighborhood analysis
        if 'Neighborhood' in self.df.columns:
            neighborhood_stats = self.df.groupby('Neighborhood')[self.target_column].agg(['mean', 'count'])
            top_neighborhood = neighborhood_stats['mean'].idxmax()
            lowest_neighborhood = neighborhood_stats['mean'].idxmin()
            price_range = neighborhood_stats['mean'].max() - neighborhood_stats['mean'].min()
            
            insights['domain_specific_insights'].append(
                f"Neighborhood location premium: {top_neighborhood} commands highest prices "
                f"while {lowest_neighborhood} has lowest - ${price_range:,.0f} spread indicates "
                f"location is a major value driver"
            )
        
        # Age analysis
        if 'Year Built' in self.df.columns:
            current_year = 2010  # Based on data
            self.df['House_Age'] = current_year - self.df['Year Built']
            age_corr = stats.pearsonr(self.df['House_Age'].dropna(), 
                                    self.df.loc[self.df['House_Age'].dropna().index, self.target_column])[0]
            
            insights['domain_specific_insights'].append(
                f"Property age impact: {age_corr:.2f} correlation suggests "
                f"{'newer homes command premium' if age_corr < -0.3 else 'age has minimal impact on pricing'}"
            )
        
        # Outlier insights
        target_q3 = self.df[self.target_column].quantile(0.75)
        target_q1 = self.df[self.target_column].quantile(0.25)
        iqr = target_q3 - target_q1
        outliers = self.df[(self.df[self.target_column] > target_q3 + 1.5 * iqr) | 
                          (self.df[self.target_column] < target_q1 - 1.5 * iqr)]
        
        if len(outliers) > 0:
            insights['data_issues'].append(
                f"Price outliers detected: {len(outliers)} properties ({len(outliers)/len(self.df)*100:.1f}%) "
                f"fall outside normal pricing range - may indicate luxury properties or data errors"
            )
        
        # Business Recommendations
        insights['business_recommendations'].extend([
            "Focus on Overall Quality, Living Area, and Neighborhood when pricing properties",
            "Implement data collection improvements for high-missing features like Pool QC, Fence",
            "Consider separate pricing models for luxury properties (top 5% price outliers)",
            "Neighborhood-based pricing strategies could capture significant location premiums"
        ])
        
        return insights
    
    def export_enhanced_report(self, filename='enhanced_housing_analysis.json'):
        """Export complete analysis report"""
        with open(filename, 'w') as f:
            json.dump(self.report, f, indent=2)
        print(f"✅ Enhanced analysis report saved to {filename}")
    
    def generate_html_report(self):
        """Generate comprehensive HTML report"""
        html_template = """
<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>Enhanced Ames Housing Data Analysis Report</title>
    <style>
        body { font-family: 'Segoe UI', Tahoma, Geneva, Verdana, sans-serif; margin: 0; padding: 20px; background: #f8f9fa; }
        .container { max-width: 1400px; margin: 0 auto; background: white; padding: 30px; border-radius: 10px; box-shadow: 0 2px 10px rgba(0,0,0,0.1); }
        .header { text-align: center; margin-bottom: 40px; padding: 20px; background: linear-gradient(135deg, #667eea 0%, #764ba2 100%); color: white; border-radius: 8px; }
        .section { margin: 30px 0; padding: 20px; border-left: 4px solid #667eea; background: #f8f9fa; border-radius: 5px; }
        .metric-grid { display: grid; grid-template-columns: repeat(auto-fit, minmax(250px, 1fr)); gap: 20px; margin: 20px 0; }
        .metric-card { background: white; padding: 20px; border-radius: 8px; box-shadow: 0 2px 5px rgba(0,0,0,0.1); text-align: center; }
        .metric-value { font-size: 2em; font-weight: bold; color: #667eea; }
        .metric-label { color: #666; margin-top: 5px; }
        .insight-box { background: #e8f4fd; padding: 15px; border-left: 4px solid #0066cc; margin: 10px 0; border-radius: 5px; }
        .warning-box { background: #fff3cd; padding: 15px; border-left: 4px solid #ffc107; margin: 10px 0; border-radius: 5px; }
        .success-box { background: #d4edda; padding: 15px; border-left: 4px solid #28a745; margin: 10px 0; border-radius: 5px; }
        .visualization { text-align: center; margin: 20px 0; }
        .visualization img { max-width: 100%; height: auto; border-radius: 8px; box-shadow: 0 2px 10px rgba(0,0,0,0.1); }
        .correlation-item { display: flex; justify-content: space-between; align-items: center; padding: 10px; background: white; margin: 5px 0; border-radius: 5px; }
        .correlation-bar { height: 20px; background: linear-gradient(90deg, #667eea, #764ba2); border-radius: 10px; }
        table { width: 100%; border-collapse: collapse; margin: 20px 0; }
        th, td { padding: 12px; text-align: left; border-bottom: 1px solid #ddd; }
        th { background: #667eea; color: white; }
        .highlight { background: #fff3cd; font-weight: bold; }
        .footer { text-align: center; margin-top: 40px; padding: 20px; color: #666; border-top: 1px solid #eee; }
    </style>
</head>
<body>
    <div class="container">
        <div class="header">
            <h1>🏠 Enhanced Ames Housing Data Analysis</h1>
            <p>Comprehensive Data Profiling & Predictive Insights</p>
            <p>Generated on: {timestamp}</p>
        </div>
        
        <div class="section">
            <h2>📊 A. Basic Data Overview</h2>
            <div class="metric-grid">
                <div class="metric-card">
                    <div class="metric-value">{total_rows}</div>
                    <div class="metric-label">Total Properties</div>
                </div>
                <div class="metric-card">
                    <div class="metric-value">{total_cols}</div>
                    <div class="metric-label">Features</div>
                </div>
                <div class="metric-card">
                    <div class="metric-value">{numeric_cols}</div>
                    <div class="metric-label">Numeric Features</div>
                </div>
                <div class="metric-card">
                    <div class="metric-value">{categorical_cols}</div>
                    <div class="metric-label">Categorical Features</div>
                </div>
            </div>
            {missing_analysis}
        </div>
        
        <div class="section">
            <h2>🔍 B. Data Quality Assessment</h2>
            <div class="metric-card" style="margin: 20px 0;">
                <div class="metric-value">{quality_score}/100</div>
                <div class="metric-label">Overall Data Quality Score</div>
            </div>
            {quality_analysis}
            {missing_viz}
        </div>
        
        <div class="section">
            <h2>📈 C. Descriptive Statistics & Target Analysis</h2>
            {target_analysis}
        </div>
        
        <div class="section">
            <h2>🔗 D. Relationship & Correlation Insights</h2>
            {correlation_analysis}
            {correlation_viz}
        </div>
        
        <div class="section">
            <h2>📊 E. Advanced Visualizations</h2>
            {visualizations}
        </div>
        
        <div class="section">
            <h2>🎯 F. Automated Narrative Insights</h2>
            {narrative_insights}
        </div>
        
        <div class="footer">
            <p>Generated by Enhanced DataScout Profiler | Advanced Analytics & AI-Powered Insights</p>
        </div>
    </div>
</body>
</html>
        """
        
        # This is a template structure - full implementation would populate all placeholders
        return html_template


if __name__ == "__main__":
    # Load the Ames Housing dataset
    df = pd.read_csv('data/AmesHousing.csv')
    
    # Initialize enhanced profiler
    profiler = EnhancedDataProfiler(df, target_column='SalePrice')
    
    # Generate comprehensive analysis
    print("🚀 Starting Enhanced Data Profiling...")
    report = profiler.generate_comprehensive_profile()
    
    # Export results
    profiler.export_enhanced_report('enhanced_ames_analysis.json')
    
    print("✨ Enhanced profiling complete! Check the output files for detailed insights.")
"""
Comprehensive Data Profiling Module for DataScout
Enhanced profiling that addresses gaps in existing reports and provides
deep, context-aware insights for the Ames Housing dataset.
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import scipy.stats as stats
from scipy.stats import chi2_contingency, f_oneway
import warnings
from typing import Dict, List, Tuple, Any, Optional, Union
import base64
from io import BytesIO
from datetime import datetime
import logging
from dataclasses import dataclass
from collections import Counter

warnings.filterwarnings('ignore')
plt.style.use('seaborn-v0_8')
sns.set_palette("husl")

logger = logging.getLogger(__name__)

@dataclass
class ProfileConfig:
    """Configuration for comprehensive profiling."""
    target_column: str = 'SalePrice'
    correlation_threshold: float = 0.7
    outlier_threshold: float = 3.0
    missing_threshold: float = 0.05
    cardinality_threshold: int = 50
    visualize: bool = True
    generate_html: bool = True

class ComprehensiveProfiler:
    """
    Enhanced data profiler that provides deep, context-aware insights
    specifically designed for the Ames Housing dataset.
    
    Addresses gaps in existing reports:
    - Shallow analysis -> Deep statistical insights
    - Generic outputs -> Context-aware housing domain insights  
    - Missing relationships -> Comprehensive correlation analysis
    - No target focus -> Target-driven feature analysis
    - Poor visualizations -> Rich, interactive visualizations
    """
    
    def __init__(self, config: Optional[ProfileConfig] = None):
        self.config = config or ProfileConfig()
        self.df = None
        self.profile_results = {}
        self.visualizations = {}
        self.insights = []
        
        # Housing-specific context knowledge
        self.housing_context = {
            'quality_features': ['OverallQual', 'OverallCond', 'ExterQual', 'ExterCond', 'BsmtQual', 
                               'BsmtCond', 'HeatingQC', 'KitchenQual', 'FireplaceQu', 'GarageQual'],
            'size_features': ['GrLivArea', 'TotalBsmtSF', '1stFlrSF', '2ndFlrSF', 'GarageArea', 
                            'WoodDeckSF', 'OpenPorchSF', 'LotArea', 'LotFrontage'],
            'age_features': ['YearBuilt', 'YearRemodAdd', 'GarageYrBlt', 'YrSold'],
            'categorical_features': ['Neighborhood', 'MSSubClass', 'MSZoning', 'Street', 'LotShape',
                                   'Condition1', 'BldgType', 'HouseStyle', 'RoofStyle', 'Exterior1st'],
            'price_drivers': ['GrLivArea', 'OverallQual', 'TotalBsmtSF', 'GarageCars', 'FullBath',
                            'YearBuilt', 'Fireplaces', 'LotFrontage']
        }
    
    def load_data(self, data_path: str) -> pd.DataFrame:
        """Load and validate the dataset."""
        logger.info(f"Loading dataset from {data_path}")
        
        try:
            self.df = pd.read_csv(data_path)
            logger.info(f"Dataset loaded successfully: {self.df.shape}")
            
            # Validate target column exists
            if self.config.target_column not in self.df.columns:
                logger.warning(f"Target column '{self.config.target_column}' not found")
                # Try to infer target column
                price_cols = [col for col in self.df.columns if 'price' in col.lower() or 'value' in col.lower()]
                if price_cols:
                    self.config.target_column = price_cols[0]
                    logger.info(f"Using inferred target column: {self.config.target_column}")
            
            return self.df
            
        except Exception as e:
            logger.error(f"Error loading dataset: {str(e)}")
            raise
    
    def generate_comprehensive_profile(self) -> Dict[str, Any]:
        """Generate the complete comprehensive profile."""
        logger.info("Starting comprehensive data profiling...")
        
        if self.df is None:
            raise ValueError("No data loaded. Call load_data() first.")
        
        # A. Basic Data Overview
        logger.info("Generating basic data overview...")
        self.profile_results['basic_overview'] = self._generate_basic_overview()
        
        # B. Data Quality Report  
        logger.info("Analyzing data quality...")
        self.profile_results['data_quality'] = self._generate_data_quality_report()
        
        # C. Descriptive Statistics
        logger.info("Computing descriptive statistics...")
        self.profile_results['descriptive_stats'] = self._generate_descriptive_statistics()
        
        # D. Relationship & Correlation Insights
        logger.info("Analyzing relationships and correlations...")
        self.profile_results['relationships'] = self._generate_relationship_analysis()
        
        # E. Visualization Section
        if self.config.visualize:
            logger.info("Creating visualizations...")
            self.profile_results['visualizations'] = self._generate_visualizations()
        
        # F. Automated Narrative Insights
        logger.info("Generating narrative insights...")
        self.profile_results['narrative_insights'] = self._generate_narrative_insights()
        
        # Housing-specific insights
        logger.info("Generating housing domain insights...")
        self.profile_results['housing_insights'] = self._generate_housing_specific_insights()
        
        logger.info("Comprehensive profiling completed successfully")
        return self.profile_results
    
    def _generate_basic_overview(self) -> Dict[str, Any]:
        """A. Basic Data Overview with enhanced metrics."""
        basic_overview = {}
        
        # Dataset shape and basic info
        basic_overview['shape'] = {
            'rows': len(self.df),
            'columns': len(self.df.columns),
            'memory_usage_mb': self.df.memory_usage(deep=True).sum() / 1024**2
        }
        
        # Missing values analysis (detailed)
        missing_data = self.df.isnull().sum()
        missing_pct = (missing_data / len(self.df)) * 100
        
        missing_analysis = {
            'total_missing_values': int(missing_data.sum()),
            'total_missing_percentage': float(missing_pct.sum() / len(self.df.columns)),
            'columns_with_missing': int((missing_data > 0).sum()),
            'missing_by_column': {
                col: {
                    'count': int(missing_data[col]),
                    'percentage': float(missing_pct[col])
                }
                for col in missing_data.index if missing_data[col] > 0
            }
        }
        basic_overview['missing_data'] = missing_analysis
        
        # Duplicate rows analysis
        duplicates = self.df.duplicated()
        basic_overview['duplicates'] = {
            'duplicate_rows': int(duplicates.sum()),
            'duplicate_percentage': float(duplicates.sum() / len(self.df) * 100),
            'unique_rows': int(len(self.df) - duplicates.sum())
        }
        
        # Unique values per column (cardinality analysis)
        cardinality_analysis = {}
        for col in self.df.columns:
            unique_count = self.df[col].nunique()
            unique_ratio = unique_count / len(self.df)
            
            # Classify column type based on cardinality
            if unique_count == 1:
                col_type = 'constant'
            elif unique_count == len(self.df):
                col_type = 'identifier'
            elif unique_ratio < 0.05:
                col_type = 'categorical'
            elif self.df[col].dtype in ['int64', 'float64']:
                col_type = 'numeric'
            else:
                col_type = 'mixed'
            
            cardinality_analysis[col] = {
                'unique_values': int(unique_count),
                'unique_ratio': float(unique_ratio),
                'inferred_type': col_type
            }
        
        basic_overview['cardinality'] = cardinality_analysis
        
        # Data types summary
        dtype_summary = {}
        for dtype in self.df.dtypes.value_counts().index:
            cols = self.df.select_dtypes(include=[dtype]).columns.tolist()
            dtype_summary[str(dtype)] = {
                'count': len(cols),
                'columns': cols
            }
        
        basic_overview['data_types'] = dtype_summary
        
        return basic_overview
    
    def _generate_data_quality_report(self) -> Dict[str, Any]:
        """B. Data Quality Report with advanced quality metrics."""
        quality_report = {}
        
        # Missing value patterns (missing data matrix analysis)
        missing_patterns = {}
        missing_df = self.df.isnull()
        
        # Most common missing patterns
        pattern_counts = missing_df.value_counts().head(10)
        missing_patterns['common_patterns'] = {
            f"Pattern_{i+1}": {
                'count': int(count),
                'percentage': float(count / len(self.df) * 100),
                'pattern': [col for col, val in pattern.items() if val]
            }
            for i, (pattern, count) in enumerate(pattern_counts.items())
        }
        
        # Columns with high missing data correlation
        missing_corr = missing_df.corr()
        high_missing_corr = {}
        for col1 in missing_corr.columns:
            for col2 in missing_corr.columns:
                if col1 < col2 and abs(missing_corr.loc[col1, col2]) > 0.5:
                    high_missing_corr[f"{col1}__{col2}"] = float(missing_corr.loc[col1, col2])
        
        missing_patterns['missing_correlations'] = high_missing_corr
        quality_report['missing_patterns'] = missing_patterns
        
        # Constant/near-constant values
        constant_analysis = {}
        for col in self.df.columns:
            if self.df[col].dtype in ['int64', 'float64']:
                # For numeric columns, check coefficient of variation
                if self.df[col].std() != 0:
                    cv = self.df[col].std() / abs(self.df[col].mean())
                    if cv < 0.01:  # Very low variation
                        constant_analysis[col] = {
                            'type': 'near_constant_numeric',
                            'coefficient_of_variation': float(cv),
                            'most_common_value': float(self.df[col].mode().iloc[0]) if not self.df[col].mode().empty else None,
                            'value_frequency': int((self.df[col] == self.df[col].mode().iloc[0]).sum()) if not self.df[col].mode().empty else 0
                        }
            else:
                # For categorical columns
                value_counts = self.df[col].value_counts()
                if len(value_counts) <= 2:  # Binary or constant
                    most_common_freq = value_counts.iloc[0] / len(self.df)
                    if most_common_freq > 0.95:  # 95% same value
                        constant_analysis[col] = {
                            'type': 'near_constant_categorical',
                            'most_common_value': value_counts.index[0],
                            'most_common_frequency': float(most_common_freq),
                            'unique_values': int(len(value_counts))
                        }
        
        quality_report['constant_values'] = constant_analysis
        
        # Outlier detection for numeric columns
        outlier_analysis = {}
        numeric_cols = self.df.select_dtypes(include=[np.number]).columns
        
        for col in numeric_cols:
            if self.df[col].notna().sum() > 0:  # Skip if all missing
                # IQR method
                Q1 = self.df[col].quantile(0.25)
                Q3 = self.df[col].quantile(0.75)
                IQR = Q3 - Q1
                lower_bound = Q1 - 1.5 * IQR
                upper_bound = Q3 + 1.5 * IQR
                iqr_outliers = ((self.df[col] < lower_bound) | (self.df[col] > upper_bound)).sum()
                
                # Z-score method
                z_scores = np.abs(stats.zscore(self.df[col].dropna()))
                z_outliers = (z_scores > self.config.outlier_threshold).sum()
                
                outlier_analysis[col] = {
                    'iqr_outliers': int(iqr_outliers),
                    'iqr_percentage': float(iqr_outliers / len(self.df) * 100),
                    'z_score_outliers': int(z_outliers),
                    'z_score_percentage': float(z_outliers / len(self.df) * 100),
                    'bounds': {
                        'iqr_lower': float(lower_bound),
                        'iqr_upper': float(upper_bound),
                        'z_threshold': float(self.config.outlier_threshold)
                    }
                }
        
        quality_report['outliers'] = outlier_analysis
        
        # Inconsistent categorical labels
        categorical_cols = self.df.select_dtypes(include=['object']).columns
        inconsistency_analysis = {}
        
        for col in categorical_cols:
            values = self.df[col].dropna().astype(str)
            
            # Check for case inconsistencies
            case_issues = {}
            value_groups = {}
            for val in values.unique():
                lower_val = val.lower()
                if lower_val not in value_groups:
                    value_groups[lower_val] = []
                value_groups[lower_val].append(val)
            
            for lower_val, variants in value_groups.items():
                if len(variants) > 1:
                    case_issues[lower_val] = variants
            
            # Check for whitespace/formatting issues
            whitespace_issues = []
            for val in values.unique():
                if val != val.strip() or '  ' in val:
                    whitespace_issues.append(val)
            
            if case_issues or whitespace_issues:
                inconsistency_analysis[col] = {
                    'case_inconsistencies': case_issues,
                    'whitespace_issues': whitespace_issues,
                    'total_unique_values': int(values.nunique()),
                    'potential_duplicates': sum(len(variants) - 1 for variants in case_issues.values())
                }
        
        quality_report['categorical_inconsistencies'] = inconsistency_analysis
        
        # Overall quality score
        total_issues = (
            len(missing_patterns['common_patterns']) +
            len(constant_analysis) +
            sum(1 for col in outlier_analysis if outlier_analysis[col]['iqr_percentage'] > 5) +
            len(inconsistency_analysis)
        )
        
        quality_score = max(0, 100 - (total_issues * 5))  # Penalize 5 points per issue type
        
        quality_report['overall_quality'] = {
            'quality_score': float(quality_score),
            'total_issues': int(total_issues),
            'assessment': 'Excellent' if quality_score >= 90 else 
                        'Good' if quality_score >= 70 else 
                        'Fair' if quality_score >= 50 else 'Poor'
        }
        
        return quality_report
    
    def _generate_descriptive_statistics(self) -> Dict[str, Any]:
        """C. Enhanced Descriptive Statistics for all feature types."""
        desc_stats = {}
        
        # Numeric features - comprehensive statistics
        numeric_cols = self.df.select_dtypes(include=[np.number]).columns
        numeric_stats = {}
        
        for col in numeric_cols:
            if self.df[col].notna().sum() > 0:
                data = self.df[col].dropna()
                
                # Basic statistics
                basic_stats = {
                    'count': int(len(data)),
                    'mean': float(data.mean()),
                    'median': float(data.median()),
                    'mode': float(data.mode().iloc[0]) if not data.mode().empty else None,
                    'std': float(data.std()),
                    'var': float(data.var()),
                    'min': float(data.min()),
                    'max': float(data.max()),
                    'range': float(data.max() - data.min()),
                }
                
                # Distribution characteristics
                skewness = float(data.skew())
                kurtosis = float(data.kurtosis())
                
                distribution_stats = {
                    'skewness': skewness,
                    'kurtosis': kurtosis,
                    'is_normal': bool(abs(skewness) < 0.5 and abs(kurtosis) < 3),
                    'distribution_shape': (
                        'highly_right_skewed' if skewness > 1 else
                        'moderately_right_skewed' if skewness > 0.5 else
                        'highly_left_skewed' if skewness < -1 else
                        'moderately_left_skewed' if skewness < -0.5 else
                        'approximately_symmetric'
                    )
                }
                
                # Quantiles
                quantile_stats = {
                    'q5': float(data.quantile(0.05)),
                    'q10': float(data.quantile(0.10)),
                    'q25': float(data.quantile(0.25)),
                    'q75': float(data.quantile(0.75)),
                    'q90': float(data.quantile(0.90)),
                    'q95': float(data.quantile(0.95)),
                    'iqr': float(data.quantile(0.75) - data.quantile(0.25))
                }
                
                numeric_stats[col] = {
                    **basic_stats,
                    **distribution_stats,
                    **quantile_stats
                }
        
        desc_stats['numeric_features'] = numeric_stats
        
        # Categorical features - enhanced frequency analysis
        categorical_cols = self.df.select_dtypes(include=['object']).columns
        categorical_stats = {}
        
        for col in categorical_cols:
            if self.df[col].notna().sum() > 0:
                value_counts = self.df[col].value_counts()
                total_count = len(self.df[col].dropna())
                
                # Top categories with percentages
                top_categories = {}
                for i, (category, count) in enumerate(value_counts.head(10).items()):
                    top_categories[f"rank_{i+1}"] = {
                        'category': str(category),
                        'count': int(count),
                        'percentage': float(count / total_count * 100)
                    }
                
                # Cardinality analysis
                cardinality_analysis = {
                    'unique_count': int(value_counts.count()),
                    'cardinality_ratio': float(value_counts.count() / total_count),
                    'entropy': float(-sum((p := count/total_count) * np.log2(p) for count in value_counts.values)),
                    'concentration': float(value_counts.iloc[0] / total_count)  # Concentration in top category
                }
                
                # Category distribution analysis
                category_distribution = {
                    'is_balanced': bool(cardinality_analysis['concentration'] < 0.5),
                    'dominance_level': (
                        'highly_concentrated' if cardinality_analysis['concentration'] > 0.8 else
                        'moderately_concentrated' if cardinality_analysis['concentration'] > 0.5 else
                        'well_distributed'
                    ),
                    'rare_categories_count': int((value_counts < 5).sum()),  # Categories with < 5 occurrences
                    'rare_categories_percentage': float((value_counts < 5).sum() / len(value_counts) * 100)
                }
                
                categorical_stats[col] = {
                    'total_observations': int(total_count),
                    'top_categories': top_categories,
                    'cardinality': cardinality_analysis,
                    'distribution': category_distribution
                }
        
        desc_stats['categorical_features'] = categorical_stats
        
        # DateTime features analysis (if any)
        datetime_stats = {}
        
        # Try to identify potential datetime columns
        potential_datetime_cols = []
        for col in self.df.columns:
            if any(keyword in col.lower() for keyword in ['date', 'time', 'year', 'month', 'day']):
                potential_datetime_cols.append(col)
        
        # For Ames Housing, we know some year columns
        year_cols = ['YearBuilt', 'YearRemodAdd', 'GarageYrBlt', 'YrSold', 'MoSold']
        available_year_cols = [col for col in year_cols if col in self.df.columns]
        
        for col in available_year_cols:
            if self.df[col].notna().sum() > 0:
                data = self.df[col].dropna()
                
                datetime_analysis = {
                    'min_year': int(data.min()),
                    'max_year': int(data.max()),
                    'year_range': int(data.max() - data.min()),
                    'most_common_year': int(data.mode().iloc[0]) if not data.mode().empty else None,
                    'trend_analysis': {
                        'years_covered': int(len(data.unique())),
                        'average_year': float(data.mean()),
                        'recent_bias': float((data > data.mean()).sum() / len(data))  # Proportion of recent years
                    }
                }
                
                datetime_stats[col] = datetime_analysis
        
        if datetime_stats:
            desc_stats['datetime_features'] = datetime_stats
        
        return desc_stats
    
    def _generate_relationship_analysis(self) -> Dict[str, Any]:
        """D. Comprehensive Relationship & Correlation Analysis."""
        relationships = {}
        
        # Target variable analysis
        target_col = self.config.target_column
        if target_col not in self.df.columns:
            logger.warning(f"Target column {target_col} not found")
            return relationships
        
        # Numeric-Numeric correlations with target
        numeric_cols = self.df.select_dtypes(include=[np.number]).columns
        target_correlations = {}
        
        if target_col in numeric_cols:
            correlations_pearson = {}
            correlations_spearman = {}
            
            for col in numeric_cols:
                if col != target_col and self.df[col].notna().sum() > 0:
                    # Pearson correlation
                    pearson_corr, pearson_p = stats.pearsonr(
                        self.df[col].dropna(), 
                        self.df.loc[self.df[col].notna(), target_col]
                    )
                    
                    # Spearman correlation  
                    spearman_corr, spearman_p = stats.spearmanr(
                        self.df[col].dropna(),
                        self.df.loc[self.df[col].notna(), target_col]
                    )
                    
                    correlations_pearson[col] = {
                        'correlation': float(pearson_corr),
                        'p_value': float(pearson_p),
                        'significant': bool(pearson_p < 0.05),
                        'strength': self._interpret_correlation_strength(abs(pearson_corr))
                    }
                    
                    correlations_spearman[col] = {
                        'correlation': float(spearman_corr), 
                        'p_value': float(spearman_p),
                        'significant': bool(spearman_p < 0.05),
                        'strength': self._interpret_correlation_strength(abs(spearman_corr))
                    }
            
            # Rank features by correlation strength
            pearson_ranked = sorted(correlations_pearson.items(), 
                                  key=lambda x: abs(x[1]['correlation']), reverse=True)
            
            target_correlations = {
                'pearson': correlations_pearson,
                'spearman': correlations_spearman,
                'top_correlated_features': [
                    {
                        'feature': feature,
                        'pearson_correlation': float(data['correlation']),
                        'spearman_correlation': float(correlations_spearman[feature]['correlation']),
                        'strength': data['strength']
                    }
                    for feature, data in pearson_ranked[:10]  # Top 10
                ]
            }
        
        relationships['target_correlations'] = target_correlations
        
        # Categorical-Numeric relationships (ANOVA F-tests)
        categorical_cols = self.df.select_dtypes(include=['object']).columns
        categorical_numeric_relationships = {}
        
        if target_col in numeric_cols:
            for cat_col in categorical_cols:
                if self.df[cat_col].notna().sum() > 0:
                    # Prepare data for ANOVA
                    groups = []
                    categories = self.df[cat_col].dropna().unique()
                    
                    # Only test if we have at least 2 categories and enough data
                    if len(categories) >= 2:
                        for category in categories:
                            category_data = self.df[
                                (self.df[cat_col] == category) & (self.df[target_col].notna())
                            ][target_col].values
                            
                            if len(category_data) >= 3:  # Minimum sample size
                                groups.append(category_data)
                        
                        if len(groups) >= 2:
                            try:
                                f_stat, p_value = f_oneway(*groups)
                                
                                # Calculate effect size (eta-squared)
                                # eta_squared = SS_between / SS_total
                                category_means = {}
                                category_counts = {}
                                for category in categories:
                                    cat_data = self.df[
                                        (self.df[cat_col] == category) & (self.df[target_col].notna())
                                    ][target_col]
                                    if len(cat_data) > 0:
                                        category_means[category] = float(cat_data.mean())
                                        category_counts[category] = int(len(cat_data))
                                
                                categorical_numeric_relationships[cat_col] = {
                                    'f_statistic': float(f_stat),
                                    'p_value': float(p_value),
                                    'significant': bool(p_value < 0.05),
                                    'effect_size': self._calculate_eta_squared(groups),
                                    'categories_tested': int(len(groups)),
                                    'category_means': category_means,
                                    'category_counts': category_counts,
                                    'interpretation': self._interpret_anova_result(f_stat, p_value)
                                }
                                
                            except Exception as e:
                                logger.warning(f"ANOVA failed for {cat_col}: {str(e)}")
        
        relationships['categorical_numeric'] = categorical_numeric_relationships
        
        # Categorical-Categorical relationships (Chi-square tests)
        categorical_relationships = {}
        
        if target_col in categorical_cols:
            for cat_col in categorical_cols:
                if cat_col != target_col and self.df[cat_col].notna().sum() > 0:
                    # Create contingency table
                    contingency_table = pd.crosstab(self.df[cat_col].fillna('Missing'), 
                                                  self.df[target_col].fillna('Missing'))
                    
                    # Only test if we have enough data
                    if contingency_table.shape[0] > 1 and contingency_table.shape[1] > 1:
                        try:
                            chi2, p_value, dof, expected = chi2_contingency(contingency_table)
                            
                            # Calculate Cramer's V
                            n = contingency_table.sum().sum()
                            cramers_v = np.sqrt(chi2 / (n * (min(contingency_table.shape) - 1)))
                            
                            categorical_relationships[cat_col] = {
                                'chi_square': float(chi2),
                                'p_value': float(p_value),
                                'degrees_of_freedom': int(dof),
                                'cramers_v': float(cramers_v),
                                'significant': bool(p_value < 0.05),
                                'association_strength': self._interpret_cramers_v(cramers_v),
                                'contingency_shape': list(contingency_table.shape)
                            }
                            
                        except Exception as e:
                            logger.warning(f"Chi-square test failed for {cat_col}: {str(e)}")
        
        relationships['categorical_categorical'] = categorical_relationships
        
        # Feature correlation matrix (numeric only)
        numeric_df = self.df[numeric_cols].select_dtypes(include=[np.number])
        if len(numeric_df.columns) > 1:
            correlation_matrix = numeric_df.corr()
            
            # Find strong correlations (excluding diagonal and duplicates)
            strong_correlations = {}
            for i in range(len(correlation_matrix.columns)):
                for j in range(i+1, len(correlation_matrix.columns)):
                    col1 = correlation_matrix.columns[i]
                    col2 = correlation_matrix.columns[j]
                    corr_val = correlation_matrix.iloc[i, j]
                    
                    if abs(corr_val) >= self.config.correlation_threshold:
                        strong_correlations[f"{col1}__{col2}"] = {
                            'correlation': float(corr_val),
                            'strength': self._interpret_correlation_strength(abs(corr_val)),
                            'features': [col1, col2]
                        }
            
            relationships['feature_correlations'] = {
                'strong_correlations': strong_correlations,
                'correlation_matrix_available': True,
                'matrix_size': list(correlation_matrix.shape)
            }
        
        return relationships
    
    def _interpret_correlation_strength(self, correlation: float) -> str:
        """Interpret correlation strength."""
        if correlation >= 0.8:
            return 'very_strong'
        elif correlation >= 0.6:
            return 'strong'
        elif correlation >= 0.4:
            return 'moderate'
        elif correlation >= 0.2:
            return 'weak'
        else:
            return 'very_weak'
    
    def _interpret_cramers_v(self, cramers_v: float) -> str:
        """Interpret Cramer's V strength."""
        if cramers_v >= 0.5:
            return 'strong'
        elif cramers_v >= 0.3:
            return 'moderate'
        elif cramers_v >= 0.1:
            return 'weak'
        else:
            return 'negligible'
    
    def _interpret_anova_result(self, f_stat: float, p_value: float) -> str:
        """Interpret ANOVA results."""
        if p_value < 0.001:
            return 'highly_significant_difference'
        elif p_value < 0.01:
            return 'very_significant_difference'
        elif p_value < 0.05:
            return 'significant_difference'
        else:
            return 'no_significant_difference'
    
    def _calculate_eta_squared(self, groups: List[np.ndarray]) -> float:
        """Calculate eta-squared effect size for ANOVA."""
        all_data = np.concatenate(groups)
        grand_mean = np.mean(all_data)
        
        # Sum of squares between groups
        ss_between = sum(len(group) * (np.mean(group) - grand_mean)**2 for group in groups)
        
        # Total sum of squares
        ss_total = np.sum((all_data - grand_mean)**2)
        
        return float(ss_between / ss_total) if ss_total > 0 else 0.0
    
    def _generate_visualizations(self) -> Dict[str, str]:
        """E. Generate comprehensive visualizations."""
        visualizations = {}
        
        plt.style.use('seaborn-v0_8')
        
        # 1. Missing value heatmap
        if self.df.isnull().sum().sum() > 0:
            missing_viz = self._create_missing_value_heatmap()
            if missing_viz:
                visualizations['missing_value_heatmap'] = missing_viz
        
        # 2. Target variable distribution
        target_col = self.config.target_column
        if target_col in self.df.columns:
            target_dist = self._create_target_distribution_plot()
            if target_dist:
                visualizations['target_distribution'] = target_dist
        
        # 3. Correlation heatmap
        correlation_heatmap = self._create_correlation_heatmap()
        if correlation_heatmap:
            visualizations['correlation_heatmap'] = correlation_heatmap
        
        # 4. Top feature correlations with target
        target_corr_plot = self._create_target_correlation_plot()
        if target_corr_plot:
            visualizations['target_correlations'] = target_corr_plot
        
        # 5. Distribution plots for key numeric features
        numeric_dist_plots = self._create_numeric_distributions()
        if numeric_dist_plots:
            visualizations['numeric_distributions'] = numeric_dist_plots
        
        # 6. Categorical feature plots
        categorical_plots = self._create_categorical_plots()
        if categorical_plots:
            visualizations['categorical_plots'] = categorical_plots
        
        # 7. Outlier detection plots
        outlier_plots = self._create_outlier_plots()
        if outlier_plots:
            visualizations['outlier_analysis'] = outlier_plots
        
        # 8. Housing-specific visualizations
        housing_viz = self._create_housing_specific_plots()
        visualizations.update(housing_viz)
        
        return visualizations
    
    def _create_missing_value_heatmap(self) -> Optional[str]:
        """Create missing value heatmap."""
        try:
            fig, ax = plt.subplots(figsize=(12, 8))
            
            # Get columns with missing values
            missing_data = self.df.isnull()
            cols_with_missing = missing_data.columns[missing_data.any()].tolist()
            
            if not cols_with_missing:
                return None
            
            # Create heatmap
            sns.heatmap(missing_data[cols_with_missing], 
                       cbar=True, yticklabels=False, 
                       cmap='viridis_r', ax=ax)
            
            ax.set_title('Missing Value Patterns\n(Yellow = Missing, Purple = Present)', 
                        fontsize=14, fontweight='bold')
            ax.set_xlabel('Features', fontsize=12)
            ax.set_ylabel('Observations (samples)', fontsize=12)
            
            plt.tight_layout()
            return self._fig_to_base64(fig)
            
        except Exception as e:
            logger.error(f"Error creating missing value heatmap: {str(e)}")
            return None
    
    def _create_target_distribution_plot(self) -> Optional[str]:
        """Create target variable distribution plot."""
        try:
            target_col = self.config.target_column
            if target_col not in self.df.columns:
                return None
                
            fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(15, 10))
            
            data = self.df[target_col].dropna()
            
            # Histogram with KDE
            ax1.hist(data, bins=50, density=True, alpha=0.7, color='skyblue', edgecolor='black')
            ax1.set_title(f'{target_col} Distribution', fontweight='bold')
            ax1.set_xlabel(target_col)
            ax1.set_ylabel('Density')
            
            # Add KDE
            try:
                from scipy.stats import gaussian_kde
                kde = gaussian_kde(data)
                x_range = np.linspace(data.min(), data.max(), 100)
                ax1.plot(x_range, kde(x_range), 'red', linewidth=2, label='KDE')
                ax1.legend()
            except:
                pass
            
            # Box plot
            ax2.boxplot(data, vert=True)
            ax2.set_title(f'{target_col} Box Plot', fontweight='bold')
            ax2.set_ylabel(target_col)
            
            # Q-Q plot for normality
            stats.probplot(data, dist="norm", plot=ax3)
            ax3.set_title('Q-Q Plot (Normality Check)', fontweight='bold')
            
            # Log-transformed distribution (for price data)
            if (data > 0).all():
                log_data = np.log(data)
                ax4.hist(log_data, bins=50, density=True, alpha=0.7, color='lightgreen', edgecolor='black')
                ax4.set_title(f'Log({target_col}) Distribution', fontweight='bold')
                ax4.set_xlabel(f'Log({target_col})')
                ax4.set_ylabel('Density')
            
            plt.tight_layout()
            return self._fig_to_base64(fig)
            
        except Exception as e:
            logger.error(f"Error creating target distribution plot: {str(e)}")
            return None
    
    def _create_correlation_heatmap(self) -> Optional[str]:
        """Create correlation heatmap for numeric features."""
        try:
            numeric_cols = self.df.select_dtypes(include=[np.number]).columns
            if len(numeric_cols) < 2:
                return None
            
            # Calculate correlation matrix
            corr_matrix = self.df[numeric_cols].corr()
            
            fig, ax = plt.subplots(figsize=(12, 10))
            
            # Create mask for upper triangle
            mask = np.triu(np.ones_like(corr_matrix, dtype=bool))
            
            # Generate heatmap
            sns.heatmap(corr_matrix, mask=mask, annot=True, fmt='.2f',
                       center=0, cmap='RdBu_r', square=True,
                       linewidths=0.5, cbar_kws={"shrink": .8}, ax=ax)
            
            ax.set_title('Feature Correlation Matrix\n(Stronger colors = Higher correlation)', 
                        fontsize=14, fontweight='bold')
            
            plt.tight_layout()
            return self._fig_to_base64(fig)
            
        except Exception as e:
            logger.error(f"Error creating correlation heatmap: {str(e)}")
            return None
    
    def _create_target_correlation_plot(self) -> Optional[str]:
        """Create target correlation plot showing top correlations."""
        try:
            target_col = self.config.target_column
            if target_col not in self.df.columns:
                return None
                
            numeric_cols = self.df.select_dtypes(include=[np.number]).columns
            correlations = []
            
            for col in numeric_cols:
                if col != target_col:
                    corr = self.df[col].corr(self.df[target_col])
                    if not np.isnan(corr):
                        correlations.append((col, corr))
            
            # Sort by absolute correlation
            correlations.sort(key=lambda x: abs(x[1]), reverse=True)
            top_correlations = correlations[:15]  # Top 15
            
            if not top_correlations:
                return None
            
            features, corr_values = zip(*top_correlations)
            
            fig, ax = plt.subplots(figsize=(12, 8))
            
            colors = ['red' if corr < 0 else 'green' for corr in corr_values]
            bars = ax.barh(range(len(features)), corr_values, color=colors, alpha=0.7)
            
            ax.set_yticks(range(len(features)))
            ax.set_yticklabels(features)
            ax.set_xlabel('Correlation Coefficient', fontsize=12)
            ax.set_title(f'Top Features Correlated with {target_col}', 
                        fontsize=14, fontweight='bold')
            ax.axvline(x=0, color='black', linestyle='-', alpha=0.3)
            ax.grid(True, alpha=0.3)
            
            # Add correlation values on bars
            for i, (bar, corr) in enumerate(zip(bars, corr_values)):
                ax.text(corr + 0.01 if corr > 0 else corr - 0.01, i, 
                       f'{corr:.3f}', va='center', 
                       ha='left' if corr > 0 else 'right', fontweight='bold')
            
            plt.tight_layout()
            return self._fig_to_base64(fig)
            
        except Exception as e:
            logger.error(f"Error creating target correlation plot: {str(e)}")
            return None
    
    def _create_numeric_distributions(self) -> Optional[str]:
        """Create distribution plots for key numeric features."""
        try:
            # Select key numeric features (top correlated with target + some important ones)
            numeric_cols = self.df.select_dtypes(include=[np.number]).columns
            target_col = self.config.target_column
            
            # Get top correlated features
            if target_col in numeric_cols:
                correlations = []
                for col in numeric_cols:
                    if col != target_col:
                        corr = abs(self.df[col].corr(self.df[target_col]))
                        if not np.isnan(corr):
                            correlations.append((col, corr))
                
                correlations.sort(key=lambda x: x[1], reverse=True)
                top_features = [col for col, _ in correlations[:6]]  # Top 6
            else:
                top_features = list(numeric_cols[:6])
            
            if not top_features:
                return None
            
            fig, axes = plt.subplots(2, 3, figsize=(18, 12))
            axes = axes.flatten()
            
            for i, col in enumerate(top_features):
                if i < len(axes):
                    data = self.df[col].dropna()
                    
                    if len(data) > 0:
                        # Histogram with KDE
                        axes[i].hist(data, bins=30, density=True, alpha=0.7, 
                                   color='skyblue', edgecolor='black')
                        
                        # Add statistics text
                        stats_text = f'Mean: {data.mean():.2f}\nStd: {data.std():.2f}\nSkew: {data.skew():.2f}'
                        axes[i].text(0.70, 0.95, stats_text, transform=axes[i].transAxes,
                                   verticalalignment='top', bbox=dict(boxstyle='round', 
                                   facecolor='white', alpha=0.8))
                        
                        axes[i].set_title(f'{col} Distribution', fontweight='bold')
                        axes[i].set_xlabel(col)
                        axes[i].set_ylabel('Density')
                        axes[i].grid(True, alpha=0.3)
            
            # Remove empty subplots
            for i in range(len(top_features), len(axes)):
                axes[i].remove()
            
            plt.tight_layout()
            return self._fig_to_base64(fig)
            
        except Exception as e:
            logger.error(f"Error creating numeric distributions: {str(e)}")
            return None
    
    def _create_categorical_plots(self) -> Optional[str]:
        """Create plots for key categorical features."""
        try:
            categorical_cols = self.df.select_dtypes(include=['object']).columns
            
            # Select interesting categorical columns (low to medium cardinality)
            interesting_cats = []
            for col in categorical_cols:
                unique_count = self.df[col].nunique()
                if 2 <= unique_count <= 15:  # Reasonable for plotting
                    interesting_cats.append(col)
            
            interesting_cats = interesting_cats[:4]  # Top 4
            
            if not interesting_cats:
                return None
            
            fig, axes = plt.subplots(2, 2, figsize=(16, 12))
            axes = axes.flatten()
            
            for i, col in enumerate(interesting_cats):
                if i < len(axes):
                    value_counts = self.df[col].value_counts().head(10)  # Top 10 categories
                    
                    # Create bar plot
                    bars = axes[i].bar(range(len(value_counts)), value_counts.values,
                                     color='lightcoral', alpha=0.7, edgecolor='black')
                    
                    axes[i].set_xticks(range(len(value_counts)))
                    axes[i].set_xticklabels(value_counts.index, rotation=45, ha='right')
                    axes[i].set_title(f'{col} Distribution', fontweight='bold')
                    axes[i].set_ylabel('Count')
                    axes[i].grid(True, alpha=0.3)
                    
                    # Add count labels on bars
                    for bar, count in zip(bars, value_counts.values):
                        axes[i].text(bar.get_x() + bar.get_width()/2., bar.get_height(),
                                   f'{count}', ha='center', va='bottom', fontweight='bold')
            
            # Remove empty subplots
            for i in range(len(interesting_cats), len(axes)):
                axes[i].remove()
            
            plt.tight_layout()
            return self._fig_to_base64(fig)
            
        except Exception as e:
            logger.error(f"Error creating categorical plots: {str(e)}")
            return None
    
    def _create_outlier_plots(self) -> Optional[str]:
        """Create outlier analysis plots."""
        try:
            # Select features with outliers
            numeric_cols = self.df.select_dtypes(include=[np.number]).columns
            outlier_features = []
            
            for col in numeric_cols:
                data = self.df[col].dropna()
                if len(data) > 0:
                    Q1 = data.quantile(0.25)
                    Q3 = data.quantile(0.75)
                    IQR = Q3 - Q1
                    outliers = ((data < Q1 - 1.5*IQR) | (data > Q3 + 1.5*IQR)).sum()
                    
                    if outliers > 0:
                        outlier_features.append((col, outliers))
            
            # Sort by number of outliers and take top 4
            outlier_features.sort(key=lambda x: x[1], reverse=True)
            top_outlier_features = [col for col, _ in outlier_features[:4]]
            
            if not top_outlier_features:
                return None
            
            fig, axes = plt.subplots(2, 2, figsize=(16, 12))
            axes = axes.flatten()
            
            for i, col in enumerate(top_outlier_features):
                if i < len(axes):
                    data = self.df[col].dropna()
                    
                    # Box plot with outlier highlighting
                    bp = axes[i].boxplot(data, patch_artist=True, 
                                       boxprops=dict(facecolor='lightblue', alpha=0.7))
                    
                    axes[i].set_title(f'{col} - Outlier Analysis', fontweight='bold')
                    axes[i].set_ylabel(col)
                    axes[i].grid(True, alpha=0.3)
                    
                    # Add outlier statistics
                    Q1 = data.quantile(0.25)
                    Q3 = data.quantile(0.75)
                    IQR = Q3 - Q1
                    outlier_count = ((data < Q1 - 1.5*IQR) | (data > Q3 + 1.5*IQR)).sum()
                    outlier_pct = outlier_count / len(data) * 100
                    
                    stats_text = f'Outliers: {outlier_count}\n({outlier_pct:.1f}%)'
                    axes[i].text(0.02, 0.98, stats_text, transform=axes[i].transAxes,
                               verticalalignment='top', bbox=dict(boxstyle='round',
                               facecolor='yellow', alpha=0.8))
            
            # Remove empty subplots
            for i in range(len(top_outlier_features), len(axes)):
                axes[i].remove()
            
            plt.tight_layout()
            return self._fig_to_base64(fig)
            
        except Exception as e:
            logger.error(f"Error creating outlier plots: {str(e)}")
            return None
    
    def _create_housing_specific_plots(self) -> Dict[str, str]:
        """Create housing domain-specific visualizations."""
        housing_plots = {}
        
        # 1. Price vs Living Area scatter plot
        if 'GrLivArea' in self.df.columns and self.config.target_column in self.df.columns:
            price_area_plot = self._create_price_vs_area_plot()
            if price_area_plot:
                housing_plots['price_vs_living_area'] = price_area_plot
        
        # 2. Overall Quality impact on price
        if 'OverallQual' in self.df.columns and self.config.target_column in self.df.columns:
            quality_price_plot = self._create_quality_price_plot()
            if quality_price_plot:
                housing_plots['quality_vs_price'] = quality_price_plot
        
        # 3. Neighborhood analysis
        if 'Neighborhood' in self.df.columns and self.config.target_column in self.df.columns:
            neighborhood_plot = self._create_neighborhood_analysis()
            if neighborhood_plot:
                housing_plots['neighborhood_analysis'] = neighborhood_plot
        
        return housing_plots
    
    def _create_price_vs_area_plot(self) -> Optional[str]:
        """Create price vs living area scatter plot."""
        try:
            fig, ax = plt.subplots(figsize=(12, 8))
            
            x = self.df['GrLivArea']
            y = self.df[self.config.target_column]
            
            # Create scatter plot
            scatter = ax.scatter(x, y, alpha=0.6, c='blue', s=30)
            
            # Add trend line
            z = np.polyfit(x.dropna(), y.dropna(), 1)
            p = np.poly1d(z)
            ax.plot(x, p(x), "r--", alpha=0.8, linewidth=2)
            
            # Calculate correlation
            correlation = x.corr(y)
            
            ax.set_xlabel('Above Ground Living Area (sq ft)', fontsize=12)
            ax.set_ylabel('Sale Price ($)', fontsize=12) 
            ax.set_title(f'Living Area vs Sale Price\nCorrelation: {correlation:.3f}', 
                        fontsize=14, fontweight='bold')
            ax.grid(True, alpha=0.3)
            
            # Format y-axis as currency
            ax.yaxis.set_major_formatter(plt.FuncFormatter(lambda x, p: f'${x/1000:.0f}K'))
            
            plt.tight_layout()
            return self._fig_to_base64(fig)
            
        except Exception as e:
            logger.error(f"Error creating price vs area plot: {str(e)}")
            return None
    
    def _create_quality_price_plot(self) -> Optional[str]:
        """Create overall quality vs price box plot."""
        try:
            fig, ax = plt.subplots(figsize=(12, 8))
            
            # Create box plot
            quality_data = []
            quality_labels = []
            
            for qual in sorted(self.df['OverallQual'].unique()):
                price_data = self.df[self.df['OverallQual'] == qual][self.config.target_column].dropna()
                if len(price_data) > 0:
                    quality_data.append(price_data)
                    quality_labels.append(f'Quality {qual}')
            
            bp = ax.boxplot(quality_data, labels=quality_labels, patch_artist=True)
            
            # Color boxes by quality level
            colors = plt.cm.RdYlGn(np.linspace(0.2, 0.8, len(bp['boxes'])))
            for patch, color in zip(bp['boxes'], colors):
                patch.set_facecolor(color)
                patch.set_alpha(0.7)
            
            ax.set_xlabel('Overall Quality Rating', fontsize=12)
            ax.set_ylabel('Sale Price ($)', fontsize=12)
            ax.set_title('House Price Distribution by Overall Quality Rating', 
                        fontsize=14, fontweight='bold')
            ax.grid(True, alpha=0.3)
            
            # Format y-axis as currency
            ax.yaxis.set_major_formatter(plt.FuncFormatter(lambda x, p: f'${x/1000:.0f}K'))
            
            plt.xticks(rotation=45)
            plt.tight_layout()
            return self._fig_to_base64(fig)
            
        except Exception as e:
            logger.error(f"Error creating quality price plot: {str(e)}")
            return None
    
    def _create_neighborhood_analysis(self) -> Optional[str]:
        """Create neighborhood price analysis."""
        try:
            # Calculate average price by neighborhood
            neighborhood_prices = self.df.groupby('Neighborhood')[self.config.target_column].agg(['mean', 'count']).reset_index()
            neighborhood_prices = neighborhood_prices[neighborhood_prices['count'] >= 5]  # At least 5 sales
            neighborhood_prices = neighborhood_prices.sort_values('mean', ascending=True)
            
            if len(neighborhood_prices) == 0:
                return None
            
            fig, ax = plt.subplots(figsize=(14, 10))
            
            bars = ax.barh(neighborhood_prices['Neighborhood'], neighborhood_prices['mean'],
                          color='lightgreen', alpha=0.7, edgecolor='black')
            
            ax.set_xlabel('Average Sale Price ($)', fontsize=12)
            ax.set_ylabel('Neighborhood', fontsize=12)
            ax.set_title('Average House Prices by Neighborhood\n(Min 5 sales per neighborhood)', 
                        fontsize=14, fontweight='bold')
            ax.grid(True, alpha=0.3)
            
            # Format x-axis as currency
            ax.xaxis.set_major_formatter(plt.FuncFormatter(lambda x, p: f'${x/1000:.0f}K'))
            
            # Add value labels
            for i, (bar, price) in enumerate(zip(bars, neighborhood_prices['mean'])):
                ax.text(price + 2000, i, f'${price/1000:.0f}K', 
                       va='center', ha='left', fontweight='bold')
            
            plt.tight_layout()
            return self._fig_to_base64(fig)
            
        except Exception as e:
            logger.error(f"Error creating neighborhood analysis: {str(e)}")
            return None
    
    def _fig_to_base64(self, fig) -> str:
        """Convert matplotlib figure to base64 string."""
        buffer = BytesIO()
        fig.savefig(buffer, format='png', dpi=300, bbox_inches='tight')
        buffer.seek(0)
        image_base64 = base64.b64encode(buffer.read()).decode('utf-8')
        plt.close(fig)
        return image_base64
    
    def _generate_narrative_insights(self) -> List[str]:
        """F. Generate automated narrative insights."""
        insights = []
        
        # Data quality insights
        quality_data = self.profile_results.get('data_quality', {})
        if 'overall_quality' in quality_data:
            quality_score = quality_data['overall_quality']['quality_score']
            insights.append(f"Data quality assessment shows a {quality_data['overall_quality']['assessment'].lower()} "
                          f"quality score of {quality_score:.1f}/100.")
        
        # Missing data insights
        basic_overview = self.profile_results.get('basic_overview', {})
        if 'missing_data' in basic_overview:
            missing_data = basic_overview['missing_data']
            total_missing_pct = missing_data['total_missing_percentage']
            
            if total_missing_pct > 10:
                insights.append(f"Dataset has significant missing data ({total_missing_pct:.1f}% overall) "
                              "that may require imputation strategies.")
            elif total_missing_pct > 5:
                insights.append(f"Dataset has moderate missing data ({total_missing_pct:.1f}% overall) "
                              "which should be addressed before modeling.")
            else:
                insights.append(f"Dataset has minimal missing data ({total_missing_pct:.1f}% overall), "
                              "indicating good data collection practices.")
        
        # Target variable insights
        relationships = self.profile_results.get('relationships', {})
        if 'target_correlations' in relationships:
            target_corrs = relationships['target_correlations']
            if 'top_correlated_features' in target_corrs:
                top_features = target_corrs['top_correlated_features'][:3]  # Top 3
                
                for i, feature_info in enumerate(top_features):
                    feature = feature_info['feature']
                    correlation = feature_info['pearson_correlation']
                    strength = feature_info['strength']
                    
                    if i == 0:  # Most correlated feature
                        insights.append(f"Feature '{feature}' shows the strongest correlation with the target "
                                      f"variable ({correlation:.3f}), indicating it's a {strength} predictor.")
                    elif abs(correlation) > 0.5:
                        insights.append(f"Feature '{feature}' has a {strength} correlation ({correlation:.3f}) "
                                      "with the target variable and should be prioritized in modeling.")
        
        # Outlier insights
        if 'outliers' in quality_data:
            outlier_data = quality_data['outliers']
            high_outlier_features = []
            
            for feature, outlier_info in outlier_data.items():
                if outlier_info['iqr_percentage'] > 5:  # More than 5% outliers
                    high_outlier_features.append((feature, outlier_info['iqr_percentage']))
            
            if high_outlier_features:
                high_outlier_features.sort(key=lambda x: x[1], reverse=True)
                top_outlier_feature, outlier_pct = high_outlier_features[0]
                insights.append(f"Feature '{top_outlier_feature}' contains {outlier_pct:.1f}% outliers, "
                              "which may indicate data quality issues or genuine extreme values requiring investigation.")
        
        # Categorical insights
        categorical_numeric = relationships.get('categorical_numeric', {})
        significant_categorical = []
        
        for cat_feature, anova_result in categorical_numeric.items():
            if anova_result['significant'] and anova_result['p_value'] < 0.001:
                significant_categorical.append((cat_feature, anova_result['effect_size']))
        
        if significant_categorical:
            significant_categorical.sort(key=lambda x: x[1], reverse=True)
            top_categorical, effect_size = significant_categorical[0]
            insights.append(f"Categorical feature '{top_categorical}' shows highly significant differences "
                          f"in target variable means (effect size: {effect_size:.3f}), making it valuable for segmentation.")
        
        # Feature correlation insights
        if 'feature_correlations' in relationships:
            strong_corrs = relationships['feature_correlations'].get('strong_correlations', {})
            
            if len(strong_corrs) > 0:
                insights.append(f"Found {len(strong_corrs)} strong feature correlations (>{self.config.correlation_threshold}), "
                              "indicating potential multicollinearity that should be addressed before modeling.")
        
        return insights
    
    def _generate_housing_specific_insights(self) -> Dict[str, Any]:
        """Generate housing domain-specific insights."""
        housing_insights = {}
        
        # Price range analysis
        if self.config.target_column in self.df.columns:
            prices = self.df[self.config.target_column].dropna()
            
            price_analysis = {
                'price_range': {
                    'min_price': float(prices.min()),
                    'max_price': float(prices.max()),
                    'median_price': float(prices.median()),
                    'price_range_spread': float(prices.max() - prices.min()),
                    'affordable_homes_pct': float((prices < prices.quantile(0.33)).sum() / len(prices) * 100),
                    'luxury_homes_pct': float((prices > prices.quantile(0.90)).sum() / len(prices) * 100)
                }
            }
            housing_insights['price_analysis'] = price_analysis
        
        # House size analysis
        size_features = ['GrLivArea', 'TotalBsmtSF', 'LotArea']
        available_size_features = [col for col in size_features if col in self.df.columns]
        
        if available_size_features:
            size_analysis = {}
            for feature in available_size_features:
                data = self.df[feature].dropna()
                if len(data) > 0:
                    size_analysis[feature] = {
                        'average_size': float(data.mean()),
                        'median_size': float(data.median()),
                        'size_variation': float(data.std() / data.mean()),  # Coefficient of variation
                        'large_homes_pct': float((data > data.quantile(0.75)).sum() / len(data) * 100)
                    }
            
            housing_insights['size_analysis'] = size_analysis
        
        # Quality distribution analysis
        quality_features = ['OverallQual', 'OverallCond']
        available_quality_features = [col for col in quality_features if col in self.df.columns]
        
        if available_quality_features:
            quality_analysis = {}
            for feature in available_quality_features:
                data = self.df[feature].dropna()
                if len(data) > 0:
                    quality_dist = data.value_counts().sort_index()
                    quality_analysis[feature] = {
                        'average_quality': float(data.mean()),
                        'quality_distribution': {int(k): int(v) for k, v in quality_dist.items()},
                        'high_quality_pct': float((data >= 7).sum() / len(data) * 100),  # Quality 7+ is high
                        'poor_quality_pct': float((data <= 4).sum() / len(data) * 100)    # Quality 4- is poor
                    }
            
            housing_insights['quality_analysis'] = quality_analysis
        
        # Age analysis
        if 'YearBuilt' in self.df.columns:
            current_year = datetime.now().year
            ages = current_year - self.df['YearBuilt'].dropna()
            
            age_analysis = {
                'average_age': float(ages.mean()),
                'median_age': float(ages.median()),
                'new_homes_pct': float((ages <= 10).sum() / len(ages) * 100),  # 10 years or newer
                'old_homes_pct': float((ages >= 50).sum() / len(ages) * 100),   # 50+ years old
                'age_distribution': {
                    'under_10_years': int((ages <= 10).sum()),
                    '10_30_years': int(((ages > 10) & (ages <= 30)).sum()),
                    '30_50_years': int(((ages > 30) & (ages <= 50)).sum()),
                    'over_50_years': int((ages > 50).sum())
                }
            }
            housing_insights['age_analysis'] = age_analysis
        
        # Neighborhood insights
        if 'Neighborhood' in self.df.columns and self.config.target_column in self.df.columns:
            neighborhood_stats = self.df.groupby('Neighborhood').agg({
                self.config.target_column: ['mean', 'median', 'count', 'std']
            }).round(2)
            
            neighborhood_stats.columns = ['avg_price', 'median_price', 'sales_count', 'price_std']
            neighborhood_stats = neighborhood_stats[neighborhood_stats['sales_count'] >= 5]  # Min 5 sales
            
            if len(neighborhood_stats) > 0:
                most_expensive = neighborhood_stats['avg_price'].idxmax()
                least_expensive = neighborhood_stats['avg_price'].idxmin()
                most_active = neighborhood_stats['sales_count'].idxmax()
                
                neighborhood_insights = {
                    'total_neighborhoods': int(len(neighborhood_stats)),
                    'most_expensive_neighborhood': {
                        'name': most_expensive,
                        'average_price': float(neighborhood_stats.loc[most_expensive, 'avg_price'])
                    },
                    'most_affordable_neighborhood': {
                        'name': least_expensive,
                        'average_price': float(neighborhood_stats.loc[least_expensive, 'avg_price'])
                    },
                    'most_active_neighborhood': {
                        'name': most_active,
                        'sales_count': int(neighborhood_stats.loc[most_active, 'sales_count'])
                    },
                    'price_variation_across_neighborhoods': float(neighborhood_stats['avg_price'].std())
                }
                
                housing_insights['neighborhood_insights'] = neighborhood_insights
        
        return housing_insights
    
    def export_html_report(self, output_path: str) -> str:
        """Export comprehensive report as HTML."""
        logger.info(f"Exporting HTML report to {output_path}")
        
        html_content = self._generate_html_report()
        
        with open(output_path, 'w', encoding='utf-8') as f:
            f.write(html_content)
        
        logger.info("HTML report exported successfully")
        return output_path
    
    def _generate_html_report(self) -> str:
        """Generate comprehensive HTML report."""
        # This would integrate with the existing HTML report generator
        # For now, returning a structured HTML template
        
        timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        
        html_template = f"""
<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>DataScout Comprehensive Profiling Report</title>
    <style>
        body {{ font-family: 'Segoe UI', Tahoma, Geneva, Verdana, sans-serif; line-height: 1.6; margin: 0; padding: 20px; background-color: #f5f5f5; }}
        .container {{ max-width: 1200px; margin: 0 auto; background: white; padding: 30px; border-radius: 10px; box-shadow: 0 0 20px rgba(0,0,0,0.1); }}
        .header {{ text-align: center; margin-bottom: 30px; padding: 20px; background: linear-gradient(135deg, #667eea 0%, #764ba2 100%); color: white; border-radius: 8px; }}
        .section {{ margin-bottom: 30px; padding: 20px; border: 1px solid #ddd; border-radius: 8px; }}
        .section h2 {{ color: #333; border-bottom: 2px solid #667eea; padding-bottom: 10px; }}
        .insight {{ background: #e8f4f8; padding: 15px; margin: 10px 0; border-left: 4px solid #667eea; border-radius: 4px; }}
        .stat-grid {{ display: grid; grid-template-columns: repeat(auto-fit, minmax(200px, 1fr)); gap: 15px; margin: 20px 0; }}
        .stat-card {{ background: #f8f9fa; padding: 15px; border-radius: 8px; text-align: center; border: 1px solid #dee2e6; }}
        .stat-number {{ font-size: 2em; font-weight: bold; color: #667eea; }}
        .visualization {{ text-align: center; margin: 20px 0; }}
        .visualization img {{ max-width: 100%; height: auto; border: 1px solid #ddd; border-radius: 8px; box-shadow: 0 2px 8px rgba(0,0,0,0.1); }}
        table {{ width: 100%; border-collapse: collapse; margin: 20px 0; }}
        th, td {{ padding: 12px; text-align: left; border-bottom: 1px solid #ddd; }}
        th {{ background-color: #f8f9fa; font-weight: bold; }}
        .footer {{ text-align: center; margin-top: 30px; padding: 20px; background-color: #f8f9fa; border-radius: 8px; color: #666; }}
    </style>
</head>
<body>
    <div class="container">
        <div class="header">
            <h1>🏠 DataScout Comprehensive Data Profiling Report</h1>
            <p>Enhanced Analysis for Ames Housing Dataset</p>
            <p>Generated on: {timestamp}</p>
        </div>
        
        <!-- Report sections would be populated here -->
        <div class="section">
            <h2>📊 Executive Summary</h2>
            <p>This comprehensive profiling report addresses gaps in existing analyses by providing:</p>
            <ul>
                <li><strong>Deep Statistical Insights:</strong> Advanced descriptive statistics beyond basic summaries</li>
                <li><strong>Context-Aware Analysis:</strong> Housing domain-specific insights and interpretations</li>
                <li><strong>Target-Driven Analysis:</strong> All features analyzed in relation to house prices</li>
                <li><strong>Comprehensive Quality Assessment:</strong> Multi-dimensional data quality evaluation</li>
                <li><strong>Rich Visualizations:</strong> Interactive and informative charts for all data aspects</li>
                <li><strong>Actionable Insights:</strong> Specific recommendations for data preparation and modeling</li>
            </ul>
        </div>
        
        <!-- Additional sections would be dynamically generated based on profile_results -->
        
        <div class="footer">
            <p><strong>DataScout Enhanced Profiler</strong> - Comprehensive, Context-Aware Data Analysis</p>
            <p>This report provides deeper insights than standard profiling tools by incorporating domain knowledge and advanced statistical analysis.</p>
        </div>
    </div>
</body>
</html>
        """
        
        return html_template


# Factory function
def create_comprehensive_profiler(config: Optional[ProfileConfig] = None) -> ComprehensiveProfiler:
    """Create and return a ComprehensiveProfiler instance."""
    return ComprehensiveProfiler(config)


# Example usage function
def analyze_ames_housing(data_path: str, output_dir: str = './reports/') -> Dict[str, Any]:
    """
    Comprehensive analysis of Ames Housing dataset.
    
    Args:
        data_path: Path to the Ames Housing CSV file
        output_dir: Directory to save reports
        
    Returns:
        Complete profiling results
    """
    logger.info("Starting comprehensive Ames Housing analysis...")
    
    # Configure for housing data
    config = ProfileConfig(
        target_column='SalePrice',
        correlation_threshold=0.6,
        outlier_threshold=3.0,
        visualize=True,
        generate_html=True
    )
    
    # Create profiler
    profiler = create_comprehensive_profiler(config)
    
    # Load data
    df = profiler.load_data(data_path)
    
    # Generate comprehensive profile
    results = profiler.generate_comprehensive_profile()
    
    # Export HTML report
    import os
    os.makedirs(output_dir, exist_ok=True)
    html_path = os.path.join(output_dir, 'comprehensive_housing_analysis.html')
    profiler.export_html_report(html_path)
    
    logger.info(f"Analysis complete. Report saved to: {html_path}")
    
    return results
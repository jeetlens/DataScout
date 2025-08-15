"""
Data Summarizer Module for DataScout
Generates descriptive statistics, correlations, and comprehensive data insights.
"""

import pandas as pd
import numpy as np
from typing import Dict, List, Any, Optional, Union, Tuple
from scipy import stats
import logging

logger = logging.getLogger(__name__)

class DataSummarizer:
    """
    Data summarizer class that generates comprehensive statistical summaries and insights.
    
    Features:
    - Descriptive statistics for numerical and categorical data
    - Correlation analysis
    - Distribution analysis
    - Data profiling and insights
    - Missing data analysis
    - Outlier detection summaries
    """
    
    def __init__(self):
        self.summary_cache = {}
        
    def generate_comprehensive_summary(self, df: pd.DataFrame) -> Dict[str, Any]:
        """
        Generate a comprehensive summary of the dataset.
        
        Args:
            df: Input DataFrame
            
        Returns:
            Dict containing comprehensive summary information
        """
        summary = {
            'basic_info': self._get_basic_info(df),
            'descriptive_stats': self._get_descriptive_statistics(df),
            'missing_data_analysis': self._analyze_missing_data(df),
            'correlation_analysis': self._analyze_correlations(df),
            'distribution_analysis': self._analyze_distributions(df),
            'categorical_analysis': self._analyze_categorical_data(df),
            'outlier_analysis': self._analyze_outliers(df),
            'data_quality_insights': self._generate_quality_insights(df)
        }
        
        logger.info(f"Generated comprehensive summary for dataset with shape {df.shape}")
        return summary
        
    def _get_basic_info(self, df: pd.DataFrame) -> Dict[str, Any]:
        """Get basic dataset information."""
        numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
        categorical_cols = df.select_dtypes(include=['object', 'category']).columns.tolist()
        datetime_cols = df.select_dtypes(include=['datetime64']).columns.tolist()
        
        return {
            'total_rows': len(df),
            'total_columns': len(df.columns),
            'column_names': list(df.columns),
            'data_types': df.dtypes.to_dict(),
            'numeric_columns': numeric_cols,
            'categorical_columns': categorical_cols,
            'datetime_columns': datetime_cols,
            'memory_usage_mb': round(df.memory_usage(deep=True).sum() / 1024 / 1024, 2),
            'dataset_size': f"{len(df)} rows x {len(df.columns)} columns"
        }
        
    def _get_descriptive_statistics(self, df: pd.DataFrame) -> Dict[str, Any]:
        """Generate descriptive statistics for numerical columns."""
        numeric_df = df.select_dtypes(include=[np.number])
        
        if numeric_df.empty:
            return {'message': 'No numerical columns found'}
            
        desc_stats = {}
        
        for column in numeric_df.columns:
            series = numeric_df[column].dropna()
            if len(series) == 0:
                continue
                
            stats_dict = {
                'count': len(series),
                'mean': round(series.mean(), 4),
                'median': round(series.median(), 4),
                'std': round(series.std(), 4),
                'min': round(series.min(), 4),
                'max': round(series.max(), 4),
                'q25': round(series.quantile(0.25), 4),
                'q75': round(series.quantile(0.75), 4),
                'iqr': round(series.quantile(0.75) - series.quantile(0.25), 4),
                'variance': round(series.var(), 4),
                'skewness': round(series.skew(), 4),
                'kurtosis': round(series.kurtosis(), 4),
                'range': round(series.max() - series.min(), 4)
            }
            
            # Add distribution insights
            if abs(stats_dict['skewness']) < 0.5:
                stats_dict['distribution_shape'] = 'approximately normal'
            elif stats_dict['skewness'] > 0.5:
                stats_dict['distribution_shape'] = 'right-skewed'
            else:
                stats_dict['distribution_shape'] = 'left-skewed'
                
            desc_stats[column] = stats_dict
            
        return desc_stats
        
    def _analyze_missing_data(self, df: pd.DataFrame) -> Dict[str, Any]:
        """Analyze missing data patterns."""
        missing_counts = df.isnull().sum()
        total_rows = len(df)
        
        missing_analysis = {
            'total_missing_values': int(missing_counts.sum()),
            'missing_percentage_overall': round((missing_counts.sum() / (total_rows * len(df.columns))) * 100, 2),
            'columns_with_missing': {},
            'missing_data_patterns': {}
        }
        
        # Analyze each column
        for column in df.columns:
            missing_count = missing_counts[column]
            if missing_count > 0:
                missing_pct = round((missing_count / total_rows) * 100, 2)
                missing_analysis['columns_with_missing'][column] = {
                    'count': int(missing_count),
                    'percentage': missing_pct,
                    'severity': self._get_missing_data_severity(missing_pct)
                }
                
        # Find missing data patterns
        if missing_counts.sum() > 0:
            missing_patterns = df.isnull().value_counts().head(5)
            missing_analysis['missing_data_patterns'] = {
                str(pattern): int(count) for pattern, count in missing_patterns.items()
            }
            
        return missing_analysis
        
    def _get_missing_data_severity(self, percentage: float) -> str:
        """Classify missing data severity."""
        if percentage == 0:
            return 'none'
        elif percentage < 5:
            return 'low'
        elif percentage < 15:
            return 'moderate'
        elif percentage < 50:
            return 'high'
        else:
            return 'critical'
            
    def _analyze_correlations(self, df: pd.DataFrame) -> Dict[str, Any]:
        """Analyze correlations between numerical variables."""
        numeric_df = df.select_dtypes(include=[np.number])
        
        if len(numeric_df.columns) < 2:
            return {'message': 'Need at least 2 numerical columns for correlation analysis'}
            
        correlation_matrix = numeric_df.corr()
        
        # Find strong correlations
        strong_correlations = []
        for i in range(len(correlation_matrix.columns)):
            for j in range(i+1, len(correlation_matrix.columns)):
                corr_value = correlation_matrix.iloc[i, j]
                if abs(corr_value) > 0.7:  # Strong correlation threshold
                    strong_correlations.append({
                        'variable1': correlation_matrix.columns[i],
                        'variable2': correlation_matrix.columns[j],
                        'correlation': round(corr_value, 4),
                        'strength': self._get_correlation_strength(abs(corr_value))
                    })
                    
        return {
            'correlation_matrix': correlation_matrix.round(4).to_dict(),
            'strong_correlations': strong_correlations,
            'average_correlation': round(correlation_matrix.values[np.triu_indices_from(correlation_matrix.values, k=1)].mean(), 4)
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
            
    def _analyze_distributions(self, df: pd.DataFrame) -> Dict[str, Any]:
        """Analyze distributions of numerical variables."""
        numeric_df = df.select_dtypes(include=[np.number])
        
        if numeric_df.empty:
            return {'message': 'No numerical columns found'}
            
        distribution_analysis = {}
        
        for column in numeric_df.columns:
            series = numeric_df[column].dropna()
            if len(series) < 3:
                continue
                
            # Normality test (Shapiro-Wilk for small samples, Anderson-Darling for larger)
            if len(series) <= 5000:
                stat, p_value = stats.shapiro(series)
                test_name = 'Shapiro-Wilk'
            else:
                # Use Anderson-Darling test for larger samples
                result = stats.anderson(series, dist='norm')
                stat, p_value = result.statistic, 0.05 if result.statistic > result.critical_values[2] else 0.1
                test_name = 'Anderson-Darling'
                
            distribution_analysis[column] = {
                'normality_test': test_name,
                'test_statistic': round(stat, 4),
                'p_value': round(p_value, 4) if p_value is not None else None,
                'is_normal': p_value > 0.05 if p_value is not None else None,
                'unique_values': int(series.nunique()),
                'unique_percentage': round((series.nunique() / len(series)) * 100, 2)
            }
            
        return distribution_analysis
        
    def _analyze_categorical_data(self, df: pd.DataFrame) -> Dict[str, Any]:
        """Analyze categorical variables."""
        categorical_df = df.select_dtypes(include=['object', 'category'])
        
        if categorical_df.empty:
            return {'message': 'No categorical columns found'}
            
        categorical_analysis = {}
        
        for column in categorical_df.columns:
            series = categorical_df[column].dropna()
            if len(series) == 0:
                continue
                
            value_counts = series.value_counts()
            
            categorical_analysis[column] = {
                'unique_values': int(series.nunique()),
                'most_frequent': str(value_counts.index[0]) if not value_counts.empty else None,
                'most_frequent_count': int(value_counts.iloc[0]) if not value_counts.empty else 0,
                'most_frequent_percentage': round((value_counts.iloc[0] / len(series)) * 100, 2) if not value_counts.empty else 0,
                'cardinality': 'high' if series.nunique() > len(series) * 0.5 else 'low',
                'top_5_values': {str(k): int(v) for k, v in value_counts.head(5).items()}
            }
            
        return categorical_analysis
        
    def _analyze_outliers(self, df: pd.DataFrame) -> Dict[str, Any]:
        """Analyze outliers in numerical columns."""
        numeric_df = df.select_dtypes(include=[np.number])
        
        if numeric_df.empty:
            return {'message': 'No numerical columns found'}
            
        outlier_analysis = {}
        
        for column in numeric_df.columns:
            series = numeric_df[column].dropna()
            if len(series) < 4:
                continue
                
            Q1 = series.quantile(0.25)
            Q3 = series.quantile(0.75)
            IQR = Q3 - Q1
            lower_bound = Q1 - 1.5 * IQR
            upper_bound = Q3 + 1.5 * IQR
            
            outliers = series[(series < lower_bound) | (series > upper_bound)]
            
            outlier_analysis[column] = {
                'outlier_count': len(outliers),
                'outlier_percentage': round((len(outliers) / len(series)) * 100, 2),
                'lower_bound': round(lower_bound, 4),
                'upper_bound': round(upper_bound, 4),
                'outlier_values': outliers.tolist()[:10] if len(outliers) <= 10 else outliers.tolist()[:10] + ['...more']
            }
            
        return outlier_analysis
        
    def _generate_quality_insights(self, df: pd.DataFrame) -> List[str]:
        """Generate data quality insights and recommendations."""
        insights = []
        
        # Check data completeness
        missing_pct = (df.isnull().sum().sum() / (len(df) * len(df.columns))) * 100
        if missing_pct > 10:
            insights.append(f"Dataset has {missing_pct:.1f}% missing values - consider data imputation")
        elif missing_pct > 0:
            insights.append(f"Dataset has {missing_pct:.1f}% missing values - generally good quality")
        else:
            insights.append("Dataset has no missing values - excellent completeness")
            
        # Check for duplicates
        duplicate_count = df.duplicated().sum()
        if duplicate_count > 0:
            insights.append(f"Found {duplicate_count} duplicate rows - consider deduplication")
            
        # Check column variety
        numeric_cols = len(df.select_dtypes(include=[np.number]).columns)
        categorical_cols = len(df.select_dtypes(include=['object', 'category']).columns)
        
        if numeric_cols == 0:
            insights.append("Dataset contains no numerical columns - limited for statistical analysis")
        elif categorical_cols == 0:
            insights.append("Dataset contains no categorical columns - purely numerical data")
        else:
            insights.append(f"Balanced dataset with {numeric_cols} numerical and {categorical_cols} categorical columns")
            
        # Check dataset size
        if len(df) < 100:
            insights.append("Small dataset - results may not be statistically significant")
        elif len(df) > 100000:
            insights.append("Large dataset - consider sampling for exploratory analysis")
        else:
            insights.append("Dataset size is appropriate for comprehensive analysis")
            
        return insights
        
    def generate_summary_report(self, df: pd.DataFrame, title: str = "Dataset Summary Report") -> Dict[str, Any]:
        """Generate a formatted summary report."""
        comprehensive_summary = self.generate_comprehensive_summary(df)
        
        report = {
            'title': title,
            'generated_at': pd.Timestamp.now().strftime('%Y-%m-%d %H:%M:%S'),
            'dataset_overview': comprehensive_summary['basic_info'],
            'key_statistics': comprehensive_summary['descriptive_stats'],
            'data_quality': {
                'missing_data': comprehensive_summary['missing_data_analysis'],
                'outliers': comprehensive_summary['outlier_analysis']
            },
            'relationships': comprehensive_summary['correlation_analysis'],
            'insights_and_recommendations': comprehensive_summary['data_quality_insights']
        }
        
        return report
        
    def compare_datasets(self, df1: pd.DataFrame, df2: pd.DataFrame, 
                        names: Tuple[str, str] = ('Dataset 1', 'Dataset 2')) -> Dict[str, Any]:
        """Compare two datasets and highlight differences."""
        summary1 = self.generate_comprehensive_summary(df1)
        summary2 = self.generate_comprehensive_summary(df2)
        
        comparison = {
            'dataset_names': names,
            'basic_comparison': {
                'rows': (summary1['basic_info']['total_rows'], summary2['basic_info']['total_rows']),
                'columns': (summary1['basic_info']['total_columns'], summary2['basic_info']['total_columns']),
                'memory_usage_mb': (summary1['basic_info']['memory_usage_mb'], summary2['basic_info']['memory_usage_mb'])
            },
            'column_differences': {
                'common_columns': list(set(df1.columns) & set(df2.columns)),
                'unique_to_first': list(set(df1.columns) - set(df2.columns)),
                'unique_to_second': list(set(df2.columns) - set(df1.columns))
            },
            'quality_comparison': {
                'missing_data_pct': (
                    summary1['missing_data_analysis']['missing_percentage_overall'],
                    summary2['missing_data_analysis']['missing_percentage_overall']
                )
            }
        }
        
        return comparison


# Factory function
def create_summarizer() -> DataSummarizer:
    """Create and return a DataSummarizer instance."""
    return DataSummarizer()


# Convenience functions
def quick_summary(df: pd.DataFrame) -> Dict[str, Any]:
    """Generate a quick summary of the dataset."""
    summarizer = DataSummarizer()
    return summarizer.generate_comprehensive_summary(df)


def generate_report(df: pd.DataFrame, title: str = "Data Summary") -> Dict[str, Any]:
    """Generate a formatted summary report."""
    summarizer = DataSummarizer()
    return summarizer.generate_summary_report(df, title)
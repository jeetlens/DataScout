"""
Data Preprocessor Module for DataScout
Handles data cleaning, validation, transformation, and preparation for analysis.
"""

import pandas as pd
import numpy as np
from typing import Dict, List, Any, Optional, Union, Tuple
from sklearn.preprocessing import StandardScaler, MinMaxScaler, LabelEncoder
from sklearn.impute import SimpleImputer
import logging

logger = logging.getLogger(__name__)

class DataPreprocessor:
    """
    Data preprocessing class that handles cleaning, validation, and transformation.
    
    Features:
    - Missing value handling
    - Data type conversion
    - Outlier detection and treatment
    - Data encoding (categorical variables)
    - Data scaling and normalization
    - Basic data validation
    """
    
    def __init__(self):
        self.scalers = {}
        self.encoders = {}
        self.imputers = {}
        
    def clean_data(self, df: pd.DataFrame, config: Optional[Dict[str, Any]] = None) -> pd.DataFrame:
        """
        Comprehensive data cleaning pipeline.
        
        Args:
            df: Input DataFrame
            config: Configuration dictionary for cleaning options
            
        Returns:
            pd.DataFrame: Cleaned DataFrame
        """
        if config is None:
            config = self._get_default_cleaning_config()
            
        df_cleaned = df.copy()
        
        # Remove duplicate rows
        if config.get('remove_duplicates', True):
            initial_rows = len(df_cleaned)
            df_cleaned = df_cleaned.drop_duplicates()
            removed_rows = initial_rows - len(df_cleaned)
            if removed_rows > 0:
                logger.info(f"Removed {removed_rows} duplicate rows")
                
        # Handle missing values
        if config.get('handle_missing', True):
            df_cleaned = self._handle_missing_values(df_cleaned, config.get('missing_strategy', 'auto'))
            
        # Fix data types
        if config.get('fix_dtypes', True):
            df_cleaned = self._fix_data_types(df_cleaned)
            
        # Clean text columns
        if config.get('clean_text', True):
            df_cleaned = self._clean_text_columns(df_cleaned)
            
        # Handle outliers
        if config.get('handle_outliers', False):
            df_cleaned = self._handle_outliers(df_cleaned, config.get('outlier_method', 'iqr'))
            
        return df_cleaned
        
    def _get_default_cleaning_config(self) -> Dict[str, Any]:
        """Get default cleaning configuration."""
        return {
            'remove_duplicates': True,
            'handle_missing': True,
            'missing_strategy': 'auto',
            'fix_dtypes': True,
            'clean_text': True,
            'handle_outliers': False,
            'outlier_method': 'iqr'
        }
        
    def _handle_missing_values(self, df: pd.DataFrame, strategy: str = 'auto') -> pd.DataFrame:
        """Handle missing values based on strategy."""
        df_result = df.copy()
        
        for column in df.columns:
            missing_count = df[column].isnull().sum()
            if missing_count == 0:
                continue
                
            missing_pct = (missing_count / len(df)) * 100
            logger.info(f"Column '{column}': {missing_count} missing values ({missing_pct:.1f}%)")
            
            # If more than 50% missing, consider dropping the column
            if missing_pct > 50:
                logger.warning(f"Column '{column}' has >50% missing values. Consider manual review.")
                
            if strategy == 'auto':
                if df[column].dtype in ['object', 'category']:
                    # Fill categorical with mode or 'Unknown'
                    mode_val = df[column].mode()
                    fill_value = mode_val[0] if not mode_val.empty else 'Unknown'
                    df_result[column] = df_result[column].fillna(fill_value)
                elif df[column].dtype in ['int64', 'float64']:
                    # Fill numerical with median
                    df_result[column] = df_result[column].fillna(df[column].median())
                elif df[column].dtype == 'datetime64[ns]':
                    # Fill datetime with forward fill
                    df_result[column] = df_result[column].fillna(method='ffill')
            elif strategy == 'drop':
                df_result = df_result.dropna(subset=[column])
            elif strategy == 'median':
                if pd.api.types.is_numeric_dtype(df[column]):
                    df_result[column] = df_result[column].fillna(df[column].median())
            elif strategy == 'mean':
                if pd.api.types.is_numeric_dtype(df[column]):
                    df_result[column] = df_result[column].fillna(df[column].mean())
            elif strategy == 'mode':
                mode_val = df[column].mode()
                if not mode_val.empty:
                    df_result[column] = df_result[column].fillna(mode_val[0])
                    
        return df_result
        
    def _fix_data_types(self, df: pd.DataFrame) -> pd.DataFrame:
        """Automatically fix data types."""
        df_result = df.copy()
        
        for column in df.columns:
            # Try to convert strings that look like numbers
            if df[column].dtype == 'object':
                # Check if it's numeric
                try:
                    # Remove common formatting characters
                    cleaned_series = df[column].astype(str).str.replace(',', '').str.replace('$', '').str.replace('%', '')
                    numeric_series = pd.to_numeric(cleaned_series, errors='coerce')
                    
                    # If most values can be converted to numeric, do it
                    non_null_count = numeric_series.notna().sum()
                    if non_null_count > len(df) * 0.8:  # 80% threshold
                        df_result[column] = numeric_series
                        logger.info(f"Converted column '{column}' to numeric")
                        continue
                except:
                    pass
                    
                # Check if it's datetime
                try:
                    datetime_series = pd.to_datetime(df[column], errors='coerce', infer_datetime_format=True)
                    non_null_count = datetime_series.notna().sum()
                    if non_null_count > len(df) * 0.8:  # 80% threshold
                        df_result[column] = datetime_series
                        logger.info(f"Converted column '{column}' to datetime")
                        continue
                except:
                    pass
                    
        return df_result
        
    def _clean_text_columns(self, df: pd.DataFrame) -> pd.DataFrame:
        """Clean text columns."""
        df_result = df.copy()
        
        for column in df.columns:
            if df[column].dtype == 'object':
                # Strip whitespace
                df_result[column] = df_result[column].astype(str).str.strip()
                
                # Replace empty strings with NaN
                df_result[column] = df_result[column].replace('', np.nan)
                df_result[column] = df_result[column].replace('nan', np.nan)
                
        return df_result
        
    def _handle_outliers(self, df: pd.DataFrame, method: str = 'iqr') -> pd.DataFrame:
        """Handle outliers in numeric columns."""
        df_result = df.copy()
        numeric_columns = df.select_dtypes(include=[np.number]).columns
        
        for column in numeric_columns:
            if method == 'iqr':
                Q1 = df[column].quantile(0.25)
                Q3 = df[column].quantile(0.75)
                IQR = Q3 - Q1
                lower_bound = Q1 - 1.5 * IQR
                upper_bound = Q3 + 1.5 * IQR
                
                outliers = ((df[column] < lower_bound) | (df[column] > upper_bound))
                outlier_count = outliers.sum()
                
                if outlier_count > 0:
                    logger.info(f"Found {outlier_count} outliers in column '{column}'")
                    # Cap outliers at bounds
                    df_result[column] = df_result[column].clip(lower_bound, upper_bound)
                    
        return df_result
        
    def encode_categorical_data(self, df: pd.DataFrame, columns: Optional[List[str]] = None) -> pd.DataFrame:
        """Encode categorical variables."""
        df_result = df.copy()
        
        if columns is None:
            # Auto-detect categorical columns
            columns = df.select_dtypes(include=['object', 'category']).columns.tolist()
            
        for column in columns:
            if column not in df.columns:
                continue
                
            unique_values = df[column].nunique()
            
            # Use different encoding strategies based on cardinality
            if unique_values <= 10:  # Low cardinality - use one-hot encoding
                dummies = pd.get_dummies(df[column], prefix=column, dummy_na=True)
                df_result = pd.concat([df_result.drop(column, axis=1), dummies], axis=1)
                logger.info(f"One-hot encoded column '{column}' ({unique_values} categories)")
            else:  # High cardinality - use label encoding
                encoder = LabelEncoder()
                df_result[f"{column}_encoded"] = encoder.fit_transform(df[column].astype(str))
                self.encoders[column] = encoder
                logger.info(f"Label encoded column '{column}' ({unique_values} categories)")
                
        return df_result
        
    def scale_numerical_data(self, df: pd.DataFrame, method: str = 'standard', 
                           columns: Optional[List[str]] = None) -> pd.DataFrame:
        """Scale numerical data."""
        df_result = df.copy()
        
        if columns is None:
            columns = df.select_dtypes(include=[np.number]).columns.tolist()
            
        if method == 'standard':
            scaler = StandardScaler()
        elif method == 'minmax':
            scaler = MinMaxScaler()
        else:
            raise ValueError(f"Unsupported scaling method: {method}")
            
        for column in columns:
            if column in df.columns:
                df_result[f"{column}_scaled"] = scaler.fit_transform(df[[column]])
                self.scalers[column] = scaler
                logger.info(f"Scaled column '{column}' using {method} scaling")
                
        return df_result
        
    def validate_data_quality(self, df: pd.DataFrame) -> Dict[str, Any]:
        """Comprehensive data quality assessment."""
        quality_report = {
            'total_rows': len(df),
            'total_columns': len(df.columns),
            'data_types': df.dtypes.to_dict(),
            'missing_data': {},
            'duplicates': 0,
            'quality_score': 0,
            'issues': [],
            'recommendations': []
        }
        
        # Check missing data
        missing_data = df.isnull().sum()
        quality_report['missing_data'] = {
            col: {'count': int(missing_data[col]), 'percentage': round((missing_data[col] / len(df)) * 100, 2)}
            for col in df.columns if missing_data[col] > 0
        }
        
        # Check duplicates
        quality_report['duplicates'] = df.duplicated().sum()
        
        # Check for potential issues
        for column in df.columns:
            # Very high missing data
            missing_pct = (df[column].isnull().sum() / len(df)) * 100
            if missing_pct > 50:
                quality_report['issues'].append(f"Column '{column}': {missing_pct:.1f}% missing data")
                quality_report['recommendations'].append(f"Consider dropping column '{column}' or investigating missing data pattern")
                
            # Single value columns
            if df[column].nunique() == 1:
                quality_report['issues'].append(f"Column '{column}': contains only one unique value")
                quality_report['recommendations'].append(f"Consider dropping column '{column}' as it provides no information")
                
        # Calculate quality score (0-100)
        base_score = 100
        
        # Penalize for missing data
        overall_missing_pct = (df.isnull().sum().sum() / (len(df) * len(df.columns))) * 100
        base_score -= min(overall_missing_pct * 2, 50)  # Max 50 point penalty
        
        # Penalize for duplicates
        duplicate_pct = (quality_report['duplicates'] / len(df)) * 100
        base_score -= min(duplicate_pct * 3, 30)  # Max 30 point penalty
        
        # Penalize for issues
        base_score -= len(quality_report['issues']) * 5  # 5 points per issue
        
        quality_report['quality_score'] = max(0, round(base_score, 1))
        
        return quality_report
        
    def get_preprocessing_summary(self, original_df: pd.DataFrame, processed_df: pd.DataFrame) -> Dict[str, Any]:
        """Generate preprocessing summary."""
        return {
            'original_shape': original_df.shape,
            'processed_shape': processed_df.shape,
            'rows_removed': original_df.shape[0] - processed_df.shape[0],
            'columns_added': processed_df.shape[1] - original_df.shape[1],
            'missing_values_before': original_df.isnull().sum().sum(),
            'missing_values_after': processed_df.isnull().sum().sum(),
            'data_types_changed': self._get_dtype_changes(original_df, processed_df)
        }
        
    def _get_dtype_changes(self, original_df: pd.DataFrame, processed_df: pd.DataFrame) -> Dict[str, Dict[str, str]]:
        """Get data type changes."""
        changes = {}
        common_columns = set(original_df.columns) & set(processed_df.columns)
        
        for column in common_columns:
            original_dtype = str(original_df[column].dtype)
            processed_dtype = str(processed_df[column].dtype)
            if original_dtype != processed_dtype:
                changes[column] = {
                    'from': original_dtype,
                    'to': processed_dtype
                }
                
        return changes


# Factory function
def create_preprocessor() -> DataPreprocessor:
    """Create and return a DataPreprocessor instance."""
    return DataPreprocessor()


# Convenience functions
def quick_clean(df: pd.DataFrame) -> pd.DataFrame:
    """Quick data cleaning with default settings."""
    preprocessor = DataPreprocessor()
    return preprocessor.clean_data(df)


def assess_data_quality(df: pd.DataFrame) -> Dict[str, Any]:
    """Quick data quality assessment."""
    preprocessor = DataPreprocessor()
    return preprocessor.validate_data_quality(df)
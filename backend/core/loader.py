"""
Data Loader Module for DataScout
Handles loading of various data formats including CSV, Excel, JSON, and basic SQL connectivity.
"""

import pandas as pd
import numpy as np
import json
import sqlite3
from pathlib import Path
from typing import Union, Dict, Any, Optional, List
from io import StringIO, BytesIO
import logging

logger = logging.getLogger(__name__)

class DataLoader:
    """
    Data loader class that supports multiple file formats and data sources.
    
    Supported formats:
    - CSV files
    - Excel files (.xlsx, .xls)
    - JSON files
    - SQLite databases
    """
    
    def __init__(self):
        self.supported_formats = {'.csv', '.xlsx', '.xls', '.json', '.sqlite', '.db'}
        
    def load_data(self, file_path: Union[str, Path], **kwargs) -> pd.DataFrame:
        """
        Load data from various file formats.
        
        Args:
            file_path: Path to the data file
            **kwargs: Additional arguments passed to specific loader functions
            
        Returns:
            pd.DataFrame: Loaded data as pandas DataFrame
            
        Raises:
            ValueError: If file format is not supported
            FileNotFoundError: If file doesn't exist
            Exception: For data loading errors
        """
        file_path = Path(file_path)
        
        if not file_path.exists():
            raise FileNotFoundError(f"File not found: {file_path}")
            
        file_ext = file_path.suffix.lower()
        
        if file_ext not in self.supported_formats:
            raise ValueError(f"Unsupported file format: {file_ext}")
            
        try:
            if file_ext == '.csv':
                return self._load_csv(file_path, **kwargs)
            elif file_ext in ['.xlsx', '.xls']:
                return self._load_excel(file_path, **kwargs)
            elif file_ext == '.json':
                return self._load_json(file_path, **kwargs)
            elif file_ext in ['.sqlite', '.db']:
                return self._load_sqlite(file_path, **kwargs)
        except Exception as e:
            logger.error(f"Error loading data from {file_path}: {str(e)}")
            raise
            
    def _load_csv(self, file_path: Path, **kwargs) -> pd.DataFrame:
        """Load CSV files with automatic encoding detection."""
        default_params = {
            'encoding': 'utf-8',
            'sep': None,  # Auto-detect separator
            'engine': 'python'
        }
        default_params.update(kwargs)
        
        try:
            df = pd.read_csv(file_path, **default_params)
            logger.info(f"Loaded CSV: {df.shape[0]} rows, {df.shape[1]} columns")
            return df
        except UnicodeDecodeError:
            # Try with different encodings
            for encoding in ['latin-1', 'cp1252', 'iso-8859-1']:
                try:
                    default_params['encoding'] = encoding
                    df = pd.read_csv(file_path, **default_params)
                    logger.info(f"Loaded CSV with {encoding} encoding: {df.shape[0]} rows, {df.shape[1]} columns")
                    return df
                except UnicodeDecodeError:
                    continue
            raise ValueError("Could not determine file encoding")
            
    def _load_excel(self, file_path: Path, **kwargs) -> pd.DataFrame:
        """Load Excel files."""
        default_params = {
            'sheet_name': 0,  # Load first sheet by default
            'engine': 'openpyxl'
        }
        default_params.update(kwargs)
        
        df = pd.read_excel(file_path, **default_params)
        logger.info(f"Loaded Excel: {df.shape[0]} rows, {df.shape[1]} columns")
        return df
        
    def _load_json(self, file_path: Path, **kwargs) -> pd.DataFrame:
        """Load JSON files and convert to DataFrame."""
        default_params = {
            'orient': 'records'  # Assume records format by default
        }
        default_params.update(kwargs)
        
        with open(file_path, 'r', encoding='utf-8') as f:
            data = json.load(f)
            
        # Handle different JSON structures
        if isinstance(data, list):
            df = pd.DataFrame(data)
        elif isinstance(data, dict):
            if 'data' in data:
                df = pd.DataFrame(data['data'])
            else:
                df = pd.DataFrame([data])
        else:
            raise ValueError("Unsupported JSON structure")
            
        logger.info(f"Loaded JSON: {df.shape[0]} rows, {df.shape[1]} columns")
        return df
        
    def _load_sqlite(self, file_path: Path, table_name: str = None, query: str = None, **kwargs) -> pd.DataFrame:
        """Load data from SQLite database."""
        conn = sqlite3.connect(file_path)
        
        try:
            if query:
                df = pd.read_sql_query(query, conn)
            elif table_name:
                df = pd.read_sql_table(table_name, conn)
            else:
                # Get first table if no table specified
                tables = pd.read_sql_query("SELECT name FROM sqlite_master WHERE type='table'", conn)
                if tables.empty:
                    raise ValueError("No tables found in database")
                first_table = tables.iloc[0]['name']
                df = pd.read_sql_table(first_table, conn)
                
            logger.info(f"Loaded SQLite data: {df.shape[0]} rows, {df.shape[1]} columns")
            return df
        finally:
            conn.close()
            
    def load_from_string(self, data_string: str, format_type: str, **kwargs) -> pd.DataFrame:
        """
        Load data from string content.
        
        Args:
            data_string: String content of the data
            format_type: Format type ('csv', 'json')
            **kwargs: Additional parameters
            
        Returns:
            pd.DataFrame: Loaded data
        """
        if format_type.lower() == 'csv':
            return pd.read_csv(StringIO(data_string), **kwargs)
        elif format_type.lower() == 'json':
            data = json.loads(data_string)
            return pd.DataFrame(data if isinstance(data, list) else [data])
        else:
            raise ValueError(f"Unsupported string format: {format_type}")
            
    def validate_data(self, df: pd.DataFrame) -> Dict[str, Any]:
        """
        Validate loaded data and return basic information.
        
        Args:
            df: DataFrame to validate
            
        Returns:
            Dict containing validation results
        """
        validation_results = {
            'is_valid': True,
            'shape': df.shape,
            'columns': list(df.columns),
            'dtypes': {str(k): str(v) for k, v in df.dtypes.to_dict().items()},
            'missing_values': {str(k): int(v) for k, v in df.isnull().sum().to_dict().items()},
            'memory_usage': int(df.memory_usage(deep=True).sum()),
            'issues': []
        }
        
        # Check for common issues
        if df.empty:
            validation_results['is_valid'] = False
            validation_results['issues'].append('DataFrame is empty')
            
        if df.columns.duplicated().any():
            validation_results['issues'].append('Duplicate column names found')
            
        if df.isnull().all().any():
            validation_results['issues'].append('Columns with all null values found')
            
        return validation_results
        
    def get_sample_data(self, df: pd.DataFrame, n_rows: int = 5) -> Dict[str, Any]:
        """
        Get sample data for preview.
        
        Args:
            df: DataFrame to sample
            n_rows: Number of rows to include in sample
            
        Returns:
            Dict containing sample data and info
        """
        # Convert sample data to ensure JSON serialization
        sample_df = df.head(n_rows)
        sample_data = []
        
        for record in sample_df.to_dict('records'):
            clean_record = {}
            for key, value in record.items():
                # Convert NumPy types to native Python types
                if isinstance(value, (np.integer, np.int64, np.int32)):
                    clean_record[str(key)] = int(value)
                elif isinstance(value, (np.floating, np.float64, np.float32)):
                    clean_record[str(key)] = float(value)
                elif pd.isna(value):
                    clean_record[str(key)] = None
                else:
                    clean_record[str(key)] = str(value)
            sample_data.append(clean_record)
        
        return {
            'sample_data': sample_data,
            'total_rows': int(len(df)),
            'total_columns': int(len(df.columns)),
            'column_names': list(df.columns)
        }


# Factory function for easy usage
def create_loader() -> DataLoader:
    """Create and return a DataLoader instance."""
    return DataLoader()


# Convenience functions
def load_csv(file_path: Union[str, Path], **kwargs) -> pd.DataFrame:
    """Convenience function to load CSV files."""
    loader = DataLoader()
    return loader.load_data(file_path, **kwargs)


def load_excel(file_path: Union[str, Path], **kwargs) -> pd.DataFrame:
    """Convenience function to load Excel files."""
    loader = DataLoader()
    return loader.load_data(file_path, **kwargs)


def load_json(file_path: Union[str, Path], **kwargs) -> pd.DataFrame:
    """Convenience function to load JSON files."""
    loader = DataLoader()
    return loader.load_data(file_path, **kwargs)
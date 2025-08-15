"""
Utility functions for API operations
"""

import numpy as np
import pandas as pd
from typing import Any, Dict, List, Union


def convert_numpy_types(obj: Any) -> Any:
    """
    Convert NumPy types to native Python types for JSON serialization.
    
    Args:
        obj: Object that may contain NumPy types
        
    Returns:
        Object with NumPy types converted to native Python types
    """
    if isinstance(obj, dict):
        return {str(k): convert_numpy_types(v) for k, v in obj.items()}
    elif isinstance(obj, list):
        return [convert_numpy_types(item) for item in obj]
    elif isinstance(obj, tuple):
        return tuple(convert_numpy_types(item) for item in obj)
    elif isinstance(obj, (np.integer, np.int64, np.int32, np.int16, np.int8)):
        return int(obj)
    elif isinstance(obj, (np.floating, np.float64, np.float32)):
        return float(obj)
    elif isinstance(obj, np.bool_):
        return bool(obj)
    elif isinstance(obj, np.ndarray):
        return obj.tolist()
    elif pd.isna(obj):
        return None
    else:
        return obj


def safe_json_response(data: Dict[str, Any]) -> Dict[str, Any]:
    """
    Ensure response data is JSON serializable by converting NumPy types.
    
    Args:
        data: Response data dictionary
        
    Returns:
        JSON-serializable response data
    """
    return convert_numpy_types(data)


def convert_dtypes_dict(dtypes_dict: Dict[str, Any]) -> Dict[str, str]:
    """
    Convert pandas dtypes dictionary to string representation.
    
    Args:
        dtypes_dict: Dictionary with pandas dtypes
        
    Returns:
        Dictionary with string representations of dtypes
    """
    return {str(k): str(v) for k, v in dtypes_dict.items()}


def convert_missing_values_dict(missing_dict: Dict[str, Any]) -> Dict[str, int]:
    """
    Convert missing values dictionary to ensure integer types.
    
    Args:
        missing_dict: Dictionary with missing value counts
        
    Returns:
        Dictionary with integer missing value counts
    """
    return {str(k): int(v) for k, v in missing_dict.items()}


def prepare_sample_data(df: pd.DataFrame, n_rows: int = 5) -> List[Dict[str, Any]]:
    """
    Prepare sample data for JSON serialization.
    
    Args:
        df: DataFrame to sample from
        n_rows: Number of rows to sample
        
    Returns:
        List of JSON-serializable dictionaries
    """
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
    
    return sample_data
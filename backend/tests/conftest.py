"""
Test fixtures and utilities for DataScout testing.
"""

import pytest
import pandas as pd
import numpy as np
import tempfile
import json
from pathlib import Path


@pytest.fixture
def sample_numeric_data():
    """Fixture providing sample numeric dataset."""
    np.random.seed(42)
    return pd.DataFrame({
        'feature1': np.random.normal(100, 15, 200),
        'feature2': np.random.normal(50, 10, 200),
        'feature3': np.random.exponential(2, 200),
        'target': np.random.normal(75, 12, 200)
    })


@pytest.fixture
def sample_mixed_data():
    """Fixture providing mixed data types dataset."""
    np.random.seed(42)
    return pd.DataFrame({
        'numeric_col': np.random.normal(0, 1, 100),
        'categorical_col': np.random.choice(['A', 'B', 'C', 'D'], 100),
        'binary_col': np.random.choice([0, 1], 100),
        'text_col': [f'item_{i}' for i in range(100)],
        'date_col': pd.date_range('2023-01-01', periods=100, freq='D')
    })


@pytest.fixture
def sample_data_with_missing():
    """Fixture providing dataset with missing values."""
    np.random.seed(42)
    data = pd.DataFrame({
        'col1': np.random.normal(0, 1, 100),
        'col2': np.random.choice(['X', 'Y', 'Z'], 100),
        'col3': np.random.uniform(0, 100, 100)
    })
    
    # Introduce missing values
    missing_indices = np.random.choice(100, 20, replace=False)
    data.loc[missing_indices, 'col1'] = np.nan
    
    missing_indices = np.random.choice(100, 15, replace=False)
    data.loc[missing_indices, 'col2'] = np.nan
    
    return data


@pytest.fixture
def sample_csv_file():
    """Fixture providing temporary CSV file."""
    data = pd.DataFrame({
        'name': ['Alice', 'Bob', 'Charlie'],
        'age': [25, 30, 35],
        'city': ['NYC', 'LA', 'Chicago']
    })
    
    with tempfile.NamedTemporaryFile(mode='w', suffix='.csv', delete=False) as f:
        data.to_csv(f.name, index=False)
        yield f.name
        
    # Cleanup
    Path(f.name).unlink(missing_ok=True)


@pytest.fixture
def sample_json_file():
    """Fixture providing temporary JSON file."""
    data = [
        {'name': 'Alice', 'age': 25, 'city': 'NYC'},
        {'name': 'Bob', 'age': 30, 'city': 'LA'},
        {'name': 'Charlie', 'age': 35, 'city': 'Chicago'}
    ]
    
    with tempfile.NamedTemporaryFile(mode='w', suffix='.json', delete=False) as f:
        json.dump(data, f)
        yield f.name
        
    # Cleanup
    Path(f.name).unlink(missing_ok=True)


@pytest.fixture
def correlated_data():
    """Fixture providing data with known correlations."""
    np.random.seed(42)
    n = 1000
    
    # Generate correlated variables
    x1 = np.random.normal(0, 1, n)
    x2 = 0.8 * x1 + 0.6 * np.random.normal(0, 1, n)  # Strong positive correlation
    x3 = -0.7 * x1 + 0.7 * np.random.normal(0, 1, n)  # Strong negative correlation
    x4 = np.random.normal(0, 1, n)  # Independent
    
    return pd.DataFrame({
        'var1': x1,
        'var2': x2,
        'var3': x3,
        'var4': x4,
        'target': 2 * x1 + x2 - x3 + np.random.normal(0, 0.5, n)
    })


@pytest.fixture
def time_series_data():
    """Fixture providing time series dataset."""
    dates = pd.date_range('2020-01-01', '2023-12-31', freq='D')
    np.random.seed(42)
    
    # Generate trend + seasonality + noise
    trend = np.linspace(100, 200, len(dates))
    seasonal = 10 * np.sin(2 * np.pi * np.arange(len(dates)) / 365.25)
    noise = np.random.normal(0, 5, len(dates))
    
    values = trend + seasonal + noise
    
    return pd.DataFrame({
        'date': dates,
        'value': values,
        'category': np.random.choice(['A', 'B', 'C'], len(dates))
    })


def assert_dataframe_equal(df1, df2, check_dtype=True):
    """Custom assertion for DataFrame equality."""
    pd.testing.assert_frame_equal(df1, df2, check_dtype=check_dtype)


def assert_dict_contains_keys(dict_obj, required_keys):
    """Assert that dictionary contains all required keys."""
    missing_keys = set(required_keys) - set(dict_obj.keys())
    assert not missing_keys, f"Missing keys: {missing_keys}"


def create_test_dataset(n_rows=100, n_numeric=3, n_categorical=2, missing_pct=0.1):
    """Create a test dataset with specified characteristics."""
    np.random.seed(42)
    data = {}
    
    # Add numeric columns
    for i in range(n_numeric):
        data[f'numeric_{i}'] = np.random.normal(i * 10, 5, n_rows)
        
    # Add categorical columns
    for i in range(n_categorical):
        categories = [f'cat_{j}' for j in range(3 + i)]
        data[f'categorical_{i}'] = np.random.choice(categories, n_rows)
        
    df = pd.DataFrame(data)
    
    # Introduce missing values
    if missing_pct > 0:
        n_missing = int(n_rows * len(df.columns) * missing_pct)
        missing_positions = np.random.choice(
            n_rows * len(df.columns), n_missing, replace=False
        )
        
        for pos in missing_positions:
            row_idx = pos // len(df.columns)
            col_idx = pos % len(df.columns)
            df.iloc[row_idx, col_idx] = np.nan
            
    return df
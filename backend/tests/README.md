# DataScout Testing Framework

This directory contains comprehensive unit tests for all DataScout core modules.

## Test Structure

- `test_core_modules.py` - Main test suite covering all core functionality
- `conftest.py` - Test fixtures and utilities
- `pytest.ini` - Pytest configuration

## Running Tests

### Run all tests
```bash
cd backend
python -m pytest tests/ -v
```

### Run specific test class
```bash
python -m pytest tests/test_core_modules.py::TestDataLoader -v
```

### Run with coverage
```bash
python -m pytest tests/ --cov=core --cov-report=html
```

## Test Coverage

The test suite covers:

### DataLoader
- ✅ CSV/JSON string loading
- ✅ Data validation
- ✅ Sample data extraction
- ✅ Error handling

### DataPreprocessor  
- ✅ Data cleaning pipeline
- ✅ Quality assessment
- ✅ Categorical encoding
- ✅ Numerical scaling

### DataSummarizer
- ✅ Comprehensive summaries
- ✅ Descriptive statistics
- ✅ Report generation
- ✅ Basic info extraction

### DataVisualizer
- ✅ Multiple chart types
- ✅ Plot recommendations
- ✅ Error handling
- ✅ Interactive/static modes

### FeatureSelector
- ✅ Feature importance analysis
- ✅ Multicollinearity detection
- ✅ Feature selection methods
- ✅ Correlation analysis

### InsightEngine
- ✅ Comprehensive insights
- ✅ Executive summaries
- ✅ Anomaly detection
- ✅ Report generation

### Integration Tests
- ✅ Full analysis pipeline
- ✅ Cross-module interactions
- ✅ Error handling chains

## Test Fixtures

Available fixtures in `conftest.py`:
- `sample_numeric_data` - Pure numerical dataset
- `sample_mixed_data` - Mixed data types
- `sample_data_with_missing` - Dataset with missing values
- `correlated_data` - Dataset with known correlations
- `time_series_data` - Time series dataset
- `sample_csv_file` - Temporary CSV file
- `sample_json_file` - Temporary JSON file

## Test Categories

- **Unit Tests**: Test individual functions and methods
- **Integration Tests**: Test module interactions
- **Error Handling**: Test exception scenarios
- **Edge Cases**: Test boundary conditions

## Performance Tests

For performance testing, add the `@pytest.mark.slow` decorator:

```python
@pytest.mark.slow
def test_large_dataset_processing():
    # Test with large dataset
    pass
```

Run without slow tests: `pytest -m "not slow"`
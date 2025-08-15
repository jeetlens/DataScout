"""
Unit Tests for DataScout Core Modules
Test suite for loader, preprocessor, summarizer, visualizer, feature_selector, and insight_engine.
"""

import pytest
import pandas as pd
import numpy as np
import tempfile
import json
from pathlib import Path

# Import core modules
import sys
sys.path.append('../core')

from core.loader import DataLoader, create_loader
from core.preprocessor import DataPreprocessor, create_preprocessor
from core.summarizer import DataSummarizer, create_summarizer
from core.visualizer import DataVisualizer, create_visualizer
from core.feature_selector import FeatureSelector, create_feature_selector
from core.insight_engine import InsightEngine, create_insight_engine


class TestDataLoader:
    """Test cases for DataLoader class."""
    
    def setup_method(self):
        """Set up test fixtures."""
        self.loader = DataLoader()
        self.sample_data = pd.DataFrame({
            'id': [1, 2, 3, 4, 5],
            'name': ['Alice', 'Bob', 'Charlie', 'Diana', 'Eve'],
            'age': [25, 30, 35, 28, 32],
            'salary': [50000, 60000, 70000, 55000, 65000]
        })
        
    def test_create_loader(self):
        """Test factory function."""
        loader = create_loader()
        assert isinstance(loader, DataLoader)
        
    def test_load_csv_from_string(self):
        """Test loading CSV from string."""
        csv_string = "name,age,city\nAlice,25,NYC\nBob,30,LA"
        result = self.loader.load_from_string(csv_string, 'csv')
        
        assert len(result) == 2
        assert list(result.columns) == ['name', 'age', 'city']
        assert result.iloc[0]['name'] == 'Alice'
        
    def test_load_json_from_string(self):
        """Test loading JSON from string."""
        json_data = [
            {"name": "Alice", "age": 25},
            {"name": "Bob", "age": 30}
        ]
        json_string = json.dumps(json_data)
        result = self.loader.load_from_string(json_string, 'json')
        
        assert len(result) == 2
        assert list(result.columns) == ['name', 'age']
        
    def test_validate_data(self):
        """Test data validation."""
        validation = self.loader.validate_data(self.sample_data)
        
        assert validation['is_valid'] == True
        assert validation['shape'] == (5, 4)
        assert len(validation['columns']) == 4
        assert 'missing_values' in validation
        
    def test_get_sample_data(self):
        """Test sample data extraction."""
        sample = self.loader.get_sample_data(self.sample_data, n_rows=3)
        
        assert len(sample['sample_data']) == 3
        assert sample['total_rows'] == 5
        assert sample['total_columns'] == 4
        
    def test_unsupported_format_error(self):
        """Test error handling for unsupported format."""
        with pytest.raises(ValueError, match="Unsupported string format"):
            self.loader.load_from_string("test", "xml")


class TestDataPreprocessor:
    """Test cases for DataPreprocessor class."""
    
    def setup_method(self):
        """Set up test fixtures."""
        self.preprocessor = DataPreprocessor()
        self.sample_data = pd.DataFrame({
            'numeric_col': [1, 2, np.nan, 4, 5],
            'categorical_col': ['A', 'B', 'A', np.nan, 'C'],
            'text_col': ['  hello  ', 'world', '', 'test', 'data'],
            'duplicate_row': [1, 2, 3, 2, 1]
        })
        
    def test_create_preprocessor(self):
        """Test factory function."""
        preprocessor = create_preprocessor()
        assert isinstance(preprocessor, DataPreprocessor)
        
    def test_clean_data_basic(self):
        """Test basic data cleaning."""
        cleaned = self.preprocessor.clean_data(self.sample_data)
        
        # Should have some missing values handled
        assert cleaned.isnull().sum().sum() <= self.sample_data.isnull().sum().sum()
        
    def test_validate_data_quality(self):
        """Test data quality validation."""
        quality = self.preprocessor.validate_data_quality(self.sample_data)
        
        assert 'total_rows' in quality
        assert 'total_columns' in quality
        assert 'missing_data' in quality
        assert 'quality_score' in quality
        assert 0 <= quality['quality_score'] <= 100
        
    def test_encode_categorical_data(self):
        """Test categorical encoding."""
        encoded = self.preprocessor.encode_categorical_data(self.sample_data)
        
        # Should have more columns due to one-hot encoding
        assert len(encoded.columns) >= len(self.sample_data.columns)
        
    def test_scale_numerical_data(self):
        """Test numerical scaling."""
        scaled = self.preprocessor.scale_numerical_data(self.sample_data)
        
        # Should have additional scaled columns
        scaled_cols = [col for col in scaled.columns if '_scaled' in col]
        assert len(scaled_cols) > 0


class TestDataSummarizer:
    """Test cases for DataSummarizer class."""
    
    def setup_method(self):
        """Set up test fixtures."""
        self.summarizer = DataSummarizer()
        self.sample_data = pd.DataFrame({
            'numeric1': [1, 2, 3, 4, 5],
            'numeric2': [10, 20, 30, 40, 50],
            'categorical': ['A', 'B', 'A', 'C', 'B'],
            'text': ['hello', 'world', 'test', 'data', 'analysis']
        })
        
    def test_create_summarizer(self):
        """Test factory function."""
        summarizer = create_summarizer()
        assert isinstance(summarizer, DataSummarizer)
        
    def test_comprehensive_summary(self):
        """Test comprehensive summary generation."""
        summary = self.summarizer.generate_comprehensive_summary(self.sample_data)
        
        required_keys = [
            'basic_info', 'descriptive_stats', 'missing_data_analysis',
            'correlation_analysis', 'categorical_analysis'
        ]
        
        for key in required_keys:
            assert key in summary
            
    def test_basic_info(self):
        """Test basic info extraction."""
        basic_info = self.summarizer._get_basic_info(self.sample_data)
        
        assert basic_info['total_rows'] == 5
        assert basic_info['total_columns'] == 4
        assert len(basic_info['numeric_columns']) == 2
        assert len(basic_info['categorical_columns']) == 2
        
    def test_descriptive_statistics(self):
        """Test descriptive statistics calculation."""
        desc_stats = self.summarizer._get_descriptive_statistics(self.sample_data)
        
        assert 'numeric1' in desc_stats
        assert 'numeric2' in desc_stats
        assert 'mean' in desc_stats['numeric1']
        assert 'std' in desc_stats['numeric1']
        
    def test_generate_summary_report(self):
        """Test summary report generation."""
        report = self.summarizer.generate_summary_report(self.sample_data, "Test Report")
        
        assert report['title'] == "Test Report"
        assert 'generated_at' in report
        assert 'dataset_overview' in report
        assert 'key_statistics' in report


class TestDataVisualizer:
    """Test cases for DataVisualizer class."""
    
    def setup_method(self):
        """Set up test fixtures."""
        self.visualizer = DataVisualizer()
        self.sample_data = pd.DataFrame({
            'x': np.random.normal(0, 1, 100),
            'y': np.random.normal(0, 1, 100),
            'category': np.random.choice(['A', 'B', 'C'], 100),
            'date': pd.date_range('2023-01-01', periods=100, freq='D')
        })
        
    def test_create_visualizer(self):
        """Test factory function."""
        visualizer = create_visualizer()
        assert isinstance(visualizer, DataVisualizer)
        
    def test_create_histogram(self):
        """Test histogram creation."""
        result = self.visualizer.create_histogram(self.sample_data, 'x')
        
        assert result['type'] == 'histogram'
        assert result['column'] == 'x'
        assert 'plot_data' in result
        assert 'statistics' in result
        
    def test_create_scatter_plot(self):
        """Test scatter plot creation."""
        result = self.visualizer.create_scatter_plot(self.sample_data, 'x', 'y')
        
        assert result['type'] == 'scatter_plot'
        assert result['x_column'] == 'x'
        assert result['y_column'] == 'y'
        assert 'correlation' in result
        
    def test_create_correlation_heatmap(self):
        """Test correlation heatmap creation."""
        result = self.visualizer.create_correlation_heatmap(self.sample_data)
        
        assert result['type'] == 'correlation_heatmap'
        assert 'correlation_matrix' in result
        
    def test_get_recommended_plots(self):
        """Test plot recommendations."""
        recommendations = self.visualizer.get_recommended_plots(self.sample_data)
        
        assert isinstance(recommendations, list)
        assert len(recommendations) > 0
        
        for rec in recommendations:
            assert 'type' in rec
            assert 'columns' in rec
            assert 'reason' in rec
            
    def test_invalid_column_error(self):
        """Test error handling for invalid column."""
        with pytest.raises(ValueError, match="Column 'invalid' not found"):
            self.visualizer.create_histogram(self.sample_data, 'invalid')


class TestFeatureSelector:
    """Test cases for FeatureSelector class."""
    
    def setup_method(self):
        """Set up test fixtures."""
        self.feature_selector = FeatureSelector()
        self.sample_data = pd.DataFrame({
            'feature1': np.random.normal(0, 1, 100),
            'feature2': np.random.normal(0, 1, 100),
            'feature3': np.random.choice(['A', 'B', 'C'], 100),
            'target': np.random.choice([0, 1], 100)
        })
        
    def test_create_feature_selector(self):
        """Test factory function."""
        selector = create_feature_selector()
        assert isinstance(selector, FeatureSelector)
        
    def test_analyze_feature_importance(self):
        """Test feature importance analysis."""
        analysis = self.feature_selector.analyze_feature_importance(
            self.sample_data, 'target'
        )
        
        required_keys = [
            'correlation_analysis', 'statistical_tests', 'multicollinearity_analysis',
            'feature_rankings', 'recommendations', 'supervised_selection'
        ]
        
        for key in required_keys:
            assert key in analysis
            
    def test_detect_multicollinearity(self):
        """Test multicollinearity detection."""
        # Create correlated features
        correlated_data = self.sample_data.copy()
        correlated_data['feature4'] = correlated_data['feature1'] * 2 + np.random.normal(0, 0.1, 100)
        
        analysis = self.feature_selector._detect_multicollinearity(correlated_data)
        
        assert 'high_correlations' in analysis
        assert 'vif_scores' in analysis
        assert 'multicollinearity_detected' in analysis
        
    def test_select_best_features(self):
        """Test feature selection."""
        result = self.feature_selector.select_best_features(
            self.sample_data, 'target', k=2
        )
        
        assert 'selected_features' in result
        assert len(result['selected_features']) <= 2
        assert 'method' in result


class TestInsightEngine:
    """Test cases for InsightEngine class."""
    
    def setup_method(self):
        """Set up test fixtures."""
        self.insight_engine = InsightEngine()
        self.sample_data = pd.DataFrame({
            'sales': np.random.normal(1000, 200, 100),
            'marketing_spend': np.random.normal(500, 100, 100),
            'region': np.random.choice(['North', 'South', 'East', 'West'], 100),
            'date': pd.date_range('2023-01-01', periods=100, freq='D')
        })
        
    def test_create_insight_engine(self):
        """Test factory function."""
        engine = create_insight_engine()
        assert isinstance(engine, InsightEngine)
        
    def test_generate_comprehensive_insights(self):
        """Test comprehensive insights generation."""
        insights = self.insight_engine.generate_comprehensive_insights(
            self.sample_data, 'sales'
        )
        
        required_keys = [
            'executive_summary', 'data_quality_insights', 'statistical_insights',
            'relationship_insights', 'anomaly_detection', 'business_recommendations'
        ]
        
        for key in required_keys:
            assert key in insights
            
    def test_executive_summary(self):
        """Test executive summary generation."""
        summary = self.insight_engine._generate_executive_summary(
            self.sample_data, 'sales'
        )
        
        assert 'dataset_overview' in summary
        assert 'key_findings' in summary
        assert 'data_health' in summary
        assert summary['dataset_overview']['total_records'] == 100
        
    def test_detect_anomalies(self):
        """Test anomaly detection."""
        anomalies = self.insight_engine._detect_anomalies(self.sample_data)
        
        assert 'statistical_outliers' in anomalies
        assert 'pattern_anomalies' in anomalies
        assert 'data_inconsistencies' in anomalies
        
    def test_generate_insight_report(self):
        """Test insight report generation."""
        report = self.insight_engine.generate_insight_report(
            self.sample_data, 'sales', title="Test Report"
        )
        
        assert report['title'] == "Test Report"
        assert 'generated_at' in report
        assert 'executive_summary' in report
        assert 'key_insights' in report
        assert 'recommendations' in report


# Integration Tests
class TestIntegration:
    """Integration tests for module interactions."""
    
    def setup_method(self):
        """Set up test fixtures."""
        self.sample_data = pd.DataFrame({
            'feature1': np.random.normal(0, 1, 100),
            'feature2': np.random.normal(0, 1, 100),
            'category': np.random.choice(['A', 'B', 'C'], 100),
            'target': np.random.choice([0, 1], 100)
        })
        
    def test_full_pipeline(self):
        """Test complete analysis pipeline."""
        # Load data (simulated)
        loader = create_loader()
        validation = loader.validate_data(self.sample_data)
        assert validation['is_valid']
        
        # Preprocess data
        preprocessor = create_preprocessor()
        cleaned_data = preprocessor.clean_data(self.sample_data)
        
        # Generate summary
        summarizer = create_summarizer()
        summary = summarizer.generate_comprehensive_summary(cleaned_data)
        assert 'basic_info' in summary
        
        # Create visualizations
        visualizer = create_visualizer()
        viz_recommendations = visualizer.get_recommended_plots(cleaned_data)
        assert len(viz_recommendations) > 0
        
        # Feature selection
        feature_selector = create_feature_selector()
        feature_analysis = feature_selector.analyze_feature_importance(
            cleaned_data, 'target'
        )
        assert 'correlation_analysis' in feature_analysis
        
        # Generate insights
        insight_engine = create_insight_engine()
        insights = insight_engine.generate_comprehensive_insights(
            cleaned_data, 'target'
        )
        assert 'executive_summary' in insights
        
    def test_error_handling_integration(self):
        """Test error handling across modules."""
        empty_data = pd.DataFrame()
        
        # Test various modules with empty data
        loader = create_loader()
        validation = loader.validate_data(empty_data)
        assert not validation['is_valid']
        
        preprocessor = create_preprocessor()
        quality = preprocessor.validate_data_quality(empty_data)
        assert quality['total_rows'] == 0


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
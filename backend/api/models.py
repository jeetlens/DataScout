"""
API Models for DataScout
Pydantic models for request/response validation.
"""

from pydantic import BaseModel, Field
from typing import Dict, List, Any, Optional, Union
from datetime import datetime


class DataUploadResponse(BaseModel):
    """Response model for data upload."""
    data_id: str
    status: str
    message: str
    data_info: Dict[str, Any]
    sample_data: List[Dict[str, Any]]
    validation: Dict[str, Any]


class DataInfo(BaseModel):
    """Basic data information model."""
    rows: int
    columns: int
    column_names: List[str]
    data_types: Dict[str, str]
    memory_usage_mb: float


class PreprocessingConfig(BaseModel):
    """Configuration for data preprocessing."""
    remove_duplicates: bool = True
    handle_missing: bool = True
    missing_strategy: str = "auto"
    fix_dtypes: bool = True
    clean_text: bool = True
    handle_outliers: bool = False
    outlier_method: str = "iqr"


class PreprocessingResponse(BaseModel):
    """Response model for data preprocessing."""
    original_data_id: str
    processed_data_id: str
    status: str
    preprocessing_summary: Dict[str, Any]
    quality_assessment: Dict[str, Any]
    message: str


class VisualizationConfig(BaseModel):
    """Configuration for visualization creation."""
    column: Optional[str] = None
    x_column: Optional[str] = None
    y_column: Optional[str] = None
    color_column: Optional[str] = None
    columns: Optional[List[str]] = None
    bins: Optional[int] = 30
    top_n: Optional[int] = 10


class VisualizationResponse(BaseModel):
    """Response model for visualization creation."""
    data_id: str
    plot_type: str
    config: Dict[str, Any]
    visualization: Dict[str, Any]


class FeatureAnalysisResponse(BaseModel):
    """Response model for feature analysis."""
    data_id: str
    target_column: Optional[str]
    feature_analysis: Dict[str, Any]
    best_features: Dict[str, Any]


class InsightResponse(BaseModel):
    """Response model for insight generation."""
    data_id: str
    target_column: Optional[str]
    business_context: Optional[str]
    comprehensive_insights: Dict[str, Any]
    insight_report: Dict[str, Any]


class CompleteAnalysisResponse(BaseModel):
    """Response model for complete analysis."""
    data_id: str
    target_column: Optional[str]
    business_context: Optional[str]
    analysis_results: Dict[str, Any]
    analysis_summary: Dict[str, Any]


class DatasetInfo(BaseModel):
    """Model for dataset information in list."""
    data_id: str
    rows: int
    columns: int
    memory_usage_mb: float


class DatasetListResponse(BaseModel):
    """Response model for dataset listing."""
    datasets: List[DatasetInfo]
    total_datasets: int


class HealthResponse(BaseModel):
    """Response model for health check."""
    status: str
    service: str
    version: str
    available_endpoints: List[str]


class ErrorResponse(BaseModel):
    """Error response model."""
    detail: str
    error_type: Optional[str] = None
    timestamp: Optional[datetime] = None
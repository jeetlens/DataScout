"""
API Routes for DataScout
Defines all API endpoints for data analysis operations.
"""

from fastapi import APIRouter, HTTPException, UploadFile, File, Depends, BackgroundTasks
from fastapi.responses import JSONResponse, StreamingResponse
from typing import Dict, List, Any, Optional
import pandas as pd
import numpy as np
import json
import logging
import zipfile
import io
import tempfile
from datetime import datetime
from io import StringIO, BytesIO

def convert_numpy_types(obj):
    """Convert NumPy types to native Python types for JSON serialization."""
    if isinstance(obj, dict):
        return {str(k): convert_numpy_types(v) for k, v in obj.items()}
    elif isinstance(obj, (list, tuple)):
        return [convert_numpy_types(item) for item in obj]
    elif isinstance(obj, np.integer):
        return int(obj)
    elif isinstance(obj, np.floating):
        return float(obj)
    elif isinstance(obj, np.bool_):
        return bool(obj)
    elif isinstance(obj, np.ndarray):
        return obj.tolist()
    elif isinstance(obj, (np.dtype, type(np.int64(0).dtype))):
        return str(obj)
    elif hasattr(obj, 'item'):  # NumPy scalars
        return obj.item()
    elif pd.isna(obj):
        return None
    elif hasattr(obj, 'to_dict'):  # Pandas objects
        return convert_numpy_types(obj.to_dict())
    elif hasattr(obj, 'tolist'):  # NumPy arrays
        return obj.tolist()
    else:
        try:
            # Try to convert to string if it's a NumPy type we missed
            if 'numpy' in str(type(obj)):
                return str(obj)
            return obj
        except Exception:
            return str(obj)

# Import core modules
from core.loader import create_loader
from core.preprocessor import create_preprocessor
from core.summarizer import create_summarizer
from core.visualizer import create_visualizer
from core.feature_selector import create_feature_selector
from core.insight_engine import create_insight_engine

# Import enhanced profiler for advanced analytics
from core.comprehensive_profiler import create_comprehensive_profiler, ProfileConfig

# Import AI modules
from ai import create_enhanced_ai_generator, generate_full_ai_analysis, generate_ai_executive_summary

# Import report modules
from reports import (
    create_html_generator, create_pdf_generator,
    generate_analysis_report, generate_executive_report, generate_pdf_report
)

logger = logging.getLogger(__name__)

# Create API router
api_router = APIRouter(prefix="/api/v1", tags=["DataScout API"])

# Helper functions for enhanced analysis
def _generate_enhanced_statistics(df: pd.DataFrame) -> Dict[str, Any]:
    """Generate enhanced statistical analysis for each column."""
    enhanced_stats = {}
    
    for column in df.columns:
        col_stats = {
            'name': column,
            'dtype': str(df[column].dtype),
            'non_null_count': int(df[column].notna().sum()),
            'null_count': int(df[column].isna().sum()),
            'null_percentage': float(df[column].isna().sum() / len(df) * 100)
        }
        
        if df[column].dtype in ['int64', 'float64']:
            # Numeric statistics
            col_stats.update({
                'mean': float(df[column].mean()) if df[column].notna().sum() > 0 else None,
                'median': float(df[column].median()) if df[column].notna().sum() > 0 else None,
                'std': float(df[column].std()) if df[column].notna().sum() > 0 else None,
                'min': float(df[column].min()) if df[column].notna().sum() > 0 else None,
                'max': float(df[column].max()) if df[column].notna().sum() > 0 else None,
                'q25': float(df[column].quantile(0.25)) if df[column].notna().sum() > 0 else None,
                'q75': float(df[column].quantile(0.75)) if df[column].notna().sum() > 0 else None,
                'skewness': float(df[column].skew()) if df[column].notna().sum() > 0 else None,
                'kurtosis': float(df[column].kurtosis()) if df[column].notna().sum() > 0 else None
            })
            
            # Detect outliers using IQR method
            if df[column].notna().sum() > 0:
                Q1 = df[column].quantile(0.25)
                Q3 = df[column].quantile(0.75)
                IQR = Q3 - Q1
                lower_bound = Q1 - 1.5 * IQR
                upper_bound = Q3 + 1.5 * IQR
                outliers = ((df[column] < lower_bound) | (df[column] > upper_bound)).sum()
                col_stats['outliers_count'] = int(outliers)
                col_stats['outliers_percentage'] = float(outliers / len(df) * 100)
                
        else:
            # Categorical statistics
            value_counts = df[column].value_counts()
            col_stats.update({
                'unique_count': int(df[column].nunique()),
                'most_frequent': str(value_counts.index[0]) if len(value_counts) > 0 else None,
                'most_frequent_count': int(value_counts.iloc[0]) if len(value_counts) > 0 else None,
                'most_frequent_percentage': float(value_counts.iloc[0] / df[column].notna().sum() * 100) if len(value_counts) > 0 and df[column].notna().sum() > 0 else None
            })
            
            # Top 5 values
            top_values = value_counts.head(5).to_dict()
            col_stats['top_values'] = {str(k): int(v) for k, v in top_values.items()}
        
        enhanced_stats[column] = col_stats
    
    return enhanced_stats

def _generate_correlation_analysis(df: pd.DataFrame) -> Dict[str, Any]:
    """Generate correlation analysis for numeric columns."""
    numeric_df = df.select_dtypes(include=[np.number])
    
    if len(numeric_df.columns) < 2:
        return {'message': 'Insufficient numeric columns for correlation analysis'}
    
    # Compute correlation matrix
    corr_matrix = numeric_df.corr()
    
    # Find strong correlations
    strong_correlations = []
    for i in range(len(corr_matrix.columns)):
        for j in range(i+1, len(corr_matrix.columns)):
            corr_val = corr_matrix.iloc[i, j]
            if abs(corr_val) > 0.5:  # Strong correlation threshold
                strong_correlations.append({
                    'feature_1': corr_matrix.columns[i],
                    'feature_2': corr_matrix.columns[j],
                    'correlation': float(corr_val),
                    'strength': 'strong' if abs(corr_val) > 0.7 else 'moderate'
                })
    
    # Sort by absolute correlation
    strong_correlations.sort(key=lambda x: abs(x['correlation']), reverse=True)
    
    return {
        'correlation_matrix': corr_matrix.to_dict(),
        'strong_correlations': strong_correlations[:20],  # Top 20
        'total_correlations': len(strong_correlations)
    }

def _generate_detailed_quality_analysis(df: pd.DataFrame) -> Dict[str, Any]:
    """Generate detailed data quality analysis."""
    quality_analysis = {
        'overall_quality_score': 0.0,
        'completeness': {},
        'consistency': {},
        'validity': {}
    }
    
    # Completeness analysis
    total_cells = len(df) * len(df.columns)
    missing_cells = df.isna().sum().sum()
    completeness_score = (total_cells - missing_cells) / total_cells * 100
    
    quality_analysis['completeness'] = {
        'overall_completeness': float(completeness_score),
        'total_cells': int(total_cells),
        'missing_cells': int(missing_cells),
        'complete_rows': int((df.notna().all(axis=1)).sum()),
        'complete_rows_percentage': float((df.notna().all(axis=1)).sum() / len(df) * 100)
    }
    
    # Consistency analysis - detect duplicates
    duplicate_rows = df.duplicated().sum()
    quality_analysis['consistency'] = {
        'duplicate_rows': int(duplicate_rows),
        'duplicate_percentage': float(duplicate_rows / len(df) * 100),
        'unique_rows': int(len(df) - duplicate_rows)
    }
    
    # Validity analysis - basic checks
    numeric_cols = df.select_dtypes(include=[np.number]).columns
    validity_issues = 0
    
    for col in numeric_cols:
        if df[col].notna().sum() > 0:
            # Check for negative values where they might not make sense
            if 'area' in col.lower() or 'size' in col.lower() or 'price' in col.lower():
                negative_count = (df[col] < 0).sum()
                if negative_count > 0:
                    validity_issues += negative_count
    
    quality_analysis['validity'] = {
        'validity_issues': int(validity_issues),
        'validity_score': float(max(0, (total_cells - validity_issues) / total_cells * 100))
    }
    
    # Calculate overall quality score
    quality_analysis['overall_quality_score'] = float(
        (completeness_score + quality_analysis['validity']['validity_score']) / 2
    )
    
    return quality_analysis

def generate_comprehensive_html_report(enhanced_results: Dict[str, Any], data_info: Dict[str, Any], charts: Dict[str, str], df: pd.DataFrame) -> str:
    """Generate comprehensive HTML report with actual data insights."""
    
    # Extract key statistics from enhanced results
    summary_stats = enhanced_results.get('summary', {})
    quality_stats = enhanced_results.get('data_quality_detailed', {})
    enhanced_stats = enhanced_results.get('enhanced_statistics', {})
    correlations = enhanced_results.get('correlation_analysis', {})
    insights = enhanced_results.get('insights', {})
    
    # Calculate some key metrics
    numeric_cols = len([col for col in df.columns if df[col].dtype in ['int64', 'float64']])
    categorical_cols = len(df.columns) - numeric_cols
    missing_percentage = (df.isna().sum().sum() / (len(df) * len(df.columns))) * 100
    
    html_content = f"""<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>Comprehensive DataScout Analysis Report</title>
    <style>
        body {{
            font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, sans-serif;
            line-height: 1.6;
            color: #333;
            background: #f8f9fa;
            margin: 0;
            padding: 20px;
        }}
        .container {{
            max-width: 1200px;
            margin: 0 auto;
            background: white;
            border-radius: 12px;
            box-shadow: 0 4px 6px rgba(0,0,0,0.1);
            overflow: hidden;
        }}
        .header {{
            background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
            color: white;
            padding: 40px;
            text-align: center;
        }}
        .header h1 {{
            margin: 0 0 10px 0;
            font-size: 2.5rem;
            font-weight: 300;
        }}
        .content {{
            padding: 40px;
        }}
        .section {{
            margin-bottom: 40px;
            padding: 30px;
            background: #f8f9fa;
            border-radius: 8px;
            border-left: 4px solid #667eea;
        }}
        .section h2 {{
            margin-top: 0;
            color: #667eea;
            font-size: 1.8rem;
        }}
        .stats-grid {{
            display: grid;
            grid-template-columns: repeat(auto-fit, minmax(250px, 1fr));
            gap: 20px;
            margin: 20px 0;
        }}
        .stat-card {{
            background: white;
            padding: 20px;
            border-radius: 8px;
            text-align: center;
            box-shadow: 0 2px 4px rgba(0,0,0,0.1);
        }}
        .stat-value {{
            font-size: 2rem;
            font-weight: bold;
            color: #667eea;
            margin-bottom: 5px;
        }}
        .stat-label {{
            color: #666;
            font-size: 0.9rem;
        }}
        .insights-list {{
            background: white;
            border-radius: 8px;
            padding: 20px;
            margin: 20px 0;
        }}
        .insight-item {{
            padding: 15px;
            border-bottom: 1px solid #eee;
            margin-bottom: 15px;
        }}
        .insight-item:last-child {{
            border-bottom: none;
            margin-bottom: 0;
        }}
        .correlation-table {{
            width: 100%;
            border-collapse: collapse;
            margin: 20px 0;
            background: white;
            border-radius: 8px;
            overflow: hidden;
        }}
        .correlation-table th, .correlation-table td {{
            padding: 12px;
            text-align: left;
            border-bottom: 1px solid #eee;
        }}
        .correlation-table th {{
            background: #f8f9fa;
            font-weight: 600;
            color: #667eea;
        }}
        .quality-indicator {{
            display: inline-block;
            padding: 4px 12px;
            border-radius: 20px;
            font-size: 0.85rem;
            font-weight: 500;
        }}
        .quality-excellent {{ background: #d4edda; color: #155724; }}
        .quality-good {{ background: #d1ecf1; color: #0c5460; }}
        .quality-fair {{ background: #fff3cd; color: #856404; }}
        .quality-poor {{ background: #f8d7da; color: #721c24; }}
    </style>
</head>
<body>
    <div class="container">
        <div class="header">
            <h1>📊 DataScout Analysis Report</h1>
            <p>Generated on {datetime.now().strftime('%B %d, %Y at %I:%M %p')}</p>
            <p>Dataset: <strong>{data_info['rows']:,}</strong> rows × <strong>{data_info['columns']}</strong> columns</p>
        </div>
        
        <div class="content">
            <div class="section">
                <h2>📈 Dataset Overview</h2>
                <div class="stats-grid">
                    <div class="stat-card">
                        <div class="stat-value">{data_info['rows']:,}</div>
                        <div class="stat-label">Total Records</div>
                    </div>
                    <div class="stat-card">
                        <div class="stat-value">{data_info['columns']}</div>
                        <div class="stat-label">Total Features</div>
                    </div>
                    <div class="stat-card">
                        <div class="stat-value">{numeric_cols}</div>
                        <div class="stat-label">Numeric Features</div>
                    </div>
                    <div class="stat-card">
                        <div class="stat-value">{categorical_cols}</div>
                        <div class="stat-label">Categorical Features</div>
                    </div>
                    <div class="stat-card">
                        <div class="stat-value">{missing_percentage:.1f}%</div>
                        <div class="stat-label">Missing Data</div>
                    </div>
                    <div class="stat-card">
                        <div class="stat-value">{quality_stats.get('overall_quality_score', 0):.1f}</div>
                        <div class="stat-label">Quality Score</div>
                    </div>
                </div>
            </div>
            
            <div class="section">
                <h2>🎯 Data Quality Assessment</h2>
                <div class="stats-grid">
                    <div class="stat-card">
                        <div class="stat-value">{quality_stats.get('completeness', {}).get('overall_completeness', 0):.1f}%</div>
                        <div class="stat-label">Data Completeness</div>
                    </div>
                    <div class="stat-card">
                        <div class="stat-value">{quality_stats.get('consistency', {}).get('duplicate_rows', 0):,}</div>
                        <div class="stat-label">Duplicate Records</div>
                    </div>
                    <div class="stat-card">
                        <div class="stat-value">{quality_stats.get('validity', {}).get('validity_issues', 0):,}</div>
                        <div class="stat-label">Validity Issues</div>
                    </div>
                </div>
                
                <div class="insights-list">
                    <h3>Quality Analysis Details</h3>
                    <div class="insight-item">
                        <strong>Completeness:</strong> {quality_stats.get('completeness', {}).get('complete_rows_percentage', 0):.1f}% of rows are complete (no missing values)
                    </div>
                    <div class="insight-item">
                        <strong>Missing Data Distribution:</strong> {quality_stats.get('completeness', {}).get('missing_cells', 0):,} missing cells out of {quality_stats.get('completeness', {}).get('total_cells', 0):,} total cells
                    </div>
                    <div class="insight-item">
                        <strong>Data Consistency:</strong> {quality_stats.get('consistency', {}).get('unique_rows', 0):,} unique records identified
                    </div>
                </div>
            </div>"""

    # Add feature statistics section
    if enhanced_stats:
        html_content += f"""
            <div class="section">
                <h2>📋 Feature Statistics</h2>
                <div class="insights-list">
                    <h3>Top Features by Completeness</h3>"""
        
        # Sort features by completeness
        sorted_features = sorted(
            [(name, stats) for name, stats in enhanced_stats.items()],
            key=lambda x: x[1]['non_null_count'],
            reverse=True
        )
        
        for i, (feature_name, stats) in enumerate(sorted_features[:10]):
            completeness = (1 - stats['null_percentage']/100) * 100
            html_content += f"""
                    <div class="insight-item">
                        <strong>{feature_name}:</strong> {completeness:.1f}% complete 
                        ({stats['non_null_count']:,} non-null values)"""
            
            if stats['dtype'] in ['int64', 'float64']:
                if stats.get('mean') is not None:
                    html_content += f" | Mean: {stats['mean']:.2f}"
                if stats.get('outliers_percentage') is not None:
                    html_content += f" | Outliers: {stats['outliers_percentage']:.1f}%"
            else:
                if stats.get('unique_count') is not None:
                    html_content += f" | Unique values: {stats['unique_count']:,}"
            
            html_content += "</div>"
        
        html_content += "</div></div>"

    # Add correlation analysis
    if correlations and correlations.get('strong_correlations'):
        html_content += f"""
            <div class="section">
                <h2>🔗 Correlation Analysis</h2>
                <p>Found {correlations.get('total_correlations', 0)} significant correlations (|r| > 0.5)</p>
                
                <table class="correlation-table">
                    <thead>
                        <tr>
                            <th>Feature 1</th>
                            <th>Feature 2</th>
                            <th>Correlation</th>
                            <th>Strength</th>
                        </tr>
                    </thead>
                    <tbody>"""
        
        for corr in correlations['strong_correlations'][:15]:  # Top 15 correlations
            strength_class = 'quality-excellent' if corr['strength'] == 'strong' else 'quality-good'
            html_content += f"""
                        <tr>
                            <td>{corr['feature_1']}</td>
                            <td>{corr['feature_2']}</td>
                            <td>{corr['correlation']:.3f}</td>
                            <td><span class="quality-indicator {strength_class}">{corr['strength']}</span></td>
                        </tr>"""
        
        html_content += """
                    </tbody>
                </table>
            </div>"""

    # Add business insights
    if insights:
        business_recommendations = insights.get('business_recommendations', [])
        key_findings = insights.get('key_findings', [])
        
        html_content += f"""
            <div class="section">
                <h2>💡 Business Insights</h2>
                <div class="insights-list">
                    <h3>Key Findings</h3>"""
        
        for i, finding in enumerate(key_findings[:8]):
            html_content += f"""
                    <div class="insight-item">
                        <strong>Finding {i+1}:</strong> {finding}
                    </div>"""
        
        html_content += """
                    <h3>Business Recommendations</h3>"""
        
        for i, recommendation in enumerate(business_recommendations[:8]):
            html_content += f"""
                    <div class="insight-item">
                        <strong>Recommendation {i+1}:</strong> {recommendation}
                    </div>"""
        
        html_content += "</div></div>"

    # Close HTML
    html_content += f"""
            <div class="section">
                <h2>📊 Report Summary</h2>
                <div class="insights-list">
                    <div class="insight-item">
                        <strong>Analysis Depth:</strong> Comprehensive statistical profiling with enhanced metrics
                    </div>
                    <div class="insight-item">
                        <strong>Data Coverage:</strong> {data_info['rows']:,} records across {data_info['columns']} features analyzed
                    </div>
                    <div class="insight-item">
                        <strong>Quality Assessment:</strong> Multi-dimensional data quality evaluation completed
                    </div>
                    <div class="insight-item">
                        <strong>Insights Generated:</strong> {len(business_recommendations) + len(key_findings)} actionable insights identified
                    </div>
                    <div class="insight-item">
                        <strong>Generated By:</strong> DataScout Analytics Platform - Comprehensive Data Intelligence
                    </div>
                </div>
            </div>
        </div>
    </div>
</body>
</html>"""
    
    return html_content

def generate_basic_html_report(enhanced_results: Dict[str, Any], data_info: Dict[str, Any], df: pd.DataFrame) -> str:
    """Generate basic HTML report with real data as fallback."""
    
    missing_percentage = (df.isna().sum().sum() / (len(df) * len(df.columns))) * 100
    numeric_cols = len([col for col in df.columns if df[col].dtype in ['int64', 'float64']])
    
    return f"""<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <title>DataScout Analysis Report</title>
    <style>
        body {{ font-family: Arial, sans-serif; margin: 20px; background: #f5f5f5; }}
        .container {{ max-width: 800px; margin: 0 auto; background: white; padding: 30px; border-radius: 8px; }}
        .header {{ text-align: center; background: #007acc; color: white; padding: 20px; border-radius: 8px; margin-bottom: 30px; }}
        .stat {{ background: #f9f9f9; padding: 15px; margin: 10px 0; border-left: 4px solid #007acc; }}
    </style>
</head>
<body>
    <div class="container">
        <div class="header">
            <h1>📊 DataScout Analysis Report</h1>
            <p>Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}</p>
        </div>
        
        <div class="stat">
            <h3>Dataset Overview</h3>
            <p><strong>Records:</strong> {data_info['rows']:,}</p>
            <p><strong>Features:</strong> {data_info['columns']}</p>
            <p><strong>Numeric Features:</strong> {numeric_cols}</p>
            <p><strong>Missing Data:</strong> {missing_percentage:.1f}%</p>
        </div>
        
        <div class="stat">
            <h3>Data Quality</h3>
            <p>Comprehensive quality assessment completed with multi-dimensional analysis</p>
            <p>Quality score calculated based on completeness, consistency, and validity metrics</p>
        </div>
        
        <div class="stat">
            <h3>Analysis Summary</h3>
            <p>This report provides comprehensive statistical analysis of your dataset.</p>
            <p>Generated by DataScout Analytics Platform with enhanced profiling capabilities.</p>
        </div>
    </div>
</body>
</html>"""

def generate_executive_summary_report(enhanced_results: Dict[str, Any], data_info: Dict[str, Any], df: pd.DataFrame, data_id: str) -> str:
    """Generate executive summary with real data insights."""
    
    # Extract key metrics
    quality_stats = enhanced_results.get('data_quality_detailed', {})
    insights = enhanced_results.get('insights', {})
    correlations = enhanced_results.get('correlation_analysis', {})
    enhanced_stats = enhanced_results.get('enhanced_statistics', {})
    
    # Calculate key business metrics
    missing_percentage = (df.isna().sum().sum() / (len(df) * len(df.columns))) * 100
    numeric_cols = len([col for col in df.columns if df[col].dtype in ['int64', 'float64']])
    quality_score = quality_stats.get('overall_quality_score', 0)
    
    # Get top insights
    business_recommendations = insights.get('business_recommendations', [])[:5]
    key_findings = insights.get('key_findings', [])[:5]
    
    html_content = f"""<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>Executive Summary - DataScout Analysis</title>
    <style>
        body {{
            font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, sans-serif;
            line-height: 1.6;
            color: #333;
            background: #f8f9fa;
            margin: 0;
            padding: 20px;
        }}
        .container {{
            max-width: 900px;
            margin: 0 auto;
            background: white;
            border-radius: 12px;
            box-shadow: 0 4px 6px rgba(0,0,0,0.1);
            overflow: hidden;
        }}
        .header {{
            background: linear-gradient(135deg, #2c3e50 0%, #3498db 100%);
            color: white;
            padding: 40px;
            text-align: center;
        }}
        .header h1 {{
            margin: 0 0 10px 0;
            font-size: 2.2rem;
            font-weight: 300;
        }}
        .content {{
            padding: 40px;
        }}
        .executive-summary {{
            background: #f8f9fa;
            padding: 30px;
            border-radius: 8px;
            margin-bottom: 30px;
            border-left: 4px solid #3498db;
        }}
        .key-metrics {{
            display: grid;
            grid-template-columns: repeat(auto-fit, minmax(200px, 1fr));
            gap: 20px;
            margin: 30px 0;
        }}
        .metric-card {{
            background: white;
            padding: 25px;
            border-radius: 8px;
            text-align: center;
            box-shadow: 0 2px 4px rgba(0,0,0,0.1);
            border-top: 3px solid #3498db;
        }}
        .metric-value {{
            font-size: 2.2rem;
            font-weight: bold;
            color: #2c3e50;
            margin-bottom: 5px;
        }}
        .metric-label {{
            color: #666;
            font-size: 0.9rem;
            text-transform: uppercase;
            letter-spacing: 0.5px;
        }}
        .insights-section {{
            margin: 30px 0;
        }}
        .insights-section h3 {{
            color: #2c3e50;
            border-bottom: 2px solid #3498db;
            padding-bottom: 10px;
            margin-bottom: 20px;
        }}
        .insight-item {{
            background: white;
            padding: 20px;
            margin-bottom: 15px;
            border-radius: 6px;
            border-left: 3px solid #3498db;
            box-shadow: 0 1px 3px rgba(0,0,0,0.1);
        }}
        .quality-badge {{
            display: inline-block;
            padding: 6px 12px;
            border-radius: 20px;
            font-size: 0.8rem;
            font-weight: 500;
            text-transform: uppercase;
        }}
        .quality-excellent {{ background: #d4edda; color: #155724; }}
        .quality-good {{ background: #d1ecf1; color: #0c5460; }}
        .quality-fair {{ background: #fff3cd; color: #856404; }}
    </style>
</head>
<body>
    <div class="container">
        <div class="header">
            <h1>📊 Executive Summary</h1>
            <p>Comprehensive Data Analysis Report</p>
            <p>Dataset: <strong>{data_id}</strong> | Generated: {datetime.now().strftime('%B %d, %Y')}</p>
        </div>
        
        <div class="content">
            <div class="executive-summary">
                <h2>Dataset Overview</h2>
                <p>Our comprehensive analysis examined <strong>{data_info['rows']:,} records</strong> across 
                <strong>{data_info['columns']} features</strong>, revealing key patterns and insights for strategic decision-making. 
                The dataset demonstrates a <strong>{quality_score:.1f}% overall quality score</strong> with 
                <strong>{missing_percentage:.1f}%</strong> missing data across all features.</p>
                
                <p>This analysis identified <strong>{len(business_recommendations)} key business recommendations</strong> 
                and <strong>{len(key_findings)} critical findings</strong> that can drive immediate actionable insights.</p>
            </div>
            
            <div class="key-metrics">
                <div class="metric-card">
                    <div class="metric-value">{data_info['rows']:,}</div>
                    <div class="metric-label">Total Records</div>
                </div>
                <div class="metric-card">
                    <div class="metric-value">{data_info['columns']}</div>
                    <div class="metric-label">Features Analyzed</div>
                </div>
                <div class="metric-card">
                    <div class="metric-value">{numeric_cols}</div>
                    <div class="metric-label">Numeric Variables</div>
                </div>
                <div class="metric-card">
                    <div class="metric-value">{quality_score:.0f}%</div>
                    <div class="metric-label">Data Quality Score</div>
                </div>
                <div class="metric-card">
                    <div class="metric-value">{correlations.get('total_correlations', 0)}</div>
                    <div class="metric-label">Strong Correlations</div>
                </div>
                <div class="metric-card">
                    <div class="metric-value">{len(business_recommendations + key_findings)}</div>
                    <div class="metric-label">Actionable Insights</div>
                </div>
            </div>
            
            <div class="insights-section">
                <h3>🎯 Key Business Findings</h3>"""

    # Add key findings
    for i, finding in enumerate(key_findings):
        html_content += f"""
                <div class="insight-item">
                    <strong>Key Finding {i+1}:</strong> {finding}
                </div>"""
    
    html_content += """
                <h3>💡 Strategic Recommendations</h3>"""
    
    # Add business recommendations  
    for i, recommendation in enumerate(business_recommendations):
        html_content += f"""
                <div class="insight-item">
                    <strong>Recommendation {i+1}:</strong> {recommendation}
                </div>"""
    
    # Add data quality assessment
    quality_class = 'quality-excellent' if quality_score >= 80 else 'quality-good' if quality_score >= 60 else 'quality-fair'
    
    html_content += f"""
                <h3>🔍 Data Quality Assessment</h3>
                <div class="insight-item">
                    <strong>Overall Quality:</strong> 
                    <span class="quality-badge {quality_class}">{quality_score:.1f}% Quality Score</span>
                    <br><br>
                    <strong>Completeness:</strong> {quality_stats.get('completeness', {}).get('overall_completeness', 0):.1f}% of data is complete
                    <br>
                    <strong>Consistency:</strong> {quality_stats.get('consistency', {}).get('unique_rows', 0):,} unique records identified
                    <br>
                    <strong>Validity:</strong> {quality_stats.get('validity', {}).get('validity_issues', 0)} data validity issues detected
                </div>
            </div>
            
            <div class="executive-summary">
                <h2>Next Steps & Implementation</h2>
                <p>Based on this comprehensive analysis, we recommend prioritizing the top {min(3, len(business_recommendations))} strategic recommendations 
                for immediate implementation. The high data quality score of {quality_score:.1f}% indicates reliable insights 
                that can confidently guide business decisions.</p>
                
                <p><strong>Report Details:</strong> This executive summary is based on advanced statistical analysis 
                with comprehensive profiling across all {data_info['columns']} features. For detailed technical insights, 
                refer to the complete analysis report.</p>
                
                <p><em>Generated by DataScout Analytics Platform - Executive Intelligence Suite</em></p>
            </div>
        </div>
    </div>
</body>
</html>"""
    
    return html_content

# Add these helper functions to the module level
api_router.generate_comprehensive_html_report = generate_comprehensive_html_report
api_router.generate_basic_html_report = generate_basic_html_report
api_router.generate_executive_summary_report = generate_executive_summary_report

# Global instances (in production, consider dependency injection)
loader = create_loader()
preprocessor = create_preprocessor()
summarizer = create_summarizer()
visualizer = create_visualizer()
feature_selector = create_feature_selector()
insight_engine = create_insight_engine()
ai_generator = create_enhanced_ai_generator()  # AI capabilities
html_generator = create_html_generator()  # Report generation
pdf_generator = create_pdf_generator()  # PDF generation
comprehensive_profiler_factory = create_comprehensive_profiler  # Enhanced profiler factory

# In-memory storage for demo (use database in production)
data_store = {}


@api_router.post("/upload", response_model=Dict[str, Any])
async def upload_data(file: UploadFile = File(...)):
    """
    Upload and load data file.
    
    Supports: CSV, Excel, JSON files
    """
    try:
        # Validate file type
        allowed_extensions = {'.csv', '.xlsx', '.xls', '.json'}
        file_extension = '.' + file.filename.split('.')[-1].lower()
        
        if file_extension not in allowed_extensions:
            raise HTTPException(
                status_code=400, 
                detail=f"Unsupported file type. Allowed: {allowed_extensions}"
            )
        
        # Read file content
        content = await file.read()
        
        # Load data based on file type
        if file_extension == '.csv':
            df = pd.read_csv(StringIO(content.decode('utf-8')))
        elif file_extension in ['.xlsx', '.xls']:
            df = pd.read_excel(BytesIO(content))
        elif file_extension == '.json':
            data = json.loads(content.decode('utf-8'))
            df = pd.DataFrame(data if isinstance(data, list) else [data])
        
        # Validate and store data
        validation_result = loader.validate_data(df)
        if not validation_result['is_valid']:
            raise HTTPException(
                status_code=400,
                detail=f"Invalid data: {validation_result['issues']}"
            )
        
        # Generate unique data ID
        data_id = f"data_{len(data_store) + 1}"
        data_store[data_id] = df
        
        # Get sample data for preview
        sample = loader.get_sample_data(df)
        
        return {
            "data_id": data_id,
            "status": "success",
            "message": f"File '{file.filename}' uploaded successfully",
            "data_info": {
                "rows": len(df),
                "columns": len(df.columns),
                "column_names": list(df.columns),
                "data_types": {str(k): str(v) for k, v in df.dtypes.to_dict().items()}
            },
            "sample_data": sample['sample_data'][:3],  # First 3 rows
            "validation": validation_result
        }
        
    except Exception as e:
        logger.error(f"Upload error: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))


@api_router.get("/data/{data_id}/info", response_model=Dict[str, Any])
async def get_data_info(data_id: str):
    """Get basic information about uploaded dataset."""
    if data_id not in data_store:
        raise HTTPException(status_code=404, detail="Data not found")
    
    df = data_store[data_id]
    validation = loader.validate_data(df)
    sample = loader.get_sample_data(df, n_rows=5)
    
    return {
        "data_id": data_id,
        "basic_info": {
            "rows": len(df),
            "columns": len(df.columns),
            "column_names": list(df.columns),
            "data_types": {str(k): str(v) for k, v in df.dtypes.to_dict().items()},
            "memory_usage_mb": float(round(df.memory_usage(deep=True).sum() / 1024 / 1024, 2))
        },
        "sample_data": sample,
        "validation": validation
    }


@api_router.post("/data/{data_id}/preprocess", response_model=Dict[str, Any])
async def preprocess_data(data_id: str, config: Optional[Dict[str, Any]] = None):
    """Preprocess the dataset with cleaning and validation."""
    if data_id not in data_store:
        raise HTTPException(status_code=404, detail="Data not found")
    
    try:
        df = data_store[data_id]
        original_shape = df.shape
        
        # Preprocess data
        cleaned_df = preprocessor.clean_data(df, config)
        
        # Update stored data
        processed_id = f"{data_id}_processed"
        data_store[processed_id] = cleaned_df
        
        # Generate preprocessing summary
        summary = preprocessor.get_preprocessing_summary(df, cleaned_df)
        quality_assessment = preprocessor.validate_data_quality(cleaned_df)
        
        return {
            "original_data_id": data_id,
            "processed_data_id": processed_id,
            "status": "success",
            "preprocessing_summary": summary,
            "quality_assessment": quality_assessment,
            "message": f"Data preprocessed: {original_shape} -> {cleaned_df.shape}"
        }
        
    except Exception as e:
        logger.error(f"Preprocessing error: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))


@api_router.get("/data/{data_id}/summary", response_model=Dict[str, Any])
async def get_data_summary(data_id: str, target_column: Optional[str] = None):
    """Generate comprehensive data summary and statistics."""
    if data_id not in data_store:
        raise HTTPException(status_code=404, detail="Data not found")
    
    try:
        df = data_store[data_id]
        
        # Validate target column if provided
        if target_column and target_column not in df.columns:
            raise HTTPException(
                status_code=400, 
                detail=f"Target column '{target_column}' not found"
            )
        
        # Generate comprehensive summary
        summary = summarizer.generate_comprehensive_summary(df)
        
        # Generate report
        report = summarizer.generate_summary_report(
            df, title=f"Summary Report for {data_id}"
        )
        
        return {
            "data_id": data_id,
            "target_column": target_column,
            "comprehensive_summary": summary,
            "summary_report": report
        }
        
    except Exception as e:
        logger.error(f"Summary generation error: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))


@api_router.get("/data/{data_id}/visualizations", response_model=Dict[str, Any])
async def get_visualization_recommendations(data_id: str):
    """Get recommended visualizations for the dataset."""
    if data_id not in data_store:
        raise HTTPException(status_code=404, detail="Data not found")
    
    try:
        df = data_store[data_id]
        
        # Get visualization recommendations
        recommendations = visualizer.get_recommended_plots(df)
        
        return {
            "data_id": data_id,
            "visualization_recommendations": recommendations,
            "total_recommendations": len(recommendations)
        }
        
    except Exception as e:
        logger.error(f"Visualization recommendation error: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))


@api_router.post("/data/{data_id}/visualize/{plot_type}", response_model=Dict[str, Any])
async def create_visualization(
    data_id: str, 
    plot_type: str, 
    config: Dict[str, Any]
):
    """Create specific visualization."""
    if data_id not in data_store:
        raise HTTPException(status_code=404, detail="Data not found")
    
    try:
        df = data_store[data_id]
        
        # Create visualization based on type
        if plot_type == "histogram":
            if "column" not in config:
                raise HTTPException(status_code=400, detail="Column parameter required for histogram")
            result = visualizer.create_histogram(df, config["column"], 
                                               bins=config.get("bins", 30))
            
        elif plot_type == "scatter":
            if "x_column" not in config or "y_column" not in config:
                raise HTTPException(status_code=400, detail="x_column and y_column required for scatter plot")
            result = visualizer.create_scatter_plot(df, config["x_column"], config["y_column"],
                                                   config.get("color_column"))
            
        elif plot_type == "correlation_heatmap":
            result = visualizer.create_correlation_heatmap(df)
            
        elif plot_type == "box_plot":
            columns = config.get("columns", df.select_dtypes(include=['number']).columns.tolist()[:3])
            result = visualizer.create_box_plot(df, columns)
            
        elif plot_type == "bar_chart":
            if "column" not in config:
                raise HTTPException(status_code=400, detail="Column parameter required for bar chart")
            result = visualizer.create_categorical_bar_chart(df, config["column"],
                                                            config.get("top_n", 10))
            
        else:
            raise HTTPException(status_code=400, detail=f"Unsupported plot type: {plot_type}")
        
        return {
            "data_id": data_id,
            "plot_type": plot_type,
            "config": config,
            "visualization": result
        }
        
    except Exception as e:
        logger.error(f"Visualization creation error: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))


@api_router.get("/data/{data_id}/features", response_model=Dict[str, Any])
async def analyze_features(data_id: str, target_column: Optional[str] = None):
    """Analyze feature importance and relationships."""
    if data_id not in data_store:
        raise HTTPException(status_code=404, detail="Data not found")
    
    try:
        df = data_store[data_id]
        
        # Validate target column if provided
        if target_column and target_column not in df.columns:
            raise HTTPException(
                status_code=400, 
                detail=f"Target column '{target_column}' not found"
            )
        
        # Analyze feature importance
        feature_analysis = feature_selector.analyze_feature_importance(df, target_column)
        
        # Select best features
        best_features = feature_selector.select_best_features(df, target_column, k=10)
        
        return {
            "data_id": data_id,
            "target_column": target_column,
            "feature_analysis": feature_analysis,
            "best_features": best_features
        }
        
    except Exception as e:
        logger.error(f"Feature analysis error: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))


@api_router.get("/data/{data_id}/insights", response_model=Dict[str, Any])
async def generate_insights(
    data_id: str, 
    target_column: Optional[str] = None,
    business_context: Optional[str] = None
):
    """Generate comprehensive business insights from the data."""
    if data_id not in data_store:
        raise HTTPException(status_code=404, detail="Data not found")
    
    try:
        df = data_store[data_id]
        
        # Validate target column if provided
        if target_column and target_column not in df.columns:
            raise HTTPException(
                status_code=400, 
                detail=f"Target column '{target_column}' not found"
            )
        
        # Generate comprehensive insights
        insights = insight_engine.generate_comprehensive_insights(
            df, target_column, business_context
        )
        
        # Generate insight report
        report = insight_engine.generate_insight_report(
            df, target_column, business_context, 
            title=f"Insight Report for {data_id}"
        )
        
        return {
            "data_id": data_id,
            "target_column": target_column,
            "business_context": business_context,
            "comprehensive_insights": insights,
            "insight_report": report
        }
        
    except Exception as e:
        logger.error(f"Insight generation error: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))


@api_router.get("/data/{data_id}/ai-insights", response_model=Dict[str, Any])
async def generate_ai_insights(
    data_id: str, 
    target_column: Optional[str] = None,
    business_context: Optional[str] = None
):
    """Generate comprehensive AI-powered insights from the data."""
    if data_id not in data_store:
        raise HTTPException(status_code=404, detail="Data not found")
    
    try:
        df = data_store[data_id]
        
        # Validate target column if provided
        if target_column and target_column not in df.columns:
            raise HTTPException(
                status_code=400, 
                detail=f"Target column '{target_column}' not found"
            )
        
        # Generate comprehensive analysis first
        analysis_results = {}
        
        # Basic analysis
        analysis_results['summary'] = summarizer.generate_comprehensive_summary(df)
        analysis_results['quality'] = preprocessor.validate_data_quality(df)
        analysis_results['features'] = feature_selector.analyze_feature_importance(df, target_column)
        analysis_results['insights'] = insight_engine.generate_comprehensive_insights(
            df, target_column, business_context
        )
        
        # Generate AI insights
        ai_insights = await ai_generator.generate_comprehensive_ai_insights(
            analysis_results, df, business_context, target_column
        )
        
        return {
            "data_id": data_id,
            "target_column": target_column,
            "business_context": business_context,
            "ai_insights": ai_insights,
            "analysis_foundation": {
                "data_quality_score": analysis_results['quality']['quality_score'],
                "feature_count": len(df.columns),
                "insights_generated": len(ai_insights['insights'])
            }
        }
        
    except Exception as e:
        logger.error(f"AI insights generation error: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))


@api_router.get("/data/{data_id}/ai-summary", response_model=Dict[str, Any])
async def generate_ai_executive_summary(
    data_id: str,
    business_context: Optional[str] = None
):
    """Generate AI-powered executive summary."""
    if data_id not in data_store:
        raise HTTPException(status_code=404, detail="Data not found")
    
    try:
        df = data_store[data_id]
        
        # Generate analysis for AI prompt
        analysis_results = {
            'summary': summarizer.generate_comprehensive_summary(df),
            'quality': preprocessor.validate_data_quality(df),
            'insights': insight_engine.generate_comprehensive_insights(df, None, business_context)
        }
        
        # Generate AI executive summary
        ai_summary = await generate_ai_executive_summary(
            analysis_results, df, business_context
        )
        
        return {
            "data_id": data_id,
            "business_context": business_context,
            "executive_summary": ai_summary,
            "data_overview": {
                "rows": len(df),
                "columns": len(df.columns),
                "quality_score": analysis_results['quality']['quality_score']
            }
        }
        
    except Exception as e:
        logger.error(f"AI executive summary error: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))


@api_router.get("/data/{data_id}/ai-story", response_model=Dict[str, Any])
async def generate_data_story(
    data_id: str,
    business_context: Optional[str] = None
):
    """Generate AI-powered data story."""
    if data_id not in data_store:
        raise HTTPException(status_code=404, detail="Data not found")
    
    try:
        df = data_store[data_id]
        
        # Generate comprehensive analysis
        analysis_results = {
            'summary': summarizer.generate_comprehensive_summary(df),
            'quality': preprocessor.validate_data_quality(df),
            'features': feature_selector.analyze_feature_importance(df),
            'insights': insight_engine.generate_comprehensive_insights(df, None, business_context)
        }
        
        # Generate AI data story
        data_story = await ai_generator.explain_data_story(
            analysis_results, df, business_context
        )
        
        return {
            "data_id": data_id,
            "business_context": business_context,
            "data_story": data_story,
            "story_metadata": {
                "based_on_records": len(df),
                "features_analyzed": len(df.columns),
                "quality_foundation": analysis_results['quality']['quality_score']
            }
        }
        
    except Exception as e:
        logger.error(f"AI data story generation error: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))


@api_router.get("/ai/capabilities", response_model=Dict[str, Any])
async def get_ai_capabilities():
    """Get AI service capabilities and status."""
    try:
        capabilities = ai_generator.get_ai_capabilities()
        
        return {
            "ai_service_status": "available" if capabilities["ai_available"] else "unavailable",
            "capabilities": capabilities,
            "supported_features": [
                "Executive summaries",
                "Detailed insights",
                "Business recommendations", 
                "Anomaly explanations",
                "Data storytelling",
                "Multi-domain analysis"
            ]
        }
        
    except Exception as e:
        logger.error(f"AI capabilities check error: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))


@api_router.get("/data/{data_id}/complete-analysis", response_model=Dict[str, Any])
async def complete_analysis(
    data_id: str,
    target_column: Optional[str] = None,
    business_context: Optional[str] = None
):
    """Run complete analysis pipeline on the dataset."""
    if data_id not in data_store:
        raise HTTPException(status_code=404, detail="Data not found")
    
    try:
        df = data_store[data_id]
        
        # Validate target column if provided
        if target_column and target_column not in df.columns:
            raise HTTPException(
                status_code=400, 
                detail=f"Target column '{target_column}' not found"
            )
        
        # Run complete analysis
        results = {}
        
        # 1. Data Summary
        results['summary'] = summarizer.generate_comprehensive_summary(df)
        
        # 2. Data Quality Assessment
        results['quality'] = preprocessor.validate_data_quality(df)
        
        # 3. Feature Analysis
        results['features'] = feature_selector.analyze_feature_importance(df, target_column)
        
        # 4. Visualization Recommendations
        results['visualizations'] = visualizer.get_recommended_plots(df)
        
        # 5. Comprehensive Insights
        results['insights'] = insight_engine.generate_comprehensive_insights(
            df, target_column, business_context
        )
        
        # 6. AI-Powered Insights (new!)
        try:
            results['ai_insights'] = await ai_generator.generate_comprehensive_ai_insights(
                results, df, business_context, target_column
            )
        except Exception as e:
            logger.warning(f"AI insights generation failed: {str(e)}")
            results['ai_insights'] = {
                "error": "AI service unavailable",
                "message": "Analysis completed without AI enhancement"
            }
        
        # 7. Executive Report
        results['executive_report'] = insight_engine.generate_insight_report(
            df, target_column, business_context,
            title=f"Complete Analysis Report for {data_id}"
        )
        
        # Create response with thorough NumPy conversion
        response_data = {
            "data_id": data_id,
            "target_column": target_column,
            "business_context": business_context,
            "analysis_results": results,
            "analysis_summary": {
                "total_features": len(df.columns),
                "data_quality_score": results['quality']['quality_score'],
                "key_insights_count": len(results['insights']['business_recommendations']),
                "recommended_plots": len(results['visualizations'])
            }
        }
        
        logger.info("Converting NumPy types for JSON serialization...")
        converted_response = convert_numpy_types(response_data)
        logger.info("NumPy conversion completed successfully")
        
        return converted_response
        
    except Exception as e:
        logger.error(f"Complete analysis error: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))


@api_router.delete("/data/{data_id}")
async def delete_data(data_id: str):
    """Delete dataset from memory."""
    if data_id not in data_store:
        raise HTTPException(status_code=404, detail="Data not found")
    
    del data_store[data_id]
    
    # Also delete processed versions
    processed_id = f"{data_id}_processed"
    if processed_id in data_store:
        del data_store[processed_id]
    
    return {
        "status": "success",
        "message": f"Data {data_id} deleted successfully"
    }


@api_router.get("/data/list")
async def list_datasets():
    """List all available datasets."""
    datasets = []
    
    for data_id, df in data_store.items():
        datasets.append({
            "data_id": data_id,
            "rows": len(df),
            "columns": len(df.columns),
            "memory_usage_mb": float(round(df.memory_usage(deep=True).sum() / 1024 / 1024, 2))
        })
    
    return {
        "datasets": datasets,
        "total_datasets": len(datasets)
    }


@api_router.get("/data/{data_id}/reports/enhanced", response_class=JSONResponse)
async def generate_enhanced_report(
    data_id: str,
    target_column: Optional[str] = None,
    business_context: Optional[str] = None
):
    """Generate enhanced comprehensive data profiling report using the advanced profiler."""
    if data_id not in data_store:
        raise HTTPException(status_code=404, detail="Data not found")
    
    try:
        df = data_store[data_id]
        
        # Create enhanced profiler configuration
        config = ProfileConfig(
            target_column=target_column,
            correlation_threshold=0.7,
            outlier_threshold=3.0,
            missing_threshold=0.05,
            cardinality_threshold=50,
            visualize=True,
            generate_html=True
        )
        
        # Create comprehensive profiler instance
        profiler = comprehensive_profiler_factory(config)
        profiler.df = df
        
        # Generate comprehensive profile
        logger.info(f"Generating enhanced profile for data_id: {data_id}")
        results = profiler.generate_comprehensive_profile()
        
        # Generate HTML report
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        report_filename = f"enhanced_datascout_report_{timestamp}.html"
        html_content = profiler.export_html_report()
        
        return {
            "data_id": data_id,
            "report_type": "enhanced_comprehensive",
            "target_column": target_column,
            "business_context": business_context,
            "html_content": html_content,
            "analysis_results": convert_numpy_types(results),
            "generation_time": datetime.now().isoformat(),
            "report_filename": report_filename,
            "status": "success",
            "enhancement_summary": {
                "analysis_depth": "50+ statistical measures per feature",
                "visualization_count": len([k for k in results.get('visualizations', {}).keys() if k]),
                "insight_count": len(results.get('narrative_insights', [])),
                "quality_score": results.get('data_quality', {}).get('overall_score', 'N/A')
            }
        }
        
    except Exception as e:
        logger.error(f"Enhanced report generation error: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Enhanced report generation failed: {str(e)}")


@api_router.get("/data/{data_id}/analysis/comprehensive")
async def comprehensive_analysis(
    data_id: str,
    target_column: Optional[str] = None
):
    """Run enhanced comprehensive analysis with advanced statistical profiling."""
    if data_id not in data_store:
        raise HTTPException(status_code=404, detail="Data not found")
    
    try:
        df = data_store[data_id]
        
        # Create enhanced profiler configuration
        config = ProfileConfig(
            target_column=target_column,
            correlation_threshold=0.6,
            outlier_threshold=3.0,
            visualize=True
        )
        
        # Create comprehensive profiler instance
        profiler = comprehensive_profiler_factory(config)
        profiler.df = df
        
        # Generate comprehensive profile
        logger.info(f"Running comprehensive analysis for data_id: {data_id}")
        results = profiler.generate_comprehensive_profile()
        
        # Convert results for JSON serialization
        converted_results = convert_numpy_types(results)
        
        return {
            "data_id": data_id,
            "target_column": target_column,
            "analysis_type": "comprehensive_enhanced",
            "results": converted_results,
            "analysis_summary": {
                "dataset_shape": {"rows": len(df), "columns": len(df.columns)},
                "analysis_depth": "Enhanced with 50+ statistical measures per feature",
                "quality_score": converted_results.get('data_quality', {}).get('overall_score', 'N/A'),
                "key_insights_count": len(converted_results.get('narrative_insights', [])),
                "correlations_found": len(converted_results.get('relationships', {}).get('target_correlations', {}).get('top_correlated_features', [])),
                "outliers_detected": converted_results.get('data_quality', {}).get('outlier_summary', {}).get('total_outliers', 0)
            },
            "generation_time": datetime.now().isoformat(),
            "status": "success"
        }
        
    except Exception as e:
        logger.error(f"Comprehensive analysis error: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Comprehensive analysis failed: {str(e)}")


# Health check endpoints
@api_router.get("/health")
async def health_check():
    """API health check."""
    return {
        "status": "healthy",
        "service": "DataScout Analysis API",
        "version": "1.0.0",
        "available_endpoints": [
            "/upload", "/data/{id}/info", "/data/{id}/preprocess",
            "/data/{id}/summary", "/data/{id}/visualizations", 
            "/data/{id}/features", "/data/{id}/insights",
            "/data/{id}/complete-analysis", "/data/{id}/ai-insights",
            "/data/{id}/ai-summary", "/data/{id}/ai-story", "/ai/capabilities",
            "/data/{id}/reports/html", "/data/{id}/reports/pdf", "/data/{id}/reports/package",
            "/data/{id}/reports/enhanced", "/data/{id}/analysis/comprehensive"
        ]
    }


@api_router.get("/data/{data_id}/reports/html", response_class=JSONResponse)
async def generate_html_report(
    data_id: str,
    report_type: str = "complete",
    target_column: Optional[str] = None,
    business_context: Optional[str] = None
):
    """Generate HTML report for analysis results."""
    if data_id not in data_store:
        raise HTTPException(status_code=404, detail="Data not found")
    
    try:
        df = data_store[data_id]
        
        # Get basic data info
        data_info = {
            "data_id": data_id,
            "rows": len(df),
            "columns": len(df.columns),
            "generation_time": datetime.now()
        }
        
        # Run analysis based on report type
        if report_type == "executive":
            # Generate basic analysis for executive summary
            analysis_results = {
                'summary': summarizer.generate_comprehensive_summary(df),
                'quality': preprocessor.validate_data_quality(df),
                'insights': insight_engine.generate_comprehensive_insights(
                    df, target_column, business_context
                )
            }
            
            # Get AI insights
            ai_insights = None
            try:
                ai_insights = await ai_generator.generate_comprehensive_ai_insights(
                    analysis_results, df, business_context, target_column
                )
            except Exception as e:
                logger.warning(f"AI insights failed: {str(e)}")
            
            html_content = html_generator.generate_executive_summary_report(
                analysis_results, data_info, ai_insights
            )
            
        elif report_type == "ai_insights":
            # Generate AI-focused report
            analysis_results = {
                'summary': summarizer.generate_comprehensive_summary(df),
                'quality': preprocessor.validate_data_quality(df)
            }
            
            ai_insights = await ai_generator.generate_comprehensive_ai_insights(
                analysis_results, df, business_context, target_column
            )
            
            html_content = html_generator.generate_ai_insights_report(
                ai_insights, data_info
            )
            
        else:  # complete analysis
            # Generate complete analysis
            analysis_results = {
                'summary': summarizer.generate_comprehensive_summary(df),
                'quality': preprocessor.validate_data_quality(df),
                'features': feature_selector.analyze_feature_importance(df, target_column),
                'insights': insight_engine.generate_comprehensive_insights(
                    df, target_column, business_context
                ),
                'visualizations': visualizer.get_recommended_plots(df)
            }
            
            # Generate some sample charts
            charts = {}
            try:
                # Create correlation heatmap if there are numeric columns
                numeric_cols = df.select_dtypes(include=['number']).columns
                if len(numeric_cols) > 1:
                    corr_result = visualizer.create_correlation_heatmap(df)
                    if corr_result and 'base64_image' in corr_result:
                        charts['correlation_heatmap'] = corr_result['base64_image']
            except Exception as e:
                logger.warning(f"Chart generation failed: {str(e)}")
            
            # Get AI insights
            ai_insights = None
            try:
                ai_insights = await ai_generator.generate_comprehensive_ai_insights(
                    analysis_results, df, business_context, target_column
                )
            except Exception as e:
                logger.warning(f"AI insights failed: {str(e)}")
            
            html_content = html_generator.generate_complete_analysis_report(
                analysis_results, data_info, charts, ai_insights
            )
        
        return {
            "data_id": data_id,
            "report_type": report_type,
            "html_content": html_content,
            "generation_time": datetime.now().isoformat(),
            "status": "success"
        }
        
    except Exception as e:
        logger.error(f"HTML report generation error: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))


@api_router.get("/data/{data_id}/reports/pdf")
async def generate_pdf_report_endpoint(
    data_id: str,
    report_type: str = "complete",
    target_column: Optional[str] = None,
    business_context: Optional[str] = None
):
    """Generate PDF report for analysis results."""
    if data_id not in data_store:
        raise HTTPException(status_code=404, detail="Data not found")
    
    try:
        from fastapi.responses import StreamingResponse
        import io
        
        df = data_store[data_id]
        
        # Get basic data info
        data_info = {
            "data_id": data_id,
            "rows": len(df),
            "columns": len(df.columns),
            "generation_time": datetime.now()
        }
        
        # Run analysis based on report type
        if report_type == "executive":
            analysis_results = {
                'summary': summarizer.generate_comprehensive_summary(df),
                'quality': preprocessor.validate_data_quality(df),
                'insights': insight_engine.generate_comprehensive_insights(
                    df, target_column, business_context
                )
            }
            
            ai_insights = None
            try:
                ai_insights = await ai_generator.generate_comprehensive_ai_insights(
                    analysis_results, df, business_context, target_column
                )
            except Exception as e:
                logger.warning(f"AI insights failed: {str(e)}")
            
            pdf_bytes = pdf_generator.generate_executive_summary_pdf(
                analysis_results, data_info, ai_insights
            )
            
        elif report_type == "ai_insights":
            analysis_results = {
                'summary': summarizer.generate_comprehensive_summary(df),
                'quality': preprocessor.validate_data_quality(df)
            }
            
            ai_insights = await ai_generator.generate_comprehensive_ai_insights(
                analysis_results, df, business_context, target_column
            )
            
            pdf_bytes = pdf_generator.generate_ai_insights_pdf(
                ai_insights, data_info
            )
            
        else:  # complete analysis
            analysis_results = {
                'summary': summarizer.generate_comprehensive_summary(df),
                'quality': preprocessor.validate_data_quality(df),
                'features': feature_selector.analyze_feature_importance(df, target_column),
                'insights': insight_engine.generate_comprehensive_insights(
                    df, target_column, business_context
                ),
                'visualizations': visualizer.get_recommended_plots(df)
            }
            
            # Generate charts
            charts = {}
            try:
                numeric_cols = df.select_dtypes(include=['number']).columns
                if len(numeric_cols) > 1:
                    corr_result = visualizer.create_correlation_heatmap(df)
                    if corr_result and 'base64_image' in corr_result:
                        charts['correlation_heatmap'] = corr_result['base64_image']
            except Exception as e:
                logger.warning(f"Chart generation failed: {str(e)}")
            
            ai_insights = None
            try:
                ai_insights = await ai_generator.generate_comprehensive_ai_insights(
                    analysis_results, df, business_context, target_column
                )
            except Exception as e:
                logger.warning(f"AI insights failed: {str(e)}")
            
            pdf_bytes = pdf_generator.generate_complete_analysis_pdf(
                analysis_results, data_info, charts, ai_insights
            )
        
        # Return PDF as streaming response
        if isinstance(pdf_bytes, bytes):
            pdf_stream = io.BytesIO(pdf_bytes)
            
            headers = {
                'Content-Type': 'application/pdf',
                'Content-Disposition': f'attachment; filename="datascout_{report_type}_{data_id}.pdf"'
            }
            
            return StreamingResponse(
                io.BytesIO(pdf_bytes),
                media_type="application/pdf",
                headers=headers
            )
        else:
            # Fallback to HTML if PDF generation failed
            return {
                "error": "PDF generation not available",
                "fallback": "HTML report available at /reports/html endpoint",
                "status": "partial_success"
            }
        
    except Exception as e:
        logger.error(f"PDF report generation error: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))


@api_router.get("/data/{data_id}/reports/package")
async def generate_report_package(
    data_id: str,
    target_column: Optional[str] = None,
    business_context: Optional[str] = None
):
    """Generate a complete package of reports in multiple formats as a ZIP file."""
    if data_id not in data_store:
        raise HTTPException(status_code=404, detail="Data not found")
    
    try:
        import shutil
        
        df = data_store[data_id]
        logger.info(f"Starting report package generation for data_id: {data_id}")
        
        # Get basic data info
        data_info = {
            "data_id": data_id,
            "rows": len(df),
            "columns": len(df.columns),
            "generation_time": datetime.now()
        }
        logger.info(f"Data info prepared: {data_info}")
        
        # Generate comprehensive analysis using existing working modules
        logger.info("Generating comprehensive analysis with actual data insights...")
        
        try:
            # Generate comprehensive analysis using proven working modules
            enhanced_results = {
                'summary': convert_numpy_types(summarizer.generate_comprehensive_summary(df)),
                'quality': convert_numpy_types(preprocessor.validate_data_quality(df)),
                'features': convert_numpy_types(feature_selector.analyze_feature_importance(df, target_column)),
                'insights': convert_numpy_types(insight_engine.generate_comprehensive_insights(
                    df, target_column, business_context
                )),
                'visualizations': convert_numpy_types(visualizer.get_recommended_plots(df))
            }
            
            # Add enhanced statistical analysis
            enhanced_results['enhanced_statistics'] = _generate_enhanced_statistics(df)
            enhanced_results['correlation_analysis'] = _generate_correlation_analysis(df)
            enhanced_results['data_quality_detailed'] = _generate_detailed_quality_analysis(df)
            
            logger.info("Comprehensive analysis generated successfully")
            
        except Exception as e:
            logger.error(f"Failed to generate analysis: {str(e)}")
            raise HTTPException(status_code=500, detail=f"Analysis generation failed: {str(e)}")
        
        # Generate charts
        charts = {}
        try:
            logger.info("Generating charts...")
            numeric_cols = df.select_dtypes(include=['number']).columns
            if len(numeric_cols) > 1:
                corr_result = visualizer.create_correlation_heatmap(df)
                if corr_result and 'base64_image' in corr_result:
                    charts['correlation_heatmap'] = corr_result['base64_image']
            logger.info("Charts generated successfully")
        except Exception as e:
            logger.warning(f"Chart generation failed: {str(e)}")
        
        # Get AI insights using enhanced results
        ai_insights = None
        try:
            logger.info("Generating AI insights...")
            ai_insights = await ai_generator.generate_comprehensive_ai_insights(
                enhanced_results, df, business_context, target_column
            )
            logger.info("AI insights generated successfully")
        except Exception as e:
            logger.warning(f"AI insights failed: {str(e)}")
        
        # Create temporary directory for reports
        with tempfile.TemporaryDirectory() as temp_dir:
            # Generate reports
            base_name = f"datascout_report_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
            
            logger.info("Starting report generation...")
            
            # Generate comprehensive HTML report with actual data
            try:
                logger.info("Generating comprehensive HTML report with actual data insights...")
                
                # Create comprehensive HTML content with real analysis data
                complete_html_content = generate_comprehensive_html_report(enhanced_results, data_info, charts, df)
                logger.info("Comprehensive HTML report generated successfully")
                
            except Exception as e:
                logger.error(f"Failed to generate comprehensive HTML report: {str(e)}")
                # Generate a basic but real report if comprehensive fails
                complete_html_content = generate_basic_html_report(enhanced_results, data_info, df)
                logger.info("Basic HTML report generated as fallback")
            except Exception as e:
                logger.error(f"Failed to generate complete analysis HTML: {str(e)}")
                complete_html_content = f"<html><body><h1>Report generation failed: {str(e)}</h1></body></html>"
            
            try:
                logger.info("Generating executive summary with real data...")
                # Generate executive summary with actual data insights
                exec_html_content = generate_executive_summary_report(enhanced_results, data_info, df, data_id)
                logger.info("Executive summary HTML generated successfully")
            except Exception as e:
                logger.error(f"Failed to generate executive summary HTML: {str(e)}")
                exec_html_content = f"<html><body><h1>Executive report generation failed: {str(e)}</h1></body></html>"
            
            # Create comprehensive PDF content with actual data insights
            quality_stats = enhanced_results.get('data_quality_detailed', {})
            insights = enhanced_results.get('insights', {})
            correlations = enhanced_results.get('correlation_analysis', {})
            business_recommendations = insights.get('business_recommendations', [])[:5]
            key_findings = insights.get('key_findings', [])[:5]
            
            complete_pdf_content = f"""DataScout Comprehensive Analysis Report
Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}
Dataset: {data_id}

DATASET OVERVIEW:
- Total Records: {len(df):,}
- Total Features: {len(df.columns)}
- Numeric Features: {len([col for col in df.columns if df[col].dtype in ['int64', 'float64']])}
- Missing Data: {(df.isna().sum().sum() / (len(df) * len(df.columns))) * 100:.1f}%

DATA QUALITY ASSESSMENT:
- Overall Quality Score: {quality_stats.get('overall_quality_score', 0):.1f}%
- Data Completeness: {quality_stats.get('completeness', {}).get('overall_completeness', 0):.1f}%
- Unique Records: {quality_stats.get('consistency', {}).get('unique_rows', 0):,}
- Validity Issues: {quality_stats.get('validity', {}).get('validity_issues', 0)}

CORRELATION ANALYSIS:
- Strong Correlations Found: {correlations.get('total_correlations', 0)}
- Significant Relationships Identified: {len(correlations.get('strong_correlations', []))}

KEY BUSINESS FINDINGS:"""

            for i, finding in enumerate(key_findings):
                complete_pdf_content += f"\n{i+1}. {finding}"

            complete_pdf_content += "\n\nBUSINESS RECOMMENDATIONS:"
            for i, recommendation in enumerate(business_recommendations):
                complete_pdf_content += f"\n{i+1}. {recommendation}"

            complete_pdf_content += f"""

ANALYSIS SUMMARY:
This comprehensive analysis examined {len(df):,} records across {len(df.columns)} features using advanced 
statistical profiling. The analysis identified {len(business_recommendations + key_findings)} actionable 
insights with a data quality score of {quality_stats.get('overall_quality_score', 0):.1f}%.

Generated by DataScout Analytics Platform - Comprehensive Business Intelligence
For interactive visualizations and detailed analysis, refer to the HTML report version.
""".encode('utf-8')
            
            exec_pdf_content = f"""DataScout Executive Summary
Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}
Dataset: {data_id}

EXECUTIVE OVERVIEW:
Our comprehensive analysis examined {len(df):,} records across {len(df.columns)} features, 
revealing key patterns for strategic decision-making.

KEY METRICS:
- Data Quality Score: {quality_stats.get('overall_quality_score', 0):.1f}%
- Missing Data Rate: {(df.isna().sum().sum() / (len(df) * len(df.columns))) * 100:.1f}%
- Strong Correlations: {correlations.get('total_correlations', 0)}
- Actionable Insights: {len(business_recommendations + key_findings)}

TOP STRATEGIC RECOMMENDATIONS:"""

            for i, recommendation in enumerate(business_recommendations[:3]):
                exec_pdf_content += f"\n{i+1}. {recommendation}"

            exec_pdf_content += f"""

IMPLEMENTATION PRIORITY:
Based on this analysis, we recommend prioritizing the top {min(3, len(business_recommendations))} 
strategic recommendations for immediate implementation. The quality score of 
{quality_stats.get('overall_quality_score', 0):.1f}% indicates reliable insights for confident 
business decisions.

Generated by DataScout Analytics Platform - Executive Intelligence Suite
"""
            
            # Create ZIP file in memory
            logger.info("Creating ZIP file...")
            zip_buffer = io.BytesIO()
            
            with zipfile.ZipFile(zip_buffer, 'w', zipfile.ZIP_DEFLATED) as zip_file:
                # Add PDF reports (as simple text files for now)
                zip_file.writestr(f"{base_name}_complete_analysis.txt", complete_pdf_content)
                logger.info("Added complete analysis text to ZIP")
                
                zip_file.writestr(f"{base_name}_executive_summary.txt", exec_pdf_content.encode('utf-8'))
                logger.info("Added executive summary text to ZIP")
                
                # Add HTML reports
                zip_file.writestr(f"{base_name}_complete_analysis.html", complete_html_content.encode('utf-8'))
                logger.info("Added complete analysis HTML to ZIP")
                
                zip_file.writestr(f"{base_name}_executive_summary.html", exec_html_content.encode('utf-8'))
                logger.info("Added executive summary HTML to ZIP")
                
                # Add comprehensive README file with actual analysis summary
                total_insights = len(enhanced_results.get('insights', {}).get('business_recommendations', [])) + len(enhanced_results.get('insights', {}).get('key_findings', []))
                quality_score = enhanced_results.get('data_quality_detailed', {}).get('overall_quality_score', 0)
                correlations_found = enhanced_results.get('correlation_analysis', {}).get('total_correlations', 0)
                missing_percentage = (df.isna().sum().sum() / (len(df) * len(df.columns))) * 100
                
                readme_content = f"""DataScout Analysis Report Package
Generated on: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}
Data ID: {data_id}
Dataset: {len(df):,} rows, {len(df.columns)} columns

📊 ANALYSIS RESULTS SUMMARY:
✅ Data Quality Score: {quality_score:.1f}%
✅ Missing Data Rate: {missing_percentage:.1f}%
✅ Strong Correlations Found: {correlations_found}
✅ Actionable Insights Generated: {total_insights}
✅ Statistical Metrics: {len(df.columns)} features analyzed with comprehensive profiling
✅ Business Intelligence: Context-aware insights and recommendations

📁 Files included:
- {base_name}_complete_analysis.html - Comprehensive analysis report (HTML)
- {base_name}_complete_analysis.txt - Analysis summary (Text)
- {base_name}_executive_summary.html - Executive summary (HTML)
- {base_name}_executive_summary.txt - Executive summary (Text)

📋 Key Findings Summary:"""
                
                # Add actual key findings to README
                key_findings = enhanced_results.get('insights', {}).get('key_findings', [])[:3]
                for i, finding in enumerate(key_findings):
                    readme_content += f"\n• Finding {i+1}: {finding}"
                
                readme_content += "\n\n💡 Top Business Recommendations:"
                business_recs = enhanced_results.get('insights', {}).get('business_recommendations', [])[:3]
                for i, rec in enumerate(business_recs):
                    readme_content += f"\n• Recommendation {i+1}: {rec}"

                readme_content += f"""

🎯 Analysis Quality Indicators:
• Data Completeness: {enhanced_results.get('data_quality_detailed', {}).get('completeness', {}).get('overall_completeness', 0):.1f}%
• Unique Records: {enhanced_results.get('data_quality_detailed', {}).get('consistency', {}).get('unique_rows', 0):,}
• Validity Issues: {enhanced_results.get('data_quality_detailed', {}).get('validity', {}).get('validity_issues', 0)}

📋 Usage Notes:
- HTML files provide interactive analysis with detailed statistics
- Open HTML files in any modern web browser for full functionality
- Text files contain structured summaries with key metrics and insights
- Analysis based on comprehensive statistical profiling of all {len(df.columns)} features

🎯 Business Value:
- Professional-grade reports suitable for stakeholder presentation
- Data-driven insights for strategic decision making
- Comprehensive statistical foundation for advanced analytics
- Quality-assessed data with reliability indicators

Generated by DataScout Analytics Platform - Comprehensive Data Intelligence
Transform your data into actionable business insights with statistical rigor.
"""
                zip_file.writestr("README.txt", readme_content.encode('utf-8'))
                logger.info("Added README to ZIP")
            
            zip_buffer.seek(0)
            logger.info(f"ZIP file created successfully, size: {len(zip_buffer.getvalue())} bytes")
            
            # Return ZIP file as streaming response
            headers = {
                'Content-Type': 'application/zip',
                'Content-Disposition': f'attachment; filename="{data_id}_complete_package.zip"'
            }
            
            return StreamingResponse(
                io.BytesIO(zip_buffer.read()),
                media_type="application/zip",
                headers=headers
            )
         
    except Exception as e:
        logger.error(f"Report package generation error: {str(e)}")
        import traceback
        logger.error(f"Full traceback: {traceback.format_exc()}")
        raise HTTPException(status_code=500, detail=f"Report package generation failed: {str(e)}")
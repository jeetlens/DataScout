"""
Data Visualizer Module for DataScout
Generates various charts and visualizations for data analysis and insights.
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots
import base64
from io import BytesIO
from typing import Dict, List, Any, Optional, Union, Tuple
import logging

logger = logging.getLogger(__name__)

# Set style for matplotlib/seaborn
plt.style.use('default')
sns.set_palette("husl")

class DataVisualizer:
    """
    Data visualization class that generates various charts and plots.
    
    Features:
    - Statistical plots (histograms, box plots, scatter plots)
    - Correlation heatmaps
    - Distribution plots
    - Time series visualizations
    - Interactive plots with Plotly
    - Export to various formats
    """
    
    def __init__(self, style: str = 'plotly_white'):
        self.plotly_template = style
        self.figure_size = (10, 6)
        self.color_palette = px.colors.qualitative.Set3
        
    def create_histogram(self, df: pd.DataFrame, column: str, bins: int = 30, 
                        interactive: bool = True) -> Dict[str, Any]:
        """Create histogram for a numerical column."""
        if column not in df.columns:
            raise ValueError(f"Column '{column}' not found in DataFrame")
            
        if not pd.api.types.is_numeric_dtype(df[column]):
            raise ValueError(f"Column '{column}' is not numerical")
            
        data = df[column].dropna()
        
        if interactive:
            fig = px.histogram(
                x=data, 
                nbins=bins,
                title=f'Distribution of {column}',
                labels={'x': column, 'y': 'Frequency'},
                template=self.plotly_template
            )
            fig.update_layout(showlegend=False)
            
            return {
                'type': 'histogram',
                'column': column,
                'plot_data': fig.to_dict(),
                'statistics': {
                    'mean': round(data.mean(), 4),
                    'median': round(data.median(), 4),
                    'std': round(data.std(), 4),
                    'count': len(data)
                }
            }
        else:
            plt.figure(figsize=self.figure_size)
            plt.hist(data, bins=bins, alpha=0.7, edgecolor='black')
            plt.title(f'Distribution of {column}')
            plt.xlabel(column)
            plt.ylabel('Frequency')
            plt.grid(True, alpha=0.3)
            
            return {
                'type': 'histogram',
                'column': column,
                'plot_data': self._matplotlib_to_base64(),
                'statistics': {
                    'mean': round(data.mean(), 4),
                    'median': round(data.median(), 4),
                    'std': round(data.std(), 4),
                    'count': len(data)
                }
            }
            
    def create_box_plot(self, df: pd.DataFrame, columns: Union[str, List[str]], 
                       interactive: bool = True) -> Dict[str, Any]:
        """Create box plots for numerical columns."""
        if isinstance(columns, str):
            columns = [columns]
            
        # Validate columns
        for col in columns:
            if col not in df.columns:
                raise ValueError(f"Column '{col}' not found in DataFrame")
            if not pd.api.types.is_numeric_dtype(df[col]):
                raise ValueError(f"Column '{col}' is not numerical")
                
        if interactive:
            fig = go.Figure()
            
            for col in columns:
                data = df[col].dropna()
                fig.add_trace(go.Box(
                    y=data,
                    name=col,
                    boxpoints='outliers'
                ))
                
            fig.update_layout(
                title='Box Plot Analysis',
                template=self.plotly_template,
                yaxis_title='Values'
            )
            
            return {
                'type': 'box_plot',
                'columns': columns,
                'plot_data': fig.to_dict()
            }
        else:
            plt.figure(figsize=self.figure_size)
            df[columns].boxplot()
            plt.title('Box Plot Analysis')
            plt.xticks(rotation=45)
            plt.grid(True, alpha=0.3)
            
            return {
                'type': 'box_plot',
                'columns': columns,
                'plot_data': self._matplotlib_to_base64()
            }
            
    def create_scatter_plot(self, df: pd.DataFrame, x_col: str, y_col: str, 
                           color_col: Optional[str] = None, interactive: bool = True) -> Dict[str, Any]:
        """Create scatter plot between two numerical columns."""
        # Validate columns
        for col in [x_col, y_col]:
            if col not in df.columns:
                raise ValueError(f"Column '{col}' not found in DataFrame")
            if not pd.api.types.is_numeric_dtype(df[col]):
                raise ValueError(f"Column '{col}' is not numerical")
                
        if color_col and color_col not in df.columns:
            raise ValueError(f"Color column '{color_col}' not found in DataFrame")
            
        # Remove rows with missing values in key columns
        plot_data = df[[x_col, y_col] + ([color_col] if color_col else [])].dropna()
        
        if interactive:
            fig = px.scatter(
                plot_data,
                x=x_col,
                y=y_col,
                color=color_col,
                title=f'{y_col} vs {x_col}',
                template=self.plotly_template,
                trendline="ols" if not color_col else None
            )
            
            return {
                'type': 'scatter_plot',
                'x_column': x_col,
                'y_column': y_col,
                'color_column': color_col,
                'plot_data': fig.to_dict(),
                'correlation': round(plot_data[x_col].corr(plot_data[y_col]), 4)
            }
        else:
            plt.figure(figsize=self.figure_size)
            if color_col:
                scatter = plt.scatter(plot_data[x_col], plot_data[y_col], 
                                    c=plot_data[color_col], alpha=0.6)
                plt.colorbar(scatter, label=color_col)
            else:
                plt.scatter(plot_data[x_col], plot_data[y_col], alpha=0.6)
                
            plt.xlabel(x_col)
            plt.ylabel(y_col)
            plt.title(f'{y_col} vs {x_col}')
            plt.grid(True, alpha=0.3)
            
            return {
                'type': 'scatter_plot',
                'x_column': x_col,
                'y_column': y_col,
                'color_column': color_col,
                'plot_data': self._matplotlib_to_base64(),
                'correlation': round(plot_data[x_col].corr(plot_data[y_col]), 4)
            }
            
    def create_correlation_heatmap(self, df: pd.DataFrame, interactive: bool = True) -> Dict[str, Any]:
        """Create correlation heatmap for numerical columns."""
        numeric_df = df.select_dtypes(include=[np.number])
        
        if len(numeric_df.columns) < 2:
            raise ValueError("Need at least 2 numerical columns for correlation heatmap")
            
        correlation_matrix = numeric_df.corr()
        
        if interactive:
            fig = px.imshow(
                correlation_matrix,
                title='Correlation Heatmap',
                template=self.plotly_template,
                color_continuous_scale='RdBu_r',
                aspect='auto'
            )
            
            fig.update_layout(
                xaxis_title='Variables',
                yaxis_title='Variables'
            )
            
            return {
                'type': 'correlation_heatmap',
                'plot_data': fig.to_dict(),
                'correlation_matrix': correlation_matrix.round(4).to_dict()
            }
        else:
            plt.figure(figsize=(12, 8))
            sns.heatmap(correlation_matrix, annot=True, cmap='RdBu_r', center=0,
                       square=True, linewidths=0.5)
            plt.title('Correlation Heatmap')
            plt.tight_layout()
            
            return {
                'type': 'correlation_heatmap',
                'plot_data': self._matplotlib_to_base64(),
                'correlation_matrix': correlation_matrix.round(4).to_dict()
            }
            
    def create_distribution_comparison(self, df: pd.DataFrame, columns: List[str], 
                                     interactive: bool = True) -> Dict[str, Any]:
        """Create overlaid distribution plots for comparing multiple columns."""
        # Validate columns
        for col in columns:
            if col not in df.columns:
                raise ValueError(f"Column '{col}' not found in DataFrame")
            if not pd.api.types.is_numeric_dtype(df[col]):
                raise ValueError(f"Column '{col}' is not numerical")
                
        if interactive:
            fig = go.Figure()
            
            for col in columns:
                data = df[col].dropna()
                fig.add_trace(go.Histogram(
                    x=data,
                    name=col,
                    opacity=0.7,
                    nbinsx=30
                ))
                
            fig.update_layout(
                title='Distribution Comparison',
                xaxis_title='Values',
                yaxis_title='Frequency',
                barmode='overlay',
                template=self.plotly_template
            )
            
            return {
                'type': 'distribution_comparison',
                'columns': columns,
                'plot_data': fig.to_dict()
            }
        else:
            plt.figure(figsize=self.figure_size)
            for col in columns:
                data = df[col].dropna()
                plt.hist(data, alpha=0.7, label=col, bins=30)
                
            plt.title('Distribution Comparison')
            plt.xlabel('Values')
            plt.ylabel('Frequency')
            plt.legend()
            plt.grid(True, alpha=0.3)
            
            return {
                'type': 'distribution_comparison',
                'columns': columns,
                'plot_data': self._matplotlib_to_base64()
            }
            
    def create_categorical_bar_chart(self, df: pd.DataFrame, column: str, 
                                   top_n: int = 10, interactive: bool = True) -> Dict[str, Any]:
        """Create bar chart for categorical data."""
        if column not in df.columns:
            raise ValueError(f"Column '{column}' not found in DataFrame")
            
        value_counts = df[column].value_counts().head(top_n)
        
        if interactive:
            fig = px.bar(
                x=value_counts.index,
                y=value_counts.values,
                title=f'Top {top_n} Values in {column}',
                labels={'x': column, 'y': 'Count'},
                template=self.plotly_template
            )
            
            return {
                'type': 'categorical_bar_chart',
                'column': column,
                'plot_data': fig.to_dict(),
                'value_counts': value_counts.to_dict()
            }
        else:
            plt.figure(figsize=self.figure_size)
            value_counts.plot(kind='bar')
            plt.title(f'Top {top_n} Values in {column}')
            plt.xlabel(column)
            plt.ylabel('Count')
            plt.xticks(rotation=45)
            plt.grid(True, alpha=0.3)
            
            return {
                'type': 'categorical_bar_chart',
                'column': column,
                'plot_data': self._matplotlib_to_base64(),
                'value_counts': value_counts.to_dict()
            }
            
    def create_time_series_plot(self, df: pd.DataFrame, date_col: str, value_col: str, 
                               interactive: bool = True) -> Dict[str, Any]:
        """Create time series plot."""
        # Validate columns
        if date_col not in df.columns:
            raise ValueError(f"Date column '{date_col}' not found in DataFrame")
        if value_col not in df.columns:
            raise ValueError(f"Value column '{value_col}' not found in DataFrame")
            
        # Ensure date column is datetime
        if not pd.api.types.is_datetime64_any_dtype(df[date_col]):
            try:
                df[date_col] = pd.to_datetime(df[date_col])
            except:
                raise ValueError(f"Cannot convert '{date_col}' to datetime")
                
        # Sort by date and remove missing values
        plot_data = df[[date_col, value_col]].dropna().sort_values(date_col)
        
        if interactive:
            fig = px.line(
                plot_data,
                x=date_col,
                y=value_col,
                title=f'{value_col} Over Time',
                template=self.plotly_template
            )
            
            return {
                'type': 'time_series',
                'date_column': date_col,
                'value_column': value_col,
                'plot_data': fig.to_dict(),
                'data_points': len(plot_data)
            }
        else:
            plt.figure(figsize=self.figure_size)
            plt.plot(plot_data[date_col], plot_data[value_col])
            plt.title(f'{value_col} Over Time')
            plt.xlabel(date_col)
            plt.ylabel(value_col)
            plt.xticks(rotation=45)
            plt.grid(True, alpha=0.3)
            plt.tight_layout()
            
            return {
                'type': 'time_series',
                'date_column': date_col,
                'value_column': value_col,
                'plot_data': self._matplotlib_to_base64(),
                'data_points': len(plot_data)
            }
            
    def create_multi_plot_dashboard(self, df: pd.DataFrame) -> Dict[str, Any]:
        """Create a dashboard with multiple plots."""
        numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
        categorical_cols = df.select_dtypes(include=['object', 'category']).columns.tolist()
        
        dashboard_plots = {}
        
        # Correlation heatmap if we have numeric columns
        if len(numeric_cols) >= 2:
            try:
                dashboard_plots['correlation_heatmap'] = self.create_correlation_heatmap(df)
            except Exception as e:
                logger.warning(f"Could not create correlation heatmap: {e}")
                
        # Distribution plots for first few numeric columns
        for col in numeric_cols[:3]:
            try:
                dashboard_plots[f'histogram_{col}'] = self.create_histogram(df, col)
            except Exception as e:
                logger.warning(f"Could not create histogram for {col}: {e}")
                
        # Bar charts for first few categorical columns
        for col in categorical_cols[:2]:
            try:
                dashboard_plots[f'bar_chart_{col}'] = self.create_categorical_bar_chart(df, col)
            except Exception as e:
                logger.warning(f"Could not create bar chart for {col}: {e}")
                
        # Scatter plot for first two numeric columns
        if len(numeric_cols) >= 2:
            try:
                dashboard_plots['scatter_plot'] = self.create_scatter_plot(
                    df, numeric_cols[0], numeric_cols[1]
                )
            except Exception as e:
                logger.warning(f"Could not create scatter plot: {e}")
                
        return {
            'type': 'dashboard',
            'plots': dashboard_plots,
            'summary': {
                'total_plots': len(dashboard_plots),
                'numeric_columns': len(numeric_cols),
                'categorical_columns': len(categorical_cols)
            }
        }
        
    def _matplotlib_to_base64(self) -> str:
        """Convert matplotlib figure to base64 string."""
        buffer = BytesIO()
        plt.savefig(buffer, format='png', bbox_inches='tight', dpi=300)
        buffer.seek(0)
        image_base64 = base64.b64encode(buffer.getvalue()).decode()
        plt.close()
        return image_base64
        
    def get_recommended_plots(self, df: pd.DataFrame) -> List[Dict[str, Any]]:
        """Get recommended plots based on data characteristics."""
        recommendations = []
        
        numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
        categorical_cols = df.select_dtypes(include=['object', 'category']).columns.tolist()
        datetime_cols = df.select_dtypes(include=['datetime64']).columns.tolist()
        
        # Histograms for numeric columns
        for col in numeric_cols[:5]:  # Limit to first 5
            recommendations.append({
                'type': 'histogram',
                'columns': [col],
                'reason': f'Shows distribution of {col}',
                'priority': 'high' if df[col].nunique() > 10 else 'medium'
            })
            
        # Correlation heatmap
        if len(numeric_cols) >= 2:
            recommendations.append({
                'type': 'correlation_heatmap',
                'columns': numeric_cols,
                'reason': 'Shows relationships between numeric variables',
                'priority': 'high'
            })
            
        # Scatter plots for highly correlated pairs
        if len(numeric_cols) >= 2:
            corr_matrix = df[numeric_cols].corr()
            for i in range(len(numeric_cols)):
                for j in range(i+1, len(numeric_cols)):
                    if abs(corr_matrix.iloc[i, j]) > 0.5:
                        recommendations.append({
                            'type': 'scatter_plot',
                            'columns': [numeric_cols[i], numeric_cols[j]],
                            'reason': f'High correlation ({corr_matrix.iloc[i, j]:.2f}) between variables',
                            'priority': 'high'
                        })
                        
        # Bar charts for categorical columns
        for col in categorical_cols[:3]:  # Limit to first 3
            unique_count = df[col].nunique()
            if unique_count <= 20:  # Only recommend for low cardinality
                recommendations.append({
                    'type': 'categorical_bar_chart',
                    'columns': [col],
                    'reason': f'Shows distribution of {col} ({unique_count} categories)',
                    'priority': 'medium'
                })
                
        # Time series plots
        for date_col in datetime_cols:
            for value_col in numeric_cols[:2]:  # First 2 numeric columns
                recommendations.append({
                    'type': 'time_series',
                    'columns': [date_col, value_col],
                    'reason': f'Shows trend of {value_col} over time',
                    'priority': 'high'
                })
                
        return recommendations


# Factory function
def create_visualizer(style: str = 'plotly_white') -> DataVisualizer:
    """Create and return a DataVisualizer instance."""
    return DataVisualizer(style)


# Convenience functions
def quick_histogram(df: pd.DataFrame, column: str) -> Dict[str, Any]:
    """Quick histogram generation."""
    visualizer = DataVisualizer()
    return visualizer.create_histogram(df, column)


def quick_correlation_heatmap(df: pd.DataFrame) -> Dict[str, Any]:
    """Quick correlation heatmap generation."""
    visualizer = DataVisualizer()
    return visualizer.create_correlation_heatmap(df)
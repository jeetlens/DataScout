"""
Insight Engine Module for DataScout
Generates automated insights, patterns, and actionable recommendations from data analysis.
"""

import pandas as pd
import numpy as np
from typing import Dict, List, Any, Optional, Union, Tuple
from scipy import stats
import logging
from .summarizer import DataSummarizer
from .visualizer import DataVisualizer
from .feature_selector import FeatureSelector

logger = logging.getLogger(__name__)

class InsightEngine:
    """
    Insight generation engine that analyzes data patterns and generates actionable insights.
    
    Features:
    - Automated pattern detection
    - Statistical anomaly identification
    - Business insight generation
    - Trend analysis
    - Recommendation systems
    - Executive summary creation
    """
    
    def __init__(self):
        self.summarizer = DataSummarizer()
        self.visualizer = DataVisualizer()
        self.feature_selector = FeatureSelector()
        self.insights_cache = {}
        
    def generate_comprehensive_insights(self, df: pd.DataFrame, 
                                      target_col: Optional[str] = None,
                                      business_context: Optional[str] = None) -> Dict[str, Any]:
        """
        Generate comprehensive insights from dataset analysis.
        
        Args:
            df: Input DataFrame
            target_col: Optional target column for supervised insights
            business_context: Optional business context for tailored insights
            
        Returns:
            Dict containing comprehensive insights and recommendations
        """
        insights = {
            'executive_summary': self._generate_executive_summary(df, target_col),
            'data_quality_insights': self._analyze_data_quality_patterns(df),
            'statistical_insights': self._generate_statistical_insights(df, target_col),
            'relationship_insights': self._analyze_relationships(df, target_col),
            'anomaly_detection': self._detect_anomalies(df),
            'trend_analysis': self._analyze_trends(df),
            'business_recommendations': self._generate_business_recommendations(df, target_col, business_context),
            'visualization_recommendations': self._recommend_visualizations(df),
            'next_steps': self._suggest_next_steps(df, target_col)
        }
        
        logger.info(f"Generated comprehensive insights for dataset with {len(df)} rows and {len(df.columns)} columns")
        return insights
        
    def _generate_executive_summary(self, df: pd.DataFrame, target_col: Optional[str] = None) -> Dict[str, Any]:
        """Generate high-level executive summary."""
        summary_stats = self.summarizer.generate_comprehensive_summary(df)
        
        # Key metrics
        total_rows = len(df)
        total_cols = len(df.columns)
        missing_pct = (df.isnull().sum().sum() / (total_rows * total_cols)) * 100
        
        # Numeric vs categorical breakdown
        numeric_cols = len(df.select_dtypes(include=[np.number]).columns)
        categorical_cols = len(df.select_dtypes(include=['object', 'category']).columns)
        
        # Data quality score
        quality_score = 100 - missing_pct - (df.duplicated().sum() / total_rows * 100)
        quality_score = max(0, min(100, quality_score))
        
        summary = {
            'dataset_overview': {
                'total_records': total_rows,
                'total_features': total_cols,
                'numeric_features': numeric_cols,
                'categorical_features': categorical_cols,
                'data_quality_score': round(quality_score, 1)
            },
            'key_findings': [],
            'data_health': self._assess_data_health(df),
            'analysis_readiness': self._assess_analysis_readiness(df, target_col)
        }
        
        # Generate key findings
        if missing_pct > 20:
            summary['key_findings'].append(f"High missing data: {missing_pct:.1f}% of values are missing")
        elif missing_pct < 5:
            summary['key_findings'].append("Excellent data completeness with minimal missing values")
            
        if numeric_cols > 0 and categorical_cols > 0:
            summary['key_findings'].append("Balanced dataset with both numerical and categorical features")
        elif numeric_cols == 0:
            summary['key_findings'].append("Categorical-only dataset - limited statistical analysis possible")
        elif categorical_cols == 0:
            summary['key_findings'].append("Numerical-only dataset - excellent for statistical modeling")
            
        return summary
        
    def _assess_data_health(self, df: pd.DataFrame) -> Dict[str, str]:
        """Assess overall data health."""
        missing_pct = (df.isnull().sum().sum() / (len(df) * len(df.columns))) * 100
        duplicate_pct = (df.duplicated().sum() / len(df)) * 100
        
        health_score = 100 - missing_pct - duplicate_pct * 2
        
        if health_score >= 90:
            health_status = "Excellent"
        elif health_score >= 75:
            health_status = "Good"
        elif health_score >= 60:
            health_status = "Fair"
        else:
            health_status = "Poor"
            
        return {
            'overall_health': health_status,
            'health_score': round(health_score, 1),
            'primary_concerns': self._identify_data_concerns(df)
        }
        
    def _identify_data_concerns(self, df: pd.DataFrame) -> List[str]:
        """Identify primary data quality concerns."""
        concerns = []
        
        missing_pct = (df.isnull().sum().sum() / (len(df) * len(df.columns))) * 100
        if missing_pct > 15:
            concerns.append("High missing data percentage")
            
        duplicate_count = df.duplicated().sum()
        if duplicate_count > len(df) * 0.05:
            concerns.append("Significant duplicate records")
            
        # Check for columns with single values
        single_value_cols = [col for col in df.columns if df[col].nunique() <= 1]
        if single_value_cols:
            concerns.append(f"{len(single_value_cols)} columns with no variance")
            
        # Check for very high cardinality categorical columns
        categorical_cols = df.select_dtypes(include=['object', 'category']).columns
        high_cardinality_cols = [col for col in categorical_cols if df[col].nunique() > len(df) * 0.5]
        if high_cardinality_cols:
            concerns.append(f"{len(high_cardinality_cols)} categorical columns with very high cardinality")
            
        return concerns if concerns else ["No major concerns identified"]
        
    def _assess_analysis_readiness(self, df: pd.DataFrame, target_col: Optional[str] = None) -> Dict[str, Any]:
        """Assess readiness for different types of analysis."""
        readiness = {
            'descriptive_analysis': 'Ready',
            'correlation_analysis': 'Not Ready',
            'predictive_modeling': 'Not Ready',
            'clustering': 'Ready' if len(df.select_dtypes(include=[np.number]).columns) >= 2 else 'Not Ready'
        }
        
        # Correlation analysis readiness
        numeric_cols = len(df.select_dtypes(include=[np.number]).columns)
        if numeric_cols >= 2:
            readiness['correlation_analysis'] = 'Ready'
            
        # Predictive modeling readiness
        if target_col and target_col in df.columns:
            missing_in_target = df[target_col].isnull().sum()
            if missing_in_target < len(df) * 0.1 and numeric_cols >= 2:
                readiness['predictive_modeling'] = 'Ready'
            elif missing_in_target < len(df) * 0.1:
                readiness['predictive_modeling'] = 'Needs Feature Engineering'
                
        return readiness
        
    def _analyze_data_quality_patterns(self, df: pd.DataFrame) -> Dict[str, Any]:
        """Analyze patterns in data quality issues."""
        quality_analysis = self.summarizer._analyze_missing_data(df)
        
        patterns = {
            'missing_data_patterns': quality_analysis.get('missing_data_patterns', {}),
            'completeness_by_column': {},
            'data_type_issues': [],
            'outlier_prevalence': {}
        }
        
        # Completeness by column
        for col in df.columns:
            completeness = (1 - df[col].isnull().sum() / len(df)) * 100
            patterns['completeness_by_column'][col] = round(completeness, 2)
            
        # Check for data type inconsistencies
        for col in df.columns:
            if df[col].dtype == 'object':
                # Check if numeric values are stored as strings
                try:
                    pd.to_numeric(df[col].dropna().head(100), errors='raise')
                    patterns['data_type_issues'].append(f"Column '{col}' appears to contain numeric data stored as text")
                except:
                    pass
                    
        # Outlier prevalence in numeric columns
        numeric_cols = df.select_dtypes(include=[np.number]).columns
        for col in numeric_cols:
            if len(df[col].dropna()) > 0:
                Q1 = df[col].quantile(0.25)
                Q3 = df[col].quantile(0.75)
                IQR = Q3 - Q1
                outliers = df[(df[col] < Q1 - 1.5 * IQR) | (df[col] > Q3 + 1.5 * IQR)][col]
                outlier_pct = len(outliers) / len(df[col].dropna()) * 100
                patterns['outlier_prevalence'][col] = round(outlier_pct, 2)
                
        return patterns
        
    def _generate_statistical_insights(self, df: pd.DataFrame, target_col: Optional[str] = None) -> Dict[str, Any]:
        """Generate insights from statistical analysis."""
        insights = {
            'distribution_insights': {},
            'central_tendency_insights': {},
            'variability_insights': {},
            'normality_insights': {}
        }
        
        numeric_cols = df.select_dtypes(include=[np.number]).columns
        
        for col in numeric_cols:
            data = df[col].dropna()
            if len(data) < 3:
                continue
                
            # Distribution insights
            skewness = data.skew()
            kurtosis = data.kurtosis()
            
            dist_insight = []
            if abs(skewness) < 0.5:
                dist_insight.append("approximately normal distribution")
            elif skewness > 0.5:
                dist_insight.append("right-skewed distribution (tail extends to the right)")
            else:
                dist_insight.append("left-skewed distribution (tail extends to the left)")
                
            if kurtosis > 3:
                dist_insight.append("heavy-tailed (more extreme values than normal)")
            elif kurtosis < -1:
                dist_insight.append("light-tailed (fewer extreme values than normal)")
                
            insights['distribution_insights'][col] = " and ".join(dist_insight)
            
            # Central tendency insights
            mean_val = data.mean()
            median_val = data.median()
            
            if abs(mean_val - median_val) / data.std() > 0.5:
                if mean_val > median_val:
                    insights['central_tendency_insights'][col] = "Mean significantly higher than median, indicating positive skew or outliers"
                else:
                    insights['central_tendency_insights'][col] = "Median significantly higher than mean, indicating negative skew or outliers"
            else:
                insights['central_tendency_insights'][col] = "Mean and median are similar, suggesting symmetric distribution"
                
            # Variability insights
            cv = data.std() / abs(data.mean()) if data.mean() != 0 else float('inf')
            if cv > 1:
                insights['variability_insights'][col] = "High variability relative to mean"
            elif cv < 0.1:
                insights['variability_insights'][col] = "Low variability, values are consistent"
            else:
                insights['variability_insights'][col] = "Moderate variability"
                
        return insights
        
    def _analyze_relationships(self, df: pd.DataFrame, target_col: Optional[str] = None) -> Dict[str, Any]:
        """Analyze relationships between variables."""
        relationship_insights = {
            'correlation_insights': [],
            'feature_importance_insights': [],
            'interaction_insights': []
        }
        
        # Correlation insights
        numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
        if len(numeric_cols) >= 2:
            corr_matrix = df[numeric_cols].corr()
            
            # Find strongest correlations
            strong_correlations = []
            for i in range(len(corr_matrix.columns)):
                for j in range(i+1, len(corr_matrix.columns)):
                    corr_val = corr_matrix.iloc[i, j]
                    if abs(corr_val) > 0.7:
                        strong_correlations.append({
                            'var1': corr_matrix.columns[i],
                            'var2': corr_matrix.columns[j],
                            'correlation': corr_val,
                            'strength': 'strong positive' if corr_val > 0 else 'strong negative'
                        })
                        
            for corr in strong_correlations[:5]:  # Top 5
                relationship_insights['correlation_insights'].append(
                    f"{corr['var1']} and {corr['var2']} show {corr['strength']} correlation ({corr['correlation']:.3f})"
                )
                
        # Feature importance insights (if target is specified)
        if target_col and target_col in df.columns:
            try:
                feature_analysis = self.feature_selector.analyze_feature_importance(df, target_col)
                
                if 'supervised_selection' in feature_analysis:
                    supervised = feature_analysis['supervised_selection']
                    if 'tree_based' in supervised:
                        top_features = list(supervised['tree_based']['top_features'].keys())[:3]
                        relationship_insights['feature_importance_insights'].append(
                            f"Most important features for predicting {target_col}: {', '.join(top_features)}"
                        )
                        
            except Exception as e:
                logger.warning(f"Could not analyze feature importance: {e}")
                
        return relationship_insights
        
    def _detect_anomalies(self, df: pd.DataFrame) -> Dict[str, Any]:
        """Detect various types of anomalies in the data."""
        anomalies = {
            'statistical_outliers': {},
            'pattern_anomalies': [],
            'data_inconsistencies': []
        }
        
        # Statistical outliers
        numeric_cols = df.select_dtypes(include=[np.number]).columns
        for col in numeric_cols:
            data = df[col].dropna()
            if len(data) < 4:
                continue
                
            Q1 = data.quantile(0.25)
            Q3 = data.quantile(0.75)
            IQR = Q3 - Q1
            
            outliers = data[(data < Q1 - 1.5 * IQR) | (data > Q3 + 1.5 * IQR)]
            if len(outliers) > 0:
                anomalies['statistical_outliers'][col] = {
                    'count': len(outliers),
                    'percentage': round(len(outliers) / len(data) * 100, 2),
                    'extreme_values': outliers.nlargest(3).tolist() + outliers.nsmallest(3).tolist()
                }
                
        # Pattern anomalies
        # Check for unusual patterns in categorical data
        categorical_cols = df.select_dtypes(include=['object', 'category']).columns
        for col in categorical_cols:
            value_counts = df[col].value_counts()
            if len(value_counts) > 1:
                # Check if one category dominates (>95%)
                if value_counts.iloc[0] / len(df) > 0.95:
                    anomalies['pattern_anomalies'].append(
                        f"Column '{col}' is dominated by single value: '{value_counts.index[0]}' ({value_counts.iloc[0]/len(df)*100:.1f}%)"
                    )
                    
        # Data inconsistencies
        for col in df.columns:
            if df[col].dtype == 'object':
                # Check for mixed case inconsistencies
                text_values = df[col].dropna().astype(str)
                if len(text_values) > 0:
                    unique_lower = text_values.str.lower().nunique()
                    unique_original = text_values.nunique()
                    if unique_lower < unique_original:
                        anomalies['data_inconsistencies'].append(
                            f"Column '{col}' has case inconsistencies (e.g., 'Apple' vs 'apple')"
                        )
                        
        return anomalies
        
    def _analyze_trends(self, df: pd.DataFrame) -> Dict[str, Any]:
        """Analyze temporal and sequential trends in the data."""
        trends = {
            'temporal_trends': {},
            'sequential_patterns': {},
            'seasonality_insights': []
        }
        
        # Look for datetime columns
        datetime_cols = df.select_dtypes(include=['datetime64']).columns
        numeric_cols = df.select_dtypes(include=[np.number]).columns
        
        for date_col in datetime_cols:
            for numeric_col in numeric_cols[:3]:  # Analyze first 3 numeric columns
                try:
                    # Sort by date and analyze trend
                    sorted_data = df[[date_col, numeric_col]].dropna().sort_values(date_col)
                    if len(sorted_data) < 3:
                        continue
                        
                    # Simple trend analysis using correlation with time
                    time_numeric = pd.to_numeric(sorted_data[date_col])
                    correlation = np.corrcoef(time_numeric, sorted_data[numeric_col])[0, 1]
                    
                    if abs(correlation) > 0.3:
                        trend_direction = "increasing" if correlation > 0 else "decreasing"
                        trends['temporal_trends'][f"{numeric_col}_over_{date_col}"] = {
                            'trend': trend_direction,
                            'strength': abs(correlation),
                            'insight': f"{numeric_col} shows {trend_direction} trend over time (correlation: {correlation:.3f})"
                        }
                        
                except Exception as e:
                    logger.warning(f"Could not analyze trend for {numeric_col} over {date_col}: {e}")
                    
        return trends
        
    def _generate_business_recommendations(self, df: pd.DataFrame, target_col: Optional[str] = None, 
                                         business_context: Optional[str] = None) -> List[str]:
        """Generate actionable business recommendations."""
        recommendations = []
        
        # Data quality recommendations
        missing_pct = (df.isnull().sum().sum() / (len(df) * len(df.columns))) * 100
        if missing_pct > 15:
            recommendations.append("Priority: Implement data collection improvements to reduce missing values")
            
        # Analysis recommendations
        numeric_cols = len(df.select_dtypes(include=[np.number]).columns)
        categorical_cols = len(df.select_dtypes(include=['object', 'category']).columns)
        
        if numeric_cols >= 3:
            recommendations.append("Opportunity: Rich numerical data enables advanced analytics and machine learning")
            
        if categorical_cols >= 2:
            recommendations.append("Insight: Multiple categorical variables allow for detailed segmentation analysis")
            
        # Sample size recommendations
        if len(df) < 100:
            recommendations.append("Caution: Small sample size may limit statistical significance of findings")
        elif len(df) > 10000:
            recommendations.append("Advantage: Large dataset enables robust statistical analysis and modeling")
            
        # Feature engineering recommendations
        if target_col and target_col in df.columns:
            recommendations.append(f"Next step: Consider feature engineering to improve {target_col} prediction accuracy")
            
        # Business context specific recommendations
        if business_context:
            if 'sales' in business_context.lower():
                recommendations.append("Business focus: Analyze seasonal patterns and customer segments for sales optimization")
            elif 'customer' in business_context.lower():
                recommendations.append("Business focus: Segment customers based on behavior patterns for targeted marketing")
            elif 'financial' in business_context.lower():
                recommendations.append("Business focus: Monitor key financial ratios and identify risk indicators")
                
        return recommendations
        
    def _recommend_visualizations(self, df: pd.DataFrame) -> List[Dict[str, str]]:
        """Recommend specific visualizations based on data characteristics."""
        viz_recommendations = self.visualizer.get_recommended_plots(df)
        
        # Enhance with business insights
        enhanced_recommendations = []
        for rec in viz_recommendations[:5]:  # Top 5 recommendations
            enhanced_rec = rec.copy()
            
            if rec['type'] == 'correlation_heatmap':
                enhanced_rec['business_value'] = "Identify which variables move together for strategic planning"
            elif rec['type'] == 'histogram':
                enhanced_rec['business_value'] = "Understand distribution patterns for quality control and benchmarking"
            elif rec['type'] == 'scatter_plot':
                enhanced_rec['business_value'] = "Discover relationships between key metrics for optimization"
            elif rec['type'] == 'time_series':
                enhanced_rec['business_value'] = "Track performance over time to identify trends and seasonality"
                
            enhanced_recommendations.append(enhanced_rec)
            
        return enhanced_recommendations
        
    def _suggest_next_steps(self, df: pd.DataFrame, target_col: Optional[str] = None) -> List[str]:
        """Suggest next analytical steps based on current data state."""
        next_steps = []
        
        # Data preparation steps
        missing_pct = (df.isnull().sum().sum() / (len(df) * len(df.columns))) * 100
        if missing_pct > 10:
            next_steps.append("1. Address missing data through imputation or collection strategies")
            
        if df.duplicated().sum() > 0:
            next_steps.append("2. Remove or investigate duplicate records")
            
        # Analysis steps
        numeric_cols = len(df.select_dtypes(include=[np.number]).columns)
        if numeric_cols >= 2:
            next_steps.append("3. Perform detailed correlation analysis to identify key relationships")
            
        if target_col:
            next_steps.append("4. Build predictive models to forecast target variable")
            next_steps.append("5. Conduct feature importance analysis for business insights")
        else:
            next_steps.append("4. Consider clustering analysis to identify natural data segments")
            
        # Advanced analysis steps
        if len(df) > 1000 and numeric_cols >= 3:
            next_steps.append("6. Explore advanced analytics: PCA, anomaly detection, or time series analysis")
            
        next_steps.append("7. Create interactive dashboards for ongoing monitoring")
        next_steps.append("8. Establish data quality monitoring and alerting systems")
        
        return next_steps
        
    def generate_insight_report(self, df: pd.DataFrame, target_col: Optional[str] = None, 
                              business_context: Optional[str] = None, title: str = "Data Insights Report") -> Dict[str, Any]:
        """Generate a comprehensive insight report."""
        insights = self.generate_comprehensive_insights(df, target_col, business_context)
        
        report = {
            'title': title,
            'generated_at': pd.Timestamp.now().strftime('%Y-%m-%d %H:%M:%S'),
            'executive_summary': insights['executive_summary'],
            'key_insights': self._extract_key_insights(insights),
            'recommendations': insights['business_recommendations'],
            'next_steps': insights['next_steps'],
            'detailed_analysis': {
                'data_quality': insights['data_quality_insights'],
                'statistical_findings': insights['statistical_insights'],
                'relationships': insights['relationship_insights'],
                'anomalies': insights['anomaly_detection']
            },
            'visualization_plan': insights['visualization_recommendations']
        }
        
        return report
        
    def _extract_key_insights(self, full_insights: Dict[str, Any]) -> List[str]:
        """Extract the most important insights for executive summary."""
        key_insights = []
        
        # Data health insight
        health_status = full_insights['executive_summary']['data_health']['overall_health']
        key_insights.append(f"Data quality assessment: {health_status}")
        
        # Statistical insights
        if full_insights['statistical_insights']['distribution_insights']:
            key_insights.append("Distribution analysis reveals data patterns for informed decision-making")
            
        # Relationship insights
        if full_insights['relationship_insights']['correlation_insights']:
            key_insights.append("Strong correlations identified between key variables")
            
        # Anomaly insights
        outlier_cols = list(full_insights['anomaly_detection']['statistical_outliers'].keys())
        if outlier_cols:
            key_insights.append(f"Outliers detected in {len(outlier_cols)} variables requiring investigation")
            
        return key_insights


# Factory function
def create_insight_engine() -> InsightEngine:
    """Create and return an InsightEngine instance."""
    return InsightEngine()


# Convenience functions
def quick_insights(df: pd.DataFrame, target_col: Optional[str] = None) -> Dict[str, Any]:
    """Generate quick insights from dataset."""
    engine = InsightEngine()
    return engine.generate_comprehensive_insights(df, target_col)


def generate_insight_report(df: pd.DataFrame, target_col: Optional[str] = None, 
                          title: str = "Data Analysis Report") -> Dict[str, Any]:
    """Generate comprehensive insight report."""
    engine = InsightEngine()
    return engine.generate_insight_report(df, target_col, title=title)
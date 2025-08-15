"""
AI Prompt Templates System for DataScout
Manages structured prompts for different types of AI-powered data analysis insights.
"""

from typing import Dict, List, Any, Optional, Union
from dataclasses import dataclass
from enum import Enum
import pandas as pd
import json
import logging

logger = logging.getLogger(__name__)

class PromptType(Enum):
    """Types of prompts available."""
    EXECUTIVE_SUMMARY = "executive_summary"
    DATA_INSIGHTS = "data_insights"
    BUSINESS_RECOMMENDATIONS = "business_recommendations"
    STATISTICAL_ANALYSIS = "statistical_analysis"
    ANOMALY_EXPLANATION = "anomaly_explanation"
    TREND_ANALYSIS = "trend_analysis"
    CORRELATION_INSIGHTS = "correlation_insights"
    FEATURE_IMPORTANCE = "feature_importance"
    DATA_QUALITY_ASSESSMENT = "data_quality_assessment"
    PREDICTIVE_INSIGHTS = "predictive_insights"

class BusinessDomain(Enum):
    """Business domains for context-aware prompts."""
    GENERAL = "general"
    SALES = "sales"
    MARKETING = "marketing"
    FINANCE = "finance"
    OPERATIONS = "operations"
    CUSTOMER_SERVICE = "customer_service"
    HR = "hr"
    SUPPLY_CHAIN = "supply_chain"
    HEALTHCARE = "healthcare"
    ECOMMERCE = "ecommerce"

@dataclass
class PromptTemplate:
    """Structure for a prompt template."""
    name: str
    prompt_type: PromptType
    template: str
    variables: List[str]
    business_domain: BusinessDomain = BusinessDomain.GENERAL
    version: str = "1.0"
    description: str = ""

class PromptTemplateManager:
    """
    Manages AI prompt templates for different analysis scenarios.
    
    Features:
    - Context-aware prompt generation
    - Business domain-specific templates
    - Dynamic variable substitution
    - Template versioning and management
    - Prompt optimization and testing
    """
    
    def __init__(self):
        self.templates = {}
        self._initialize_default_templates()
        
    def _initialize_default_templates(self):
        """Initialize default prompt templates."""
        
        # Executive Summary Template
        self.add_template(PromptTemplate(
            name="executive_summary_general",
            prompt_type=PromptType.EXECUTIVE_SUMMARY,
            template="""
As a senior data analyst, provide an executive summary of this dataset analysis:

**Dataset Overview:**
- Total Records: {total_rows:,}
- Features: {total_columns}
- Data Quality Score: {quality_score}/100
- Missing Data: {missing_percentage:.1f}%

**Key Statistical Findings:**
{statistical_summary}

**Data Quality Issues:**
{quality_issues}

**Top Insights:**
{top_insights}

Please provide:
1. A 2-3 sentence executive summary highlighting the most important findings
2. 3-5 key business insights derived from the data
3. Critical data quality concerns that need attention
4. Recommended next steps for analysis or action

Focus on actionable insights that would be valuable to business decision-makers.
""",
            variables=["total_rows", "total_columns", "quality_score", "missing_percentage", 
                      "statistical_summary", "quality_issues", "top_insights"],
            business_domain=BusinessDomain.GENERAL,
            description="General executive summary for any dataset"
        ))
        
        # Data Insights Template
        self.add_template(PromptTemplate(
            name="data_insights_detailed",
            prompt_type=PromptType.DATA_INSIGHTS,
            template="""
Analyze the following dataset characteristics and provide detailed insights:

**Dataset Profile:**
- Rows: {total_rows:,}, Columns: {total_columns}
- Numeric Features: {numeric_columns}
- Categorical Features: {categorical_columns}

**Statistical Summary:**
{descriptive_stats}

**Correlation Analysis:**
{correlation_findings}

**Distribution Patterns:**
{distribution_analysis}

**Anomalies Detected:**
{anomaly_summary}

Based on this analysis, please provide:

1. **Pattern Recognition**: What interesting patterns do you observe in the data?
2. **Statistical Significance**: Which findings are statistically significant and actionable?
3. **Data Relationships**: What are the strongest relationships between variables?
4. **Outlier Impact**: How do anomalies affect the overall analysis?
5. **Business Implications**: What do these patterns suggest for business strategy?

Provide specific, data-driven insights with quantitative support where possible.
""",
            variables=["total_rows", "total_columns", "numeric_columns", "categorical_columns",
                      "descriptive_stats", "correlation_findings", "distribution_analysis", "anomaly_summary"],
            business_domain=BusinessDomain.GENERAL,
            description="Detailed data insights analysis"
        ))
        
        # Business Recommendations Template
        self.add_template(PromptTemplate(
            name="business_recommendations_sales",
            prompt_type=PromptType.BUSINESS_RECOMMENDATIONS,
            template="""
Based on the sales data analysis, provide actionable business recommendations:

**Sales Performance Summary:**
{performance_summary}

**Key Metrics:**
{key_metrics}

**Trend Analysis:**
{trend_analysis}

**Customer Segmentation:**
{customer_segments}

**Seasonal Patterns:**
{seasonal_patterns}

Please provide specific, actionable recommendations for:

1. **Revenue Optimization**: How can we increase sales based on the data patterns?
2. **Customer Targeting**: Which customer segments should we focus on and why?
3. **Product Performance**: What products/services need attention?
4. **Seasonal Strategy**: How should we adjust strategy based on seasonal trends?
5. **Risk Mitigation**: What potential issues should we address proactively?

Each recommendation should include:
- Specific action to take
- Expected impact/outcome
- Priority level (High/Medium/Low)
- Resources needed for implementation
""",
            variables=["performance_summary", "key_metrics", "trend_analysis", 
                      "customer_segments", "seasonal_patterns"],
            business_domain=BusinessDomain.SALES,
            description="Sales-focused business recommendations"
        ))
        
        # Anomaly Explanation Template
        self.add_template(PromptTemplate(
            name="anomaly_explanation",
            prompt_type=PromptType.ANOMALY_EXPLANATION,
            template="""
Explain the anomalies detected in the dataset and their potential business impact:

**Anomalies Summary:**
{anomaly_details}

**Statistical Context:**
- Dataset size: {total_rows:,} records
- Anomaly count: {anomaly_count}
- Percentage of data: {anomaly_percentage:.2f}%

**Anomaly Breakdown by Feature:**
{feature_anomalies}

**Potential Patterns:**
{anomaly_patterns}

Please analyze and explain:

1. **Root Cause Analysis**: What might be causing these anomalies?
2. **Business Impact**: How do these anomalies affect business operations or insights?
3. **Data Quality**: Are these legitimate outliers or data quality issues?
4. **Action Required**: Should these anomalies be investigated, corrected, or excluded?
5. **Monitoring**: How can we detect similar anomalies in the future?

Provide both technical explanations and business-friendly interpretations.
""",
            variables=["anomaly_details", "total_rows", "anomaly_count", 
                      "anomaly_percentage", "feature_anomalies", "anomaly_patterns"],
            business_domain=BusinessDomain.GENERAL,
            description="Anomaly detection and explanation"
        ))
        
        # Correlation Insights Template
        self.add_template(PromptTemplate(
            name="correlation_insights",
            prompt_type=PromptType.CORRELATION_INSIGHTS,
            template="""
Analyze the correlation patterns in the dataset and provide insights:

**Correlation Matrix Summary:**
{correlation_matrix}

**Strong Correlations Found:**
{strong_correlations}

**Feature Relationships:**
{feature_relationships}

**Statistical Significance:**
{significance_tests}

Based on the correlation analysis, please explain:

1. **Relationship Strength**: Which variables are most strongly related and why?
2. **Causation vs Correlation**: What relationships might indicate causation vs mere correlation?
3. **Business Logic**: Do the correlations align with business understanding?
4. **Multicollinearity**: Are there redundant variables that could be removed?
5. **Predictive Value**: Which correlations are most useful for prediction or decision-making?

Provide insights that bridge statistical findings with business understanding.
""",
            variables=["correlation_matrix", "strong_correlations", 
                      "feature_relationships", "significance_tests"],
            business_domain=BusinessDomain.GENERAL,
            description="Correlation analysis and insights"
        ))
        
        # Trend Analysis Template
        self.add_template(PromptTemplate(
            name="trend_analysis_time_series",
            prompt_type=PromptType.TREND_ANALYSIS,
            template="""
Analyze the time-based trends in the dataset:

**Time Period:** {time_period}
**Data Points:** {data_points:,}

**Trend Summary:**
{trend_summary}

**Seasonal Patterns:**
{seasonal_patterns}

**Growth Rates:**
{growth_rates}

**Volatility Analysis:**
{volatility_analysis}

**Forecast Indicators:**
{forecast_indicators}

Please provide insights on:

1. **Trend Direction**: What is the overall trend (growth, decline, stable)?
2. **Seasonality**: What seasonal patterns are evident and what drives them?
3. **Turning Points**: Are there significant changes in trend direction?
4. **Future Outlook**: Based on current trends, what can we expect?
5. **Business Drivers**: What business factors likely influence these trends?
6. **Risk Factors**: What trends pose risks or opportunities?

Focus on actionable insights for business planning and strategy.
""",
            variables=["time_period", "data_points", "trend_summary", "seasonal_patterns",
                      "growth_rates", "volatility_analysis", "forecast_indicators"],
            business_domain=BusinessDomain.GENERAL,
            description="Time series trend analysis"
        ))
        
        # Feature Importance Template
        self.add_template(PromptTemplate(
            name="feature_importance_ml",
            prompt_type=PromptType.FEATURE_IMPORTANCE,
            template="""
Explain the feature importance analysis results:

**Target Variable:** {target_variable}
**Total Features Analyzed:** {total_features}

**Top Important Features:**
{top_features}

**Feature Importance Scores:**
{importance_scores}

**Correlation with Target:**
{target_correlations}

**Feature Selection Results:**
{selection_results}

Please explain:

1. **Key Drivers**: Which features most strongly influence the target variable?
2. **Business Meaning**: What do these important features represent in business terms?
3. **Unexpected Findings**: Are there surprising features in the top rankings?
4. **Feature Engineering**: What new features could be created from existing ones?
5. **Data Collection**: Should we collect additional data for underrepresented important factors?
6. **Model Implications**: How do these results guide model development?

Provide insights that help prioritize data collection and business focus areas.
""",
            variables=["target_variable", "total_features", "top_features", 
                      "importance_scores", "target_correlations", "selection_results"],
            business_domain=BusinessDomain.GENERAL,
            description="Feature importance analysis for ML"
        ))
        
        logger.info(f"Initialized {len(self.templates)} default prompt templates")
        
    def add_template(self, template: PromptTemplate):
        """Add a new prompt template."""
        key = f"{template.prompt_type.value}_{template.business_domain.value}_{template.name}"
        self.templates[key] = template
        logger.debug(f"Added template: {key}")
        
    def get_template(self, prompt_type: PromptType, 
                    business_domain: BusinessDomain = BusinessDomain.GENERAL,
                    template_name: Optional[str] = None) -> Optional[PromptTemplate]:
        """Get a specific template."""
        if template_name:
            key = f"{prompt_type.value}_{business_domain.value}_{template_name}"
            return self.templates.get(key)
        
        # Find any template matching type and domain
        for key, template in self.templates.items():
            if (template.prompt_type == prompt_type and 
                template.business_domain == business_domain):
                return template
                
        # Fallback to general domain
        if business_domain != BusinessDomain.GENERAL:
            return self.get_template(prompt_type, BusinessDomain.GENERAL, template_name)
            
        return None
        
    def generate_prompt(self, prompt_type: PromptType, 
                       context_data: Dict[str, Any],
                       business_domain: BusinessDomain = BusinessDomain.GENERAL,
                       template_name: Optional[str] = None) -> Optional[str]:
        """Generate a prompt with context data."""
        template = self.get_template(prompt_type, business_domain, template_name)
        if not template:
            logger.error(f"No template found for {prompt_type.value} in {business_domain.value}")
            return None
            
        try:
            # Prepare context data with safe formatting
            safe_context = self._prepare_context_data(context_data, template.variables)
            
            # Generate the prompt
            prompt = template.template.format(**safe_context)
            
            logger.info(f"Generated prompt for {template.name}")
            return prompt.strip()
            
        except KeyError as e:
            logger.error(f"Missing variable {e} for template {template.name}")
            return None
        except Exception as e:
            logger.error(f"Error generating prompt: {str(e)}")
            return None
            
    def _prepare_context_data(self, context_data: Dict[str, Any], 
                            required_variables: List[str]) -> Dict[str, Any]:
        """Prepare and validate context data for prompt generation."""
        safe_context = {}
        
        for var in required_variables:
            if var in context_data:
                value = context_data[var]
                
                # Handle different data types safely
                if isinstance(value, (dict, list)):
                    safe_context[var] = self._format_complex_data(value)
                elif isinstance(value, float):
                    safe_context[var] = f"{value:.2f}" if not pd.isna(value) else "N/A"
                elif value is None or (isinstance(value, float) and pd.isna(value)):
                    safe_context[var] = "N/A"
                else:
                    safe_context[var] = str(value)
            else:
                safe_context[var] = "N/A"
                logger.warning(f"Missing context variable: {var}")
                
        return safe_context
        
    def _format_complex_data(self, data: Union[Dict, List]) -> str:
        """Format complex data structures for prompt inclusion."""
        if isinstance(data, dict):
            if len(data) <= 5:
                return "\n".join([f"- {k}: {v}" for k, v in data.items()])
            else:
                items = list(data.items())[:5]
                formatted = "\n".join([f"- {k}: {v}" for k, v in items])
                return f"{formatted}\n- ... and {len(data) - 5} more items"
                
        elif isinstance(data, list):
            if len(data) <= 5:
                return "\n".join([f"- {item}" for item in data])
            else:
                formatted = "\n".join([f"- {item}" for item in data[:5]])
                return f"{formatted}\n- ... and {len(data) - 5} more items"
                
        return str(data)
        
    def list_templates(self, prompt_type: Optional[PromptType] = None,
                      business_domain: Optional[BusinessDomain] = None) -> List[Dict[str, Any]]:
        """List available templates with optional filtering."""
        templates = []
        
        for key, template in self.templates.items():
            if prompt_type and template.prompt_type != prompt_type:
                continue
            if business_domain and template.business_domain != business_domain:
                continue
                
            templates.append({
                "key": key,
                "name": template.name,
                "type": template.prompt_type.value,
                "domain": template.business_domain.value,
                "version": template.version,
                "description": template.description,
                "variables": template.variables
            })
            
        return templates
        
    def create_context_from_analysis(self, analysis_results: Dict[str, Any],
                                   df: Optional[pd.DataFrame] = None) -> Dict[str, Any]:
        """Create context data from analysis results for prompt generation."""
        context = {}
        
        # Basic dataset info
        if df is not None:
            context.update({
                "total_rows": len(df),
                "total_columns": len(df.columns),
                "numeric_columns": len(df.select_dtypes(include=['number']).columns),
                "categorical_columns": len(df.select_dtypes(include=['object', 'category']).columns)
            })
            
        # Extract data from analysis results
        if "basic_info" in analysis_results:
            basic_info = analysis_results["basic_info"]
            context.update({
                "total_rows": basic_info.get("total_rows", "N/A"),
                "total_columns": basic_info.get("total_columns", "N/A"),
                "numeric_columns": len(basic_info.get("numeric_columns", [])),
                "categorical_columns": len(basic_info.get("categorical_columns", []))
            })
            
        # Quality information
        if "data_quality_insights" in analysis_results:
            quality = analysis_results["data_quality_insights"]
            context.update({
                "quality_score": quality.get("quality_score", "N/A"),
                "missing_percentage": quality.get("missing_percentage_overall", "N/A"),
                "quality_issues": quality.get("primary_concerns", [])
            })
            
        # Statistical summary
        if "descriptive_stats" in analysis_results:
            context["descriptive_stats"] = analysis_results["descriptive_stats"]
            
        # Correlation findings
        if "correlation_analysis" in analysis_results:
            corr_analysis = analysis_results["correlation_analysis"]
            context.update({
                "correlation_matrix": corr_analysis.get("correlation_matrix", {}),
                "strong_correlations": corr_analysis.get("strong_correlations", []),
                "correlation_findings": corr_analysis.get("strong_correlations", [])
            })
            
        # Anomaly information
        if "anomaly_detection" in analysis_results:
            anomalies = analysis_results["anomaly_detection"]
            outliers = anomalies.get("statistical_outliers", {})
            total_anomalies = sum([details.get("count", 0) for details in outliers.values()])
            
            context.update({
                "anomaly_details": anomalies,
                "anomaly_count": total_anomalies,
                "anomaly_percentage": (total_anomalies / context.get("total_rows", 1)) * 100 if context.get("total_rows") else 0,
                "feature_anomalies": outliers,
                "anomaly_summary": anomalies.get("pattern_anomalies", [])
            })
            
        # Feature importance
        if "feature_rankings" in analysis_results:
            context.update({
                "importance_scores": analysis_results["feature_rankings"],
                "top_features": analysis_results.get("best_features", {})
            })
            
        return context
        
    def get_domain_specific_prompt(self, business_context: str) -> BusinessDomain:
        """Determine business domain from context string."""
        context_lower = business_context.lower()
        
        domain_keywords = {
            BusinessDomain.SALES: ["sales", "revenue", "customer", "purchase", "order"],
            BusinessDomain.MARKETING: ["marketing", "campaign", "advertising", "promotion", "lead"],
            BusinessDomain.FINANCE: ["finance", "financial", "budget", "cost", "profit", "expense"],
            BusinessDomain.OPERATIONS: ["operations", "production", "manufacturing", "supply", "inventory"],
            BusinessDomain.HR: ["hr", "human resources", "employee", "staff", "personnel"],
            BusinessDomain.HEALTHCARE: ["healthcare", "medical", "patient", "treatment", "clinical"],
            BusinessDomain.ECOMMERCE: ["ecommerce", "online", "website", "digital", "cart"]
        }
        
        for domain, keywords in domain_keywords.items():
            if any(keyword in context_lower for keyword in keywords):
                return domain
                
        return BusinessDomain.GENERAL


# Factory function
def create_prompt_manager() -> PromptTemplateManager:
    """Create and return a PromptTemplateManager instance."""
    return PromptTemplateManager()


# Convenience functions
def generate_executive_summary_prompt(analysis_results: Dict[str, Any], 
                                    df: Optional[pd.DataFrame] = None,
                                    business_context: Optional[str] = None) -> Optional[str]:
    """Generate executive summary prompt from analysis results."""
    manager = create_prompt_manager()
    context = manager.create_context_from_analysis(analysis_results, df)
    
    domain = BusinessDomain.GENERAL
    if business_context:
        domain = manager.get_domain_specific_prompt(business_context)
        
    return manager.generate_prompt(PromptType.EXECUTIVE_SUMMARY, context, domain)


def generate_insights_prompt(analysis_results: Dict[str, Any], 
                           df: Optional[pd.DataFrame] = None) -> Optional[str]:
    """Generate detailed insights prompt from analysis results."""
    manager = create_prompt_manager()
    context = manager.create_context_from_analysis(analysis_results, df)
    return manager.generate_prompt(PromptType.DATA_INSIGHTS, context)
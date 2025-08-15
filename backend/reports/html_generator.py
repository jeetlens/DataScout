"""
HTML Report Generator for DataScout
Generates comprehensive HTML reports from analysis results.
"""

import os
import base64
from typing import Dict, List, Any, Optional, Union
import pandas as pd
from datetime import datetime
import logging

from .report_templates import (
    ReportTemplate, ReportType, ReportStyle, get_template
)

logger = logging.getLogger(__name__)


class HTMLReportGenerator:
    """
    HTML report generator for DataScout analysis results.
    
    Features:
    - Multiple report templates and styles
    - Chart and visualization embedding
    - AI insights integration
    - Responsive design
    - Professional formatting
    """
    
    def __init__(self, template: Optional[ReportTemplate] = None):
        """Initialize HTML report generator."""
        self.template = template or get_template()
        
    def generate_complete_analysis_report(self, 
                                        analysis_results: Dict[str, Any],
                                        data_info: Dict[str, Any],
                                        charts: Optional[Dict[str, str]] = None,
                                        ai_insights: Optional[Dict[str, Any]] = None) -> str:
        """
        Generate a complete analysis report with all sections.
        
        Args:
            analysis_results: Complete analysis results from API
            data_info: Basic dataset information
            charts: Dictionary of chart names to base64 encoded images
            ai_insights: AI-generated insights
            
        Returns:
            Complete HTML report as string
        """
        logger.info("Generating complete analysis report")
        
        # Generate report sections
        sections = []
        
        # Header
        title = f"Complete Data Analysis Report"
        subtitle = f"Dataset: {data_info.get('data_id', 'Unknown')}"
        header = self.template.generate_header(title, subtitle)
        sections.append(header)
        
        # Meta information
        meta_info = self.template.generate_meta_info(data_info)
        sections.append(meta_info)
        
        # Executive Summary
        if ai_insights and 'executive_summary' in ai_insights.get('insights', {}):
            exec_content = self._generate_executive_summary_section(ai_insights['insights']['executive_summary'])
            sections.append(self.template.generate_section(
                "Executive Summary", exec_content, "📋"
            ))
        
        # Data Quality Assessment
        if 'quality' in analysis_results:
            quality_content = self._generate_quality_section(analysis_results['quality'])
            sections.append(self.template.generate_section(
                "Data Quality Assessment", quality_content, "🔍"
            ))
        
        # Statistical Summary
        if 'summary' in analysis_results:
            summary_content = self._generate_summary_section(analysis_results['summary'])
            sections.append(self.template.generate_section(
                "Statistical Analysis", summary_content, "📊"
            ))
        
        # Visualizations
        if charts:
            viz_content = self._generate_visualization_section(charts)
            sections.append(self.template.generate_section(
                "Data Visualizations", viz_content, "📈"
            ))
        
        # Feature Analysis
        if 'features' in analysis_results:
            feature_content = self._generate_feature_section(analysis_results['features'])
            sections.append(self.template.generate_section(
                "Feature Analysis", feature_content, "🎯"
            ))
        
        # Business Insights
        if 'insights' in analysis_results:
            insights_content = self._generate_insights_section(analysis_results['insights'])
            sections.append(self.template.generate_section(
                "Business Insights", insights_content, "💡"
            ))
        
        # AI-Powered Insights
        if ai_insights:
            ai_content = self._generate_ai_insights_section(ai_insights)
            sections.append(self.template.generate_section(
                "AI-Powered Insights", ai_content, "🤖"
            ))
        
        # Recommendations
        recommendations = self._extract_recommendations(analysis_results, ai_insights)
        if recommendations:
            rec_content = self.template.generate_recommendations(recommendations)
            sections.append(self.template.generate_section(
                "Recommendations", rec_content, "🎯"
            ))
        
        # Footer
        footer = self.template.generate_footer(
            f"Analysis completed for {data_info.get('rows', 0):,} records with {data_info.get('columns', 0)} features"
        )
        sections.append(footer)
        
        # Combine all sections and wrap in HTML
        content = "".join(sections)
        return self.template.wrap_html(content, title)
    
    def generate_executive_summary_report(self,
                                        analysis_results: Dict[str, Any],
                                        data_info: Dict[str, Any],
                                        ai_insights: Optional[Dict[str, Any]] = None) -> str:
        """Generate an executive summary report."""
        logger.info("Generating executive summary report")
        
        sections = []
        
        # Header
        title = "Executive Summary Report"
        subtitle = f"Dataset: {data_info.get('data_id', 'Unknown')}"
        header = self.template.generate_header(title, subtitle)
        sections.append(header)
        
        # Meta information
        meta_info = self.template.generate_meta_info(data_info)
        sections.append(meta_info)
        
        # Key Statistics
        if 'summary' in analysis_results:
            stats_content = self._generate_key_stats_section(analysis_results['summary'])
            sections.append(self.template.generate_section(
                "Key Statistics", stats_content, "📊"
            ))
        
        # Executive Summary
        if ai_insights and 'executive_summary' in ai_insights.get('insights', {}):
            exec_content = self._generate_executive_summary_section(ai_insights['insights']['executive_summary'])
            sections.append(self.template.generate_section(
                "Executive Overview", exec_content, "📋"
            ))
        
        # Key Findings
        findings_content = self._generate_key_findings_section(analysis_results, ai_insights)
        sections.append(self.template.generate_section(
            "Key Findings", findings_content, "🔍"
        ))
        
        # Recommendations
        recommendations = self._extract_recommendations(analysis_results, ai_insights)
        if recommendations:
            rec_content = self.template.generate_recommendations(recommendations[:5])  # Top 5
            sections.append(self.template.generate_section(
                "Strategic Recommendations", rec_content, "🎯"
            ))
        
        # Footer
        footer = self.template.generate_footer()
        sections.append(footer)
        
        content = "".join(sections)
        return self.template.wrap_html(content, title)
    
    def generate_ai_insights_report(self,
                                  ai_insights: Dict[str, Any],
                                  data_info: Dict[str, Any]) -> str:
        """Generate a report focused on AI insights."""
        logger.info("Generating AI insights report")
        
        sections = []
        
        # Header
        title = "AI-Powered Insights Report"
        subtitle = f"Dataset: {data_info.get('data_id', 'Unknown')}"
        header = self.template.generate_header(title, subtitle)
        sections.append(header)
        
        # Meta information
        meta_info = self.template.generate_meta_info(data_info)
        sections.append(meta_info)
        
        # AI Capabilities Overview
        if 'metadata' in ai_insights:
            capabilities_content = self._generate_ai_capabilities_section(ai_insights['metadata'])
            sections.append(self.template.generate_section(
                "AI Analysis Overview", capabilities_content, "🤖"
            ))
        
        # Individual AI Insights
        if 'insights' in ai_insights:
            ai_content = self._generate_detailed_ai_insights_section(ai_insights['insights'])
            sections.append(self.template.generate_section(
                "Detailed AI Insights", ai_content, "🧠"
            ))
        
        # Quality Assessment
        if 'quality_assessment' in ai_insights:
            quality_content = self._generate_ai_quality_section(ai_insights['quality_assessment'])
            sections.append(self.template.generate_section(
                "Insight Quality Assessment", quality_content, "📊"
            ))
        
        # Footer
        footer = self.template.generate_footer(
            "AI insights generated using advanced language models"
        )
        sections.append(footer)
        
        content = "".join(sections)
        return self.template.wrap_html(content, title)
    
    def _generate_executive_summary_section(self, executive_summary: Dict[str, Any]) -> str:
        """Generate executive summary section content."""
        content_parts = []
        
        if executive_summary.get('content'):
            content_parts.append(f"<p>{executive_summary['content']}</p>")
        
        if executive_summary.get('confidence_score'):
            confidence_badge = self.template.generate_confidence_badge(
                executive_summary['confidence_score']
            )
            content_parts.append(f"<p><strong>Confidence:</strong> {confidence_badge}</p>")
        
        return "".join(content_parts)
    
    def _generate_quality_section(self, quality_data: Dict[str, Any]) -> str:
        """Generate data quality section content."""
        content_parts = []
        
        # Quality score
        quality_score = quality_data.get('quality_score', 0)
        content_parts.append(f"""
        <div class="subsection">
            <h3>Overall Quality Score</h3>
            <div class="stats-grid">
                <div class="stat-card">
                    <div class="number">{quality_score:.0f}/100</div>
                    <div class="label">Quality Score</div>
                </div>
            </div>
        </div>
        """)
        
        # Quality details
        if 'quality_issues' in quality_data:
            issues = quality_data['quality_issues']
            if issues:
                issue_list = "".join([f"<li>{issue}</li>" for issue in issues])
                content_parts.append(f"""
                <div class="subsection">
                    <h3>Quality Issues</h3>
                    <ul>{issue_list}</ul>
                </div>
                """)
        
        # Missing data analysis
        if 'missing_data_summary' in quality_data:
            missing_data = quality_data['missing_data_summary']
            content_parts.append(f"""
            <div class="subsection">
                <h3>Missing Data Analysis</h3>
                <p>Total missing values: <strong>{missing_data.get('total_missing', 0):,}</strong></p>
                <p>Missing percentage: <strong>{missing_data.get('missing_percentage', 0):.2f}%</strong></p>
            </div>
            """)
        
        return "".join(content_parts)
    
    def _generate_summary_section(self, summary_data: Dict[str, Any]) -> str:
        """Generate statistical summary section content."""
        content_parts = []
        
        # Basic info
        if 'basic_info' in summary_data:
            basic_info = summary_data['basic_info']
            stats = {
                'Total Records': basic_info.get('rows', 0),
                'Features': basic_info.get('columns', 0),
                'Numeric Features': basic_info.get('numeric_columns', 0),
                'Categorical Features': basic_info.get('categorical_columns', 0)
            }
            content_parts.append(self.template.generate_stats_grid(stats))
        
        # Descriptive statistics table
        if 'descriptive_stats' in summary_data:
            desc_stats = summary_data['descriptive_stats']
            if isinstance(desc_stats, dict):
                # Convert to table format
                table_data = []
                for col, stats in list(desc_stats.items())[:5]:  # First 5 columns
                    if isinstance(stats, dict):
                        row = {'Feature': col}
                        row.update({k.title(): f"{v:.3f}" if isinstance(v, float) else v 
                                  for k, v in stats.items() if k in ['mean', 'std', 'min', 'max']})
                        table_data.append(row)
                
                if table_data:
                    content_parts.append(f"""
                    <div class="subsection">
                        <h3>Descriptive Statistics (Top 5 Features)</h3>
                        {self.template.generate_data_table(table_data)}
                    </div>
                    """)
        
        return "".join(content_parts)
    
    def _generate_visualization_section(self, charts: Dict[str, str]) -> str:
        """Generate visualization section with embedded charts."""
        content_parts = []
        
        chart_descriptions = {
            'correlation_heatmap': 'Correlation matrix showing relationships between numeric features',
            'distribution': 'Distribution analysis of key variables',
            'histogram': 'Histogram showing data distribution',
            'scatter': 'Scatter plot analysis',
            'box_plot': 'Box plot showing data spread and outliers'
        }
        
        for chart_name, chart_base64 in charts.items():
            description = chart_descriptions.get(chart_name, f"Analysis chart: {chart_name}")
            chart_html = self.template.generate_chart_section(
                chart_name.replace('_', ' ').title(),
                chart_base64,
                description
            )
            content_parts.append(chart_html)
        
        return "".join(content_parts)
    
    def _generate_feature_section(self, feature_data: Dict[str, Any]) -> str:
        """Generate feature analysis section content."""
        content_parts = []
        
        # Feature importance
        if 'correlation_analysis' in feature_data:
            corr_data = feature_data['correlation_analysis']
            if 'strong_correlations' in corr_data:
                strong_corrs = corr_data['strong_correlations']
                if strong_corrs:
                    content_parts.append(f"""
                    <div class="subsection">
                        <h3>Strong Correlations</h3>
                        <p>Found <strong>{len(strong_corrs)}</strong> strong correlations between features.</p>
                    </div>
                    """)
        
        # Multicollinearity
        if 'multicollinearity_analysis' in feature_data:
            multicoll = feature_data['multicollinearity_analysis']
            if 'high_vif_features' in multicoll:
                high_vif = multicoll['high_vif_features']
                if high_vif:
                    content_parts.append(f"""
                    <div class="subsection">
                        <h3>Multicollinearity Issues</h3>
                        <p><strong>{len(high_vif)}</strong> features show high multicollinearity (VIF > 5).</p>
                        <p>Consider removing or combining these features: {', '.join(high_vif[:5])}</p>
                    </div>
                    """)
        
        return "".join(content_parts) or "<p>Feature analysis completed successfully.</p>"
    
    def _generate_insights_section(self, insights_data: Dict[str, Any]) -> str:
        """Generate business insights section content."""
        content_parts = []
        
        # Executive summary from insights
        if 'executive_summary' in insights_data:
            exec_summary = insights_data['executive_summary']
            content_parts.append(f"""
            <div class="insight-box">
                <div class="title">Key Insights</div>
                <div>{exec_summary}</div>
            </div>
            """)
        
        # Business recommendations
        if 'business_recommendations' in insights_data:
            recommendations = insights_data['business_recommendations']
            if isinstance(recommendations, list) and recommendations:
                rec_items = "".join([f"<li>{rec}</li>" for rec in recommendations[:5]])
                content_parts.append(f"""
                <div class="subsection">
                    <h3>Business Recommendations</h3>
                    <ul>{rec_items}</ul>
                </div>
                """)
        
        return "".join(content_parts) or "<p>Business insights analysis completed.</p>"
    
    def _generate_ai_insights_section(self, ai_insights: Dict[str, Any]) -> str:
        """Generate AI insights section content."""
        content_parts = []
        
        insights = ai_insights.get('insights', {})
        
        # Process each AI insight type
        insight_types = {
            'detailed_insights': 'Detailed Analysis',
            'business_recommendations': 'Business Recommendations',
            'anomaly_explanation': 'Anomaly Explanations',
            'correlation_insights': 'Correlation Analysis'
        }
        
        for insight_key, insight_title in insight_types.items():
            if insight_key in insights:
                insight_data = insights[insight_key]
                if insight_data.get('content'):
                    confidence_badge = ""
                    if insight_data.get('confidence_score'):
                        confidence_badge = self.template.generate_confidence_badge(
                            insight_data['confidence_score']
                        )
                    
                    content_parts.append(f"""
                    <div class="subsection">
                        <h3>{insight_title} {confidence_badge}</h3>
                        <div class="insight-box">
                            {insight_data['content']}
                        </div>
                    </div>
                    """)
        
        return "".join(content_parts) or "<p>AI insights generation completed.</p>"
    
    def _generate_detailed_ai_insights_section(self, insights: Dict[str, Any]) -> str:
        """Generate detailed AI insights with individual sections."""
        content_parts = []
        
        for insight_type, insight_data in insights.items():
            if not insight_data.get('content'):
                continue
                
            title = insight_type.replace('_', ' ').title()
            confidence_badge = ""
            
            if insight_data.get('confidence_score'):
                confidence_badge = self.template.generate_confidence_badge(
                    insight_data['confidence_score']
                )
            
            # Add metadata if available
            metadata_info = ""
            if insight_data.get('cached'):
                metadata_info += " (Cached)"
            if insight_data.get('response_time'):
                metadata_info += f" • Response time: {insight_data['response_time']:.1f}s"
            
            content_parts.append(f"""
            <div class="subsection">
                <h3>{title} {confidence_badge}</h3>
                <div class="insight-box">
                    <div>{insight_data['content']}</div>
                    {f'<small style="color: #666; margin-top: 10px; display: block;">{metadata_info}</small>' if metadata_info else ''}
                </div>
            </div>
            """)
        
        return "".join(content_parts)
    
    def _generate_ai_capabilities_section(self, metadata: Dict[str, Any]) -> str:
        """Generate AI capabilities overview section."""
        stats = {
            'Total Insights': metadata.get('total_insights', 0),
            'Generation Errors': metadata.get('generation_errors', 0),
            'AI Service': 'Available' if metadata.get('api_available') else 'Unavailable'
        }
        
        return self.template.generate_stats_grid(stats)
    
    def _generate_ai_quality_section(self, quality_assessment: Dict[str, Any]) -> str:
        """Generate AI quality assessment section."""
        content_parts = []
        
        # Overall quality metrics
        stats = {
            'Overall Score': f"{quality_assessment.get('overall_score', 0)}/100",
            'Quality Level': quality_assessment.get('quality_level', 'Unknown'),
            'Avg Confidence': f"{quality_assessment.get('average_confidence', 0):.1f}%",
            'Coverage Score': f"{quality_assessment.get('coverage_score', 0):.1f}%"
        }
        
        content_parts.append(self.template.generate_stats_grid(stats))
        
        # Issues and recommendations
        if quality_assessment.get('issues'):
            issues = quality_assessment['issues']
            issue_list = "".join([f"<li>{issue}</li>" for issue in issues])
            content_parts.append(f"""
            <div class="subsection">
                <h3>Quality Issues</h3>
                <ul>{issue_list}</ul>
            </div>
            """)
        
        if quality_assessment.get('recommendations'):
            recommendations = quality_assessment['recommendations']
            rec_list = "".join([f"<li>{rec}</li>" for rec in recommendations])
            content_parts.append(f"""
            <div class="subsection">
                <h3>Improvement Recommendations</h3>
                <ul>{rec_list}</ul>
            </div>
            """)
        
        return "".join(content_parts)
    
    def _generate_key_stats_section(self, summary_data: Dict[str, Any]) -> str:
        """Generate key statistics for executive summary."""
        stats = {}
        
        if 'basic_info' in summary_data:
            basic_info = summary_data['basic_info']
            stats.update({
                'Records': f"{basic_info.get('rows', 0):,}",
                'Features': basic_info.get('columns', 0),
                'Quality': f"{basic_info.get('quality_score', 0)}/100" if 'quality_score' in basic_info else 'N/A'
            })
        
        if 'missing_data_analysis' in summary_data:
            missing = summary_data['missing_data_analysis']
            stats['Completeness'] = f"{100 - missing.get('missing_percentage_overall', 0):.1f}%"
        
        return self.template.generate_stats_grid(stats)
    
    def _generate_key_findings_section(self, analysis_results: Dict[str, Any], 
                                     ai_insights: Optional[Dict[str, Any]] = None) -> str:
        """Generate key findings section."""
        findings = []
        
        # Data quality findings
        if 'quality' in analysis_results:
            quality_score = analysis_results['quality'].get('quality_score', 0)
            if quality_score >= 80:
                findings.append("✅ High data quality detected - analysis results are highly reliable")
            elif quality_score >= 60:
                findings.append("⚠️ Moderate data quality - some cleaning may improve results")
            else:
                findings.append("🔴 Data quality issues detected - recommend data cleaning")
        
        # Feature findings
        if 'features' in analysis_results:
            feature_data = analysis_results['features']
            if 'correlation_analysis' in feature_data:
                strong_corrs = len(feature_data['correlation_analysis'].get('strong_correlations', []))
                if strong_corrs > 0:
                    findings.append(f"🔗 {strong_corrs} strong feature correlations identified")
        
        # AI insights findings
        if ai_insights and 'insights' in ai_insights:
            ai_count = len([i for i in ai_insights['insights'].values() if i.get('content')])
            if ai_count > 0:
                findings.append(f"🤖 {ai_count} AI-powered insights generated")
        
        if not findings:
            findings.append("✅ Analysis completed successfully")
        
        finding_items = "".join([f"<li>{finding}</li>" for finding in findings])
        return f"<ul>{finding_items}</ul>"
    
    def _extract_recommendations(self, analysis_results: Dict[str, Any],
                               ai_insights: Optional[Dict[str, Any]] = None) -> List[str]:
        """Extract recommendations from all sources."""
        recommendations = []
        
        # From business insights
        if 'insights' in analysis_results:
            business_recs = analysis_results['insights'].get('business_recommendations', [])
            if isinstance(business_recs, list):
                recommendations.extend(business_recs)
        
        # From AI insights
        if ai_insights and 'insights' in ai_insights:
            ai_recs = ai_insights['insights'].get('business_recommendations', {})
            if isinstance(ai_recs, dict) and ai_recs.get('content'):
                # Parse AI recommendations (assuming they're in text format)
                ai_content = ai_recs['content']
                if '•' in ai_content:
                    ai_rec_list = [rec.strip() for rec in ai_content.split('•') if rec.strip()]
                    recommendations.extend(ai_rec_list[1:])  # Skip first empty split
                elif '\n-' in ai_content:
                    ai_rec_list = [rec.strip()[1:].strip() for rec in ai_content.split('\n-') if rec.strip()]
                    recommendations.extend(ai_rec_list[1:])
        
        # Remove duplicates while preserving order
        seen = set()
        unique_recs = []
        for rec in recommendations:
            if rec not in seen:
                seen.add(rec)
                unique_recs.append(rec)
        
        return unique_recs[:10]  # Limit to top 10


# Factory functions
def create_html_generator(template: Optional[ReportTemplate] = None) -> HTMLReportGenerator:
    """Create and return an HTMLReportGenerator instance."""
    return HTMLReportGenerator(template)


def generate_analysis_report(analysis_results: Dict[str, Any],
                           data_info: Dict[str, Any],
                           charts: Optional[Dict[str, str]] = None,
                           ai_insights: Optional[Dict[str, Any]] = None,
                           report_style: ReportStyle = ReportStyle.PROFESSIONAL) -> str:
    """Generate a complete analysis report."""
    template = get_template(ReportType.COMPLETE_ANALYSIS, report_style)
    generator = HTMLReportGenerator(template)
    return generator.generate_complete_analysis_report(
        analysis_results, data_info, charts, ai_insights
    )


def generate_executive_report(analysis_results: Dict[str, Any],
                            data_info: Dict[str, Any],
                            ai_insights: Optional[Dict[str, Any]] = None,
                            report_style: ReportStyle = ReportStyle.PROFESSIONAL) -> str:
    """Generate an executive summary report."""
    template = get_template(ReportType.EXECUTIVE_SUMMARY, report_style)
    generator = HTMLReportGenerator(template)
    return generator.generate_executive_summary_report(
        analysis_results, data_info, ai_insights
    )
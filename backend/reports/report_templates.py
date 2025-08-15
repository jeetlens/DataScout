"""
Report Templates for DataScout
Defines templates and styling for different types of reports.
"""

from enum import Enum
from typing import Dict, Any, Optional, List
from datetime import datetime
import base64


class ReportType(Enum):
    """Types of reports that can be generated."""
    EXECUTIVE_SUMMARY = "executive_summary"
    DETAILED_ANALYSIS = "detailed_analysis" 
    AI_INSIGHTS = "ai_insights"
    COMPLETE_ANALYSIS = "complete_analysis"
    CUSTOM = "custom"


class ReportStyle(Enum):
    """Report styling options."""
    PROFESSIONAL = "professional"
    MODERN = "modern"
    MINIMAL = "minimal"
    CORPORATE = "corporate"


class ReportTemplate:
    """
    Report template manager for generating styled HTML reports.
    
    Features:
    - Professional styling with CSS
    - Chart and visualization embedding
    - Responsive design
    - Multiple report types
    - Customizable themes
    """
    
    def __init__(self, report_type: ReportType = ReportType.COMPLETE_ANALYSIS,
                 style: ReportStyle = ReportStyle.PROFESSIONAL):
        self.report_type = report_type
        self.style = style
        self.css_styles = self._get_css_styles()
        
    def _get_css_styles(self) -> str:
        """Get CSS styles based on selected theme."""
        base_styles = """
        <style>
        * {
            margin: 0;
            padding: 0;
            box-sizing: border-box;
        }
        
        body {
            font-family: 'Segoe UI', Tahoma, Geneva, Verdana, sans-serif;
            line-height: 1.6;
            color: #333;
            background-color: #f8f9fa;
        }
        
        .container {
            max-width: 1200px;
            margin: 0 auto;
            padding: 20px;
            background-color: white;
            box-shadow: 0 0 20px rgba(0,0,0,0.1);
        }
        
        .header {
            background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
            color: white;
            padding: 30px;
            text-align: center;
            margin-bottom: 30px;
            border-radius: 8px;
        }
        
        .header h1 {
            font-size: 2.5em;
            margin-bottom: 10px;
            font-weight: 300;
        }
        
        .header .subtitle {
            font-size: 1.2em;
            opacity: 0.9;
        }
        
        .meta-info {
            background-color: #f8f9fa;
            padding: 20px;
            border-radius: 8px;
            margin-bottom: 30px;
            border-left: 4px solid #667eea;
        }
        
        .meta-info h3 {
            color: #667eea;
            margin-bottom: 15px;
        }
        
        .meta-grid {
            display: grid;
            grid-template-columns: repeat(auto-fit, minmax(200px, 1fr));
            gap: 15px;
        }
        
        .meta-item {
            background: white;
            padding: 15px;
            border-radius: 6px;
            border: 1px solid #e9ecef;
        }
        
        .meta-item .label {
            font-weight: 600;
            color: #6c757d;
            font-size: 0.9em;
            margin-bottom: 5px;
        }
        
        .meta-item .value {
            font-size: 1.1em;
            color: #333;
        }
        
        .section {
            margin-bottom: 40px;
            padding: 25px;
            background: white;
            border-radius: 8px;
            border: 1px solid #e9ecef;
        }
        
        .section-header {
            display: flex;
            align-items: center;
            margin-bottom: 20px;
            padding-bottom: 15px;
            border-bottom: 2px solid #f1f3f4;
        }
        
        .section-icon {
            width: 40px;
            height: 40px;
            background: #667eea;
            border-radius: 50%;
            display: flex;
            align-items: center;
            justify-content: center;
            margin-right: 15px;
            font-size: 18px;
            color: white;
        }
        
        .section h2 {
            color: #333;
            font-size: 1.8em;
            font-weight: 600;
        }
        
        .subsection {
            margin-bottom: 25px;
        }
        
        .subsection h3 {
            color: #667eea;
            font-size: 1.3em;
            margin-bottom: 15px;
            padding-bottom: 8px;
            border-bottom: 1px solid #e9ecef;
        }
        
        .chart-container {
            text-align: center;
            margin: 25px 0;
            padding: 20px;
            background: #fafbfc;
            border-radius: 8px;
            border: 1px solid #e9ecef;
        }
        
        .chart-container img {
            max-width: 100%;
            height: auto;
            border-radius: 4px;
            box-shadow: 0 2px 8px rgba(0,0,0,0.1);
        }
        
        .chart-title {
            font-weight: 600;
            margin-bottom: 15px;
            color: #333;
            font-size: 1.1em;
        }
        
        .stats-grid {
            display: grid;
            grid-template-columns: repeat(auto-fit, minmax(150px, 1fr));
            gap: 15px;
            margin: 20px 0;
        }
        
        .stat-card {
            background: #667eea;
            color: white;
            padding: 20px;
            border-radius: 8px;
            text-align: center;
        }
        
        .stat-card .number {
            font-size: 2em;
            font-weight: bold;
            margin-bottom: 5px;
        }
        
        .stat-card .label {
            font-size: 0.9em;
            opacity: 0.9;
        }
        
        .insight-box {
            background: #e8f4fd;
            border: 1px solid #b8daff;
            border-radius: 8px;
            padding: 20px;
            margin: 15px 0;
        }
        
        .insight-box.warning {
            background: #fff3cd;
            border-color: #ffeaa7;
        }
        
        .insight-box.success {
            background: #d4edda;
            border-color: #c3e6cb;
        }
        
        .insight-box .title {
            font-weight: 600;
            margin-bottom: 10px;
            color: #333;
        }
        
        .data-table {
            width: 100%;
            border-collapse: collapse;
            margin: 20px 0;
            background: white;
        }
        
        .data-table th,
        .data-table td {
            padding: 12px;
            text-align: left;
            border-bottom: 1px solid #e9ecef;
        }
        
        .data-table th {
            background: #f8f9fa;
            font-weight: 600;
            color: #495057;
        }
        
        .data-table tr:hover {
            background: #f8f9fa;
        }
        
        .recommendations {
            background: #f8f9fa;
            border-left: 4px solid #28a745;
            padding: 20px;
            margin: 20px 0;
            border-radius: 0 8px 8px 0;
        }
        
        .recommendations h4 {
            color: #28a745;
            margin-bottom: 15px;
        }
        
        .recommendations ul {
            padding-left: 20px;
        }
        
        .recommendations li {
            margin-bottom: 8px;
        }
        
        .footer {
            text-align: center;
            padding: 30px;
            margin-top: 40px;
            background: #f8f9fa;
            color: #6c757d;
            border-radius: 8px;
            border-top: 3px solid #667eea;
        }
        
        .confidence-score {
            display: inline-block;
            padding: 4px 12px;
            border-radius: 20px;
            font-size: 0.9em;
            font-weight: 600;
        }
        
        .confidence-high {
            background: #d4edda;
            color: #155724;
        }
        
        .confidence-medium {
            background: #fff3cd;
            color: #856404;
        }
        
        .confidence-low {
            background: #f8d7da;
            color: #721c24;
        }
        
        @media (max-width: 768px) {
            .container {
                padding: 10px;
            }
            
            .header h1 {
                font-size: 2em;
            }
            
            .meta-grid {
                grid-template-columns: 1fr;
            }
            
            .stats-grid {
                grid-template-columns: repeat(2, 1fr);
            }
        }
        </style>
        """
        
        # Add style-specific variations
        if self.style == ReportStyle.MODERN:
            base_styles += """
            <style>
            .header {
                background: linear-gradient(135deg, #ff6b6b 0%, #4ecdc4 100%);
            }
            .section-icon {
                background: #4ecdc4;
            }
            .stat-card {
                background: #4ecdc4;
            }
            </style>
            """
        elif self.style == ReportStyle.CORPORATE:
            base_styles += """
            <style>
            .header {
                background: linear-gradient(135deg, #2c3e50 0%, #34495e 100%);
            }
            .section-icon {
                background: #34495e;
            }
            .stat-card {
                background: #34495e;
            }
            </style>
            """
        
        return base_styles
    
    def generate_header(self, title: str, subtitle: str = "", 
                       generation_time: Optional[datetime] = None) -> str:
        """Generate report header."""
        if generation_time is None:
            generation_time = datetime.now()
            
        return f"""
        <div class="header">
            <h1>{title}</h1>
            {f'<div class="subtitle">{subtitle}</div>' if subtitle else ''}
        </div>
        """
    
    def generate_meta_info(self, data_info: Dict[str, Any]) -> str:
        """Generate metadata information section."""
        meta_items = []
        
        # Standard metadata items
        standard_fields = {
            "data_id": "Dataset ID",
            "rows": "Total Records",
            "columns": "Features",
            "quality_score": "Data Quality",
            "analysis_type": "Analysis Type",
            "generation_time": "Generated On"
        }
        
        for key, label in standard_fields.items():
            if key in data_info:
                value = data_info[key]
                if key == "quality_score":
                    value = f"{value}/100" if isinstance(value, (int, float)) else value
                elif key == "generation_time":
                    if isinstance(value, datetime):
                        value = value.strftime("%Y-%m-%d %H:%M:%S")
                        
                meta_items.append(f"""
                <div class="meta-item">
                    <div class="label">{label}</div>
                    <div class="value">{value}</div>
                </div>
                """)
        
        return f"""
        <div class="meta-info">
            <h3>📊 Dataset Overview</h3>
            <div class="meta-grid">
                {''.join(meta_items)}
            </div>
        </div>
        """
    
    def generate_section(self, title: str, content: str, icon: str = "📈") -> str:
        """Generate a report section with title and content."""
        return f"""
        <div class="section">
            <div class="section-header">
                <div class="section-icon">{icon}</div>
                <h2>{title}</h2>
            </div>
            {content}
        </div>
        """
    
    def generate_chart_section(self, title: str, chart_base64: str, 
                             description: str = "") -> str:
        """Generate a section with embedded chart."""
        chart_html = f"""
        <div class="chart-container">
            <div class="chart-title">{title}</div>
            <img src="data:image/png;base64,{chart_base64}" alt="{title}"/>
            {f'<p style="margin-top: 15px; color: #666;">{description}</p>' if description else ''}
        </div>
        """
        return chart_html
    
    def generate_stats_grid(self, stats: Dict[str, Any]) -> str:
        """Generate statistics grid."""
        stat_cards = []
        
        for label, value in stats.items():
            # Format value appropriately
            if isinstance(value, float):
                if value > 1000:
                    formatted_value = f"{value:,.0f}"
                else:
                    formatted_value = f"{value:.2f}"
            elif isinstance(value, int):
                formatted_value = f"{value:,}"
            else:
                formatted_value = str(value)
                
            stat_cards.append(f"""
            <div class="stat-card">
                <div class="number">{formatted_value}</div>
                <div class="label">{label.replace('_', ' ').title()}</div>
            </div>
            """)
        
        return f"""
        <div class="stats-grid">
            {''.join(stat_cards)}
        </div>
        """
    
    def generate_insight_box(self, title: str, content: str, 
                           box_type: str = "info") -> str:
        """Generate an insight box with styling."""
        return f"""
        <div class="insight-box {box_type}">
            <div class="title">{title}</div>
            <div>{content}</div>
        </div>
        """
    
    def generate_data_table(self, data: List[Dict[str, Any]], 
                          headers: Optional[List[str]] = None) -> str:
        """Generate a data table."""
        if not data:
            return "<p>No data available</p>"
            
        if headers is None:
            headers = list(data[0].keys()) if data else []
        
        header_row = "".join([f"<th>{header}</th>" for header in headers])
        
        rows = []
        for row in data[:10]:  # Limit to first 10 rows
            cells = []
            for header in headers:
                value = row.get(header, "N/A")
                if isinstance(value, float):
                    value = f"{value:.3f}" if abs(value) < 1 else f"{value:.2f}"
                cells.append(f"<td>{value}</td>")
            rows.append(f"<tr>{''.join(cells)}</tr>")
        
        return f"""
        <table class="data-table">
            <thead>
                <tr>{header_row}</tr>
            </thead>
            <tbody>
                {''.join(rows)}
            </tbody>
        </table>
        {f'<p><em>Showing first 10 of {len(data)} records</em></p>' if len(data) > 10 else ''}
        """
    
    def generate_recommendations(self, recommendations: List[str]) -> str:
        """Generate recommendations section."""
        if not recommendations:
            return ""
            
        rec_items = "".join([f"<li>{rec}</li>" for rec in recommendations])
        
        return f"""
        <div class="recommendations">
            <h4>💡 Key Recommendations</h4>
            <ul>
                {rec_items}
            </ul>
        </div>
        """
    
    def generate_confidence_badge(self, score: float) -> str:
        """Generate confidence score badge."""
        if score >= 80:
            css_class = "confidence-high"
            label = "High Confidence"
        elif score >= 60:
            css_class = "confidence-medium" 
            label = "Medium Confidence"
        else:
            css_class = "confidence-low"
            label = "Low Confidence"
            
        return f"""
        <span class="confidence-score {css_class}">
            {label} ({score:.0f}%)
        </span>
        """
    
    def generate_footer(self, additional_info: str = "") -> str:
        """Generate report footer."""
        return f"""
        <div class="footer">
            <p><strong>DataScout</strong> - Automated Data Analysis & AI Insight Platform</p>
            <p>Generated on {datetime.now().strftime("%Y-%m-%d at %H:%M:%S")}</p>
            {f'<p>{additional_info}</p>' if additional_info else ''}
        </div>
        """
    
    def wrap_html(self, content: str, title: str = "DataScout Report") -> str:
        """Wrap content in complete HTML document."""
        return f"""
        <!DOCTYPE html>
        <html lang="en">
        <head>
            <meta charset="UTF-8">
            <meta name="viewport" content="width=device-width, initial-scale=1.0">
            <title>{title}</title>
            {self.css_styles}
        </head>
        <body>
            <div class="container">
                {content}
            </div>
        </body>
        </html>
        """


# Factory functions
def get_template(report_type: ReportType = ReportType.COMPLETE_ANALYSIS,
                style: ReportStyle = ReportStyle.PROFESSIONAL) -> ReportTemplate:
    """Get a report template with specified type and style."""
    return ReportTemplate(report_type, style)


def create_custom_template(custom_css: str = "") -> ReportTemplate:
    """Create a custom template with additional CSS."""
    template = ReportTemplate()
    if custom_css:
        template.css_styles += f"<style>{custom_css}</style>"
    return template
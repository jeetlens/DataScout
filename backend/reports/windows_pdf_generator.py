"""
Windows-Compatible PDF Report Generator for DataScout
Uses ReportLab for reliable Windows PDF generation without external dependencies.
"""

import os
import logging
from typing import Dict, Any, Optional, Union
from io import BytesIO
from datetime import datetime
import tempfile

logger = logging.getLogger(__name__)

# Try ReportLab import
try:
    from reportlab.lib.pagesizes import letter, A4
    from reportlab.platypus import SimpleDocTemplate, Paragraph, Spacer, Table, TableStyle, PageBreak
    from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
    from reportlab.lib.units import inch
    from reportlab.lib import colors
    from reportlab.lib.enums import TA_CENTER, TA_LEFT, TA_JUSTIFY
    from reportlab.graphics.shapes import Drawing
    from reportlab.graphics.charts.lineplots import LinePlot
    from reportlab.graphics.charts.barcharts import VerticalBarChart
    REPORTLAB_AVAILABLE = True
    logger.info("ReportLab is available for PDF generation")
except ImportError as e:
    logger.warning(f"ReportLab not available: {e}")
    REPORTLAB_AVAILABLE = False

class WindowsCompatiblePDFGenerator:
    """
    Windows-compatible PDF generator using ReportLab.
    Fallback for environments where WeasyPrint cannot be installed.
    """
    
    def __init__(self):
        """Initialize the PDF generator."""
        if not REPORTLAB_AVAILABLE:
            raise ImportError("ReportLab is required for PDF generation. Install with: pip install reportlab")
        
        self.styles = getSampleStyleSheet()
        self._setup_custom_styles()
    
    def _setup_custom_styles(self):
        """Set up custom paragraph styles."""
        # Title style
        self.styles.add(ParagraphStyle(
            name='CustomTitle',
            parent=self.styles['Title'],
            fontSize=24,
            spaceAfter=30,
            textColor=colors.HexColor('#1D1D1F'),
            alignment=TA_CENTER
        ))
        
        # Subtitle style
        self.styles.add(ParagraphStyle(
            name='CustomSubtitle',
            parent=self.styles['Heading1'],
            fontSize=18,
            spaceAfter=20,
            textColor=colors.HexColor('#007AFF'),
            alignment=TA_LEFT
        ))
        
        # Section header style
        self.styles.add(ParagraphStyle(
            name='SectionHeader',
            parent=self.styles['Heading2'],
            fontSize=14,
            spaceAfter=12,
            spaceBefore=20,
            textColor=colors.HexColor('#1D1D1F'),
            alignment=TA_LEFT
        ))
        
        # Body text style
        self.styles.add(ParagraphStyle(
            name='CustomBody',
            parent=self.styles['Normal'],
            fontSize=10,
            spaceAfter=6,
            textColor=colors.HexColor('#1D1D1F'),
            alignment=TA_JUSTIFY
        ))
    
    def generate_pdf_from_data(self, 
                              analysis_results: Dict[str, Any],
                              data_info: Dict[str, Any],
                              ai_insights: Optional[Dict[str, Any]] = None,
                              output_path: Optional[str] = None) -> Union[bytes, str]:
        """
        Generate PDF report from analysis data.
        
        Args:
            analysis_results: Analysis results from DataScout
            data_info: Basic dataset information
            ai_insights: Optional AI insights
            output_path: Optional output file path
            
        Returns:
            PDF bytes if no output_path, otherwise file path
        """
        # Create PDF buffer or file
        if output_path:
            pdf_buffer = output_path
        else:
            pdf_buffer = BytesIO()
        
        # Create PDF document
        doc = SimpleDocTemplate(
            pdf_buffer,
            pagesize=A4,
            rightMargin=72,
            leftMargin=72,
            topMargin=72,
            bottomMargin=72
        )
        
        # Build PDF content
        story = []
        
        # Title page
        story.extend(self._create_title_page(data_info))
        story.append(PageBreak())
        
        # Executive summary
        story.extend(self._create_executive_summary(analysis_results, data_info))
        
        # Dataset overview
        story.extend(self._create_dataset_overview(data_info, analysis_results))
        
        # Statistical analysis
        if 'descriptive_stats' in analysis_results:
            story.extend(self._create_statistical_section(analysis_results['descriptive_stats']))
        
        # AI insights (if available)
        if ai_insights:
            story.extend(self._create_ai_insights_section(ai_insights))
        
        # Footer
        story.extend(self._create_footer())
        
        # Build PDF
        doc.build(story)
        
        if output_path:
            logger.info(f"PDF report saved to {output_path}")
            return output_path
        else:
            pdf_bytes = pdf_buffer.getvalue()
            pdf_buffer.close()
            logger.info("PDF report generated in memory")
            return pdf_bytes
    
    def _create_title_page(self, data_info: Dict[str, Any]) -> list:
        """Create title page content."""
        content = []
        
        # Main title
        content.append(Paragraph("DataScout Analysis Report", self.styles['CustomTitle']))
        content.append(Spacer(1, 50))
        
        # Dataset info
        dataset_name = data_info.get('filename', 'Unknown Dataset')
        content.append(Paragraph(f"Dataset: {dataset_name}", self.styles['CustomSubtitle']))
        content.append(Spacer(1, 30))
        
        # Basic stats
        rows = data_info.get('total_rows', 'N/A')
        cols = data_info.get('total_columns', 'N/A')
        content.append(Paragraph(f"Dimensions: {rows} rows × {cols} columns", self.styles['CustomBody']))
        content.append(Spacer(1, 20))
        
        # Generation date
        current_date = datetime.now().strftime("%B %d, %Y at %I:%M %p")
        content.append(Paragraph(f"Generated on: {current_date}", self.styles['CustomBody']))
        content.append(Spacer(1, 50))
        
        return content
    
    def _create_executive_summary(self, analysis_results: Dict[str, Any], data_info: Dict[str, Any]) -> list:
        """Create executive summary section."""
        content = []
        
        content.append(Paragraph("Executive Summary", self.styles['CustomSubtitle']))
        content.append(Spacer(1, 12))
        
        # Key findings
        summary_text = f"""
        This report presents a comprehensive analysis of the {data_info.get('filename', 'dataset')} 
        containing {data_info.get('total_rows', 'N/A')} records and {data_info.get('total_columns', 'N/A')} variables.
        
        Key characteristics of the dataset include {len(data_info.get('numeric_columns', []))} numerical variables 
        and {len(data_info.get('categorical_columns', []))} categorical variables.
        """
        
        content.append(Paragraph(summary_text, self.styles['CustomBody']))
        content.append(Spacer(1, 20))
        
        return content
    
    def _create_dataset_overview(self, data_info: Dict[str, Any], analysis_results: Dict[str, Any]) -> list:
        """Create dataset overview section."""
        content = []
        
        content.append(Paragraph("Dataset Overview", self.styles['CustomSubtitle']))
        content.append(Spacer(1, 12))
        
        # Create overview table
        table_data = [
            ['Attribute', 'Value'],
            ['Total Rows', str(data_info.get('total_rows', 'N/A'))],
            ['Total Columns', str(data_info.get('total_columns', 'N/A'))],
            ['Numeric Columns', str(len(data_info.get('numeric_columns', [])))],
            ['Categorical Columns', str(len(data_info.get('categorical_columns', [])))],
            ['Memory Usage (MB)', str(data_info.get('memory_usage_mb', 'N/A'))],
        ]
        
        table = Table(table_data, colWidths=[3*inch, 2*inch])
        table.setStyle(TableStyle([
            ('BACKGROUND', (0, 0), (-1, 0), colors.HexColor('#007AFF')),
            ('TEXTCOLOR', (0, 0), (-1, 0), colors.whitesmoke),
            ('ALIGN', (0, 0), (-1, -1), 'LEFT'),
            ('FONTNAME', (0, 0), (-1, 0), 'Helvetica-Bold'),
            ('FONTSIZE', (0, 0), (-1, 0), 12),
            ('BOTTOMPADDING', (0, 0), (-1, 0), 12),
            ('BACKGROUND', (0, 1), (-1, -1), colors.HexColor('#F5F5F7')),
            ('GRID', (0, 0), (-1, -1), 1, colors.HexColor('#E5E5E7'))
        ]))
        
        content.append(table)
        content.append(Spacer(1, 20))
        
        return content
    
    def _create_statistical_section(self, descriptive_stats: Dict[str, Any]) -> list:
        """Create statistical analysis section."""
        content = []
        
        content.append(Paragraph("Statistical Analysis", self.styles['CustomSubtitle']))
        content.append(Spacer(1, 12))
        
        if 'numerical_summary' in descriptive_stats:
            content.append(Paragraph("Numerical Variables Summary", self.styles['SectionHeader']))
            
            # Create stats table for first few numerical columns
            num_stats = descriptive_stats['numerical_summary']
            if num_stats and isinstance(num_stats, dict):
                # Take first 5 columns to avoid overcrowding
                cols = list(num_stats.keys())[:5]
                
                if cols:
                    table_data = [['Statistic'] + cols]
                    
                    stats_to_show = ['mean', 'std', 'min', 'max', 'count']
                    for stat in stats_to_show:
                        row = [stat.title()]
                        for col in cols:
                            if col in num_stats and stat in num_stats[col]:
                                value = num_stats[col][stat]
                                if isinstance(value, float):
                                    row.append(f"{value:.2f}")
                                else:
                                    row.append(str(value))
                            else:
                                row.append('N/A')
                        table_data.append(row)
                    
                    table = Table(table_data)
                    table.setStyle(TableStyle([
                        ('BACKGROUND', (0, 0), (-1, 0), colors.HexColor('#007AFF')),
                        ('TEXTCOLOR', (0, 0), (-1, 0), colors.whitesmoke),
                        ('ALIGN', (0, 0), (-1, -1), 'CENTER'),
                        ('FONTNAME', (0, 0), (-1, 0), 'Helvetica-Bold'),
                        ('FONTSIZE', (0, 0), (-1, -1), 8),
                        ('GRID', (0, 0), (-1, -1), 1, colors.HexColor('#E5E5E7'))
                    ]))
                    
                    content.append(table)
                    content.append(Spacer(1, 20))
        
        return content
    
    def _create_ai_insights_section(self, ai_insights: Dict[str, Any]) -> list:
        """Create AI insights section."""
        content = []
        
        content.append(Paragraph("AI-Powered Insights", self.styles['CustomSubtitle']))
        content.append(Spacer(1, 12))
        
        # Executive summary from AI
        if 'executive_summary' in ai_insights:
            content.append(Paragraph("Executive Summary", self.styles['SectionHeader']))
            exec_summary = ai_insights['executive_summary'].get('content', 'No executive summary available.')
            content.append(Paragraph(exec_summary, self.styles['CustomBody']))
            content.append(Spacer(1, 15))
        
        # Key insights
        if 'insights' in ai_insights:
            content.append(Paragraph("Key Insights", self.styles['SectionHeader']))
            insights = ai_insights['insights'].get('content', 'No insights available.')
            content.append(Paragraph(insights, self.styles['CustomBody']))
            content.append(Spacer(1, 15))
        
        # Recommendations
        if 'recommendations' in ai_insights:
            content.append(Paragraph("Recommendations", self.styles['SectionHeader']))
            recommendations = ai_insights['recommendations'].get('content', 'No recommendations available.')
            content.append(Paragraph(recommendations, self.styles['CustomBody']))
            content.append(Spacer(1, 15))
        
        return content
    
    def _create_footer(self) -> list:
        """Create footer content."""
        content = []
        
        content.append(Spacer(1, 30))
        content.append(Paragraph("Generated by DataScout - Automated Data Analysis Platform", 
                                self.styles['CustomBody']))
        
        return content


# Factory function
def create_windows_pdf_generator() -> WindowsCompatiblePDFGenerator:
    """Create and return a Windows-compatible PDF generator."""
    return WindowsCompatiblePDFGenerator()


# Convenience function
def generate_windows_pdf_report(analysis_results: Dict[str, Any],
                               data_info: Dict[str, Any],
                               ai_insights: Optional[Dict[str, Any]] = None,
                               output_path: Optional[str] = None) -> Union[bytes, str]:
    """Generate PDF report using Windows-compatible method."""
    generator = create_windows_pdf_generator()
    return generator.generate_pdf_from_data(analysis_results, data_info, ai_insights, output_path)
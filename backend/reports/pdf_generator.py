"""
PDF Report Generator for DataScout
Generates PDF reports from HTML reports using WeasyPrint.
"""

import io
import os
import tempfile
from typing import Dict, List, Any, Optional, Union
import pandas as pd
from datetime import datetime
import logging

try:
    from weasyprint import HTML, CSS
    from weasyprint.text.fonts import FontConfiguration
    WEASYPRINT_AVAILABLE = True
except (ImportError, OSError) as e:
    # Fallback for environments without WeasyPrint or missing system libraries
    WEASYPRINT_AVAILABLE = False
    HTML = None
    CSS = None
    FontConfiguration = None
    print(f"WeasyPrint not available ({e}). PDF generation will use alternative method.")

from .html_generator import HTMLReportGenerator, create_html_generator
from .report_templates import ReportTemplate, ReportType, ReportStyle, get_template

logger = logging.getLogger(__name__)


class PDFReportGenerator:
    """
    PDF report generator for DataScout analysis results.
    
    Features:
    - HTML to PDF conversion using WeasyPrint
    - Professional PDF styling
    - Chart and visualization embedding
    - Multiple report formats
    - Optimized for printing and digital viewing
    """
    
    def __init__(self, html_generator: Optional[HTMLReportGenerator] = None):
        """Initialize PDF report generator."""
        self.html_generator = html_generator or create_html_generator()
        self.font_config = FontConfiguration() if WEASYPRINT_AVAILABLE else None
        
    def generate_pdf_from_html(self, html_content: str, 
                              output_path: Optional[str] = None) -> Union[bytes, str]:
        """
        Generate PDF from HTML content.
        
        Args:
            html_content: HTML content to convert
            output_path: Optional file path to save PDF
            
        Returns:
            PDF bytes if no output_path, otherwise file path
        """
        if not WEASYPRINT_AVAILABLE:
            return self._generate_pdf_fallback(html_content, output_path)
            
        try:
            # Add PDF-specific CSS
            pdf_css = self._get_pdf_css()
            
            # Create HTML document
            html_doc = HTML(string=html_content)
            
            # Generate PDF
            if output_path:
                html_doc.write_pdf(
                    output_path,
                    stylesheets=[CSS(string=pdf_css)],
                    font_config=self.font_config
                )
                logger.info(f"PDF report saved to {output_path}")
                return output_path
            else:
                pdf_bytes = html_doc.write_pdf(
                    stylesheets=[CSS(string=pdf_css)],
                    font_config=self.font_config
                )
                logger.info("PDF report generated in memory")
                return pdf_bytes
                
        except Exception as e:
            logger.error(f"PDF generation error: {str(e)}")
            return self._generate_pdf_fallback(html_content, output_path)
    
    def generate_complete_analysis_pdf(self,
                                     analysis_results: Dict[str, Any],
                                     data_info: Dict[str, Any],
                                     charts: Optional[Dict[str, str]] = None,
                                     ai_insights: Optional[Dict[str, Any]] = None,
                                     output_path: Optional[str] = None) -> Union[bytes, str]:
        """Generate a complete analysis PDF report."""
        logger.info("Generating complete analysis PDF report")
        
        # Generate HTML content
        html_content = self.html_generator.generate_complete_analysis_report(
            analysis_results, data_info, charts, ai_insights
        )
        
        # Convert to PDF
        return self.generate_pdf_from_html(html_content, output_path)
    
    def generate_executive_summary_pdf(self,
                                     analysis_results: Dict[str, Any],
                                     data_info: Dict[str, Any],
                                     ai_insights: Optional[Dict[str, Any]] = None,
                                     output_path: Optional[str] = None) -> Union[bytes, str]:
        """Generate an executive summary PDF report."""
        logger.info("Generating executive summary PDF report")
        
        # Generate HTML content
        html_content = self.html_generator.generate_executive_summary_report(
            analysis_results, data_info, ai_insights
        )
        
        # Convert to PDF
        return self.generate_pdf_from_html(html_content, output_path)
    
    def generate_ai_insights_pdf(self,
                               ai_insights: Dict[str, Any],
                               data_info: Dict[str, Any],
                               output_path: Optional[str] = None) -> Union[bytes, str]:
        """Generate an AI insights PDF report."""
        logger.info("Generating AI insights PDF report")
        
        # Generate HTML content
        html_content = self.html_generator.generate_ai_insights_report(
            ai_insights, data_info
        )
        
        # Convert to PDF
        return self.generate_pdf_from_html(html_content, output_path)
    
    def _get_pdf_css(self) -> str:
        """Get PDF-specific CSS styling."""
        return """
        @page {
            size: A4;
            margin: 1.5cm;
            @top-center {
                content: "DataScout Analysis Report";
                font-size: 10pt;
                color: #666;
            }
            @bottom-center {
                content: "Page " counter(page) " of " counter(pages);
                font-size: 9pt;
                color: #666;
            }
        }
        
        body {
            font-size: 10pt;
            line-height: 1.4;
        }
        
        .container {
            box-shadow: none;
            padding: 0;
            margin: 0;
        }
        
        .header {
            page-break-inside: avoid;
            margin-bottom: 20px;
        }
        
        .section {
            page-break-inside: avoid;
            margin-bottom: 25px;
        }
        
        .section h2 {
            page-break-after: avoid;
            font-size: 14pt;
        }
        
        .subsection h3 {
            page-break-after: avoid;
            font-size: 12pt;
        }
        
        .chart-container {
            page-break-inside: avoid;
            text-align: center;
            margin: 15px 0;
        }
        
        .chart-container img {
            max-width: 100%;
            max-height: 400px;
        }
        
        .data-table {
            font-size: 9pt;
            page-break-inside: avoid;
        }
        
        .stats-grid {
            display: flex;
            flex-wrap: wrap;
            gap: 10px;
        }
        
        .stat-card {
            flex: 1;
            min-width: 120px;
            page-break-inside: avoid;
        }
        
        .insight-box {
            page-break-inside: avoid;
            margin: 10px 0;
        }
        
        .recommendations {
            page-break-inside: avoid;
        }
        
        .footer {
            page-break-inside: avoid;
            margin-top: 30px;
        }
        
        /* Ensure proper breaks */
        .page-break {
            page-break-before: always;
        }
        
        .no-break {
            page-break-inside: avoid;
        }
        """
    
    def _generate_pdf_fallback(self, html_content: str, 
                              output_path: Optional[str] = None) -> Union[bytes, str]:
        """
        Fallback PDF generation method when WeasyPrint is not available.
        Uses a simpler HTML-to-PDF approach or saves as HTML.
        """
        logger.warning("Using fallback PDF generation method")
        
        try:
            # Try using pdfkit as fallback
            try:
                import pdfkit
                
                options = {
                    'page-size': 'A4',
                    'margin-top': '1.5cm',
                    'margin-right': '1.5cm',
                    'margin-bottom': '1.5cm',
                    'margin-left': '1.5cm',
                    'encoding': "UTF-8",
                    'no-outline': None,
                    'enable-local-file-access': None
                }
                
                if output_path:
                    pdfkit.from_string(html_content, output_path, options=options)
                    return output_path
                else:
                    pdf_bytes = pdfkit.from_string(html_content, False, options=options)
                    return pdf_bytes
                    
            except ImportError:
                # Final fallback: save as HTML with PDF-like styling
                logger.warning("No PDF libraries available, saving as HTML")
                
                if output_path:
                    # Change extension to .html
                    html_path = output_path.replace('.pdf', '.html')
                    with open(html_path, 'w', encoding='utf-8') as f:
                        f.write(html_content)
                    return html_path
                else:
                    return html_content.encode('utf-8')
                    
        except Exception as e:
            logger.error(f"Fallback PDF generation failed: {str(e)}")
            raise
    
    def create_report_package(self,
                            analysis_results: Dict[str, Any],
                            data_info: Dict[str, Any],
                            charts: Optional[Dict[str, str]] = None,
                            ai_insights: Optional[Dict[str, Any]] = None,
                            output_dir: Optional[str] = None) -> Dict[str, str]:
        """
        Generate a complete package of reports in multiple formats.
        
        Returns:
            Dictionary mapping report types to file paths
        """
        if output_dir is None:
            output_dir = tempfile.mkdtemp(prefix="datascout_reports_")
            
        os.makedirs(output_dir, exist_ok=True)
        
        report_files = {}
        base_name = f"datascout_report_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
        
        try:
            # Complete Analysis Report
            complete_pdf_path = os.path.join(output_dir, f"{base_name}_complete.pdf")
            self.generate_complete_analysis_pdf(
                analysis_results, data_info, charts, ai_insights, complete_pdf_path
            )
            report_files['complete_analysis'] = complete_pdf_path
            
            # Executive Summary Report  
            exec_pdf_path = os.path.join(output_dir, f"{base_name}_executive.pdf")
            self.generate_executive_summary_pdf(
                analysis_results, data_info, ai_insights, exec_pdf_path
            )
            report_files['executive_summary'] = exec_pdf_path
            
            # AI Insights Report (if available)
            if ai_insights:
                ai_pdf_path = os.path.join(output_dir, f"{base_name}_ai_insights.pdf")
                self.generate_ai_insights_pdf(
                    ai_insights, data_info, ai_pdf_path
                )
                report_files['ai_insights'] = ai_pdf_path
                
            # HTML versions for web viewing
            html_complete_path = os.path.join(output_dir, f"{base_name}_complete.html")
            html_content = self.html_generator.generate_complete_analysis_report(
                analysis_results, data_info, charts, ai_insights
            )
            with open(html_complete_path, 'w', encoding='utf-8') as f:
                f.write(html_content)
            report_files['html_complete'] = html_complete_path
            
            logger.info(f"Report package created in {output_dir}")
            return report_files
            
        except Exception as e:
            logger.error(f"Error creating report package: {str(e)}")
            raise


# Factory functions
def create_pdf_generator(html_generator: Optional[HTMLReportGenerator] = None) -> PDFReportGenerator:
    """Create and return a PDFReportGenerator instance."""
    return PDFReportGenerator(html_generator)


def generate_pdf_report(analysis_results: Dict[str, Any],
                       data_info: Dict[str, Any],
                       report_type: str = "complete",
                       charts: Optional[Dict[str, str]] = None,
                       ai_insights: Optional[Dict[str, Any]] = None,
                       output_path: Optional[str] = None,
                       report_style: ReportStyle = ReportStyle.PROFESSIONAL) -> Union[bytes, str]:
    """
    Generate a PDF report of the specified type.
    
    Args:
        analysis_results: Analysis results from DataScout
        data_info: Basic dataset information
        report_type: Type of report ('complete', 'executive', 'ai_insights')
        charts: Optional charts dictionary
        ai_insights: Optional AI insights
        output_path: Optional output file path
        report_style: Report styling
        
    Returns:
        PDF bytes if no output_path, otherwise file path
    """
    # Create HTML generator with appropriate template
    if report_type == "executive":
        template = get_template(ReportType.EXECUTIVE_SUMMARY, report_style)
    elif report_type == "ai_insights":
        template = get_template(ReportType.AI_INSIGHTS, report_style)
    else:
        template = get_template(ReportType.COMPLETE_ANALYSIS, report_style)
    
    html_generator = HTMLReportGenerator(template)
    pdf_generator = PDFReportGenerator(html_generator)
    
    # Generate appropriate report type
    if report_type == "executive":
        return pdf_generator.generate_executive_summary_pdf(
            analysis_results, data_info, ai_insights, output_path
        )
    elif report_type == "ai_insights" and ai_insights:
        return pdf_generator.generate_ai_insights_pdf(
            ai_insights, data_info, output_path
        )
    else:
        return pdf_generator.generate_complete_analysis_pdf(
            analysis_results, data_info, charts, ai_insights, output_path
        )
"""
Report Generation Package for DataScout
Provides comprehensive report generation in multiple formats (HTML, PDF).
Windows-compatible with fallback PDF generation.
"""

from .html_generator import (
    HTMLReportGenerator, create_html_generator,
    generate_analysis_report, generate_executive_report
)

from .report_templates import (
    ReportTemplate, ReportType, ReportStyle,
    get_template, create_custom_template
)

# Try to import PDF generators with Windows compatibility fallback
PDF_GENERATOR_AVAILABLE = False
PDFReportGenerator = None
create_pdf_generator = None
generate_pdf_report = None

try:
    # Try main PDF generator first
    from .pdf_generator import (
        PDFReportGenerator, create_pdf_generator, generate_pdf_report
    )
    PDF_GENERATOR_AVAILABLE = True
except ImportError as e:
    print(f"Main PDF generator not available: {e}")
    try:
        # Fallback to Windows-compatible PDF generator
        from .windows_pdf_generator import (
            WindowsCompatiblePDFGenerator as PDFReportGenerator,
            create_windows_pdf_generator as create_pdf_generator,
            generate_windows_pdf_report as generate_pdf_report
        )
        PDF_GENERATOR_AVAILABLE = True
        print("Using Windows-compatible PDF generator")
    except ImportError as e2:
        print(f"Windows PDF generator not available: {e2}")
        # Create dummy functions for environments without PDF capabilities
        class PDFReportGenerator:
            def __init__(self, *args, **kwargs):
                raise ImportError("No PDF generation libraries available. Install reportlab: pip install reportlab")
        
        def create_pdf_generator(*args, **kwargs):
            raise ImportError("No PDF generation libraries available. Install reportlab: pip install reportlab")
        
        def generate_pdf_report(*args, **kwargs):
            raise ImportError("No PDF generation libraries available. Install reportlab: pip install reportlab")

__all__ = [
    # HTML Generator
    "HTMLReportGenerator", "create_html_generator",
    "generate_analysis_report", "generate_executive_report",
    
    # PDF Generator
    "PDFReportGenerator", "create_pdf_generator",
    "generate_pdf_report", "PDF_GENERATOR_AVAILABLE",
    
    # Templates
    "ReportTemplate", "ReportType", "ReportStyle",
    "get_template", "create_custom_template"
]
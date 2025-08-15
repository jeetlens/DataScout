"""
AI Package for DataScout
Provides AI-powered insights using Google Gemini API and structured prompt templates.
"""

# Import prompt templates (these don't require external AI libraries)
try:
    from .prompt_templates import (
        PromptTemplateManager, PromptType, BusinessDomain, 
        create_prompt_manager, generate_executive_summary_prompt,
        generate_insights_prompt
    )
    PROMPT_TEMPLATES_AVAILABLE = True
except ImportError as e:
    print(f"Prompt templates not available: {e}")
    PROMPT_TEMPLATES_AVAILABLE = False

# Optional AI client imports
try:
    from .gemini_client import (
        GeminiClient, AIInsightGenerator, AIResponse, ModelType,
        create_gemini_client, create_ai_generator, quick_ai_insight
    )
    GEMINI_AVAILABLE = True
except ImportError as e:
    print(f"Gemini client not available: {e}")
    GEMINI_AVAILABLE = False

try:
    from .ai_insight_generator import (
        EnhancedAIInsightGenerator, create_enhanced_ai_generator,
        generate_ai_executive_summary, generate_full_ai_analysis
    )
    AI_INSIGHTS_AVAILABLE = True
except ImportError as e:
    print(f"Enhanced AI insights not available: {e}")
    AI_INSIGHTS_AVAILABLE = False

# Create fallback functions when AI is not available
if not GEMINI_AVAILABLE or not AI_INSIGHTS_AVAILABLE:
    class MockAIGenerator:
        async def generate_comprehensive_ai_insights(self, *args, **kwargs):
            return {
                "insights": ["AI services currently unavailable"],
                "recommendations": ["Install AI dependencies: pip install google-generativeai openai"],
                "status": "AI_UNAVAILABLE"
            }
    
    def create_enhanced_ai_generator():
        return MockAIGenerator()
    
    async def generate_ai_executive_summary(*args, **kwargs):
        return {
            "summary": "AI executive summary unavailable",
            "status": "AI_UNAVAILABLE"
        }
    
    async def generate_full_ai_analysis(*args, **kwargs):
        return {
            "analysis": "AI analysis unavailable", 
            "status": "AI_UNAVAILABLE"
        }

__all__ = [
    # Always available
    "create_enhanced_ai_generator", "generate_ai_executive_summary", "generate_full_ai_analysis"
]

# Add imports based on availability
if PROMPT_TEMPLATES_AVAILABLE:
    __all__.extend([
        "PromptTemplateManager", "PromptType", "BusinessDomain",
        "create_prompt_manager", "generate_executive_summary_prompt",
        "generate_insights_prompt"
    ])

if GEMINI_AVAILABLE:
    __all__.extend([
        "GeminiClient", "AIInsightGenerator", "AIResponse", "ModelType",
        "create_gemini_client", "create_ai_generator", "quick_ai_insight"
    ])

if AI_INSIGHTS_AVAILABLE:
    __all__.extend([
        "EnhancedAIInsightGenerator"
    ])
"""
AI Insight Generator for DataScout
Integrates prompt templates with AI services to generate intelligent data insights.
"""

import asyncio
from typing import Dict, List, Any, Optional, Union
import pandas as pd
import logging
from datetime import datetime

from .prompt_templates import (
    PromptTemplateManager, PromptType, BusinessDomain, 
    create_prompt_manager
)
from .gemini_client import (
    GeminiClient, AIInsightGenerator as BaseAIGenerator, 
    AIResponse, create_ai_generator
)

logger = logging.getLogger(__name__)

class EnhancedAIInsightGenerator:
    """
    Enhanced AI insight generator that combines all AI capabilities.
    
    Features:
    - Intelligent prompt selection and generation
    - Multi-domain business insights
    - Confidence scoring and validation
    - Batch processing capabilities
    - Integration with core analysis modules
    """
    
    def __init__(self, api_key: Optional[str] = None):
        """Initialize enhanced AI insight generator."""
        self.prompt_manager = create_prompt_manager()
        self.ai_generator = create_ai_generator(api_key)
        self.client = self.ai_generator.client
        
    async def generate_comprehensive_ai_insights(self, 
                                               analysis_results: Dict[str, Any],
                                               df: Optional[pd.DataFrame] = None,
                                               business_context: Optional[str] = None,
                                               target_column: Optional[str] = None) -> Dict[str, Any]:
        """
        Generate comprehensive AI-powered insights from analysis results.
        
        Args:
            analysis_results: Results from data analysis pipeline
            df: Original DataFrame (optional)
            business_context: Business context for domain-specific insights
            target_column: Target variable for supervised insights
            
        Returns:
            Dictionary containing all AI-generated insights
        """
        logger.info("Starting comprehensive AI insight generation")
        
        # Determine business domain
        domain = BusinessDomain.GENERAL
        if business_context:
            domain = self.prompt_manager.get_domain_specific_prompt(business_context)
            
        # Create context data from analysis results
        context = self.prompt_manager.create_context_from_analysis(analysis_results, df)
        
        # Generate different types of insights in parallel
        insight_tasks = []
        
        # 1. Executive Summary
        exec_prompt = self.prompt_manager.generate_prompt(
            PromptType.EXECUTIVE_SUMMARY, context, domain
        )
        if exec_prompt:
            insight_tasks.append(
                self._generate_insight_with_metadata("executive_summary", exec_prompt, 
                                                   self.ai_generator.generate_executive_summary)
            )
        
        # 2. Detailed Data Insights
        data_prompt = self.prompt_manager.generate_prompt(
            PromptType.DATA_INSIGHTS, context, domain
        )
        if data_prompt:
            insight_tasks.append(
                self._generate_insight_with_metadata("detailed_insights", data_prompt,
                                                   self.ai_generator.generate_detailed_insights)
            )
        
        # 3. Business Recommendations
        if domain != BusinessDomain.GENERAL or business_context:
            rec_prompt = self.prompt_manager.generate_prompt(
                PromptType.BUSINESS_RECOMMENDATIONS, context, domain
            )
            if rec_prompt:
                insight_tasks.append(
                    self._generate_insight_with_metadata("business_recommendations", rec_prompt,
                                                       self.ai_generator.generate_recommendations)
                )
        
        # 4. Anomaly Explanations (if anomalies exist)
        if context.get("anomaly_count", 0) > 0:
            anomaly_prompt = self.prompt_manager.generate_prompt(
                PromptType.ANOMALY_EXPLANATION, context, domain
            )
            if anomaly_prompt:
                insight_tasks.append(
                    self._generate_insight_with_metadata("anomaly_explanation", anomaly_prompt,
                                                       self.ai_generator.explain_anomalies)
                )
        
        # 5. Correlation Insights (if correlations exist)
        if context.get("strong_correlations"):
            corr_prompt = self.prompt_manager.generate_prompt(
                PromptType.CORRELATION_INSIGHTS, context, domain
            )
            if corr_prompt:
                insight_tasks.append(
                    self._generate_insight_with_metadata("correlation_insights", corr_prompt,
                                                       self.ai_generator.generate_detailed_insights)
                )
        
        # 6. Feature Importance (if target column specified)
        if target_column and context.get("importance_scores"):
            feature_prompt = self.prompt_manager.generate_prompt(
                PromptType.FEATURE_IMPORTANCE, context, domain
            )
            if feature_prompt:
                insight_tasks.append(
                    self._generate_insight_with_metadata("feature_importance", feature_prompt,
                                                       self.ai_generator.generate_detailed_insights)
                )
        
        # Execute all insight generation tasks
        results = await asyncio.gather(*insight_tasks, return_exceptions=True)
        
        # Process results
        ai_insights = {
            "generation_timestamp": datetime.now().isoformat(),
            "business_domain": domain.value,
            "business_context": business_context,
            "target_column": target_column,
            "insights": {},
            "metadata": {
                "total_insights": len([r for r in results if not isinstance(r, Exception)]),
                "generation_errors": len([r for r in results if isinstance(r, Exception)]),
                "api_available": self.client.api_available
            }
        }
        
        # Process successful results
        for result in results:
            if isinstance(result, Exception):
                logger.error(f"Insight generation error: {str(result)}")
                continue
                
            insight_type, response, metadata = result
            ai_insights["insights"][insight_type] = {
                "content": response.content,
                "confidence_score": self._calculate_confidence_score(response),
                "response_time": response.response_time,
                "cached": response.cached,
                "error": response.error,
                "metadata": metadata
            }
        
        # Generate overall quality assessment
        ai_insights["quality_assessment"] = self._assess_insight_quality(ai_insights["insights"])
        
        logger.info(f"Generated {ai_insights['metadata']['total_insights']} AI insights")
        return ai_insights
        
    async def _generate_insight_with_metadata(self, insight_type: str, prompt: str, 
                                            generator_func) -> tuple:
        """Generate insight with metadata tracking."""
        start_time = datetime.now()
        
        try:
            response = await generator_func(prompt)
            
            metadata = {
                "insight_type": insight_type,
                "prompt_length": len(prompt),
                "generation_time": (datetime.now() - start_time).total_seconds(),
                "model_used": response.model_used
            }
            
            return insight_type, response, metadata
            
        except Exception as e:
            logger.error(f"Error generating {insight_type}: {str(e)}")
            raise
            
    def _calculate_confidence_score(self, response: AIResponse) -> float:
        """Calculate confidence score for AI response."""
        if response.error:
            return 0.0
            
        confidence = 100.0
        
        # Penalize for very short responses
        if len(response.content) < 100:
            confidence -= 30
        
        # Penalize for generic responses
        generic_indicators = [
            "I cannot", "I don't have", "insufficient", "more information needed",
            "unable to determine", "not enough data"
        ]
        
        content_lower = response.content.lower()
        generic_count = sum(1 for indicator in generic_indicators if indicator in content_lower)
        confidence -= generic_count * 15
        
        # Reward for structured responses
        structure_indicators = ["1.", "2.", "3.", "•", "-", "**", "###"]
        structure_count = sum(1 for indicator in structure_indicators if indicator in response.content)
        confidence += min(structure_count * 5, 20)
        
        # Penalize for slow responses
        if response.response_time and response.response_time > 20:
            confidence -= 10
            
        return max(0.0, min(100.0, confidence))
        
    def _assess_insight_quality(self, insights: Dict[str, Any]) -> Dict[str, Any]:
        """Assess overall quality of generated insights."""
        if not insights:
            return {
                "overall_score": 0.0,
                "quality_level": "Poor",
                "issues": ["No insights generated"],
                "recommendations": ["Check API configuration and data quality"]
            }
        
        # Calculate average confidence
        confidences = [insight.get("confidence_score", 0) for insight in insights.values()]
        avg_confidence = sum(confidences) / len(confidences) if confidences else 0
        
        # Assess coverage
        expected_insights = ["executive_summary", "detailed_insights"]
        coverage = sum(1 for expected in expected_insights if expected in insights) / len(expected_insights)
        
        # Calculate overall score
        overall_score = (avg_confidence * 0.7) + (coverage * 100 * 0.3)
        
        # Determine quality level
        if overall_score >= 80:
            quality_level = "Excellent"
        elif overall_score >= 60:
            quality_level = "Good"
        elif overall_score >= 40:
            quality_level = "Fair"
        else:
            quality_level = "Poor"
            
        # Identify issues and recommendations
        issues = []
        recommendations = []
        
        if avg_confidence < 50:
            issues.append("Low confidence in AI responses")
            recommendations.append("Review data quality and completeness")
            
        if coverage < 0.5:
            issues.append("Limited insight coverage")
            recommendations.append("Ensure sufficient data for comprehensive analysis")
            
        errors = [insight for insight in insights.values() if insight.get("error")]
        if errors:
            issues.append(f"{len(errors)} insights had generation errors")
            recommendations.append("Check AI service availability and configuration")
            
        return {
            "overall_score": round(overall_score, 1),
            "quality_level": quality_level,
            "average_confidence": round(avg_confidence, 1),
            "coverage_score": round(coverage * 100, 1),
            "issues": issues,
            "recommendations": recommendations
        }
        
    async def generate_targeted_insight(self, prompt_type: PromptType,
                                      analysis_results: Dict[str, Any],
                                      df: Optional[pd.DataFrame] = None,
                                      business_context: Optional[str] = None) -> Dict[str, Any]:
        """Generate a specific type of AI insight."""
        domain = BusinessDomain.GENERAL
        if business_context:
            domain = self.prompt_manager.get_domain_specific_prompt(business_context)
            
        context = self.prompt_manager.create_context_from_analysis(analysis_results, df)
        
        prompt = self.prompt_manager.generate_prompt(prompt_type, context, domain)
        if not prompt:
            return {
                "error": f"Could not generate prompt for {prompt_type.value}",
                "success": False
            }
            
        # Select appropriate generation method
        if prompt_type == PromptType.EXECUTIVE_SUMMARY:
            response = await self.ai_generator.generate_executive_summary(prompt)
        elif prompt_type == PromptType.BUSINESS_RECOMMENDATIONS:
            response = await self.ai_generator.generate_recommendations(prompt)
        elif prompt_type == PromptType.ANOMALY_EXPLANATION:
            response = await self.ai_generator.explain_anomalies(prompt)
        else:
            response = await self.ai_generator.generate_detailed_insights(prompt)
            
        return {
            "content": response.content,
            "prompt_type": prompt_type.value,
            "confidence_score": self._calculate_confidence_score(response),
            "response_time": response.response_time,
            "cached": response.cached,
            "error": response.error,
            "success": not response.error
        }
        
    async def explain_data_story(self, analysis_results: Dict[str, Any],
                               df: Optional[pd.DataFrame] = None,
                               business_context: Optional[str] = None) -> Dict[str, Any]:
        """Generate a cohesive data story from analysis results."""
        domain = BusinessDomain.GENERAL
        if business_context:
            domain = self.prompt_manager.get_domain_specific_prompt(business_context)
            
        context = self.prompt_manager.create_context_from_analysis(analysis_results, df)
        
        # Create a comprehensive story prompt
        story_prompt = f"""
Create a cohesive data story from this analysis:

**Dataset Overview:**
- Records: {context.get('total_rows', 'N/A'):,}
- Features: {context.get('total_columns', 'N/A')}
- Quality Score: {context.get('quality_score', 'N/A')}/100

**Key Findings:**
{self._format_key_findings(analysis_results)}

**Statistical Highlights:**
{self._format_statistical_highlights(analysis_results)}

Please create a compelling data story that:
1. Sets the context and explains what the data represents
2. Walks through the most important discoveries in logical order
3. Explains the business implications of each finding
4. Concludes with actionable recommendations
5. Uses clear, non-technical language suitable for business stakeholders

Structure this as a narrative that tells the story of what the data reveals.
"""
        
        response = await self.ai_generator.generate_detailed_insights(story_prompt)
        
        return {
            "data_story": response.content,
            "confidence_score": self._calculate_confidence_score(response),
            "response_time": response.response_time,
            "cached": response.cached,
            "error": response.error,
            "success": not response.error
        }
        
    def _format_key_findings(self, analysis_results: Dict[str, Any]) -> str:
        """Format key findings for story generation."""
        findings = []
        
        # Quality insights
        if "data_quality_insights" in analysis_results:
            quality = analysis_results["data_quality_insights"]
            missing_pct = quality.get("missing_percentage_overall", 0)
            if missing_pct > 0:
                findings.append(f"- Data completeness: {100-missing_pct:.1f}% complete")
                
        # Correlation insights
        if "correlation_analysis" in analysis_results:
            strong_corrs = analysis_results["correlation_analysis"].get("strong_correlations", [])
            if strong_corrs:
                findings.append(f"- Found {len(strong_corrs)} strong correlations between variables")
                
        # Anomaly insights
        if "anomaly_detection" in analysis_results:
            outliers = analysis_results["anomaly_detection"].get("statistical_outliers", {})
            if outliers:
                findings.append(f"- Detected anomalies in {len(outliers)} variables")
                
        return "\n".join(findings) if findings else "No specific findings to highlight"
        
    def _format_statistical_highlights(self, analysis_results: Dict[str, Any]) -> str:
        """Format statistical highlights for story generation."""
        highlights = []
        
        if "descriptive_stats" in analysis_results:
            stats = analysis_results["descriptive_stats"]
            for col, col_stats in list(stats.items())[:3]:  # Top 3 columns
                if isinstance(col_stats, dict) and "mean" in col_stats:
                    highlights.append(f"- {col}: Average {col_stats['mean']:.2f}, Range {col_stats['min']:.2f} to {col_stats['max']:.2f}")
                    
        return "\n".join(highlights) if highlights else "Statistical analysis completed"
        
    def get_ai_capabilities(self) -> Dict[str, Any]:
        """Get information about AI capabilities and status."""
        return {
            "ai_available": self.client.api_available,
            "model_type": self.client.model_type.value if self.client.model_type else None,
            "supported_insights": [ptype.value for ptype in PromptType],
            "supported_domains": [domain.value for domain in BusinessDomain],
            "cache_stats": self.client.get_usage_stats(),
            "prompt_templates": len(self.prompt_manager.templates)
        }


# Factory function
def create_enhanced_ai_generator(api_key: Optional[str] = None) -> EnhancedAIInsightGenerator:
    """Create and return an EnhancedAIInsightGenerator instance."""
    return EnhancedAIInsightGenerator(api_key)


# Convenience functions
async def generate_ai_executive_summary(analysis_results: Dict[str, Any],
                                      df: Optional[pd.DataFrame] = None,
                                      business_context: Optional[str] = None,
                                      api_key: Optional[str] = None) -> Dict[str, Any]:
    """Generate AI-powered executive summary."""
    generator = create_enhanced_ai_generator(api_key)
    return await generator.generate_targeted_insight(
        PromptType.EXECUTIVE_SUMMARY, analysis_results, df, business_context
    )


async def generate_full_ai_analysis(analysis_results: Dict[str, Any],
                                  df: Optional[pd.DataFrame] = None,
                                  business_context: Optional[str] = None,
                                  target_column: Optional[str] = None,
                                  api_key: Optional[str] = None) -> Dict[str, Any]:
    """Generate comprehensive AI analysis."""
    generator = create_enhanced_ai_generator(api_key)
    return await generator.generate_comprehensive_ai_insights(
        analysis_results, df, business_context, target_column
    )
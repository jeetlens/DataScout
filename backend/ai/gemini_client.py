"""
Gemini API Integration for DataScout
Google Gemini API client for AI-powered data analysis insights.
"""

import google.generativeai as genai
from typing import Dict, List, Any, Optional, Union
import asyncio
import logging
import os
from dataclasses import dataclass
from enum import Enum
import time
import json
from pathlib import Path

logger = logging.getLogger(__name__)

class ModelType(Enum):
    """Available Gemini models."""
    GEMINI_PRO = "gemini-pro"
    GEMINI_PRO_VISION = "gemini-pro-vision"

@dataclass
class AIResponse:
    """Structure for AI API responses."""
    content: str
    model_used: str
    tokens_used: Optional[int] = None
    response_time: Optional[float] = None
    confidence_score: Optional[float] = None
    error: Optional[str] = None
    cached: bool = False

class GeminiClient:
    """
    Google Gemini API client for generating AI insights.
    
    Features:
    - Secure API key management
    - Response caching and optimization
    - Error handling and retry logic
    - Rate limiting and throttling
    - Multiple model support
    """
    
    def __init__(self, api_key: Optional[str] = None, 
                 model_type: ModelType = ModelType.GEMINI_PRO,
                 cache_dir: Optional[str] = None):
        """
        Initialize Gemini client.
        
        Args:
            api_key: Google AI API key (or use GOOGLE_AI_API_KEY env var)
            model_type: Gemini model to use
            cache_dir: Directory for caching responses
        """
        self.model_type = model_type
        self.cache_dir = Path(cache_dir) if cache_dir else Path("./ai_cache")
        self.cache_dir.mkdir(exist_ok=True)
        
        # Initialize API
        self._setup_api(api_key)
        self._setup_model()
        
        # Response cache
        self.response_cache = {}
        self.load_cache()
        
        # Rate limiting
        self.last_request_time = 0
        self.min_request_interval = 1.0  # Minimum seconds between requests
        
    def _setup_api(self, api_key: Optional[str]):
        """Setup Google AI API configuration."""
        # Get API key from parameter or environment
        if api_key:
            self.api_key = api_key
        else:
            self.api_key = os.getenv('GOOGLE_AI_API_KEY')
            
        if not self.api_key:
            logger.warning("No Google AI API key provided. AI features will be limited.")
            self.api_available = False
            return
            
        try:
            genai.configure(api_key=self.api_key)
            self.api_available = True
            logger.info("Google Gemini API configured successfully")
        except Exception as e:
            logger.error(f"Failed to configure Gemini API: {str(e)}")
            self.api_available = False
            
    def _setup_model(self):
        """Initialize the Gemini model."""
        if not self.api_available:
            self.model = None
            return
            
        try:
            self.model = genai.GenerativeModel(self.model_type.value)
            logger.info(f"Initialized {self.model_type.value} model")
        except Exception as e:
            logger.error(f"Failed to initialize model: {str(e)}")
            self.model = None
            self.api_available = False
            
    def _rate_limit(self):
        """Enforce rate limiting between requests."""
        current_time = time.time()
        time_since_last = current_time - self.last_request_time
        
        if time_since_last < self.min_request_interval:
            sleep_time = self.min_request_interval - time_since_last
            time.sleep(sleep_time)
            
        self.last_request_time = time.time()
        
    def _get_cache_key(self, prompt: str, parameters: Dict[str, Any]) -> str:
        """Generate cache key for prompt and parameters."""
        import hashlib
        
        cache_data = {
            "prompt": prompt,
            "parameters": parameters,
            "model": self.model_type.value
        }
        
        cache_string = json.dumps(cache_data, sort_keys=True)
        return hashlib.md5(cache_string.encode()).hexdigest()
        
    def load_cache(self):
        """Load response cache from disk."""
        cache_file = self.cache_dir / "response_cache.json"
        
        if cache_file.exists():
            try:
                with open(cache_file, 'r') as f:
                    self.response_cache = json.load(f)
                logger.info(f"Loaded {len(self.response_cache)} cached responses")
            except Exception as e:
                logger.warning(f"Failed to load cache: {str(e)}")
                self.response_cache = {}
        else:
            self.response_cache = {}
            
    def save_cache(self):
        """Save response cache to disk."""
        cache_file = self.cache_dir / "response_cache.json"
        
        try:
            with open(cache_file, 'w') as f:
                json.dump(self.response_cache, f, indent=2)
            logger.debug("Saved response cache")
        except Exception as e:
            logger.warning(f"Failed to save cache: {str(e)}")
            
    async def generate_insight(self, prompt: str, 
                             temperature: float = 0.3,
                             max_tokens: Optional[int] = None,
                             use_cache: bool = True) -> AIResponse:
        """
        Generate AI insight from prompt.
        
        Args:
            prompt: Input prompt for AI
            temperature: Response creativity (0.0-1.0)
            max_tokens: Maximum response length
            use_cache: Whether to use cached responses
            
        Returns:
            AIResponse object with generated content
        """
        if not self.api_available or not self.model:
            return AIResponse(
                content="AI service unavailable. Please check API configuration.",
                model_used=self.model_type.value,
                error="API not available"
            )
            
        # Check cache
        parameters = {
            "temperature": temperature,
            "max_tokens": max_tokens
        }
        
        cache_key = self._get_cache_key(prompt, parameters)
        
        if use_cache and cache_key in self.response_cache:
            cached_response = self.response_cache[cache_key]
            logger.info("Using cached AI response")
            return AIResponse(
                content=cached_response["content"],
                model_used=cached_response["model_used"],
                tokens_used=cached_response.get("tokens_used"),
                response_time=cached_response.get("response_time"),
                cached=True
            )
            
        # Generate new response
        try:
            self._rate_limit()
            start_time = time.time()
            
            # Configure generation parameters
            generation_config = genai.types.GenerationConfig(
                temperature=temperature,
                max_output_tokens=max_tokens,
            )
            
            # Generate response
            response = await asyncio.get_event_loop().run_in_executor(
                None, 
                lambda: self.model.generate_content(
                    prompt,
                    generation_config=generation_config
                )
            )
            
            response_time = time.time() - start_time
            
            # Extract content
            if response.candidates and response.candidates[0].content:
                content = response.candidates[0].content.parts[0].text
            else:
                content = "No response generated"
                
            # Create response object
            ai_response = AIResponse(
                content=content,
                model_used=self.model_type.value,
                tokens_used=self._estimate_tokens(prompt + content),
                response_time=response_time
            )
            
            # Cache the response
            if use_cache:
                self.response_cache[cache_key] = {
                    "content": content,
                    "model_used": self.model_type.value,
                    "tokens_used": ai_response.tokens_used,
                    "response_time": response_time,
                    "timestamp": time.time()
                }
                self.save_cache()
                
            logger.info(f"Generated AI response in {response_time:.2f}s")
            return ai_response
            
        except Exception as e:
            logger.error(f"Error generating AI response: {str(e)}")
            return AIResponse(
                content=f"Error generating AI insight: {str(e)}",
                model_used=self.model_type.value,
                error=str(e)
            )
            
    def _estimate_tokens(self, text: str) -> int:
        """Rough estimation of token count."""
        # Approximate: 1 token ≈ 4 characters for English text
        return len(text) // 4
        
    async def generate_batch_insights(self, prompts: List[str],
                                    temperature: float = 0.3,
                                    max_tokens: Optional[int] = None,
                                    use_cache: bool = True) -> List[AIResponse]:
        """
        Generate insights for multiple prompts.
        
        Args:
            prompts: List of prompts to process
            temperature: Response creativity
            max_tokens: Maximum response length
            use_cache: Whether to use cached responses
            
        Returns:
            List of AIResponse objects
        """
        results = []
        
        for i, prompt in enumerate(prompts):
            logger.info(f"Processing prompt {i+1}/{len(prompts)}")
            
            response = await self.generate_insight(
                prompt, temperature, max_tokens, use_cache
            )
            results.append(response)
            
            # Add delay between batch requests
            if i < len(prompts) - 1:
                await asyncio.sleep(1.0)
                
        return results
        
    def validate_response_quality(self, response: AIResponse, 
                                min_length: int = 50) -> Dict[str, Any]:
        """
        Validate the quality of AI response.
        
        Args:
            response: AIResponse to validate
            min_length: Minimum expected response length
            
        Returns:
            Validation results
        """
        validation = {
            "is_valid": True,
            "issues": [],
            "quality_score": 100.0
        }
        
        # Check for errors
        if response.error:
            validation["is_valid"] = False
            validation["issues"].append(f"API Error: {response.error}")
            validation["quality_score"] -= 50
            
        # Check response length
        if len(response.content) < min_length:
            validation["issues"].append("Response too short")
            validation["quality_score"] -= 20
            
        # Check for generic/unhelpful responses
        generic_phrases = [
            "I cannot", "I'm sorry", "I don't have", "insufficient data",
            "more information needed", "unable to determine"
        ]
        
        content_lower = response.content.lower()
        generic_count = sum(1 for phrase in generic_phrases if phrase in content_lower)
        
        if generic_count > 2:
            validation["issues"].append("Response appears generic or unhelpful")
            validation["quality_score"] -= 30
            
        # Check response time (if available)
        if response.response_time and response.response_time > 30:
            validation["issues"].append("Slow response time")
            validation["quality_score"] -= 10
            
        validation["quality_score"] = max(0, validation["quality_score"])
        
        return validation
        
    def get_usage_stats(self) -> Dict[str, Any]:
        """Get API usage statistics."""
        total_cached = len(self.response_cache)
        cache_size_mb = sum(len(json.dumps(v)) for v in self.response_cache.values()) / 1024 / 1024
        
        return {
            "api_available": self.api_available,
            "model_type": self.model_type.value,
            "cached_responses": total_cached,
            "cache_size_mb": round(cache_size_mb, 2),
            "cache_directory": str(self.cache_dir)
        }
        
    def clear_cache(self, older_than_days: Optional[int] = None):
        """
        Clear response cache.
        
        Args:
            older_than_days: Only clear responses older than N days
        """
        if older_than_days is None:
            # Clear all cache
            self.response_cache = {}
            logger.info("Cleared all cached responses")
        else:
            # Clear old responses
            current_time = time.time()
            cutoff_time = current_time - (older_than_days * 24 * 3600)
            
            old_keys = [
                key for key, value in self.response_cache.items()
                if value.get("timestamp", 0) < cutoff_time
            ]
            
            for key in old_keys:
                del self.response_cache[key]
                
            logger.info(f"Cleared {len(old_keys)} cached responses older than {older_than_days} days")
            
        self.save_cache()


class AIInsightGenerator:
    """
    High-level AI insight generator that combines prompts with Gemini API.
    """
    
    def __init__(self, api_key: Optional[str] = None,
                 model_type: ModelType = ModelType.GEMINI_PRO):
        """Initialize AI insight generator."""
        self.client = GeminiClient(api_key, model_type)
        
    async def generate_executive_summary(self, prompt: str) -> AIResponse:
        """Generate executive summary with optimized parameters."""
        return await self.client.generate_insight(
            prompt,
            temperature=0.2,  # Lower temperature for factual summaries
            max_tokens=1000
        )
        
    async def generate_detailed_insights(self, prompt: str) -> AIResponse:
        """Generate detailed insights with balanced creativity."""
        return await self.client.generate_insight(
            prompt,
            temperature=0.4,  # Moderate temperature for insights
            max_tokens=2000
        )
        
    async def generate_recommendations(self, prompt: str) -> AIResponse:
        """Generate business recommendations with higher creativity."""
        return await self.client.generate_insight(
            prompt,
            temperature=0.6,  # Higher temperature for creative recommendations
            max_tokens=1500
        )
        
    async def explain_anomalies(self, prompt: str) -> AIResponse:
        """Explain anomalies with factual approach."""
        return await self.client.generate_insight(
            prompt,
            temperature=0.1,  # Very low temperature for factual explanations
            max_tokens=1200
        )


# Factory functions
def create_gemini_client(api_key: Optional[str] = None) -> GeminiClient:
    """Create and return a GeminiClient instance."""
    return GeminiClient(api_key)


def create_ai_generator(api_key: Optional[str] = None) -> AIInsightGenerator:
    """Create and return an AIInsightGenerator instance."""
    return AIInsightGenerator(api_key)


# Convenience functions
async def quick_ai_insight(prompt: str, api_key: Optional[str] = None) -> str:
    """Generate quick AI insight from prompt."""
    client = create_gemini_client(api_key)
    response = await client.generate_insight(prompt)
    return response.content


async def generate_summary_with_ai(analysis_results: Dict[str, Any], 
                                 df_info: Dict[str, Any],
                                 api_key: Optional[str] = None) -> AIResponse:
    """Generate AI-powered summary from analysis results."""
    from .prompt_templates import generate_executive_summary_prompt
    
    # Generate appropriate prompt
    prompt = generate_executive_summary_prompt(analysis_results, None)
    if not prompt:
        return AIResponse(
            content="Unable to generate prompt for AI analysis",
            model_used="gemini-pro",
            error="Prompt generation failed"
        )
    
    # Generate AI response
    generator = create_ai_generator(api_key)
    return await generator.generate_executive_summary(prompt)
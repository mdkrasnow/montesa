# AI.py
import os
from dotenv import load_dotenv
import logging
import json
import nest_asyncio
from typing import List, Dict, Any, Optional, Union
from datetime import date, datetime
import google.generativeai as genai
from copy import deepcopy
from concurrent.futures import ThreadPoolExecutor, as_completed
import threading
from dataclasses import dataclass

dotenv_path = os.path.join(os.path.dirname(__file__), '.env')
load_dotenv(dotenv_path)

# Initialize nest_asyncio
nest_asyncio.apply()

# Logger setup
handler = logging.StreamHandler()
handler.setFormatter(logging.Formatter(
    '%(asctime)s - %(name)s - %(levelname)s - %(message)s'))
logger = logging.getLogger(__name__)
logger.addHandler(handler)
logger.setLevel(logging.DEBUG)

# Global ThreadPoolExecutor for parallel API calls
executor = ThreadPoolExecutor(max_workers=int(os.getenv("THREADPOOL_MAX_WORKERS", "20")))

# Configure APIs
genai.configure(api_key=os.getenv("GOOGLE_API_KEY"))

# === API CLIENT CONFIGURATION ===
@dataclass
class ModelConfig:
    """Configuration for AI model selection and parameters"""
    provider: str = "google"  # google, openai, openrouter, groq
    model: str = "gemini-2.0-flash"
    temperature: float = 1.0
    max_tokens: Optional[int] = None
    reasoning_effort: Optional[str] = None  # low, medium, high (for reasoning models)
    json_mode: bool = False
    json_schema: Optional[Dict] = None
    use_structured_output: bool = False

# Initialize API clients
clients = {}

# OpenAI Client
USE_OPENAI = os.getenv("USE_OPENAI", "false").lower() == "true"
if USE_OPENAI:
    try:
        from openai import OpenAI
        openai_api_key = os.getenv("OPENAI_API_KEY")
        if openai_api_key:
            clients["openai"] = OpenAI(api_key=openai_api_key)
            logger.info("OpenAI client initialized")
        else:
            logger.warning("OPENAI_API_KEY environment variable not set")
            USE_OPENAI = False
    except ImportError:
        logger.warning("OpenAI library not installed. Install with: pip install openai")
        USE_OPENAI = False

# OpenRouter Client
USE_OPENROUTER = os.getenv("USE_OPENROUTER", "false").lower() == "true"
if USE_OPENROUTER:
    try:
        from openai import OpenAI  # OpenRouter uses OpenAI-compatible API
        openrouter_api_key = os.getenv("OPENROUTER_API_KEY")
        if openrouter_api_key:
            clients["openrouter"] = OpenAI(
                api_key=openrouter_api_key,
                base_url="https://openrouter.ai/api/v1"
            )
            logger.info("OpenRouter client initialized")
        else:
            logger.warning("OPENROUTER_API_KEY environment variable not set")
            USE_OPENROUTER = False
    except ImportError:
        logger.warning("OpenAI library not installed. Install with: pip install openai")
        USE_OPENROUTER = False

# Groq Client (existing)
USE_GROQ = os.getenv("USE_GROQ", "true").lower() == "true"
if USE_GROQ:
    try:
        from groq import Groq
        groq_api_key = os.getenv("GROQ_API_KEY")
        if groq_api_key:
            clients["groq"] = Groq(api_key=groq_api_key)
            logger.info("Groq client initialized")
        else:
            logger.warning("GROQ_API_KEY environment variable not set")
            USE_GROQ = False
    except ImportError:
        logger.warning("Groq library not installed. Install with: pip install groq")
        USE_GROQ = False

# === RESPONSE WRAPPER CLASSES ===
class GroqResponsePart:
    def __init__(self, text: str):
        self.text = text

class GroqResponse:
    def __init__(self, content: str):
        self.parts = [GroqResponsePart(content)]
        self.text = content

class OpenAIResponsePart:
    def __init__(self, text: str):
        self.text = text

class OpenAIResponse:
    def __init__(self, content: str, parsed_data: Optional[Any] = None):
        self.parts = [OpenAIResponsePart(content)]
        self.text = content
        self.parsed = parsed_data

# Add thread local storage for request-specific configuration
thread_local = threading.local()

# === DEFAULT MODEL CONFIGURATIONS ===
DEFAULT_MODELS = {
    "google": "gemini-2.0-flash",
    "openai": "gpt-4o",
    "openrouter": "anthropic/claude-3.5-sonnet",
    "groq": "llama-3.1-8b-instant"
}

REASONING_MODELS = {
    "openai": ["o1", "o1-mini", "o1-preview"],
    "google": ["gemini-2.5-pro", "gemini-2.5-flash"],  # Built-in reasoning
    "openrouter": ["openai/o1", "openai/o1-mini", "anthropic/claude-3.5-sonnet"]
}

def is_reasoning_model(provider: str, model: str) -> bool:
    """Check if the specified model is a reasoning model"""
    if provider not in REASONING_MODELS:
        return False
    
    reasoning_models = REASONING_MODELS[provider]
    return any(reasoning_model in model for reasoning_model in reasoning_models)

def prepare_messages_for_provider(messages: Any, provider: str) -> List[Dict]:
    """Convert messages to the appropriate format for each provider"""
    if isinstance(messages, str):
        return [{"role": "user", "content": messages}]
    
    if isinstance(messages, list):
        processed_messages = []
        for msg in messages:
            if isinstance(msg, dict):
                # Handle different message formats
                if provider in ["openai", "openrouter", "groq"]:
                    # OpenAI-compatible format
                    if 'parts' in msg:
                        # Convert from Gemini format
                        content = ""
                        for part in msg['parts']:
                            if 'text' in part:
                                content += part['text']
                        processed_messages.append({
                            "role": msg.get("role", "user"),
                            "content": content
                        })
                    else:
                        processed_messages.append(msg)
                else:
                    # Google format (existing logic)
                    if 'parts' not in msg:
                        if 'content' in msg:
                            if isinstance(msg['content'], str):
                                msg['parts'] = [{"text": str(msg['content'])}]
                            elif isinstance(msg['content'], dict):
                                msg['parts'] = [msg['content']]
                            del msg['content']
                        elif 'text' in msg:
                            msg['parts'] = [{"text": str(msg['text'])}]
                            del msg['text']
                    processed_messages.append(msg)
            else:
                # String message
                if provider in ["openai", "openrouter", "groq"]:
                    processed_messages.append({"role": "user", "content": str(msg)})
                else:
                    processed_messages.append({"role": "user", "parts": [{"text": str(msg)}]})
        
        return processed_messages
    
    # Single string case
    if provider in ["openai", "openrouter", "groq"]:
        return [{"role": "user", "content": str(messages)}]
    else:
        return [{"role": "user", "parts": [{"text": str(messages)}]}]

def create_openai_response_format(config: ModelConfig) -> Optional[Dict]:
    """Create response format for OpenAI-compatible APIs"""
    if config.json_schema and config.use_structured_output:
        return {
            "type": "json_schema",
            "json_schema": {
                "name": "structured_response",
                "schema": config.json_schema,
                "strict": True
            }
        }
    elif config.json_mode:
        return {"type": "json_object"}
    
    return None

def create_google_generation_config(config: ModelConfig) -> Optional[Any]:
    """Create generation config for Google Gemini"""
    generation_config = {}
    
    if config.temperature is not None:
        generation_config["temperature"] = config.temperature
    
    if config.max_tokens:
        generation_config["max_output_tokens"] = config.max_tokens
    
    if config.json_mode or config.use_structured_output:
        generation_config["response_mime_type"] = "application/json"
        if config.json_schema:
            generation_config["response_schema"] = config.json_schema
    
    # Create GenerationConfig object if we have configuration
    if generation_config:
        return genai.GenerationConfig(**generation_config)
    
    return None

def generate_ai_content(
    messages: Any, 
    generation_config: Optional[Any] = None,
    model_config: Optional[ModelConfig] = None,
    model_override: Optional[str] = None
) -> Any:
    """
    Universal wrapper for generating AI content using multiple providers.
    
    Args:
        messages: String or list of message dicts
        generation_config: Legacy generation config (for backward compatibility)
        model_config: ModelConfig object specifying provider and parameters
        model_override: Optional model name to override default
    
    Returns:
        Response object with .text and .parts attributes
    """
    
    # Handle backward compatibility
    if model_config is None:
        # Extract provider from environment or default to google
        provider = getattr(thread_local, 'provider', os.getenv("DEFAULT_PROVIDER", "google"))
        
        # Create default config based on legacy parameters
        model_config = ModelConfig(
            provider=provider,
            model=model_override or DEFAULT_MODELS.get(provider, "gemini-2.0-flash")
        )
        
        # Extract parameters from legacy generation_config
        if generation_config:
            if hasattr(generation_config, "temperature"):
                model_config.temperature = generation_config.temperature
            if hasattr(generation_config, "response_mime_type") and generation_config.response_mime_type == "application/json":
                model_config.json_mode = True
    
    # Apply model override if provided
    if model_override:
        model_config.model = model_override
    
    # Route to appropriate provider
    if model_config.provider == "openai" and USE_OPENAI:
        return _generate_openai_content(messages, model_config)
    elif model_config.provider == "openrouter" and USE_OPENROUTER:
        return _generate_openrouter_content(messages, model_config)
    elif model_config.provider == "groq" and USE_GROQ:
        return _generate_groq_content(messages, model_config)
    elif model_config.provider == "google":
        return _generate_google_content(messages, model_config)
    else:
        # Fallback to Google
        logger.warning(f"Provider {model_config.provider} not available, falling back to Google")
        model_config.provider = "google"
        return _generate_google_content(messages, model_config)

def _generate_openai_content(messages: Any, config: ModelConfig) -> OpenAIResponse:
    """Generate content using OpenAI API"""
    client = clients["openai"]
    processed_messages = prepare_messages_for_provider(messages, "openai")
    
    # Prepare request parameters
    request_params = {
        "model": config.model,
        "messages": processed_messages,
        "temperature": config.temperature
    }
    
    if config.max_tokens:
        if is_reasoning_model("openai", config.model):
            request_params["max_completion_tokens"] = config.max_tokens
        else:
            request_params["max_tokens"] = config.max_tokens
    
    # Add reasoning effort for reasoning models
    if is_reasoning_model("openai", config.model) and config.reasoning_effort:
        request_params["reasoning_effort"] = config.reasoning_effort
    
    # Add response format
    response_format = create_openai_response_format(config)
    if response_format:
        request_params["response_format"] = response_format
    
    # Use structured output parsing if available and requested
    if config.use_structured_output and config.json_schema:
        try:
            # Try using the beta parse method for structured outputs
            from pydantic import BaseModel, create_model
            
            # Create a dynamic Pydantic model from JSON schema
            # This is a simplified approach - for production, you'd want more robust schema parsing
            response = client.beta.chat.completions.parse(
                **request_params,
                response_format=response_format
            )
            
            content = response.choices[0].message.content
            parsed_data = response.choices[0].message.parsed if hasattr(response.choices[0].message, 'parsed') else None
            
            return OpenAIResponse(content, parsed_data)
        except Exception as e:
            logger.warning(f"Structured output parsing failed, falling back to regular completion: {e}")
    
    # Regular completion
    response = client.chat.completions.create(**request_params)
    content = response.choices[0].message.content
    
    return OpenAIResponse(content)

def _generate_openrouter_content(messages: Any, config: ModelConfig) -> OpenAIResponse:
    """Generate content using OpenRouter API"""
    client = clients["openrouter"]
    processed_messages = prepare_messages_for_provider(messages, "openrouter")
    
    # Prepare request parameters
    request_params = {
        "model": config.model,
        "messages": processed_messages,
        "temperature": config.temperature
    }
    
    if config.max_tokens:
        request_params["max_tokens"] = config.max_tokens
    
    # Add response format for JSON output
    response_format = create_openai_response_format(config)
    if response_format:
        request_params["response_format"] = response_format
    
    # OpenRouter headers for better service
    extra_headers = {
        "HTTP-Referer": os.getenv("APP_URL", "https://localhost:3000"),
        "X-Title": os.getenv("APP_NAME", "AI Content Generator")
    }
    
    response = client.chat.completions.create(
        **request_params,
        extra_headers=extra_headers
    )
    
    content = response.choices[0].message.content
    return OpenAIResponse(content)

def _generate_groq_content(messages: Any, config: ModelConfig) -> GroqResponse:
    """Generate content using Groq API"""
    client = clients["groq"]
    processed_messages = prepare_messages_for_provider(messages, "groq")
    
    # Prepare request parameters
    request_params = {
        "model": config.model,
        "messages": processed_messages,
        "temperature": config.temperature
    }
    
    if config.max_tokens:
        request_params["max_tokens"] = config.max_tokens
    
    # Add response format for JSON output
    if config.json_mode or config.use_structured_output:
        request_params["response_format"] = {"type": "json_object"}
    
    response = client.chat.completions.create(**request_params)
    content = response.choices[0].message.content
    
    return GroqResponse(content)

def _generate_google_content(messages: Any, config: ModelConfig) -> Any:
    """Generate content using Google Gemini API"""
    # Handle reasoning models
    model_name = config.model
    if "gemini-2.5" in model_name and config.reasoning_effort:
        # For Gemini 2.5 models, reasoning is built-in, no special parameters needed
        logger.info(f"Using Gemini 2.5 reasoning model: {model_name}")
    
    model = genai.GenerativeModel(model_name)
    processed_messages = prepare_messages_for_provider(messages, "google")
    
    # Create generation config
    generation_config = create_google_generation_config(config)
    
    # Generate content
    if isinstance(processed_messages, list) and len(processed_messages) > 1:
        return model.generate_content(processed_messages, generation_config=generation_config)
    else:
        # Single message
        content = processed_messages[0] if isinstance(processed_messages, list) else processed_messages
        return model.generate_content(content, generation_config=generation_config)

def upload_multimodal_file(file_path: str) -> Any:
    """
    Upload a file (audio or video) to Gemini File API and return the file reference.
    Supports large files beyond inline limits.

    Args:
        file_path (str): Local filesystem path to the media file.

    Returns:
        Any: File reference object usable in generate_content messages.
    """
    return genai.upload_file(file_path)

# === CONVENIENCE FUNCTIONS FOR DIFFERENT USE CASES ===

def generate_with_openai(
    messages: Any,
    model: str = "gpt-4o",
    temperature: float = 1.0,
    max_tokens: Optional[int] = None,
    json_mode: bool = False,
    json_schema: Optional[Dict] = None,
    reasoning_effort: Optional[str] = None
) -> OpenAIResponse:
    """Convenience function for OpenAI generation"""
    config = ModelConfig(
        provider="openai",
        model=model,
        temperature=temperature,
        max_tokens=max_tokens,
        json_mode=json_mode,
        json_schema=json_schema,
        use_structured_output=bool(json_schema),
        reasoning_effort=reasoning_effort
    )
    return generate_ai_content(messages, model_config=config)

def generate_with_openrouter(
    messages: Any,
    model: str = "anthropic/claude-3.5-sonnet",
    temperature: float = 1.0,
    max_tokens: Optional[int] = None,
    json_mode: bool = False,
    json_schema: Optional[Dict] = None
) -> OpenAIResponse:
    """Convenience function for OpenRouter generation"""
    config = ModelConfig(
        provider="openrouter",
        model=model,
        temperature=temperature,
        max_tokens=max_tokens,
        json_mode=json_mode,
        json_schema=json_schema,
        use_structured_output=bool(json_schema)
    )
    return generate_ai_content(messages, model_config=config)

def generate_with_google(
    messages: Any,
    model: str = "gemini-2.0-flash",
    temperature: float = 1.0,
    max_tokens: Optional[int] = None,
    json_mode: bool = False,
    json_schema: Optional[Dict] = None,
    reasoning_effort: Optional[str] = None
) -> Any:
    """Convenience function for Google Gemini generation"""
    config = ModelConfig(
        provider="google",
        model=model,
        temperature=temperature,
        max_tokens=max_tokens,
        json_mode=json_mode,
        json_schema=json_schema,
        use_structured_output=bool(json_schema),
        reasoning_effort=reasoning_effort
    )
    return generate_ai_content(messages, model_config=config)

def generate_with_reasoning(
    messages: Any,
    provider: str = "openai",
    reasoning_effort: str = "medium",
    temperature: float = 0.1,
    json_schema: Optional[Dict] = None
) -> Any:
    """Convenience function for reasoning model generation"""
    
    if provider == "openai":
        model = "o1"
    elif provider == "google":
        model = "gemini-2.5-pro"
    else:
        raise ValueError(f"Reasoning models not available for provider: {provider}")
    
    config = ModelConfig(
        provider=provider,
        model=model,
        temperature=temperature,
        reasoning_effort=reasoning_effort,
        json_schema=json_schema,
        use_structured_output=bool(json_schema)
    )
    
    return generate_ai_content(messages, model_config=config)

# === BACKWARD COMPATIBILITY ===
# Keep the original function signature for backward compatibility
def generate_ai_content_legacy(messages: Any, generation_config: Optional[Any] = None):
    """Legacy function for backward compatibility"""
    return generate_ai_content(messages, generation_config=generation_config)
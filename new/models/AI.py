import os
from dotenv import load_dotenv
import logging
import json
import nest_asyncio
from typing import List, Dict, Any, Optional
from datetime import date, datetime
import google.generativeai as genai
from copy import deepcopy
from concurrent.futures import ThreadPoolExecutor, as_completed
import threading

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

# === OPENAI API CONFIGURATION ===
USE_OPENAI = os.getenv("USE_OPENAI", "true").lower() == "true"
if USE_OPENAI:
    from openai import OpenAI
    # Initialize OpenAI client
    openai_api_key = os.getenv("OPENAI_API_KEY")
    if openai_api_key:
        openai_client = OpenAI(api_key=openai_api_key)
    else:
        logger.warning("OPENAI_API_KEY environment variable not set")
        USE_OPENAI = False

class OpenAIResponsePart:
    def __init__(self, text: str):
        self.text = text

class OpenAIResponse:
    def __init__(self, content: str):
        self.parts = [OpenAIResponsePart(content)]
        self.text = content

# Add thread local storage for request-specific useOpenAI value
thread_local = threading.local()

def generate_ai_content(messages: Any, generation_config: Optional[Any] = None, score_list: Optional[List[str]] = None):
    """
    Wrapper for generating AI content using either Gemini or OpenAI.
    Accepts either:
      - A string prompt, which will be wrapped as a single-user message.
      - A list of message dicts, where each dict can include text or media content.
    Supports audio and video file MIME types inline or via pre-upload.

    Args:
        messages (Any): String or list of message dicts.
        generation_config (Optional[Any]): Configuration for generation,
                                           including temperature and response_mime_type.
    """
    # Get the current thread's useOpenAI value, defaulting to environment variable if not set
    use_openai = True

    if use_openai:
        # Map generation configuration to OpenAI parameters
        temperature = generation_config.temperature if generation_config and hasattr(generation_config, "temperature") else 0
        response_format = {"type": "json_object"} if generation_config and hasattr(generation_config, "response_mime_type") and generation_config.response_mime_type == "application/json" else None
        
        # Convert messages to OpenAI format
        openai_messages = []
        if isinstance(messages, str):
            # Single string prompt
            openai_messages = [{"role": "user", "content": messages}]
        elif isinstance(messages, list):
            for msg in messages:
                if isinstance(msg, dict):
                    # Convert message format
                    role = msg.get("role", "user")
                    content = ""
                    
                    if "content" in msg:
                        if isinstance(msg["content"], str):
                            content = msg["content"]
                        elif isinstance(msg["content"], dict):
                            content = json.dumps(msg["content"])
                    elif "parts" in msg:
                        # Extract text from parts
                        text_parts = []
                        for part in msg["parts"]:
                            if isinstance(part, dict) and "text" in part:
                                text_parts.append(part["text"])
                            elif isinstance(part, str):
                                text_parts.append(part)
                        content = " ".join(text_parts)
                    elif "text" in msg:
                        content = msg["text"]
                    
                    openai_messages.append({"role": role, "content": content})
                else:
                    # If the message is a raw string, wrap it as a user message
                    openai_messages.append({"role": "user", "content": str(msg)})
        
        # Prepare API call parameters
        api_params = {
            "model": "gpt-4o",
            "messages": openai_messages,
            "temperature": temperature
        }
        
        # Add response format if JSON is requested
        if response_format:
            api_params["response_format"] = response_format
            # Ensure the messages contain "JSON" for JSON mode requirement
            if not any("json" in str(msg.get("content", "")).lower() for msg in openai_messages):
                # Add system message requesting JSON output
                api_params["messages"].insert(0, {
                    "role": "system", 
                    "content": "You are a helpful assistant designed to output JSON. Your output must have the following structure: {\"analysis\": <analysis>, \"score\": <score>}, with the score in this list: " + str(score_list) + "."
                })
        
        # Call OpenAI's chat completions API
        try:
            response_data = openai_client.chat.completions.create(**api_params)
            content = response_data.choices[0].message.content
            return OpenAIResponse(content)
        except Exception as e:
            logger.error(f"OpenAI API error: {e}")
            raise e
    else:
        # Use Gemini multimodal capabilities
        model = genai.GenerativeModel("models/gemini-2.0-flash")
        if isinstance(messages, list):
            # Ensure each message dict uses 'parts' for text content
            processed_messages = []
            for msg in messages:
                if isinstance(msg, dict):
                    if 'parts' not in msg:  # Only convert if not already in parts format
                        if 'content' in msg:
                            if isinstance(msg['content'], str):
                                # Wrap text content in a Part
                                msg['parts'] = [{"text": str(msg['content'])}]
                            elif isinstance(msg['content'], dict):
                                # Content is already a media dict (e.g. audio); put it in parts
                                msg['parts'] = [msg['content']]
                            del msg['content']  # Remove old key after moving to parts
                        elif 'text' in msg:
                            # Wrap 'text' field content in a Part
                            msg['parts'] = [{"text": str(msg['text'])}]
                            del msg['text']
                    processed_messages.append(msg)
                else:
                    # If the message is a raw string, wrap it as a user text part
                    processed_messages.append({"role": "user", "parts": [{"text": str(msg)}]})
            return model.generate_content(processed_messages, generation_config=generation_config)
        else:
            # Wrap single string prompt as a user message with text part
            return model.generate_content(
                [{"role": "user", "parts": [{"text": str(messages)}]}],
                generation_config=generation_config
            )

def upload_multimodal_file(file_path: str) -> Any:
    """
    Upload a file (audio or video) to Gemini File API and return the file reference.
    Supports large files beyond inline limits.

    Args:
        file_path (str): Local filesystem path to the media file.

    Returns:
        Any: File reference object usable in generate_content messages.
    """
    # Use the File API to upload the media file
    return genai.upload_file(file_path)
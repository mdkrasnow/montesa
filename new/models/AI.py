
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

# Notes on groq speed:
# llama-3.1-8b-instant is the fastest model for this task
# llama3-8192 is actually slower than llama-3.1-8b-instant
# llama-3.1-70b-versatile is not worth using -- almost as slow as gemini-2.0-flash
# llama-3.3-70b-specdec is not available yet -- once it is, it will be the fastest model

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

# === NEW GLOBAL VARIABLE AND GROQ WRAPPER CLASSES/HELPERS ===
USE_GROQ = os.getenv("USE_GROQ", "true").lower() == "true"
if USE_GROQ:
    from groq import Groq
    # Initialize Groq client without proxies parameter
    groq_api_key = os.getenv("GROQ_API_KEY")
    if groq_api_key:
        groq_client = Groq(api_key=groq_api_key)
    else:
        logger.warning("GROQ_API_KEY environment variable not set")
        USE_GROQ = False

class GroqResponsePart:
    def __init__(self, text: str):
        self.text = text

class GroqResponse:
    def __init__(self, content: str):
        self.parts = [GroqResponsePart(content)]
        self.text = content

# Add thread local storage for request-specific useGroq value
thread_local = threading.local()

def generate_ai_content(messages: Any, generation_config: Optional[Any] = None):
    """
    Wrapper for generating AI content using either Gemini or Groq.
    Accepts either:
      - A string prompt, which will be wrapped as a single-user message.
      - A list of message dicts, where each dict can include text or media content.
    Supports audio and video file MIME types inline or via pre-upload.

    Args:
        messages (Any): String or list of message dicts.
        generation_config (Optional[Any]): Configuration for generation,
                                           including temperature and response_mime_type.
    """
    # Get the current thread's useGroq value, defaulting to environment variable if not set
    use_groq = False

    if use_groq:
        # Map generation configuration to Groq parameters
        temperature = generation_config.temperature if generation_config and hasattr(generation_config, "temperature") else 1
        response_format = {"type": "json_object"} if generation_config and hasattr(generation_config, "response_mime_type") and generation_config.response_mime_type == "application/json" else None
        # Call Groq's chat completions API
        response_data = groq_client.chat.completions.create(
            messages=[{"role": "system", "content": msg["content"] if isinstance(msg["content"], str) else json.dumps(msg["content"])} for msg in messages],
            model="llama-3.3-70b-versatile",
            temperature=temperature,
            response_format=response_format
        )
        content = response_data.choices[0].message.content
        return GroqResponse(content)
    else:
        # Use Gemini multimodal capabilities
        model = genai.GenerativeModel("gemini-2.0-flash")
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
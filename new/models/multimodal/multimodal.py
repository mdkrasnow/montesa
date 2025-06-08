# File: basemodel_evaluator.py

import logging
import json
from copy import deepcopy
from concurrent.futures import ThreadPoolExecutor
import concurrent.futures
from typing import Dict, Any, Optional

# Import utility functions for LLM interaction
from ..AI import generate_ai_content
from ..PromptTemplates import (
    create_generic_component_prompt,
    create_domain_summary_prompt,
    create_overall_summary_prompt
)

logger = logging.getLogger(__name__)


class MultiModalModelEvaluator:
    """
    Flexible Framework Evaluation Pipeline implemented with OOP.
    This class provides the core methods for generating evaluations from
    low-inference observation notes through component-level scoring,
    domain summaries, and an overall evaluation summary.
    """

    def __init__(self, framework: Dict[str, Any]):
        """
        Initialize the evaluator with a given framework configuration.

        Args:
            framework (Dict[str, Any]): The framework structure and metadata.
        """
        self.framework = framework
        self.audio_bytes: Optional[bytes] = None
        self.audio_mime_type: Optional[str] = None

    def _generate_ai_response_json(self, messages: Any) -> Dict[str, Any]:
        """
        Generate AI content with JSON response parsing, using multimodal messages.

        Args:
            messages (Any): Either a string prompt or a list of message dicts,
                            where each dict can include text or media content.

        Returns:
            Dict[str, Any]: Parsed JSON result or raw text on error.
        """
        try:
            # If messages is a string, wrap into a single-user message
            if isinstance(messages, str):
                msg_list = [{"role": "user", "content": messages}]
                if self.audio_bytes and self.audio_mime_type:
                    msg_list.append({
                        "role": "user",
                        "content": {
                            "mime_type": self.audio_mime_type,
                            "data": self.audio_bytes
                        }
                    })
                response = generate_ai_content(
                    msg_list,
                    generation_config={"temperature": 0, "response_mime_type": "application/json"}
                )
            else:
                # Assume messages is already a list; append audio if provided
                msg_list = messages.copy()
                if self.audio_bytes and self.audio_mime_type:
                    msg_list.append({
                        "role": "user",
                        "content": {
                            "mime_type": self.audio_mime_type,
                            "data": self.audio_bytes
                        }
                    })
                response = generate_ai_content(
                    msg_list,
                    generation_config={"temperature": 0, "response_mime_type": "application/json"}
                )

            if hasattr(response, "parts"):
                response_text = "".join(part.text for part in response.parts).strip()
                try:
                    return json.loads(response_text)
                except json.JSONDecodeError:
                    return {"__raw_text": response_text}
            else:
                logger.warning(f"Unexpected LLM response format: {type(response)}")
                return {"__error": "Invalid response format from LLM"}
        except Exception as e:
            logger.error(f"Error generating AI content: {str(e)}")
            return {"__error": str(e)}

    def generate_component_evaluation(
        self,
        prompt: str,
        component: Dict[str, Any]
    ) -> Dict[str, Any]:
        """
        Generate a component-level evaluation using the provided prompt.

        Args:
            prompt (str): The LLM prompt for evaluation.
            component (Dict[str, Any]): Component configuration.

        Returns:
            Dict[str, Any]: Component evaluation result.
        """
        score_list = component.get("scoreList", [])
        try:
            # Build messages list starting with the component prompt
            messages = [{"role": "user", "content": prompt}]
            result = self._generate_ai_response_json(messages)
            if "__error" in result:
                return {
                    "score": score_list[0] if score_list else None,
                    "summary": "Failed to generate evaluation",
                    "error": result.get("__error")
                }

            if "score" in result:
                raw_score = result.get("score")
                # Ensure the AI returned a valid score in the list
                if raw_score not in score_list:
                    logger.warning(f"AI returned invalid score '{raw_score}' not in {score_list}.")
                    chosen_score = score_list[0] if score_list else None
                else:
                    chosen_score = raw_score
                return {
                    "score": chosen_score,
                    "summary": result.get("summary", "No analysis provided")
                }
            else:
                # If the JSON parsing failed, raw text available under "__raw_text"
                raw_text = result.get("__raw_text", "")
                logger.warning(f"Invalid JSON in component evaluation: {raw_text[:100]}...")
                return {
                    "score": score_list[0] if score_list else None,
                    "summary": raw_text or "No analysis provided"
                }

        except Exception as e:
            logger.error(f"Error generating component evaluation: {str(e)}")
            return {
                "score": score_list[0] if score_list else None,
                "summary": "Failed to generate evaluation",
                "error": str(e)
            }

    def generate_domain_summary(self, prompt: str) -> Dict[str, Any]:
        """
        Generate a domain-level summary using the provided prompt.

        Args:
            prompt (str): The LLM prompt for domain summary.

        Returns:
            Dict[str, Any]: Domain summary result.
        """
        try:
            messages = [{"role": "user", "content": prompt}]
            result = self._generate_ai_response_json(messages)
            if "__error" in result:
                return {
                    "summary": "Failed to generate summary",
                    "error": result.get("__error")
                }

            if "summary" in result:
                return {
                    "summary": result.get("summary", "No summary provided"),
                    "raw_response": result
                }
            else:
                raw_text = result.get("__raw_text", "")
                return {
                    "summary": raw_text,
                    "raw_response": raw_text
                }

        except Exception as e:
            logger.error(f"Error generating domain summary: {str(e)}")
            return {
                "summary": "Failed to generate summary",
                "error": str(e)
            }

    def generate_overall_summary(self, prompt: str) -> Dict[str, Any]:
        """
        Generate an overall evaluation summary using the provided prompt.

        Args:
            prompt (str): The LLM prompt for overall summary.

        Returns:
            Dict[str, Any]: Overall summary result.
        """
        try:
            messages = [{"role": "user", "content": prompt}]
            result = self._generate_ai_response_json(messages)
            if "__error" in result:
                return {
                    "summary": "Failed to generate summary",
                    "error": result.get("__error")
                }

            if "summary" in result:
                return {
                    "summary": result.get("summary", "No summary provided"),
                    "raw_response": result
                }
            else:
                raw_text = result.get("__raw_text", "")
                return {
                    "summary": raw_text,
                    "raw_response": raw_text
                }

        except Exception as e:
            logger.error(f"Error generating overall summary: {str(e)}")
            return {
                "summary": "Failed to generate summary",
                "error": str(e)
            }

    def evaluate(self, text: str, audio_file_path: Optional[str] = None) -> Dict[str, Any]:
        """
        Generate an evaluation based on the flexible framework structure,
        optionally including an audio file for context.

        Args:
            text (str): The observation text (transcript).
            audio_file_path (Optional[str]): Path to the audio file to be included.

        Returns:
            Dict[str, Any]: The generated evaluation result.
        """
        try:
            # If an audio file path is provided, load its bytes and infer MIME type
            if audio_file_path:
                with open(audio_file_path, "rb") as f:
                    self.audio_bytes = f.read()
                # Infer MIME type from file extension (assuming mp3 or wav)
                if audio_file_path.lower().endswith(".mp3"):
                    self.audio_mime_type = "audio/mpeg"
                elif audio_file_path.lower().endswith(".wav"):
                    self.audio_mime_type = "audio/wav"
                else:
                    # Default to generic audio if extension is unfamiliar
                    self.audio_mime_type = "application/octet-stream"

            # Step 1: Initialize evaluation structure
            evaluation: Dict[str, Any] = {
                "domains": {},
                "metadata": {
                    "framework_id": self.framework.get("framework_id"),
                    "framework_name": self.framework.get("name", "Teaching Framework"),
                },
            }

            # Step 2: Extract framework structure
            structure = self.framework.get("structure", {})
            domains_list = structure.get("domains", [])

            # Map domain IDs -> component definitions
            domain_component_map: Dict[str, Any] = {}
            for domain in domains_list:
                domain_id = str(domain.get("id", domain.get("number", domain.get("name", ""))))
                domain_component_map[domain_id] = domain.get("components", [])

            # Step 3: Schedule component evaluations
            component_futures: Dict[(str, str), concurrent.futures.Future] = {}
            with ThreadPoolExecutor() as executor:
                for domain_id, components in domain_component_map.items():
                    for component in components:
                        comp_id = str(
                            component.get(
                                "id",
                                component.get("name",
                                              component.get("number", component.get("description", "")))
                            )
                        )
                        prompt = create_generic_component_prompt(component, text, self.framework)

                        if component.get("isManuallyScored", False):
                            # If manually scored, skip AI generation
                            placeholder = {
                                "score": None,
                                "summary": "",
                                "isManuallyScored": True,
                                "modified": False,
                            }
                            component_futures[(domain_id, comp_id)] = executor.submit(lambda p=placeholder: p)
                            continue

                        future = executor.submit(self.generate_component_evaluation, prompt, component)
                        component_futures[(domain_id, comp_id)] = future

                # Step 4: Initialize domain entries in evaluation
                for domain in domains_list:
                    domain_id = str(domain.get("id", domain.get("number", domain.get("name", ""))))
                    evaluation["domains"][domain_id] = {
                        "name": domain.get("name", ""),
                        "components": {},
                        "weight": domain.get("weight", 1.0),
                        "isManuallyScored": domain.get("isManuallyScored", False),
                    }

                # Step 5: Collect component results
                for (domain_id, comp_id), future in component_futures.items():
                    try:
                        result = future.result()
                        result["isManuallyScored"] = evaluation["domains"][domain_id]["components"].get(
                            comp_id, {}
                        ).get("isManuallyScored", result.get("isManuallyScored", False))
                        evaluation["domains"][domain_id]["components"][comp_id] = result
                    except Exception as exc:
                        logger.error(f"Component {comp_id} in domain {domain_id} error: {exc}")
                        original_component = next(
                            (
                                c
                                for c in domain_component_map.get(domain_id, [])
                                if str(c.get("id", c.get("number", c.get("description", "")))) == comp_id
                            ),
                            {"scoreList": []},
                        )
                        default_score = original_component.get("scoreList", [None])[0]
                        evaluation["domains"][domain_id]["components"][comp_id] = {
                            "score": default_score,
                            "summary": "Failed to generate evaluation for this component.",
                        }

                # Step 6: Schedule domain summaries
                summary_futures: Dict[str, concurrent.futures.Future] = {}
                for domain_id in domain_component_map:
                    domain_def = next(
                        (d for d in domains_list if str(d.get("id", d.get("number", d.get("name", "")))) == domain_id),
                        {}
                    )

                    if domain_def.get("isManuallyScored", False):
                        logger.info(f"Skipping summary for manually scored domain: {domain_id}")
                        evaluation["domains"][domain_id]["summary"] = ""
                        continue

                    domain_components = evaluation["domains"][domain_id]["components"]
                    prompt = create_domain_summary_prompt(domain_def, self.framework, domain_components)
                    summary_futures[domain_id] = executor.submit(self.generate_domain_summary, prompt)

                # Step 7: Collect domain summaries
                for domain_id, future in summary_futures.items():
                    try:
                        domain_summary = future.result()
                        evaluation["domains"][domain_id]["summary"] = domain_summary.get("summary", "")
                    except Exception as exc:
                        logger.error(f"Domain {domain_id} summary error: {exc}")
                        evaluation["domains"][domain_id]["summary"] = ""

            # Step 8: Generate overall summary
            overall_prompt = create_overall_summary_prompt(self.framework, evaluation["domains"])
            overall_summary = self.generate_overall_summary(overall_prompt)
            evaluation["summary"] = overall_summary.get("summary", "")

            return {"success": True, "evaluation": evaluation}

        except Exception as e:
            logger.error(f"Error generating framework-based evaluation: {str(e)}")
            return {"success": False, "error": str(e)}


# File: ai.py

from flask import Flask, request, jsonify, Blueprint, redirect, make_response, send_file, Response
from supabase import Client, create_client
import os
import tempfile
from dotenv import load_dotenv
import logging
import json
import nest_asyncio
from typing import List, Dict, Any, Optional
from datetime import date, datetime
import google.generativeai as genai
from ..DanielsonType import DanielsonEvaluation
from decorators import token_required
from copy import deepcopy
from concurrent.futures import ThreadPoolExecutor, as_completed
import threading
from decorators import eval_paywall
import requests
import asyncio
import io
import base64
from PIL import Image
from pdf2image import convert_from_path
from openai import AsyncOpenAI, OpenAI
from index_config import async_route

# Notes on groq speed:
# llama-3.1-8b-instant is the fastest model for this task
# llama3-8192 is actually slower than llama-3.1-8b-instant
# llama-3.1-70b-versatile is not worth using -- almost as slow as gemini-2.0-flash
# llama-3.3-70b-specdec is not available yet -- once it is, it will be the fastest model

# Load environment variables
load_dotenv()

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
        # Directly pass the list of messages (text + media) if provided
        if isinstance(messages, list):
            return model.generate_content(messages, generation_config=generation_config)
        else:
            # Wrap single string prompt
            return model.generate_content([{"role": "user", "content": messages}], generation_config=generation_config)

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

# Example endpoint demonstrating how to accept video and audio uploads,
# combine with transcript text, and send to Gemini for analysis.
@swiftscope_bp.route("/analyze_multimodal", methods=["POST"])
@token_required
@eval_paywall
def analyze_multimodal():
    """
    Expects:
      - transcript_text: the classroom observation transcript as text in JSON.
      - optional audio_file: multipart form upload of audio (mp3 or wav).
      - optional video_file: multipart form upload of video (mp4 or mov).

    This endpoint packages text, audio, and video into a multimodal Gemini request.
    """
    try:
        transcript_text = request.form.get("transcript_text", "")
        audio_ref = None
        video_ref = None

        # If audio file provided, upload via File API
        if "audio_file" in request.files:
            audio_file = request.files["audio_file"]
            temp_audio_path = os.path.join(tempfile.gettempdir(), audio_file.filename)
            audio_file.save(temp_audio_path)
            audio_ref = upload_multimodal_file(temp_audio_path)

        # If video file provided, upload via File API
        if "video_file" in request.files:
            video_file = request.files["video_file"]
            temp_video_path = os.path.join(tempfile.gettempdir(), video_file.filename)
            video_file.save(temp_video_path)
            video_ref = upload_multimodal_file(temp_video_path)

        # Build messages list
        messages = []
        if transcript_text:
            messages.append({"role": "user", "content": transcript_text})

        if audio_ref:
            messages.append({
                "role": "user",
                "content": {
                    "mime_type": "audio/mpeg" if audio_ref.name.lower().endswith(".mp3") else "audio/wav",
                    "data": audio_ref
                }
            })

        if video_ref:
            messages.append({
                "role": "user",
                "content": {
                    "mime_type": "video/mp4" if video_ref.name.lower().endswith(".mp4") else "video/quicktime",
                    "data": video_ref
                }
            })

        # Send to Gemini for combined analysis
        gen_response = generate_ai_content(
            messages,
            generation_config={"temperature": 0.7, "response_mime_type": "application/json"}
        )

        # Parse and return JSON or raw text
        if hasattr(gen_response, "parts"):
            response_text = "".join(part.text for part in gen_response.parts).strip()
            try:
                result_json = json.loads(response_text)
            except json.JSONDecodeError:
                result_json = {"__raw_text": response_text}
        else:
            result_json = {"__error": "Invalid response format"}

        return jsonify({"success": True, "analysis": result_json})

    except Exception as e:
        logger.error(f"Error in multimodal analysis endpoint: {str(e)}")
        return jsonify({"success": False, "error": str(e)}), 500

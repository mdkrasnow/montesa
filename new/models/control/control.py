# control.py - Modified for batch processing with rate limiting

import logging
import json
import time
from collections import deque
from copy import deepcopy
from concurrent.futures import ThreadPoolExecutor
import concurrent.futures
from typing import Dict, Any
import threading
import re

# Import utility functions for LLM interaction
from ..AI import generate_ai_content
from ..PromptTemplates import (
    create_domain_batch_prompt,  # New batch prompt function
)

logger = logging.getLogger(__name__)

class RateLimiter:
    """
    Thread-safe rate limiter to enforce API call limits.
    """
    def __init__(self, max_calls_per_minute: int = 3):
        self.max_calls = max_calls_per_minute
        self.calls = deque()
        self.lock = threading.Lock()
    
    def wait_if_needed(self):
        """
        Wait if necessary to maintain the rate limit.
        """
        with self.lock:
            now = time.time()
            
            # Remove calls older than 1 minute
            while self.calls and now - self.calls[0] > 60:
                self.calls.popleft()
            
            # If we've hit the limit, wait until we can make another call
            if len(self.calls) >= self.max_calls:
                sleep_time = 60 - (now - self.calls[0]) + 0.1  # Add small buffer
                if sleep_time > 0:
                    logger.info(f"Rate limiting: sleeping for {sleep_time:.1f} seconds")
                    time.sleep(sleep_time)
                    # Clean up old calls after sleeping
                    now = time.time()
                    while self.calls and now - self.calls[0] > 60:
                        self.calls.popleft()
            
            # Record this call
            self.calls.append(now)

# Global rate limiter instance
_global_rate_limiter = None

class BaseModelEvaluator:
    """
    Flexible Framework Evaluation Pipeline implemented with OOP.
    Now uses batch processing to evaluate all components within a domain simultaneously.
    """

    def __init__(self, framework: Dict[str, Any], enable_throttling: bool = False):
        """
        Initialize the evaluator with a given framework configuration.

        Args:
            framework (Dict[str, Any]): The framework structure and metadata.
            enable_throttling (bool): Whether to enable API rate limiting (3 calls per minute).
        """
        self.framework = framework
        self.enable_throttling = enable_throttling
        
        # Initialize global rate limiter if throttling is enabled
        global _global_rate_limiter
        if enable_throttling and _global_rate_limiter is None:
            _global_rate_limiter = RateLimiter(max_calls_per_minute=3)
            logger.info("Rate limiting enabled: 3 API calls per minute")

    def _extract_json_from_markdown(self, text: str) -> str:
        """
        Extract JSON content from markdown code blocks.
        
        Args:
            text (str): Response text that may contain markdown-wrapped JSON
            
        Returns:
            str: Clean JSON text without markdown wrapper
        """
        text = text.strip()
        
        # Pattern 1: ```json...``` blocks
        json_match = re.search(r'```json\s*\n(.*?)\n```', text, re.DOTALL)
        if json_match:
            return json_match.group(1).strip()
        
        # Pattern 2: Generic ```...``` blocks (assuming JSON content)
        generic_match = re.search(r'```\s*\n(.*?)\n```', text, re.DOTALL)
        if generic_match:
            return generic_match.group(1).strip()
        
        # Pattern 3: Simple prefix/suffix stripping for edge cases
        if text.startswith('```json'):
            lines = text.split('\n')
            if len(lines) > 2 and lines[-1].strip() == '```':
                return '\n'.join(lines[1:-1]).strip()
        elif text.startswith('```'):
            lines = text.split('\n')
            if len(lines) > 2 and lines[-1].strip() == '```':
                return '\n'.join(lines[1:-1]).strip()
        
        # No markdown wrapper found, return as-is
        return text

    def _generate_ai_response_json(self, prompt: str) -> Dict[str, Any]:
        """
        Generate AI content with JSON response parsing and optional rate limiting.
        Now includes robust markdown-aware JSON parsing.

        Args:
            prompt (str): The LLM prompt.

        Returns:
            Dict[str, Any]: Parsed JSON result or raw text on error.
        """
        # Apply rate limiting if enabled
        if self.enable_throttling and _global_rate_limiter is not None:
            _global_rate_limiter.wait_if_needed()
        
        try:
            response = generate_ai_content(
                prompt,
                generation_config={"temperature": 0, "response_mime_type": "application/json"}
            )
            if hasattr(response, "parts"):
                response_text = "".join(part.text for part in response.parts).strip()
                
                # Extract JSON from potential markdown wrapper
                clean_json_text = self._extract_json_from_markdown(response_text)
                
                try:
                    return json.loads(clean_json_text)
                except json.JSONDecodeError as e:
                    print(f"🔍 DEBUG: ❌ JSON decode error in _generate_ai_response_json: {e}")
                    print(f"🔍 DEBUG: Original response preview: {response_text[:200]}...")
                    print(f"🔍 DEBUG: Cleaned JSON preview: {clean_json_text[:200]}...")
                    return {"__raw_text": response_text}
            else:
                error_msg = f"❌ Unexpected LLM response format: {type(response)}"
                print(f"🔍 DEBUG: {error_msg}")
                logger.warning(error_msg)
                return {"__error": error_msg}
        except Exception as e:
            error_msg = f"❌ Error in _generate_ai_response_json: {type(e).__name__}: {e}"
            print(f"🔍 DEBUG: {error_msg}")
            return {"__error": str(e)}

    def generate_domain_batch_evaluation(
        self,
        prompt: str,
        domain: Dict[str, Any]
    ) -> Dict[str, Any]:
        """
        Generate evaluations for all components in a domain using a single API call.

        Args:
            prompt (str): The LLM prompt for batch evaluation.
            domain (Dict[str, Any]): Domain configuration with all components.

        Returns:
            Dict[str, Any]: Dictionary mapping component IDs to their evaluations.
        """
        components = domain.get("components", [])
        result = {}
        domain_name = domain.get("name", "Unknown Domain")
        
        try:
            print(f"🔍 DEBUG: Starting batch evaluation for domain: {domain_name} ({len(components)} components)")
            ai_response = self._generate_ai_response_json(prompt)
            
            if "__error" in ai_response:
                error_msg = f"❌ AI response error in domain {domain_name}: {ai_response.get('__error')}"
                print(f"🔍 DEBUG: {error_msg}")
                # If batch failed, return default scores for all components
                for component in components:
                    comp_id = str(
                        component.get(
                            "id",
                            component.get(
                                "name",
                                component.get("number", component.get("description", ""))
                            )
                        )
                    )
                    score_list = component.get("scoreList", [])
                    result[comp_id] = {
                        "score": score_list[0] if score_list else None,
                        "analysis": "❌ Failed to generate evaluation",
                        "error": ai_response.get("__error")
                    }
                return result

            # Parse the batch response using numbered keys
            for i, component in enumerate(components, 1):
                comp_id = str(
                    component.get(
                        "id",
                        component.get(
                            "name",
                            component.get("number", component.get("description", ""))
                        )
                    )
                )
                score_list = component.get("scoreList", [])
                component_key = f"component_{i}"
                
                if component_key in ai_response:
                    comp_eval = ai_response[component_key]
                    
                    # Verify the component_id matches (optional validation)
                    returned_comp_id = comp_eval.get("component_id", "")
                    if returned_comp_id and returned_comp_id != comp_id:
                        print(f"🔍 DEBUG: ❌ Component ID mismatch for {component_key}: expected '{comp_id}', got '{returned_comp_id}'")
                        logger.warning(f"Component ID mismatch for {component_key}: expected '{comp_id}', got '{returned_comp_id}'")
                    
                    raw_score = comp_eval.get("score")
                    
                    # Validate score is in the allowed list
                    try:
                        raw_score = int(raw_score) if isinstance(raw_score, str) and raw_score.isdigit() else raw_score
                    except ValueError:
                        pass
                    
                    if raw_score not in score_list:
                        print(f"🔍 DEBUG: ❌ Invalid score '{raw_score}' for {comp_id} in {domain_name}, using default")
                        logger.warning(f"AI returned invalid score '{raw_score}' for {comp_id}, using default.")
                        chosen_score = score_list[0] if score_list else None
                    else:
                        chosen_score = raw_score
                    
                    result[comp_id] = {
                        "score": chosen_score,
                        "analysis": comp_eval.get("analysis", "No analysis provided")
                    }
                    print(f"🔍 DEBUG: ✅ Successfully parsed {component_key} for {comp_id}")
                else:
                    # Component missing from response, use default
                    print(f"🔍 DEBUG: ❌ Component key {component_key} (ID: {comp_id}) missing from AI response in {domain_name}")
                    logger.warning(f"Component key {component_key} missing from batch response.")
                    result[comp_id] = {
                        "score": score_list[0] if score_list else None,
                        "analysis": "❌ No evaluation provided in batch response"
                    }

        except Exception as e:
            error_msg = f"❌ Exception in generate_domain_batch_evaluation for {domain_name}: {type(e).__name__}: {e}"
            print(f"🔍 DEBUG: {error_msg}")
            logger.error(error_msg)
            # Return default scores for all components on error
            for component in components:
                comp_id = str(
                    component.get(
                        "id",
                        component.get(
                            "name",
                            component.get("number", component.get("description", ""))
                        )
                    )
                )
                score_list = component.get("scoreList", [])
                result[comp_id] = {
                    "score": score_list[0] if score_list else None,
                    "analysis": "❌ Failed to generate evaluation",
                    "error": str(e)
                }

        print(f"🔍 DEBUG: Completed batch evaluation for domain: {domain_name} - {len(result)} components processed")
        return result

    def evaluate(self, text: str) -> Dict[str, Any]:
        """
        Generate an evaluation based on the flexible framework structure using batch processing.
        """
        try:
            print(f"🔍 DEBUG: Starting evaluation - text length: {len(text)} chars")
            
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
            print(f"🔍 DEBUG: Found {len(domains_list)} domains in framework")

            # Step 3: Schedule DOMAIN batch evaluations (not individual components)
            domain_futures: Dict[str, concurrent.futures.Future] = {}
            with ThreadPoolExecutor() as executor:
                
                # Initialize domain entries first
                for domain in domains_list:
                    domain_id = str(domain.get("id", domain.get("number", domain.get("name", ""))))
                    evaluation["domains"][domain_id] = {
                        "name": domain.get("name", ""),
                        "components": {},
                        "weight": domain.get("weight", 1.0),
                        "isManuallyScored": domain.get("isManuallyScored", False),
                    }
                
                print(f"🔍 DEBUG: Starting batch evaluations for {len(domains_list)} domains")
                
                # Schedule batch evaluation for each domain
                for domain in domains_list:
                    domain_id = str(domain.get("id", domain.get("number", domain.get("name", ""))))
                    domain_name = domain.get("name", "")
                    
                    if domain.get("isManuallyScored", False):
                        # Skip AI generation for manually scored domains
                        print(f"🔍 DEBUG: Skipping manually scored domain: {domain_name}")
                        logger.info(f"Skipping batch evaluation for manually scored domain: {domain_id}")
                        # Initialize manually scored components
                        for component in domain.get("components", []):
                            comp_id = str(
                                component.get(
                                    "id",
                                    component.get(
                                        "name",
                                        component.get("number", component.get("description", ""))
                                    )
                                )
                            )
                            evaluation["domains"][domain_id]["components"][comp_id] = {
                                "score": None,
                                "analysis": "",
                                "isManuallyScored": True,
                                "modified": False,
                            }
                        continue
                    
                    print(f"🔍 DEBUG: Scheduling batch evaluation for domain: {domain_name}")
                    # Create batch prompt for the entire domain
                    try:
                        prompt = create_domain_batch_prompt(domain, text, self.framework)
                        future = executor.submit(self.generate_domain_batch_evaluation, prompt, domain)
                        domain_futures[domain_id] = future
                    except Exception as e:
                        error_msg = f"❌ Error creating prompt for domain {domain_name}: {type(e).__name__}: {e}"
                        print(f"🔍 DEBUG: {error_msg}")
                        logger.error(error_msg)

                print(f"🔍 DEBUG: Collecting batch results from {len(domain_futures)} domains")
                
                # Step 4: Collect domain batch results
                for domain_id, future in domain_futures.items():
                    domain_name = evaluation["domains"][domain_id]["name"]
                    try:
                        print(f"🔍 DEBUG: Collecting results for domain: {domain_name}")
                        batch_results = future.result()
                        # Distribute the batch results to individual components
                        for comp_id, comp_eval in batch_results.items():
                            evaluation["domains"][domain_id]["components"][comp_id] = comp_eval
                        print(f"🔍 DEBUG: Successfully processed {len(batch_results)} components for domain: {domain_name}")
                    except Exception as exc:
                        error_msg = f"❌ Domain {domain_name} batch evaluation error: {type(exc).__name__}: {exc}"
                        print(f"🔍 DEBUG: {error_msg}")
                        logger.error(error_msg)
                        # Fill with default scores for all components in this domain
                        domain_def = next(
                            (d for d in domains_list if str(d.get("id", d.get("number", d.get("name", "")))) == domain_id),
                            {}
                        )
                        for component in domain_def.get("components", []):
                            comp_id = str(
                                component.get(
                                    "id",
                                    component.get(
                                        "name",
                                        component.get("number", component.get("description", ""))
                                    )
                                )
                            )
                            score_list = component.get("scoreList", [])
                            evaluation["domains"][domain_id]["components"][comp_id] = {
                                "score": score_list[0] if score_list else None,
                                "analysis": "❌ Failed to generate evaluation for this component.",
                                "error": str(exc)
                            }

            print(f"🔍 DEBUG: Evaluation completed successfully")
            return {"success": True, "evaluation": evaluation}

        except Exception as e:
            error_msg = f"❌ Error in evaluate method: {type(e).__name__}: {e}"
            print(f"🔍 DEBUG: {error_msg}")
            logger.error(error_msg)
            return {"success": False, "error": str(e)}
# BaseModelEvaluator.py
#
# Significant upgrade: introduces an “LLM-as-Judge” post-processing pass
# for every component evaluation.  After the first LLM call produces a
# score + analysis, we build a second prompt that asks a *separate*
# LLM instance to critique that output against the rubric and (if
# needed) recommend a revised score / analysis.  This pattern—popular-
# ized in the research literature under names such as “LLM-as-Judge”
# (EvidentlyAI Guide, Feb 2025) and “self-critique / Reflexion”—
# consistently boosts accuracy and reliability. :contentReference[oaicite:0]{index=0}

import logging
import json
from copy import deepcopy
from concurrent.futures import ThreadPoolExecutor
import concurrent.futures
from typing import Dict, Any

# Import utility functions for LLM interaction and prompt builders
from ..AI import generate_ai_content
from ..PromptTemplates import (
    create_generic_component_prompt,
    create_domain_summary_prompt,
    create_overall_summary_prompt,
)

logger = logging.getLogger(__name__)


class BaseModelEvaluator:
    """
    Flexible Framework Evaluation Pipeline implemented with OOP.
    Upgraded with a two-stage “LLM-as-Judge” loop for each component:
      1. **Draft pass** – model reasons through rubric & evidence.
      2. **Critique pass** – independent judge model reviews the
         draft JSON and, if warranted, revises score / analysis.

    References:
      - “LLM-as-a-Judge: A Complete Guide” EvidentlyAI (2025-02-19)
      - “Reflexion: Language Agents with Verbal RL” Shinn et al. 2023
      - “Self-Critique Improves Logic & Reasoning” PromptEng Blog 2023
    """

    def __init__(self, framework: Dict[str, Any]):
        """
        Initialize the evaluator with a given framework configuration.

        Args:
            framework (Dict[str, Any]): The framework structure and metadata.
        """
        self.framework = framework

    # ────────────────────────────────────────────────────────────────
    # Internal helpers
    # ────────────────────────────────────────────────────────────────

    def _generate_ai_response_json(self, prompt: str) -> Dict[str, Any]:
        """
        Generate AI content with JSON response parsing.

        Args:
            prompt (str): The LLM prompt.

        Returns:
            Dict[str, Any]: Parsed JSON result or raw text on error.
        """
        try:
            response = generate_ai_content(
                prompt,
                generation_config={
                    "temperature": 0,
                    "response_mime_type": "application/json",
                },
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
            # Print the error instead of using logger.error to avoid threading I/O issues in Jupyter
            print(f"Error generating AI content: {e}")
            return {"__error": str(e)}

    def _build_critique_prompt(
        self,
        component: Dict[str, Any],
        draft_json: Dict[str, Any],
    ) -> str:
        """
        Construct a second-pass prompt that asks an independent model
        to critique the initial component evaluation and (optionally)
        propose an improved score / analysis.

        Returns:
            str: Critique prompt requesting strict JSON output.
        """
        comp_id = str(
            component.get(
                "id",
                component.get(
                    "name",
                    component.get("number", component.get("description", "")),
                ),
            )
        )
        comp_description = component.get("description", "")
        score_list = component.get("scoreList", [])

        rubric_text = (
            component.get("rubric")
            or component.get("rubric_description", "")
            or "Follow the framework rubric criteria."
        )

        critique_prompt = (
            f"You are acting as an impartial **LLM-as-Judge** reviewing a "
            f"peer evaluator’s JSON for a teacher-observation component.\n\n"
            f"### Component Metadata\n"
            f"- ID: {comp_id}\n"
            f"- Description: {comp_description}\n"
            f"- Allowed scores: {score_list}\n\n"
            f"### Rubric Excerpt\n{rubric_text}\n\n"
            f"### Candidate Evaluation (JSON)\n```json\n"
            f"{json.dumps(draft_json, indent=2)}\n```\n\n"
            "# Your Tasks\n"
            "1. Verify the analysis cites appropriate evidence and aligns with the rubric.\n"
            "2. Decide whether the chosen score is well-justified (output `accept`) "
            "or should be revised (`revise`).\n"
            "3. If revising, supply a better `recommended_score` chosen from the "
            "allowed list **and** a concise `revised_analysis` (≤ 60 words) that "
            "improves clarity or alignment.\n\n"
            "### Strict JSON Response Template (no extra keys / text!)\n"
            "{\n"
            '  "verdict": "accept" | "revise",\n'
            '  "recommended_score": <number|null>,\n'
            '  "revised_analysis": "<string|null>"\n'
            "}\n"
        )
        return critique_prompt

    # ────────────────────────────────────────────────────────────────
    # Public APIs
    # ────────────────────────────────────────────────────────────────

    def generate_component_evaluation(
        self,
        prompt: str,
        component: Dict[str, Any],
    ) -> Dict[str, Any]:
        """
        Generate a component-level evaluation followed by a critique-and-revise
        pass. Returns the **final** score / analysis accepted after judging.

        Args:
            prompt (str): The LLM draft prompt.
            component (Dict[str, Any]): Component configuration.

        Returns:
            Dict[str, Any]: Final component evaluation.
        """
        score_list = component.get("scoreList", [])

        # ── Pass 1: draft evaluation ────────────────────────────────
        draft_result = self._generate_ai_response_json(prompt)

        if "__error" in draft_result:
            return {
                "score": score_list[0] if score_list else None,
                "analysis": "Failed to generate evaluation",
                "error": draft_result.get("__error"),
            }

        # Normalize / validate draft JSON
        draft_score = draft_result.get("score")
        try:
            draft_score_int = int(draft_score)
        except Exception:
            draft_score_int = draft_score  # leave as-is (could be string label)

        if draft_score_int not in score_list:
            logger.warning(
                f"LLM draft returned invalid score '{draft_score_int}' "
                f"not in {score_list}. Overriding with first option."
            )
            draft_score_int = score_list[0] if score_list else None

        draft_analysis = draft_result.get("analysis", "No analysis provided")

        # Prepare a dict so the judge sees a *clean* structure
        draft_json_clean = {
            "analysis": draft_analysis,
            "score": draft_score_int,
        }

        # ── Pass 2: critique / judge ────────────────────────────────
        critique_prompt = self._build_critique_prompt(component, draft_json_clean)
        critique_result = self._generate_ai_response_json(critique_prompt)

        final_score = draft_score_int
        final_analysis = draft_analysis

        try:
            if (
                isinstance(critique_result, dict)
                and critique_result.get("verdict") == "revise"
            ):
                recommended = critique_result.get("recommended_score")
                try:
                    recommended_int = int(recommended)
                except Exception:
                    recommended_int = recommended
                if recommended_int in score_list:
                    final_score = recommended_int
                revised_analysis = critique_result.get("revised_analysis")
                if isinstance(revised_analysis, str) and revised_analysis.strip():
                    final_analysis = revised_analysis.strip()
        except Exception as exc:
            logger.error(f"Crtitique parsing error: {exc}")

        # Return final accepted result
        return {
            "score": final_score,
            "analysis": final_analysis,
            "draft": draft_json_clean,
            "judge_feedback": critique_result,
        }

    # ────────────────────────────────────────────────────────────────
    # Domain & overall summaries (unchanged)
    # ────────────────────────────────────────────────────────────────

    def generate_domain_summary(self, prompt: str) -> Dict[str, Any]:
        """
        Generate a domain-level summary using the provided prompt.

        Args:
            prompt (str): The LLM prompt for domain summary.

        Returns:
            Dict[str, Any]: Domain summary result.
        """
        try:
            result = self._generate_ai_response_json(prompt)
            if "__error" in result:
                return {
                    "summary": "Failed to generate summary",
                    "error": result.get("__error"),
                }

            if "summary" in result:
                return {
                    "summary": result.get("summary", "No summary provided"),
                    "raw_response": result,
                }
            else:
                raw_text = result.get("__raw_text", "")
                return {"summary": raw_text, "raw_response": raw_text}

        except Exception as e:
            logger.error(f"Error generating domain summary: {str(e)}")
            return {"summary": "Failed to generate summary", "error": str(e)}

    def generate_overall_summary(self, prompt: str) -> Dict[str, Any]:
        """
        Generate an overall evaluation summary using the provided prompt.

        Args:
            prompt (str): The LLM prompt for overall summary.

        Returns:
            Dict[str, Any]: Overall summary result.
        """
        try:
            result = self._generate_ai_response_json(prompt)
            if "__error" in result:
                return {
                    "summary": "Failed to generate summary",
                    "error": result.get("__error"),
                }

            if "summary" in result:
                return {
                    "summary": result.get("summary", "No summary provided"),
                    "raw_response": result,
                }
            else:
                raw_text = result.get("__raw_text", "")
                return {"summary": raw_text, "raw_response": raw_text}

        except Exception as e:
            logger.error(f"Error generating overall summary: {str(e)}")
            return {"summary": "Failed to generate summary", "error": str(e)}

    # ────────────────────────────────────────────────────────────────
    # Main execution pipeline (evaluate)
    # ────────────────────────────────────────────────────────────────

    def evaluate(self, text: str) -> Dict[str, Any]:
        """
        Generate a complete evaluation based on the flexible framework
        structure.  Component-level evaluations now include a second-
        pass judge for improved scoring fidelity.
        """
        try:
            # Step 1 – skeleton
            evaluation: Dict[str, Any] = {
                "domains": {},
                "metadata": {
                    "framework_id": self.framework.get("framework_id"),
                    "framework_name": self.framework.get("name", "Teaching Framework"),
                },
            }

            # Step 2 – framework structure
            structure = self.framework.get("structure", {})
            domains_list = structure.get("domains", [])

            # Map domain IDs → component definitions
            domain_component_map: Dict[str, Any] = {}
            for domain in domains_list:
                domain_id = str(
                    domain.get("id", domain.get("number", domain.get("name", "")))
                )
                domain_component_map[domain_id] = domain.get("components", [])

            # Step 3 – schedule component evaluations (parallel)
            component_futures: Dict[(str, str), concurrent.futures.Future] = {}
            with ThreadPoolExecutor() as executor:
                for domain_id, components in domain_component_map.items():
                    for component in components:
                        comp_id = str(
                            component.get(
                                "id",
                                component.get(
                                    "name",
                                    component.get("number", component.get("description", "")),
                                ),
                            )
                        )

                        prompt = create_generic_component_prompt(
                            component, text, self.framework, ""
                        )

                        if component.get("isManuallyScored", False):
                            # Manual components—skip AI
                            placeholder = {
                                "score": None,
                                "analysis": "",
                                "isManuallyScored": True,
                                "modified": False,
                            }
                            component_futures[(domain_id, comp_id)] = executor.submit(
                                lambda p=placeholder: p
                            )
                            continue

                        future = executor.submit(
                            self.generate_component_evaluation, prompt, component
                        )
                        component_futures[(domain_id, comp_id)] = future

                # Step 4 – init domain entries
                for domain in domains_list:
                    domain_id = str(
                        domain.get("id", domain.get("number", domain.get("name", "")))
                    )
                    evaluation["domains"][domain_id] = {
                        "name": domain.get("name", ""),
                        "components": {},
                        "weight": domain.get("weight", 1.0),
                        "isManuallyScored": domain.get("isManuallyScored", False),
                    }

                # Step 5 – gather component results
                for (domain_id, comp_id), future in component_futures.items():
                    try:
                        result = future.result()
                        evaluation["domains"][domain_id]["components"][comp_id] = result
                    except Exception as exc:
                        logger.error(
                            f"Component {comp_id} in domain {domain_id} error: {exc}"
                        )
                        original_component = next(
                            (
                                c
                                for c in domain_component_map.get(domain_id, [])
                                if str(
                                    c.get(
                                        "id", c.get("number", c.get("description", ""))
                                    )
                                )
                                == comp_id
                            ),
                            {"scoreList": []},
                        )
                        default_score = original_component.get("scoreList", [None])[0]
                        evaluation["domains"][domain_id]["components"][comp_id] = {
                            "score": default_score,
                            "analysis": "Failed to generate evaluation for this component.",
                        }

                # Step 6 – schedule domain summaries
                summary_futures: Dict[str, concurrent.futures.Future] = {}
                for domain_id in domain_component_map:
                    domain_def = next(
                        (
                            d
                            for d in domains_list
                            if str(
                                d.get("id", d.get("number", d.get("name", "")))
                            )
                            == domain_id
                        ),
                        {},
                    )
                    if domain_def.get("isManuallyScored", False):
                        logger.info(
                            f"Skipping summary for manually scored domain: {domain_id}"
                        )
                        evaluation["domains"][domain_id]["summary"] = ""
                        continue

                    domain_components = evaluation["domains"][domain_id]["components"]
                    prompt = create_domain_summary_prompt(
                        domain_def, self.framework, domain_components
                    )
                    summary_futures[domain_id] = executor.submit(
                        self.generate_domain_summary, prompt
                    )

                # Step 7 – collect domain summaries
                for domain_id, future in summary_futures.items():
                    try:
                        domain_summary = future.result()
                        evaluation["domains"][domain_id]["summary"] = domain_summary.get(
                            "summary", ""
                        )
                    except Exception as exc:
                        logger.error(f"Domain {domain_id} summary error: {exc}")
                        evaluation["domains"][domain_id]["summary"] = ""

            # Step 8 – overall summary
            overall_prompt = create_overall_summary_prompt(
                self.framework, evaluation["domains"]
            )
            overall_summary = self.generate_overall_summary(overall_prompt)
            evaluation["summary"] = overall_summary.get("summary", "")

            return {"success": True, "evaluation": evaluation}

        except Exception as e:
            logger.error(f"Error generating framework-based evaluation: {str(e)}")
            return {"success": False, "error": str(e)}

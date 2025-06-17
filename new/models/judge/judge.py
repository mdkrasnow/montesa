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


class JudgeModelEvaluator:
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

        critique_prompt = f"""You are an expert **LLM-as-Judge** specializing in educational assessment validation. Your role is to rigorously evaluate peer evaluators' analyses of teacher observation data to ensure accuracy, consistency, and rubric alignment.

## EVALUATION CONTEXT

### Component Details
- **ID**: {comp_id}
- **Description**: {comp_description}  
- **Valid Score Range**: {score_list}

### Assessment Rubric
{rubric_text}

### Candidate Evaluation Under Review
```json
{json.dumps(draft_json, indent=2)}
```

## EVALUATION FRAMEWORK

You must systematically assess the candidate evaluation using this step-by-step process:

### Step 1: Evidence Analysis
- Does the analysis cite specific, observable evidence from the classroom observation?
- Is the evidence directly relevant to this component's criteria?
- Are claims supported rather than speculative?

### Step 2: Rubric Alignment Check  
- Does the analysis demonstrate clear understanding of the rubric descriptors?
- Is the interpretation of evidence consistent with rubric language and intent?
- Are any rubric criteria misapplied or overlooked?

### Step 3: Score Justification Review
- Is the assigned score logically derived from the evidence presented?
- Does the score match the performance level described in the analysis?
- Would an independent evaluator reach a similar conclusion based on this analysis?

### Step 4: Quality Assessment
- Is the analysis clear, specific, and professionally written?
- Does it provide actionable insights rather than vague generalizations?
- Is the length appropriate (substantive but concise)?

## DECISION CRITERIA

**ACCEPT** the evaluation if:
- Evidence is specific, relevant, and well-documented
- Rubric alignment is accurate and demonstrates understanding
- Score is well-justified by the evidence and analysis
- Analysis quality meets professional standards

**REVISE** the evaluation if:
- Evidence is vague, irrelevant, or insufficient
- Rubric interpretation is flawed or misaligned  
- Score doesn't match the evidence or analysis
- Analysis lacks clarity or contains significant errors

## OUTPUT REQUIREMENTS

Provide your response in this exact JSON format with no additional text:

```json
{{
  "analysis": "<your systematic evaluation of the candidate's analysis and score using the 4-step framework above>",
  "verdict": "accept|revise", 
  "recommended_score": <number from {score_list} or null if accepting>,
  "revised_analysis": "<60-word maximum improvement or null if accepting>"
}}
```

### Validation Rules:
- `analysis` must contain your step-by-step reasoning through the evaluation framework (reference Steps 1-4 explicitly)
- `verdict` must be exactly "accept" or "revise"
- `recommended_score` must be from the valid range {score_list} if revising, null if accepting
- `revised_analysis` must be ≤60 words if revising, null if accepting
- No additional keys or explanatory text outside the JSON structure

**Important**: Use the `analysis` field to show your systematic evaluation process before reaching your verdict. This ensures transparency and consistency in your judgment.

## EXAMPLES

### Example 1 - ACCEPT case:
If candidate analysis states: "Teacher used think-pair-share strategy, allowing 2 minutes individual reflection before paired discussion. Observed 85% student engagement during the 8-minute activity."

Response:
```json
{{
  "analysis": "Step 1: Evidence is specific and observable (think-pair-share, timing, engagement percentage). Step 2: Aligns with rubric criteria for student engagement strategies. Step 3: Score matches the described performance level. Step 4: Analysis is clear and professional.",
  "verdict": "accept",
  "recommended_score": null,
  "revised_analysis": null
}}
```

### Example 2 - REVISE case:  
If candidate analysis states: "Teacher seemed to engage students well and the lesson was good."

Response:
```json
{{
  "analysis": "Step 1: Evidence is vague and subjective ('seemed', 'good'). Step 2: No clear rubric alignment demonstrated. Step 3: Score cannot be justified from limited evidence. Step 4: Analysis lacks specificity and professional rigor.",
  "verdict": "revise", 
  "recommended_score": 2,
  "revised_analysis": "Teacher implemented interactive strategies with observable student participation, though specific engagement metrics and strategy details need documentation for complete evaluation."
}}
```

Think through each step systematically before rendering your final judgment."""
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

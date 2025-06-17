import logging
import json
from copy import deepcopy
from concurrent.futures import ThreadPoolExecutor
import concurrent.futures
from typing import Dict, Any

# Import utility functions for LLM interaction
from ..AI import generate_ai_content
from ..PromptTemplates import (
    create_generic_component_prompt,
    create_domain_summary_prompt,
    create_overall_summary_prompt
)

logger = logging.getLogger(__name__)

class ChainModelEvaluator:
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
            # Print the error instead of using logger.error to avoid threading I/O issues in Jupyter
            print(f"Error generating AI content: {e}")
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
            result = self._generate_ai_response_json(prompt)
            if "__error" in result:
                return {
                    "score": score_list[0] if score_list else None,
                    "analysis": "Failed to generate evaluation",
                    "error": result.get("__error")
                }

            if "score" in result:
                raw_score = result.get("score")
                try:
                    raw_score = int(raw_score)
                except ValueError:
                    pass
                
                # Ensure the AI returned a valid score in the list
                if raw_score not in score_list:
                    logger.warning(f"AI returned invalid score '{raw_score}' not in {score_list}.")
                    chosen_score = score_list[0] if score_list else None
                else:
                    chosen_score = raw_score
                return {
                    "score": chosen_score,
                    "analysis": result.get("analysis", "No analysis provided")
                }
            else:
                # If the JSON parsing failed, raw text available under "__raw_text"
                raw_text = result.get("__raw_text", "")
                logger.warning(f"Invalid JSON in component evaluation: {raw_text[:100]}...")
                return {
                    "score": score_list[0] if score_list else None,
                    "analysis": raw_text or "No analysis provided"
                }

        except Exception as e:
            logger.error(f"Error generating component evaluation: {str(e)}")
            return {
                "score": score_list[0] if score_list else None,
                "analysis": "Failed to generate evaluation",
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
            result = self._generate_ai_response_json(prompt)
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
            result = self._generate_ai_response_json(prompt)
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

    def evaluate(self, text: str) -> Dict[str, Any]:
        """
        Generate an evaluation based on the flexible framework structure.
        """
        try:
            # New Step: Detailed Transcript Analysis
            analysis_prompt = (
    "You are an expert AI analyst trained on the TEACH Classroom Observation Tool (Teach Manual, pp. 17–31) :contentReference[oaicite:0]{index=0}. "
    "Use the TEACH framework’s four components—Time on Task, Classroom Culture, Instruction, and Socioemotional Skills—to analyze the transcript below. "
    "Generate a JSON report strictly matching this schema:\n"
    "{\n"
    "  \"transcript_summary\": string,  // Detailed narrative overview of the lesson (2 paragraphs)\n"
    "  \"time_on_task\": {\n"
    "    \"activity_provision\": \"Y/N\",  // Snapshot-based: see element 0.1 (p. 17) :contentReference[oaicite:1]{index=1}\n"
    "    \"student_on_task\": \"L/M/H\"   // Snapshot-based: element 0.2 (p. 17) :contentReference[oaicite:2]{index=2}\n"
    "  },\n"
    "  \"classroom_culture\": {\n"
    "    \"supportive_learning_environment\": [\"quotes\"],  // Elements 1.1–1.4 (p. 19) :contentReference[oaicite:3]{index=3}\n"
    "    \"behavioral_expectations\": [\"quotes\"]           // Elements 2.1–2.3 (p. 21) :contentReference[oaicite:4]{index=4}\n"
    "  },\n"
    "  \"instruction\": {\n"
    "    \"lesson_facilitation\": [\"quotes\"],       // Elements 3.1–3.4 (p. 23) :contentReference[oaicite:5]{index=5}\n"
    "    \"checks_for_understanding\": [\"quotes\"],  // Elements 4.1–4.3 (p. 24) :contentReference[oaicite:6]{index=6}\n"
    "    \"feedback\": [\"quotes\"],                  // Elements 5.1–5.2 (p. 25) :contentReference[oaicite:7]{index=7}\n"
    "    \"critical_thinking\": [\"quotes\"]          // Elements 6.1–6.3 (p. 26) :contentReference[oaicite:8]{index=8}\n"
    "  },\n"
    "  \"socioemotional_skills\": {\n"
    "    \"autonomy\": [\"quotes\"],                 // Elements 7.1–7.3 (p. 28) :contentReference[oaicite:9]{index=9}\n"
    "    \"perseverance\": [\"quotes\"],             // Elements 8.1–8.3 (p. 30) :contentReference[oaicite:10]{index=10}\n"
    "    \"social_collaborative_skills\": [\"quotes\"] // Elements 9.1–9.3 (p. 31) :contentReference[oaicite:11]{index=11}\n"
    "  },\n"
    "  \"inferences\": string \"Make detailed inferences about the data which is not explicitly provided in the transcript but is likely true. (2 paragraphs) \n"
    "}\n\n"
    "Transcript:\n"
) + text + "\n\nRespond *only* with JSON conforming to the schema above."

            transcript_analysis = self._generate_ai_response_json(analysis_prompt)

            # Step 1: Initialize evaluation structure, including transcript analysis
            evaluation: Dict[str, Any] = {
                "transcript_analysis": transcript_analysis,
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
                                component.get(
                                    "name",
                                    component.get("number", component.get("description", ""))
                                )
                            )
                        )
                        # Pass the transcript analysis JSON as context
                        context_str = json.dumps(transcript_analysis)
                        prompt = create_generic_component_prompt(
                            component,
                            text,
                            self.framework,
                            context_str
                        )
                        if component.get("isManuallyScored", False):
                            placeholder = {
                                "score": None,
                                "analysis": "",
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
                                c for c in domain_component_map.get(domain_id, [])
                                if str(c.get("id", c.get("number", c.get("description", "")))) == comp_id
                            ),
                            {"scoreList": []},
                        )
                        default_score = original_component.get("scoreList", [None])[0]
                        evaluation["domains"][domain_id]["components"][comp_id] = {
                            "score": default_score,
                            "analysis": "Failed to generate evaluation for this component.",
                        }

                # Step 6: Schedule domain summaries
                summary_futures: Dict[str, concurrent.futures.Future] = {}
                for domain_id in domain_component_map:
                    domain_def = next(
                        (d for d in domains_list
                         if str(d.get("id", d.get("number", d.get("name", "")))) == domain_id),
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

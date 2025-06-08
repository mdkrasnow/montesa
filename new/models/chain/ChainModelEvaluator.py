import logging
import json
from copy import deepcopy
from concurrent.futures import ThreadPoolExecutor
import concurrent.futures
from typing import Dict, Any

# Import utility functions for analysis and LLM interaction
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
    domain summaries, and an overall evaluation summary, including enhancement
    of component feedback.
    """

    def __init__(self, framework: Dict[str, Any]):
        """
        Initialize the evaluator with a given framework configuration.

        Args:
            framework (Dict[str, Any]): The framework structure and metadata.
        """
        self.framework = framework

    def analyze_observation_context(self, text: str) -> Dict[str, Any]:
        """
        Analyze the observation context in a framework-agnostic way.
        Currently delegates to the Danielson analyzer for backward compatibility.

        Args:
            text (str): The observation text to analyze.

        Returns:
            Dict[str, Any]: Analysis result or error.
        """
        prompt = f"""
            You are an expert school leader. Analyze the following classroom observation text to identify and summarize key instructional and professional practices observed.

            ### Your analysis should cover:
            1. **Instructional Design and Preparation**:
            - Was the lesson well-prepared and appropriate for the students' developmental level and content area?
            - Was there evidence of clear objectives, differentiated instruction, or assessment strategies?

            2. **Learning Environment**:
            - Describe the classroom culture and climate.
            - Was there evidence of mutual respect, student engagement, behavior management, and routines?

            3. **Instructional Delivery**:
            - Comment on teacher clarity, questioning strategies, student engagement, pacing, and checks for understanding.
            - Highlight how instructional practices supported learning.

            4. **Professionalism and Reflection**:
            - Note any indicators of teacher reflection, collaboration, or responsiveness to student needs.
            - Identify evidence of professionalism or areas for growth.

            ### Additional Aspects to Address:
            - **Time in the Classroom**: Was the observer present for the beginning, middle, end, or entire session? If unclear, make reasonable inferences.
            - **Evaluation Type**: Try to determine if the context is Early Childhood, Special Education, or General Education based on explicit or implicit cues.
            - **Grade Level and Subject Area**: Use context clues to deduce the general grade level and content area. Suggest instructional best practices appropriate to that level.
            
            ### Your Task:
            1. **Structured Evaluation**: Organize your analysis using the categories above.
            2. **Evidence Collection**: Include at least 3–5 direct **word-for-word** quotes from the observation text for each area to support your analysis.
            3. **Identify Gaps**: Note where evidence is missing, vague, or unclear.
            4. **Inference and Interpretation**: When details are not explicitly stated, use observable behaviors to infer effectiveness (e.g., student focus may suggest strong engagement).

            ### Formatting Instructions:
            - Clearly separate direct evidence from inferred insights.
            - Use bullet points or numbered lists for clarity and structure.

            **Observation Text to Analyze:**
            {text}

            Deliver a comprehensive and detailed evaluation suitable for supporting reflective practice and educator growth.
            """


        
        logger.debug(f"Preprocessing Danielson context with text: {text}")
        
        try:
            response = generate_ai_content(prompt)
            if response.parts:
                # Join all parts of the response
                analysis_text = ''.join(part.text for part in response.parts)
                logger.info(f"Preprocessing response: {analysis_text}")
                return {"analysis": analysis_text, "error": None}
            else:
                logger.error("No content in AI response")
                return {"analysis": "", "error": "No content generated"}
        except Exception as e:
            logger.error(f"Error in preprocessing: {str(e)}")
            return {"analysis": "", "error": str(e)}

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
            result = self._generate_ai_response_json(prompt)
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
# ! THIS CODE WAS REMOVED BECAUSE FOR THE RESEARCH PROJECT WE ARE NOT INTERESTED IN THE EVALUATION FEEDBACK. IT IS MAINTAINED AS A COMMENT IF WE WANT TO BRING IT BACK

#     def restructure_component_feedback(
#         self,
#         text: str,
#         evidence: str,
#         component_id: str
#     ) -> str:
#         """
#         Restructure feedback for a single component using AI in a framework-agnostic way.

#         Args:
#             text (str): Original feedback text.
#             evidence (str): Original observation text containing evidence.
#             component_id (str): Component identifier.

#         Returns:
#             str: Restructured feedback.
#         """
#         example_evaluations = [
#             """**Performance Analysis**
# The teacher demonstrates elements of proficient practice, particularly in eliciting student thinking around measurement, but lacks a clear articulation of lesson objectives and a structured approach to address potential student misconceptions. The teacher effectively prompts students to identify potential errors in measurement ("maybe they didn't do it in centimeters," "maybe they didn't start at the beginning of the ruler") which acknowledges possible student struggles. The teacher also uses questioning to encourage students to compare different units of measure ("How was measuring in inches the same as measuring in centimeters?"), promoting mathematical discourse. However, the observation lacks evidence of proactive planning for these misconceptions or alignment to specific learning goals. The focus seems to be on surface-level comparisons of tools rather than deep conceptual understanding of measurement principles. The absence of clearly defined objectives and pre-planned strategies to address misconceptions limits the depth of student learning.

# **Growth Path**
# To elevate the lesson and deepen student understanding, the teacher should explicitly state the learning objectives and pre-plan strategies to address potential misconceptions during the lesson. This could be accomplished by writing a measurable learning objective on the board, such as "Students will be able to accurately measure objects using inches and centimeters, identifying and correcting common errors such as starting at the wrong end of the ruler or using the wrong unit." This will ensure that all activities and questioning are aligned towards a clear goal and improve student focus and understanding of lesson expectations. The teacher should create a misconception chart with columns for the anticipated misconception, the reasoning behind the misconception, and a pre-planned teacher response.
# By articulating a clear learning objective and anticipating student misconceptions through a pre-planned strategy, the teacher will create a more focused and effective learning experience, resulting in higher student success and comprehension as demonstrated by increased accuracy in measurements and the ability to articulate the reasoning behind their approach.

# To build a stronger conceptual foundation for understanding measurement, the teacher should incorporate manipulatives and activities that allow students to explore the relationship between different units of measure and the importance of precise measurement techniques. This could be accomplished by providing students with various objects to measure using both inches and centimeters, prompting them to convert between the two units and discuss the impact of small errors in measurement on the final result. For example, after students measure an object, ask them, "If you started measuring half an inch further along the ruler, how would that affect your measurement in centimeters? Why?" This will help students move beyond rote memorization of measurement techniques to a deeper understanding of the underlying principles, leading to improved problem-solving skills and a stronger grasp of measurement concepts, which can be observed through more consistent accuracy in student measurements and detailed explanations of their reasoning.""",
#             """Performance Analysis
# The teacher demonstrates elements of proficient practice in facilitating student understanding of evaporation through guided inquiry and experimentation. The teacher effectively builds on students' prior knowledge by asking questions that prompt them to articulate their understanding of evaporation ("What do we know about evaporation? Where does the water go?"). This questioning strategy successfully elicits student thinking about the process and encourages students to develop hypotheses about factors that influence evaporation rates. The teacher also effectively supports students in designing and conducting a simple experiment to test their hypothesis about heat's effect on evaporation, guiding them to set up comparative conditions and measure results. However, the observation lacks evidence of explicit connections to broader scientific concepts or vocabulary development beyond the basic mechanics of evaporation. While the teacher guides students to collect data using appropriate tools (ruler to measure water levels), there is limited evidence of supporting students in systematic data collection or analysis of variables that might affect their results.
# Growth Path
# To elevate this lesson and deepen student understanding of evaporation as a scientific process, the teacher should incorporate explicit vocabulary development and connect the concept to the broader water cycle. This could be accomplished by creating a visual anchor chart with scientific terminology such as "evaporation," "condensation," and "water vapor," referring to these terms consistently throughout the investigation, and prompting students to use this vocabulary in their explanations. The teacher should also guide students to document their observations more systematically by creating a simple data table that tracks water levels in both cups at regular time intervals, allowing students to see patterns in the rate of evaporation and develop a more quantitative understanding of the process.
# To further strengthen students' scientific thinking skills, the teacher should expand the investigation to include additional variables that might affect evaporation rates beyond sunlight, such as surface area, air movement, or initial water temperature. This could be accomplished by designing follow-up investigations where students predict, test, and analyze how manipulating these variables affects evaporation rates. For example, after completing the sun/shade experiment, the teacher might ask, "What would happen if we used a wide, shallow dish instead of a cup? How might that affect how quickly the water evaporates?" This would help students develop a more nuanced understanding of evaporation as a process influenced by multiple factors, leading to improved scientific reasoning skills and the ability to apply these concepts to real-world contexts, which can be observed through more sophisticated student explanations that incorporate multiple variables and accurate scientific terminology when discussing evaporation phenomena.""",
#             """Performance Analysis
# The teacher demonstrates elements of proficient practice in facilitating scientific inquiry about plant growth, particularly through guiding students to design a controlled experiment. The teacher effectively uses questioning to help students identify variables that need to be controlled ("What variables should we control?"), which prompts students to consider experimental design elements such as plant type, pot size, and soil consistency. The teacher also guides students to determine appropriate measurement parameters ("What will we measure?") and data collection frequency ("How often should we collect data?"), supporting the development of systematic observation skills. Additionally, the teacher effectively connects the experiment to previous data, encouraging students to analyze patterns and draw evidence-based conclusions. However, the observation lacks evidence of explicit teaching around scientific vocabulary related to plant biology or photosynthesis. While students are engaged in the experimental design process, there is limited indication that the teacher is connecting these observations to deeper scientific concepts about how plants utilize light and water at a cellular level.
# Growth Path
# To elevate this lesson and deepen student understanding of plant biology, the teacher should incorporate explicit scientific vocabulary and connect experimental observations to fundamental biological processes. This could be accomplished by introducing key terms such as "photosynthesis," "chlorophyll," and "transpiration" when discussing why plants respond differently to varying light and water conditions, and creating visual models that illustrate how plants use these resources at the cellular level. The teacher should also guide students to make more precise predictions before conducting experiments by having them articulate the expected relationship between specific variables (light/water) and specific plant responses (height/leaf color/leaf count), recording these predictions in their science notebooks to reference when analyzing results.

# To further strengthen students' scientific thinking, the teacher should expand the investigation to include quantitative measurements and graphical representation of data. This could be accomplished by introducing simple measuring tools like rulers and graduated cylinders to measure plant height and water volume precisely, and teaching students to create line graphs that show growth patterns over time under different conditions. For example, after collecting several data points, the teacher might ask, "How could we display our measurements to make it easier to see patterns in plant growth rates?" This would help students develop more sophisticated data analysis skills and a deeper understanding of the relationship between variables, resulting in improved scientific reasoning and the ability to support claims with quantitative evidence, which can be observed through more detailed student explanations that incorporate scientific terminology and reference specific data points when discussing the factors affecting plant growth.
# """
#         ]

#         prompt = f"""
# # You are a teacher coach. Analyze this teacher observation feedback for component {component_id} and restructure it following the "Performance Analysis, Growth Path" approach. Your feedback should be concise, evidence-based, and focused on student learning outcomes.

# # Original Feedback:
# {text}

# # Original Evidence/Low Inference Notes:
# {evidence}

# # Guidelines for effective feedback:
# 1. Follow the "Performance Analysis, Growth Path" structure:
#    - Performance Analysis: Analyze teacher practices with specific evidence, highlighting both strengths and areas for improvement. Include 3-5 direct quotes from observation notes.
#    - Growth Path: Provide 2-3 specific, actionable next steps that:
#       a) Clearly state what the teacher should do
#       b) Explain how to implement the suggestion with concrete examples
#       c) Connect to expected student outcomes

# 2. For Performance Analysis:
#    - Identify 2-3 specific teacher actions/strategies with direct evidence
#    - Analyze how these actions impacted student learning (positive or negative)
#    - Balance recognition of effective practices with identification of growth areas
#    - Use evaluative language that connects to teaching standards

# 3. For Growth Path:
#    - Make recommendations specific and implementable
#    - Include "This could be accomplished by..." with detailed examples
#    - Explain why each recommendation would improve student learning
#    - Describe what improved outcomes would look like

# 4. General guidelines:
#    - Use evidence-based, specific language rather than general descriptions
#    - Highlight differentiation practices when present
#    - Connect observations to content standards when relevant
#    - Maintain a growth-oriented, constructive tone

# # Example Evaluations (for reference):
# {example_evaluations[0]}

# {example_evaluations[1]}

# {example_evaluations[2]}

# # YOU MUST NOT INCLUDE ANY OTHER METADATA OR DISCUSSION IN YOUR RESPONSE. JUST THE IMPROVED COMPONENT SUMMARY.

# # Your improved component summary:
# """

#         try:
#             response = generate_ai_content(prompt)
#             if hasattr(response, "parts"):
#                 enhanced_text = "".join(part.text for part in response.parts)
#                 return enhanced_text
#             else:
#                 logger.error("No content generated for restructure_component_feedback")
#                 return text
#         except Exception as e:
#             logger.error(f"Error restructuring feedback for component {component_id}: {str(e)}")
#             return text

    # def enhance_evaluation_feedback(
    #     self,
    #     evaluation: Dict[str, Any],
    #     text: str
    # ) -> Dict[str, Any]:
    #     """
    #     Enhance evaluation feedback for components with detailed evidence and actionable suggestions.

    #     Args:
    #         evaluation (dict): The evaluation data to enhance.
    #         text (str): The observation text.

    #     Returns:
    #         dict: The enhanced evaluation data.
    #     """
    #     domains = evaluation.get("domains", {})
    #     futures: Dict[(str, str), concurrent.futures.Future] = {}

    #     with ThreadPoolExecutor() as executor:
    #         # Queue up enhancement tasks for all components
    #         for domain_id, domain_data in domains.items():
    #             for comp_id, comp_data in domain_data.get("components", {}).items():
    #                 # Skip components manually scored (and not modified)
    #                 if comp_data.get("isManuallyScored", False) and not comp_data.get("modified", False):
    #                     continue
    #                 original_feedback = comp_data.get("summary", "")
    #                 future = executor.submit(
    #                     self.restructure_component_feedback,
    #                     original_feedback,
    #                     text,
    #                     comp_id,
    #                 )
    #                 futures[(domain_id, comp_id)] = future

    #         # Collect enhancement results
    #         for (domain_id, comp_id), future in futures.items():
    #             try:
    #                 enhanced_summary = future.result()
    #                 if isinstance(enhanced_summary, str) and enhanced_summary.strip():
    #                     domains[domain_id]["components"][comp_id]["summary"] = enhanced_summary
    #             except Exception as e:
    #                 logger.error(
    #                     f"Error enhancing summary for component {comp_id} in domain {domain_id}: {str(e)}"
    #                 )

    #     return evaluation

    def evaluate(self, text: str) -> Dict[str, Any]:
        """
        Generate an evaluation based on the flexible framework structure.

        Args:
            text (str): The observation text.

        Returns:
            Dict[str, Any]: The generated evaluation result.
        """
        try:
            # Step 1: Context analysis
            context_result = self.analyze_observation_context(text)
            if context_result.get("error"):
                return {"success": False, "error": context_result["error"]}
            context = context_result.get("analysis", {})

            # Step 2: Initialize evaluation structure
            evaluation: Dict[str, Any] = {
                "domains": {},
                "metadata": {
                    "framework_id": self.framework.get("framework_id"),
                    "framework_name": self.framework.get("name", "Teaching Framework"),
                },
            }

            # Step 3: Extract framework structure
            structure = self.framework.get("structure", {})
            domains_list = structure.get("domains", [])

            # Map domain IDs -> component definitions
            domain_component_map: Dict[str, Any] = {}
            for domain in domains_list:
                domain_id = str(domain.get("id", domain.get("number", domain.get("name", ""))))
                domain_component_map[domain_id] = domain.get("components", [])

            # Step 4: Schedule component evaluations
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
                        prompt = create_generic_component_prompt(component, text, self.framework, context)

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

                # Step 5: Initialize domain entries in evaluation
                for domain in domains_list:
                    domain_id = str(domain.get("id", domain.get("number", domain.get("name", ""))))
                    evaluation["domains"][domain_id] = {
                        "name": domain.get("name", ""),
                        "components": {},
                        "weight": domain.get("weight", 1.0),
                        "isManuallyScored": domain.get("isManuallyScored", False),
                    }

                # Step 6: Collect component results
                domain_component_results: Dict[str, Dict[str, Any]] = {
                    domain_id: {} for domain_id in domain_component_map
                }
                for (domain_id, comp_id), future in component_futures.items():
                    try:
                        result = future.result()
                        result["isManuallyScored"] = evaluation["domains"][domain_id]["components"].get(
                            comp_id, {}
                        ).get("isManuallyScored", result.get("isManuallyScored", False))
                        evaluation["domains"][domain_id]["components"][comp_id] = result
                        domain_component_results[domain_id][comp_id] = result
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

                # Step 7: Schedule domain summaries
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

                # Step 8: Collect domain summaries
                for domain_id, future in summary_futures.items():
                    try:
                        domain_summary = future.result()
                        evaluation["domains"][domain_id]["summary"] = domain_summary.get("summary", "")
                    except Exception as exc:
                        logger.error(f"Domain {domain_id} summary error: {exc}")
                        evaluation["domains"][domain_id]["summary"] = ""

            # Step 9: Calculate domain scores
            # ! THIS CODE WAS REMOVED BECAUSE FOR THE RESEARCH PROJECT WE ARE NOT INTERESTED IN THE OVERALL SCORES. IT IS MAINTAINED AS A COMMENT IF WE WANT TO BRING IT BACK
            # for domain_id, domain_data in evaluation["domains"].items():
            #     domain_def = next(
            #         (d for d in domains_list if str(d.get("id", d.get("number", d.get("name", "")))) == domain_id),
            #         {}
            #     )
            #     domain_components = {
            #         comp_id: comp for comp_id, comp in domain_data["components"].items()
            #     }
            #     domain_data["domainScore"] = calculate_domain_score(
            #         domain_components, domain_def, self.framework, skip_manually_scored=True
            #     )

            # Step 10: Generate overall summary
            overall_prompt = create_overall_summary_prompt(self.framework, evaluation["domains"])
            overall_summary = self.generate_overall_summary(overall_prompt)
            evaluation["summary"] = overall_summary.get("summary", "")
            # evaluation["summaryScores"] = calculate_overall_scores(evaluation["domains"], self.framework)

            # Step 11: Enhance component-level feedback
            # ! THIS CODE WAS REMOVED BECAUSE FOR THE RESEARCH PROJECT WE ARE NOT INTERESTED IN THE EVALUATION FEEDBACK. IT IS MAINTAINED AS A COMMENT IF WE WANT TO BRING IT BACK
            # evaluation = self.enhance_evaluation_feedback(evaluation, text)
            return {"success": True, "evaluation": evaluation}

        except Exception as e:
            logger.error(f"Error generating framework-based evaluation: {str(e)}")
            return {"success": False, "error": str(e)}

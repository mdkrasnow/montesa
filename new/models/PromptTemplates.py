# PromptTemplates.py
"""
Framework-agnostic prompt templates for evaluation generation.

This module provides templates for creating LLM prompts that can adapt to any
framework structure, replacing hardcoded Danielson-specific prompts.
"""

from typing import Dict, Any, List
import logging
from ._context.teach_1_manual import manual_text

# Configure logging
logger = logging.getLogger(__name__)


def create_domain_batch_prompt(domain: Dict[str, Any], observation_text: str, framework: Dict[str, Any]) -> str:
    """
    Create a prompt for evaluating ALL components in a domain simultaneously.
    This replaces individual component API calls with a single batch call per domain.

    Args:
        domain (Dict): The domain details including all components.
        observation_text (str): The observation text to evaluate.
        framework (Dict): The framework structure.

    Returns:
        str: A prompt that instructs the LLM to output JSON with simple numbered keys.
    """
    framework_name = framework.get("name", "Teaching Framework")
    domain_name = domain.get("name", "Domain")
    domain_description = domain.get("description", "")
    components = domain.get("components", [])

    # Base instructions
    base_prompt = (
        f"You are an expert evaluator using the {framework_name}. "
        f"Evaluate the teacher observation for the domain: {domain_name} - {domain_description}.\n\n"
        f"You must evaluate ALL {len(components)} components in this domain simultaneously.\n\n"
    )

    # Include rubric once for the entire domain
    rubric_text = manual_text
    base_prompt += f"# Rubric Criteria:\n{rubric_text}\n\n"

    # List all components with their scoring options
    components_prompt = "# Components to Evaluate:\n\n"
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
        comp_name = component.get("name", comp_id)
        comp_description = component.get("description", "")
        score_list = component.get("scoreList", [])
        
        components_prompt += (
            f"{i}. **Component {i}**: {comp_name}\n"
            f"   ID: {comp_id}\n"
            f"   Description: {comp_description}\n"
            f"   Score Options: {score_list}\n\n"
        )

    # JSON structure specification with simple numbered keys
    json_structure = "# Required JSON Output Structure:\n\n```json\n{\n"
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
        comp_name = component.get("name", comp_id)
        score_list = component.get("scoreList", [])
        
        json_structure += f'  "component_{i}": {{\n'
        json_structure += f'    "component_id": "{comp_id}",\n'
        json_structure += f'    "component_name": "{comp_name}",\n'
        json_structure += f'    "analysis": "Your detailed analysis for this component",\n'
        json_structure += f'    "score": "Selected score from: {score_list}"\n'
        json_structure += f'  }}'
        
        # Add comma if not the last component
        if i < len(components):
            json_structure += ','
        json_structure += '\n'
    
    json_structure += '}\n```\n\n'

    # Final instructions
    instructions = (
        f"# Classroom Observation Transcript:\n{observation_text}\n\n"
        "# Instructions:\n"
        "1. For EACH component listed above, provide a detailed analysis explaining your reasoning\n"
        "2. Select the most appropriate score from each component's scoreList\n"
        "3. Consider the rubric criteria when making your evaluations\n"
        "4. Your response must be valid JSON with the EXACT structure shown above\n"
        "5. Use simple numbered keys: component_1, component_2, etc. (case-sensitive)\n"
        "6. Include the component_id and component_name fields exactly as shown\n"
        "7. Do not include any text outside the JSON structure\n"
        "8. Ensure every component is evaluated - missing components will cause errors\n\n"
        "Provide thorough analysis for each component while being concise and evidence-based.\n"
    )

    # Assemble the complete prompt
    full_prompt = base_prompt + components_prompt + json_structure + instructions
    
    return full_prompt

def create_control_component_prompt(component: Dict[str, Any], observation_text: str, context: str) -> str:
    """
    Dynamically create a prompt for evaluating a component in any framework that
    iterates through each possible score option, providing chain-of-thought analysis
    and selecting the most appropriate score.

    Args:
        component (Dict): The component details including description and scoring parameters.
        observation_text (str): The observation text to evaluate.
        framework (Dict): The framework structure.
        context (str): Additional context information.

    Returns:
        str: A prompt that instructs the LLM to output JSON with 'analysis' and 'score'.
    """
    # Get framework metadata
    component_description = component.get("description", "")
    score_list = component.get("scoreList", [])

    # Base instructions
    base_prompt = (
        f"You are an expert evaluator using the Teach 1 Framework from the World Bank. "
        f"Evaluate the teacher observation for the component: {component_description}."
    )

    # Inject rubric criteria if provided
    rubric_text = manual_text

    # Enumerate possible scores
    score_prompt = "Possible score options:\n"
    for option in score_list:
        score_prompt += f"- {option}\n"

    # Assemble the final prompt with chain-of-thought and strict JSON requirements
    prompt = (
        f"{base_prompt}\n\n"
        f"{score_prompt}\n\n"
        f"# Context:\n{context}\n\n"
        f"# Transcript from classroom observation:\n{observation_text}\n\n"
        "# Instructions:\n"
        "For each score option, think step by step and provide your reasoning under "
        "analysis.<score_option>. After reviewing all options, set the final chosen "
        "score under the key 'score'.\n"
        "Output must be pure JSON with exactly two top-level keys:\n"
        "  - analysis: an object mapping each score to its chain-of-thought justification\n"
        "  - score: the selected score option\n"
        "Do not include any tokens, commentary, or formatting outside this JSON."
        "In your analysis, you must iterate through each possible score option and provide analysis as for why or why not the score should be given."
        "After reviewing all options, set the most likely chosen score under the key 'score'."
        f"Rubric Criteria:\n{rubric_text}\n\n"
        f"{base_prompt}\n\n"
        f"{score_prompt}\n\n"
        f"# Context:\n{context}\n\n"
        f"# Transcript from classroom observation:\n{observation_text}\n\n"
        "# Instructions:\n"
        "For each score option, think step by step and provide your reasoning under "
        "analysis.<score_option>. After reviewing all options, set the final chosen "
        "score under the key 'score'.\n"
        "Output must be pure JSON with exactly two top-level keys:\n"
        "  - analysis: an object mapping each score to its chain-of-thought justification\n"
        "  - score: the selected score option\n"
        "Do not include any tokens, commentary, or formatting outside this JSON."
        "In your analysis, you must iterate through each possible score option and provide analysis as for why or why not the score should be given."
        "After reviewing all options, set the most likely chosen score under the key 'score'."

    )
    return prompt

def create_generic_component_prompt(component: Dict[str, Any], observation_text: str, framework: Dict[str, Any], context: str) -> str:
    """
    Dynamically create a prompt for evaluating a component in any framework that
    iterates through each possible score option, providing chain-of-thought analysis
    and selecting the most appropriate score.

    Args:
        component (Dict): The component details including description and scoring parameters.
        observation_text (str): The observation text to evaluate.
        framework (Dict): The framework structure.
        context (str): Additional context information.

    Returns:
        str: A prompt that instructs the LLM to output JSON with 'analysis' and 'score'.
    """
    # Get framework metadata
    framework_name = framework.get("name", "Teaching Framework")
    score_list = component.get("scoreList", [])
    component_description = component.get("description", "")

    # Base instructions
    base_prompt = (
        f"You are an expert evaluator using the {framework_name}. "
        f"Evaluate the teacher observation for the component: {component_description}."
    )

    # Inject rubric criteria if provided
    rubric_text = component.get("rubric") or component.get("rubric_description", "")
    if rubric_text:
        base_prompt += f"\n\nRubric Criteria:\n{rubric_text}\n"

    # Enumerate possible scores
    score_prompt = "Possible score options:\n"
    for option in score_list:
        score_prompt += f"- {option}\n"

    # Assemble the final prompt with chain-of-thought and strict JSON requirements
    prompt = (
        f"{base_prompt}\n\n"
        f"{score_prompt}\n\n"
        f"# Context:\n{context}\n\n"
        f"# Transcript from classroom observation:\n{observation_text}\n\n"
        "# Instructions:\n"
        "For each score option, think step by step and provide your reasoning under "
        "analysis.<score_option>. After reviewing all options, set the final chosen "
        "score under the key 'score'.\n"
        "Output must be pure JSON with exactly two top-level keys:\n"
        "  - analysis: an object mapping each score to its chain-of-thought justification\n"
        "  - score: the selected score option\n"
        "Do not include any tokens, commentary, or formatting outside this JSON."
        "In your analysis, you must iterate through each possible score option and provide analysis as for why or why not the score should be given."
        "After reviewing all options, set the most likely chosen score under the key 'score'."
    )
    return prompt

def create_domain_summary_prompt(domain: Dict[str, Any], framework: Dict[str, Any], component_evaluations: Dict[str, Any]) -> str:
    """
    Create a prompt for generating a domain-level summary.

    Args:
        domain (Dict): The domain details.
        framework (Dict): The framework structure.
        component_evaluations (Dict): Evaluations of components in this domain.

    Returns:
        str: A prompt for generating the domain summary.
    """
    framework_name = framework.get("name", "Teaching Framework")
    domain_name = domain.get("name", "Domain")
    domain_description = domain.get("description", "")

    # Format component evaluations for inclusion in the prompt
    component_summaries = []
    for component in domain.get("components", []):
        component_id = str(component.get("id", component.get("description", "")))
        if component_id in component_evaluations:
            comp_eval = component_evaluations[component_id]
            component_summaries.append(
                f"Component: {component.get('description', '')}\n"
                f"Score: {comp_eval.get('score', 'N/A')}\n"
                f"Summary: {comp_eval.get('summary', 'No summary')}"
            )

    component_text = "\n\n".join(component_summaries)

    prompt = (
        f"You are an expert evaluator using the {framework_name}. "
        f"Create a summary for the domain: {domain_name} - {domain_description}.\n\n"
        f"Component Evaluations:\n{component_text}\n\n"
        "Guidelines for effective domain summary:\n"
        "1. Keep your summary succinct and digestible (one paragraph maximum)\n"
        "2. Identify 2-3 key strengths across components with specific evidence\n"
        "3. Highlight ONE high-leverage growth area that will have the greatest impact on student learning\n"
        "4. Provide specific action steps for improvement in the growth area\n"
        "5. If evidence of differentiation exists, emphasize how the teacher addresses needs of diverse learners\n"
        "6. Connect feedback to specific, observable student outcomes\n"
        "7. If information is insufficient, clearly state which component has the most adequate evidence\n\n"
        "Provide a JSON output with a single key: 'summary'.\n"
        "The summary should follow the guidelines above and synthesize component evaluations "
        "into a cohesive domain assessment.\n"
        "Do not include any commentary outside the JSON structure.\n"
    )
    return prompt

def create_overall_summary_prompt(framework: Dict[str, Any], domain_evaluations: Dict[str, Any]) -> str:
    """
    Create a prompt for generating an overall evaluation summary.

    Args:
        framework (Dict): The framework structure.
        domain_evaluations (Dict): Evaluations of all domains.

    Returns:
        str: A prompt for generating the overall summary.
    """
    framework_name = framework.get("name", "Teaching Framework")

    # Format domain summaries for inclusion in the prompt
    domain_summaries = []
    for domain_id, domain_data in domain_evaluations.items():
        domain_summaries.append(
            f"Domain: {domain_data.get('name', '')}\n"
            f"Score: {domain_data.get('domainScore', 'N/A')}\n"
            f"Summary: {domain_data.get('summary', 'No summary')}"
        )

    domain_text = "\n\n".join(domain_summaries)

    prompt = (
        f"Create a concise overall summary of this teacher evaluation based on the {framework_name}.\n\n"
        f"Domain Summaries:\n{domain_text}\n\n"
        "Guidelines for the summary:\n"
        "1. Follow the 'Glow and Grow' structure:\n"
        "   - Glow: Identify the strongest domain and 2-3 specific effective practices\n"
        "   - Grow: Identify ONE high-leverage area for growth with the greatest potential impact\n"
        "2. Provide 2-3 specific, actionable next steps that connect to student outcomes\n"
        "3. If evidence shows differentiation, highlight how the teacher addresses diverse learners\n"
        "4. Connect feedback to standards/progression frameworks when relevant\n"
        "5. Keep the overall summary to two paragraphs maximum (one for strengths, one for growth)\n"
        "6. Do not include score values in the summary\n\n"
        "Provide a JSON response with one key: 'summary'.\n"
        "The summary should follow the guidelines above and provide a holistic view of the teacher's performance.\n"
    )
    return prompt


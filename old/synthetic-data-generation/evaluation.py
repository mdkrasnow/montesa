from flask import request, jsonify
from . import swiftscope_bp
from supabase import create_client, Client
import os, json
from decorators import token_required
from typing import Any, Optional
from google.generativeai import GenerativeModel
import google.generativeai as genai
from groq import Groq
import importlib.util

# === NEW GLOBAL VARIABLE AND GROQ WRAPPER CLASSES/HELPERS ===
USE_GROQ = os.getenv("USE_GROQ", "true").lower() == "true"
if USE_GROQ:
    groq_client = Groq(api_key=os.getenv("GROQ_API_KEY"))

class GroqResponsePart:
    def __init__(self, text: str):
        self.text = text

class GroqResponse:
    def __init__(self, content: str):
        self.parts = [GroqResponsePart(content)]
        self.text = content


def load_example_data():
    """
    Load inputs.py (defines `inputs` list) and outputs.json (mapping index -> JSON output),
    pair them by index, and return a list of (input_text, expected_output_dict).
    """
    base_dir = os.path.join(os.path.dirname(__file__), 'data', 'example-data')
    # Load inputs.py dynamically
    inputs_path = os.path.join(base_dir, 'inputs.py')
    spec = importlib.util.spec_from_file_location("example_inputs", inputs_path)
    example_inputs = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(example_inputs)
    inputs_list = getattr(example_inputs, 'inputs', [])
    # Load outputs.json
    outputs_path = os.path.join(base_dir, 'outputs.json')
    with open(outputs_path, 'r') as f:
        outputs_dict = json.load(f)
    # Pair by index
    examples = []
    for idx, text_input in enumerate(inputs_list):
        key = str(idx)
        output_json = outputs_dict.get(key)
        if output_json is not None:
            examples.append((text_input, output_json))
    return examples

def generate_input_data():
    """
    Generate a brand-new example input as JSON with two keys:
      - description: a brief description of what the evaluation will be like
      - content: the actual observation text

    Returns:
      {
        "success": true,
        "description": "...",
        "content": "..."
      }
    """
    # 1. Load our examples
    examples = load_example_data()
    if not examples:
        return jsonify({"success": False, "error": "No example data available"}), 500

    # 2. Build the few‑shot context
    examples_context = ""
    for idx, (inp, _) in enumerate(examples):
        examples_context += f"\nExample {idx}:\n{inp}\n"

    # 3. Prompt the LLM for valid JSON output
    prompt = f"""
You are provided with several example classroom observation texts:
{examples_context}

Task: Create one *new* classroom observation input that closely matches the style, tone, and structure of the provided examples.

Respond *only* with valid JSON in the following format (no extra keys, text, or commentary):

{{
  "description": "<A concise description of the classroom observation>",
  "content": "<The full observation text>"
}}
"""
    # 4. Call the LLM, requesting JSON
    response = generate_ai_content(
        prompt,
        generation_config={
            "temperature": 0.7,
            "response_mime_type": "application/json"
        }
    )

    # 5. Extract and parse the JSON response
    try:
        raw = (
            ''.join(part.text for part in response.parts).strip()
            if hasattr(response, 'parts')
            else response.text.strip()
        )
        data = json.loads(raw)

        # 6. Validate keys
        if not all(key in data for key in ("description", "content")):
            raise ValueError("Model JSON is missing required keys")

        return jsonify({"success": True, **data}), 200

    except Exception as e:
        return jsonify({
            "success": False,
            "error": f"Invalid model response: {e}",
            "raw": raw
        }), 500




# Single-call evaluation creator 
def create_evaluation():
    """
    Generate a full evaluation using a single LLM call based on the Teach Primary framework.
    """
    data = request.json or {}
    text = data.get('text')
    framework_id = data.get('frameworkId')
    if not text or not framework_id:
        return jsonify({"success": False, "error": "Missing required data"}), 400
    
    framework_json = {
        "framework_id": "Teach",
        "name": "Teach Primary",
        "structure": {
            "domains": [
                {
                    "id": "time_on_task",
                    "name": "Time on Task",
                    "weight": 0.2,
                    "components": [
                        {
                            "id": "snapshot_1",
                            "name": "Snapshot 1",
                            "description": "First 1-10s snapshot of teacher action and student on-task count",
                            "score_min": 1,
                            "score_max": 5
                        },
                        {
                            "id": "snapshot_2",
                            "name": "Snapshot 2",
                            "description": "Second 1-10s snapshot",
                            "score_min": 1,
                            "score_max": 5
                        },
                        {
                            "id": "snapshot_3",
                            "name": "Snapshot 3",
                            "description": "Third 1-10s snapshot",
                            "score_min": 1,
                            "score_max": 5
                        }
                    ]
                },
                {
                    "id": "classroom_culture",
                    "name": "Classroom Culture",
                    "weight": 0.267,
                    "components": [
                        { "id": "culture_1", "name": "Positive Environment", "score_min": 1, "score_max": 5 },
                        { "id": "culture_2", "name": "Behavioral Expectations", "score_min": 1, "score_max": 5 }
                    ]
                },
                {
                    "id": "instruction",
                    "name": "Instruction",
                    "weight": 0.267,
                    "components": [
                        { "id": "instr_1", "name": "Facilitating Understanding", "score_min": 1, "score_max": 5 },
                        { "id": "instr_2", "name": "Checks for Understanding", "score_min": 1, "score_max": 5 }
                    ]
                },
                {
                    "id": "socioemotional_skills",
                    "name": "Socioemotional Skills",
                    "weight": 0.267,
                    "components": [
                        { "id": "se_1", "name": "Autonomy", "score_min": 1, "score_max": 5 },
                        { "id": "se_2", "name": "Collaboration", "score_min": 1, "score_max": 5 }
                    ]
                }
            ]
        }
    }

    # Load example data
    examples = load_example_data()
    examples_context = ""
    for idx, (inp, outp) in enumerate(examples):
        examples_context += f"\nExample {idx} Input:\n{inp}\nExpected Output:\n```json\n{json.dumps(outp, indent=2)}\n```\n"


    # Build a single prompt
    prompt = f"""
Generate a complete evaluation for the following classroom observation using the Teach Primary framework.

# Examples: 
{examples_context}

Output must be valid JSON in this format:

{{
  "success": true,
  "evaluation": {{
    "domains": {{
      "<domain_id>": {{
        "name": "<domain_name>",
        "components": {{
          "<component_id>": {{
            "score": <int 1-5>,
            "summary": "<brief analysis>"
          }},
          ...
        }},
        "domainScore": <float>,
        "domainSummary": "<brief summary>"
      }},
      ...
    }},
    "summary": "<overall summary of teaching practice>",
    "summaryScores": {{
      "overallScore": <float>,
      "domainWeights": {{ "<domain_id>": <weight>, ... }}
    }}
  }}
}}

Here is the framework definition:
```json
{json.dumps(framework_json, indent=2)}
```

Observation text: 
{text}
"""

    # Single LLM call
    response = generate_ai_content(
        prompt,
        generation_config={"temperature": 0, "response_mime_type": "application/json"}
    )

    # Parse and return
    try:
        raw = ''.join(part.text for part in response.parts).strip() if hasattr(response, 'parts') else response.text
        evaluation_result = json.loads(raw)
        return jsonify(evaluation_result)
    except Exception as e:
        return jsonify({"success": False, "error": f"Invalid model response: {e}", "raw": raw}), 500
    


def generate_ai_content(prompt: str, generation_config: Optional[Any] = None):
    """
    Wrapper for generating AI content using either Gemini or Groq.
    For Groq, we use the "llama-3.3-70b-versatile" model and map the generation
    parameters (e.g. temperature and JSON output) per the Groq API documentation.
    """
    # Get the current thread's useGroq value, defaulting to environment variable if not set
    use_groq = False
    
    if use_groq:
        # Map generation configuration to Groq parameters
        temperature = generation_config.temperature if generation_config and hasattr(generation_config, "temperature") else 1
        response_format = {"type": "json_object"} if generation_config and hasattr(generation_config, "response_mime_type") and generation_config.response_mime_type == "application/json" else None
        # Call Groq's chat completions API (see documentation for available parameters)
        response_data = groq_client.chat.completions.create(
            messages=[{
                "role": "system",
                "content": prompt
            }],
            model="llama-3.3-70b-versatile",
            temperature=temperature,
            response_format=response_format
        )
        content = response_data.choices[0].message.content
        return GroqResponse(content)
    else:
        # Use Gemini as before
        model = genai.GenerativeModel("gemini-2.0-flash")
        return model.generate_content(prompt, generation_config=generation_config)
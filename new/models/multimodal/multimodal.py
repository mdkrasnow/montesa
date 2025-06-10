import logging
import json
from typing import Dict, Any

# Import base evaluator and multimodal utilities
from ..base.BaseModelEvaluator import BaseModelEvaluator
from ..AI import generate_ai_content, upload_multimodal_file

logger = logging.getLogger(__name__)

class MultiModalModelEvaluator(BaseModelEvaluator):
    """
    Extension of BaseModelEvaluator to handle both transcript text and
    an audio file as inputs for multimodal evaluation.
    """

    def __init__(self, framework: Dict[str, Any]):
        """
        Initialize the multimodal evaluator with the given framework.

        Args:
            framework (Dict[str, Any]): The framework structure and metadata.
        """
        super().__init__(framework)
        self.audio_file_ref = None  # Will hold reference to uploaded audio

    def _generate_ai_response_json(self, prompt: str) -> Dict[str, Any]:
        """
        Override to bundle the audio file reference with each LLM call.

        Args:
            prompt (str): The LLM prompt text.

        Returns:
            Dict[str, Any]: Parsed JSON or raw text on error.
        """
        try:
            # If no audio has been uploaded, fallback to base behavior
            if not self.audio_file_ref:
                logger.warning("No audio file uploaded. Falling back to text-only evaluation.")
                return super()._generate_ai_response_json(prompt)

            # Prepare multimodal message sequence: prompt plus audio
            messages = [prompt, self.audio_file_ref]
            response = generate_ai_content(
                messages,
                generation_config={"temperature": 0, "response_mime_type": "application/json"}
            )

            # Assemble and parse JSON response as in BaseModelEvaluator
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
            # Print instead of logger.error to avoid threading I/O issues
            print(f"Error generating multimodal AI content: {e}")
            return {"__error": str(e)}

    def evaluate(self, text: str, audio_file_path: str) -> Dict[str, Any]:
        """
        Run the full evaluation pipeline using both transcript text and audio file.

        Args:
            text (str): Transcript or text input.
            audio_file_path (str): Local path to the audio file.

        Returns:
            Dict[str, Any]: Evaluation result with domain/component scores and summaries.
        """
        # Upload the audio file before any LLM calls
        try:
            self.audio_file_ref = upload_multimodal_file(audio_file_path)
        except Exception as e:
            logger.error(f"Error uploading audio file: {e}")
            return {"success": False, "error": f"Audio upload failed: {e}"}
        finally:
            # ensure we clear the reference
            self.audio_file_ref = None

        return super().evaluate(text)
        

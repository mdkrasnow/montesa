#!/usr/bin/env python
"""
Script to format synthetic data properly for LLM training and then train an LLM on that data.
This script handles:
1. Formatting data into the required format for LLM fine-tuning
2. Training an LLM on the formatted data using AutoTrain
"""

import os
import json
import argparse
from pathlib import Path
import logging
from typing import List, Dict, Any, Optional, Union
import importlib.util

from autotrain.params import LLMTrainingParams
from autotrain.project import AutoTrainProject

# Set up logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)

logger = logging.getLogger(__name__)

def load_synthetic_data(base_dir: str) -> tuple:
    """
    Load synthetic_inputs.py and synthetic_outputs.json, and return them as usable data.
    
    Args:
        base_dir: Directory containing the synthetic data files
        
    Returns:
        tuple: (inputs_list, outputs_dict) containing the loaded data
    """
    try:
        # Path to synthetic inputs file
        inputs_path = os.path.join(base_dir, 'synthetic_inputs.py')
        
        # Check if the file exists
        if not os.path.exists(inputs_path):
            logger.error(f"synthetic_inputs.py not found at {inputs_path}")
            return [], {}
            
        # Load inputs.py dynamically
        spec = importlib.util.spec_from_file_location("synthetic_inputs", inputs_path)
        synthetic_inputs = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(synthetic_inputs)
        
        # Get the inputs list from the module
        inputs_list = getattr(synthetic_inputs, 'inputs', [])
        
        # Load outputs.json
        outputs_path = os.path.join(base_dir, 'synthetic_outputs.json')
        
        # Check if the file exists
        if not os.path.exists(outputs_path):
            logger.error(f"synthetic_outputs.json not found at {outputs_path}")
            return inputs_list, {}
            
        with open(outputs_path, 'r') as f:
            outputs_dict = json.load(f)
            
        return inputs_list, outputs_dict
        
    except Exception as e:
        logger.error(f"Error loading synthetic data: {str(e)}")
        return [], {}

def format_data_for_training(inputs: List[Dict], outputs: Dict, format_type: str = "chatml") -> List[Dict]:
    """
    Format the synthetic data for training an LLM.
    
    Args:
        inputs: List of input dictionaries (containing 'description' and 'content')
        outputs: Dictionary mapping index -> evaluation outputs
        format_type: Format to use (chatml, zephyr, or none)
        
    Returns:
        List of formatted data entries ready for training
    """
    formatted_data = []
    
    for idx, input_data in enumerate(inputs):
        idx_str = str(idx)
        
        # Skip if no corresponding output exists
        if idx_str not in outputs:
            logger.warning(f"No output found for input index {idx}")
            continue
            
        output_data = outputs[idx_str]
        
        # Get the observation text and evaluation
        observation_text = input_data.get("content", "")
        evaluation = output_data
        
        # Format as JSONL with conversations according to the specified chat template
        if format_type.lower() in ["chatml", "zephyr"]:
            # Create conversation format with system, user, and assistant messages
            conversation = [
                {"role": "system", "content": "You are an AI assistant that evaluates classroom teaching observations and provides structured feedback based on educational frameworks."},
                {"role": "user", "content": observation_text},
                {"role": "assistant", "content": json.dumps(evaluation, indent=2)}
            ]
            
            formatted_data.append(conversation)
        else:
            # Plain text format (not recommended for this use case)
            formatted_entry = {
                "text": f"Observation:\n{observation_text}\n\nEvaluation:\n{json.dumps(evaluation, indent=2)}"
            }
            formatted_data.append(formatted_entry)
    
    return formatted_data

def save_formatted_data(data: List, output_dir: str, format_type: str = "jsonl") -> str:
    """
    Save the formatted data to a file.
    
    Args:
        data: List of formatted data entries
        output_dir: Directory to save the formatted data
        format_type: File format to save as (jsonl or csv)
        
    Returns:
        str: Path to the saved data file
    """
    os.makedirs(output_dir, exist_ok=True)
    
    if format_type.lower() == "jsonl":
        output_path = os.path.join(output_dir, "train.jsonl")
        
        with open(output_path, 'w') as f:
            for entry in data:
                f.write(json.dumps(entry) + "\n")
                
    elif format_type.lower() == "csv":
        # Not recommended for this use case, but included for completeness
        output_path = os.path.join(output_dir, "train.csv")
        
        import csv
        with open(output_path, 'w', newline='') as f:
            writer = csv.writer(f)
            # Write header
            writer.writerow(["text"])
            # Write data
            for entry in data:
                if isinstance(entry, list):  # Conversation format
                    writer.writerow([json.dumps(entry)])
                else:  # Plain text format
                    writer.writerow([entry.get("text", "")])
    else:
        raise ValueError(f"Unsupported format type: {format_type}")
        
    logger.info(f"Saved formatted data to {output_path}")
    return output_path

def train_model(data_dir: str, model_name: str, output_dir: str, chat_template: str = "chatml",
               push_to_hub: bool = False, username: Optional[str] = None, token: Optional[str] = None) -> None:
    """
    Train an LLM on the formatted data using AutoTrain.
    
    Args:
        data_dir: Directory containing the formatted training data
        model_name: Name of the pre-trained model to fine-tune
        output_dir: Directory to save the trained model
        chat_template: Chat template to use (chatml, zephyr, tokenizer, or none)
        push_to_hub: Whether to push the trained model to Hugging Face Hub
        username: Hugging Face username (required if push_to_hub is True)
        token: Hugging Face token (required if push_to_hub is True)
    """
    # Define training parameters
    params = LLMTrainingParams(
        model=model_name,
        project_name=output_dir,
        data_path=data_dir,
        train_split="train",
        valid_split=None,  # No validation data for this example
        chat_template=chat_template,
        text_column="text" if chat_template == "none" else "messages",
        
        # Training hyperparameters
        trainer="sft",  # Using Supervised Fine-Tuning trainer
        epochs=3,
        batch_size=2,
        lr=3e-5,
        warmup_ratio=0.1,
        
        # Optimization parameters
        peft=True,  # Use Parameter-Efficient Fine-Tuning
        quantization="int4",
        target_modules="all-linear",
        gradient_accumulation=4,
        mixed_precision="fp16",
        
        # Model parameters
        block_size=1024,
        model_max_length=2048,
        padding="right",
        
        # Optimizer and scheduler
        optimizer="adamw_torch",
        scheduler="linear",
        
        # Logging and saving
        log="tensorboard",
        
        # HuggingFace Hub parameters
        push_to_hub=push_to_hub,
        username=username,
        token=token
    )
    
    # Create and run the AutoTrain project
    project = AutoTrainProject(params=params, backend="local", process=True)
    project.create()
    
    logger.info(f"Training completed. Model saved to {output_dir}")

def main():
    parser = argparse.ArgumentParser(description="Format and train an LLM on synthetic classroom observation data")
    
    parser.add_argument("--data-dir", type=str, required=True, 
                        help="Directory containing synthetic_inputs.py and synthetic_outputs.json")
    parser.add_argument("--output-dir", type=str, default="trained_llm",
                        help="Directory to save the formatted data and trained model")
    parser.add_argument("--model", type=str, default="google/flan-t5-base",
                        help="Pre-trained model to fine-tune")
    parser.add_argument("--chat-template", type=str, default="chatml",
                        choices=["chatml", "zephyr", "tokenizer", "none"],
                        help="Chat template to use for formatting the data")
    parser.add_argument("--format-type", type=str, default="jsonl",
                        choices=["jsonl", "csv"],
                        help="File format for saving the formatted data")
    parser.add_argument("--push-to-hub", action="store_true",
                        help="Push the trained model to Hugging Face Hub")
    parser.add_argument("--username", type=str, default=None,
                        help="Hugging Face username (required if push-to-hub is True)")
    parser.add_argument("--token", type=str, default=None,
                        help="Hugging Face token (required if push-to-hub is True)")
    
    args = parser.parse_args()
    
    # Ensure the required environment variables are set if pushing to Hub
    if args.push_to_hub:
        if not args.username:
            args.username = os.environ.get("HF_USERNAME")
        if not args.token:
            args.token = os.environ.get("HF_TOKEN")
        
        if not args.username or not args.token:
            logger.error("HF_USERNAME and HF_TOKEN must be provided when push-to-hub is enabled")
            return
    
    # Set up directories
    data_dir = args.data_dir
    formatted_data_dir = os.path.join(args.output_dir, "formatted_data")
    model_output_dir = os.path.join(args.output_dir, "model")
    
    # Load synthetic data
    inputs, outputs = load_synthetic_data(data_dir)
    
    if not inputs or not outputs:
        logger.error("Failed to load synthetic data. Exiting.")
        return
    
    # Format the data for training
    logger.info("Formatting data for training...")
    formatted_data = format_data_for_training(inputs, outputs, args.chat_template)
    
    # Save the formatted data
    data_path = save_formatted_data(formatted_data, formatted_data_dir, args.format_type)
    
    # Train the model
    logger.info(f"Training model {args.model} on formatted data...")
    train_model(
        data_dir=formatted_data_dir,
        model_name=args.model,
        output_dir=model_output_dir,
        chat_template=args.chat_template,
        push_to_hub=args.push_to_hub,
        username=args.username,
        token=args.token
    )
    
    logger.info("Done!")

if __name__ == "__main__":
    main()
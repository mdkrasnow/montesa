# synthetic_data_generator.py

"""
Script for concurrently generating synthetic classroom observation inputs and their corresponding evaluations,
then appending both to synthetic_inputs.py and synthetic_outputs.json with thread-safe I/O.
"""

from flask import Flask
import os
import json
import threading
from concurrent.futures import ThreadPoolExecutor, as_completed
import argparse
from .evaluation import generate_input_data, create_evaluation

# Flask app used for test_request_context
app = Flask(__name__)

# Lock to ensure thread-safe file operations
file_lock = threading.Lock()

# Base directory for synthetic data files
def get_base_dir():
    base_dir = os.path.join(os.path.dirname(__file__), "data")
    os.makedirs(base_dir, exist_ok=True)
    return base_dir


def append_single_synthetic():
    # 1. Generate a new synthetic input
    resp_tuple = generate_input_data()
    if isinstance(resp_tuple, tuple):
        response_obj, status = resp_tuple
    else:
        response_obj = resp_tuple
        status = response_obj.status_code
    data = response_obj.get_json()
    if status != 200 or not data.get('success'):
        raise RuntimeError(f"Input generation failed: {data}")

    new_input = {
        "description": data["description"],
        "content": data["content"]
    }

    # 2. Generate evaluation using the new input
    with app.test_request_context(json={"text": new_input["content"], "frameworkId": "Teach"}):
        eval_resp = create_evaluation()
    eval_data = eval_resp.get_json()
    if not eval_data.get("success"):
        raise RuntimeError(f"Evaluation generation failed: {eval_data}")

    evaluation = eval_data["evaluation"]

    # 3. Append to files with thread safety
    base_dir = get_base_dir()
    inputs_path = os.path.join(base_dir, "synthetic_inputs.py")
    outputs_path = os.path.join(base_dir, "synthetic_outputs.json")

    with file_lock:
        # Append to synthetic_inputs.py
        with open(inputs_path, "a") as f_inputs:
            f_inputs.write(f"{json.dumps(new_input)},\n")

        # Load existing outputs or init
        if os.path.exists(outputs_path):
            with open(outputs_path, "r") as f_out:
                outputs = json.load(f_out)
        else:
            outputs = {}
        next_index = str(len(outputs))
        outputs[next_index] = evaluation
        with open(outputs_path, "w") as f_out:
            json.dump(outputs, f_out, indent=2)

    return {"success": True, "index": next_index, "input": new_input, "evaluation": evaluation}


def main(count: int, workers: int = None):
    max_workers = workers or (os.cpu_count() or 4)
    results = []

    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        futures = [executor.submit(append_single_synthetic) for _ in range(count)]
        for future in as_completed(futures):
            try:
                res = future.result()
                results.append(res)
                print("Generated synthetic data index", res["index"] )
            except Exception as e:
                print("Error during generation:", e)

    return results


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Generate synthetic classroom observations concurrently"
    )
    parser.add_argument(
        "--count", type=int, default=1,
        help="Number of synthetic examples to generate concurrently"
    )
    parser.add_argument(
        "--workers", type=int, default=None,
        help="Number of worker threads (defaults to CPU count)"
    )
    args = parser.parse_args()
    results = main(count=args.count, workers=args.workers)
    print(json.dumps(results, indent=2))

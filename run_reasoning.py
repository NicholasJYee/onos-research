"""
Run reasoning experiment: Convert doctor-patient conversations to structured
JSON using the reasoning prompt, then generate a clinic note from that JSON.
"""

from pathlib import Path
from typing import Dict, Any
from datetime import datetime
import json
import re
import time
import pandas as pd
import ollama

from run_baseline_experiment import (
    load_prompt_template,
    load_transcript,
    generate_medical_note,
)


def extract_answer_json(text: str) -> str:
    """Extract JSON payload from markdown code block, <answer> tags, or raw JSON."""

    # Check for <answer> tags (case-insensitive, handle whitespace)
    answer_match = re.search(r"<answer>([\s\S]*?)</answer>", text, re.IGNORECASE)
    if answer_match:
        return answer_match.group(1).strip()
    
    # Check for ```json code blocks
    json_block_match = re.search(r"```json\s*(.*?)\s*```", text, re.DOTALL)
    if json_block_match:
        return json_block_match.group(1).strip()
    
    
    # If no tags, validate that the text itself is valid JSON
    text_stripped = text.strip()
    try:
        json.loads(text_stripped)
        return text_stripped
    except (json.JSONDecodeError, ValueError):
        # If not valid JSON, raise error and print the text
        print(f"ERROR: Text is not valid JSON:\n{text}")
        return None


def generate_reasoning_json(system_prompt: str, transcript: str, model: str) -> Dict[str, Any]:
    """Run reasoning prompt to produce structured JSON from the conversation."""
    base_prompt = f"{system_prompt}\n\nConversation:\n{transcript}"
    expected_hint = (
        "Return exactly one JSON object wrapped in <answer>...</answer> following the schema. "
        "Do not include any extra text."
    )

    attempts = 0
    total_start = time.time()
    last_raw = ""

    while attempts < 5:
        attempts += 1
        print(f"Attempt {attempts} of 5...")
        if attempts == 1:
            prompt = base_prompt
        else:
            prompt = (
                f"{system_prompt}\n\n"
                f"The latest output did not include a JSON inside <answer>...</answer> tags.\n"
                f"Latest output:\n---\n{last_raw}\n---\n"
                f"Expected: {expected_hint}\n\n"
                f"Conversation:\n{transcript}"
            )
            print(f"Latest output: {last_raw}")

        response = ollama.generate(model=model, prompt=prompt)
        last_raw = response["response"].strip()
        json_content = extract_answer_json(last_raw)

        if json_content:
            duration = time.time() - total_start
            return {
                "json": json_content,
                "raw": last_raw,
                "duration_seconds": duration,
                "attempts": attempts,
                "error": None,
            }

    duration = time.time() - total_start
    return {
        "json": "",
        "raw": last_raw,
        "duration_seconds": duration,
        "attempts": attempts,
        "error": "Failed to produce JSON inside <answer> after 5 attempts.",
    }


def run_reasoning_stage(
    excel_file: str = "data/Fake OSCEs.xlsx",
    reasoning_template_path: str = "methods/templates/reasoning_prompt.yaml",
    output_dir: Path = None,
):
    """
    Stage 1:
    - Use reasoning prompt to transform conversation transcripts into JSON.
    - Save reasoning JSON outputs and intermediate results (notes are saved later).
    
    Args:
        excel_file: Path to Excel file with transcripts
        reasoning_template_path: Path to reasoning prompt template
        output_dir: Optional output directory. If provided, will resume/continue existing work.
                   If None, creates a new timestamped directory.
    """
    models = [
        "gemma3:27b",
        "qwen3:32b",
        "llama3.1:8b",
        "mistral-small3.1:24b",
        "qwen2.5:3b",
        "llama3.2:3b",
        "deepseek-r1:32b",
        "deepseek-r1:7b",
        "gemma3:4b",
    ]

    base_path = Path(__file__).parent
    excel_path = base_path / excel_file
    reasoning_path = base_path / reasoning_template_path

    if output_dir is None:
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        output_dir = base_path / "outputs" / "reasoning" / timestamp
        output_dir.mkdir(parents=True, exist_ok=True)
        print(f"Output directory: {output_dir}")
    else:
        output_dir = Path(output_dir)
        if not output_dir.is_absolute():
            output_dir = base_path / output_dir
        print(f"Using existing output directory: {output_dir}")

    print("Loading reasoning prompt...")
    reasoning_config = load_prompt_template(str(reasoning_path))
    reasoning_system_prompt = reasoning_config.get("system_prompt")
    reasoning_name = reasoning_config.get("name")
    reasoning_version = reasoning_config.get("version")

    print(f"Reasoning prompt: {reasoning_name} v{reasoning_version}")

    print(f"Loading Excel file: {excel_path}")
    try:
        df = pd.read_excel(excel_path)
    except Exception as exc:
        print(f"Error loading Excel file: {exc}")
        return

    print(f"Found {len(df)} transcripts in Excel file")

    model_dirs: Dict[str, Dict[str, Path]] = {}
    for model in models:
        base_model_dir = output_dir / model.replace(":", "_")
        reasoning_dir = base_model_dir / "reasoning"
        reasoning_dir.mkdir(parents=True, exist_ok=True)
        model_dirs[model] = {"reasoning": reasoning_dir}

    results_file = output_dir / "results.json"
    
    # Load existing results if resuming, otherwise create new
    if results_file.exists():
        print(f"Loading existing results from {results_file}...")
        with open(results_file, "r", encoding="utf-8") as f:
            results = json.load(f)
    else:
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        results: Dict[str, Any] = {
            "experiment_info": {
                "reasoning_prompt": {"name": reasoning_name, "version": reasoning_version},
                "timestamp": timestamp,
                "models": models,
                "total_transcripts": len(df),
            },
            "results": [],
        }
        with open(results_file, "w", encoding="utf-8") as f:
            json.dump(results, f, indent=2, ensure_ascii=False)

    for idx, row in df.iterrows():
        transcript_file = row.get("transcription_file", "")
        if pd.isna(transcript_file) or not transcript_file:
            print(f"Row {idx + 1}: No transcription_file, skipping...")
            continue

        transcript_path = excel_path.parent / str(transcript_file)
        print(f"\nRow {idx + 1}: Processing {transcript_file}...")
        try:
            transcript = load_transcript(str(transcript_path))
        except Exception as exc:
            print(f"  Error loading transcript: {exc}")
            continue

        pathology = row.get("pathology", "")
        visit_type = row.get("visit_type", "")

        # Check if reasoning JSON already exists for this transcript
        row_result = None
        for existing_result in results["results"]:
            if existing_result.get("transcription_file") == str(transcript_file):
                row_result = existing_result
                break     
        if row_result is None: # If no row_result (i.e. no reasoning JSON) exists for this transcript, create a new one
            row_result = {
                "row_index": int(idx),
                "transcription_file": str(transcript_file),
                "transcript_path": str(transcript_path),
                "pathology": pathology,
                "visit_type": visit_type,
                "models": {},
            }
            results["results"].append(row_result)

        for model in models:
            transcript_name = Path(transcript_file).stem
            reasoning_file = model_dirs[model]["reasoning"] / f"{transcript_name}.json"
            
            # Skip if reasoning file already exists
            if reasoning_file.exists():
                print(f"  Reasoning JSON already exists for {model}, skipping...")
                continue
            
            print(f"  Generating reasoning JSON with {model}...")
            reasoning_result = generate_reasoning_json(reasoning_system_prompt, transcript, model)
            
            if reasoning_result["error"]:
                print(f"    Error (reasoning): {reasoning_result['error']}")
                with open(results_file, "w", encoding="utf-8") as f:
                    json.dump(results, f, indent=2, ensure_ascii=False)
                continue

            # Save the generated JSON
            with open(reasoning_file, "w", encoding="utf-8") as f:
                f.write(reasoning_result["json"])
            print(
                f"    Saved reasoning JSON ({len(reasoning_result['json'])} chars) "
                f"in {reasoning_result['duration_seconds']:.2f}s to {reasoning_file}"
            )
            
            model_entry = row_result["models"].get(model, {}).copy()
            # Preserve any existing data (e.g., notes) while adding reasoning
            if "reasoning" not in model_entry:
                model_entry["reasoning"] = reasoning_result
            row_result["models"][model] = model_entry
            
            # Update results file after each model completes
            with open(results_file, "w", encoding="utf-8") as f:
                json.dump(results, f, indent=2, ensure_ascii=False)

    print("\n=== Reasoning Stage Complete ===")
    print(f"Reasoning outputs saved to: {output_dir}/<model>/reasoning/")

    return output_dir

def generate_notes_from_reasoning(
    output_dir: Path,
    note_template_path: str = "methods/templates/baseline_prompt.yaml",
) -> Dict[str, Any]:
    """
    Stage 2:
    - Load reasoning results.json from Stage 1.
    - Generate clinic notes and persist updates to the same results.json.
    """
    results_path = output_dir / "results.json"
    with open(results_path, "r", encoding="utf-8") as f:
        results = json.load(f)

    models = results["experiment_info"]["models"]

    note_path = Path(note_template_path)
    if not note_path.is_absolute():
        note_path = Path(__file__).parent / note_path
    note_config = load_prompt_template(str(note_path))
    note_system_prompt = note_config["system_prompt"].format(input_type="reasoning JSON")
    note_name = note_config.get("name")
    note_version = note_config.get("version")

    # Persist baseline prompt metadata for traceability
    results["experiment_info"]["note_prompt"] = {
        "name": note_name,
        "version": note_version,
    }
    with open(results_path, "w", encoding="utf-8") as f:
        json.dump(results, f, indent=2, ensure_ascii=False)

    for row in results["results"]:
        transcript_file = row.get("transcription_file", "")
        transcript_name = Path(transcript_file).stem

        for model in models:
            model_data = row["models"].get(model)
            if not model_data or model_data["reasoning"]["error"]:
                raise ValueError(f"Error generating reasoning JSON for {transcript_file} with {model}")

            # Skip note generation if note file already exists
            notes_dir = output_dir / model.replace(":", "_") / "notes"
            note_file = notes_dir / f"{transcript_name}.txt"
            if note_file.exists():
                print(f"  Note file already exists for {model}, skipping...")
                continue
            
            # Load JSON from saved file instead of results.json
            reasoning_dir = output_dir / model.replace(":", "_") / "reasoning"
            reasoning_file = reasoning_dir / f"{transcript_name}.json"
            try:
                with open(reasoning_file, "r", encoding="utf-8") as f:
                    reasoning_json = f.read()
            except FileNotFoundError:
                print(f"  Error: Reasoning JSON file not found: {reasoning_file}")
                continue

            notes_dir.mkdir(parents=True, exist_ok=True)

            print(f"Generating clinic note from reasoning JSON with {model} for {transcript_file}...")
            note_result = generate_medical_note(
                note_system_prompt, reasoning_json, model, input_type="reasoning"
            )
            model_data["note"] = note_result

            if note_result["error"]:
                print(f"  Error (note): {note_result['error']}")
            else:
                note_file = notes_dir / f"{transcript_name}.txt"
                with open(note_file, "w", encoding="utf-8") as f:
                    f.write(note_result["note"])
                print(
                    f"  Saved clinic note ({len(note_result['note'])} chars) "
                    f"in {note_result['duration_seconds']:.2f}s to {note_file}"
                )
            
            # Update results file after each model note generation completes
            with open(results_path, "w", encoding="utf-8") as f:
                json.dump(results, f, indent=2, ensure_ascii=False)

    print("\n=== Note Generation Complete ===")
    print(f"Updated results saved to: {results_path}")
    return results


if __name__ == "__main__":
    output_dir = run_reasoning_stage(
        excel_file="data/Fake OSCEs.xlsx",
        reasoning_template_path="methods/templates/reasoning_prompt.yaml",
        output_dir="outputs/reasoning/20251209_112109"
    )
    
    # output_dir = Path("outputs/reasoning/20251209_110602")
    generate_notes_from_reasoning(
        output_dir=output_dir,
        note_template_path="methods/templates/baseline_prompt.yaml",
    )


"""
Run chunked reasoning experiment: Chunk transcripts, generate/update reasoning JSON
with each chunk, then generate clinic notes from the final reasoning JSON.

Combines chunking approach from run_chunking.py with reasoning approach from run_reasoning.py.
"""

from pathlib import Path
import pandas as pd
import json
import ollama
from typing import List, Dict, Any
from datetime import datetime
import time

from methods.chunking.chunk_transcript import chunk_by_words
from run_baseline_experiment import (
    load_prompt_template,
    load_transcript,
    generate_medical_note,
)
from run_reasoning import extract_answer_json
from run_chunking import load_chunking_config


def generate_reasoning_json_from_chunk(
    system_prompt: str,
    chunk: str,
    model: str,
    previous_json: str = None,
    update_reasoning_instructions: str = None
) -> Dict[str, Any]:
    """
    Generate or update reasoning JSON from a chunk.
    
    Args:
        system_prompt: The reasoning system prompt
        chunk: Current conversation chunk
        model: Ollama model to use
        previous_json: Previous reasoning JSON (if updating)
        update_reasoning_instructions: Instructions for updating reasoning JSON with subsequent chunks
    
    Returns:
        Dictionary with JSON content, raw response, timing, and error info
    """
    if previous_json is None:
        # First chunk: generate initial JSON
        base_prompt = f"{system_prompt}\n\nConversation:\n{chunk}"
    else:
        # Subsequent chunks: update existing JSON
        update_instruction = (
            f"{update_reasoning_instructions}\n\n"
            f"Previous Reasoning JSON:\n{previous_json}\n\n"
            f"New Conversation Chunk:\n{chunk}"
        )
        base_prompt = f"{system_prompt}\n\n{update_instruction}"
    
    expected_hint = (
        "Return exactly one JSON object wrapped in <answer>...</answer> following the schema. "
        "Do not include any extra text."
    )

    attempts = 0
    total_start = time.time()
    last_raw = ""

    while attempts < 5:
        attempts += 1
        if attempts == 1:
            prompt = base_prompt
        else:
            prompt = (
                f"{system_prompt}\n\n"
                f"The latest output did not include a JSON inside <answer>...</answer> tags.\n"
                f"Latest output:\n---\n{last_raw}\n---\n"
                f"Expected: {expected_hint}\n\n"
                f"{update_reasoning_instructions}\n\n"
                f"Previous Reasoning JSON:\n{previous_json}\n\n"
                f"New Conversation Chunk:\n{chunk}"
            )

        try:
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
        except Exception as e:
            last_raw = f"Error during generation: {str(e)}"

    duration = time.time() - total_start
    return {
        "json": previous_json if previous_json else "",
        "raw": last_raw,
        "duration_seconds": duration,
        "attempts": attempts,
        "error": "Failed to produce JSON inside <answer> after 5 attempts.",
    }


def generate_note_from_json(
    note_system_prompt: str,
    reasoning_json: str,
    model: str,
    previous_note: str = None,
    update_note_instructions: str = None
) -> Dict[str, Any]:
    """
    Generate or update clinic note from reasoning JSON.
    
    Args:
        note_system_prompt: The note generation system prompt
        reasoning_json: Current reasoning JSON
        model: Ollama model to use
        previous_note: Previous clinic note (if updating)
        update_note_instructions: Instructions for updating the note with subsequent chunks
    
    Returns:
        Dictionary with note content, timing, and error info
    """
    if previous_note is None:
        # First note: generate from JSON
        full_prompt = f"{note_system_prompt}\n\nReasoning (in JSON format):\n{reasoning_json}"
    else:
        # Subsequent notes: update existing note
        full_prompt = (
            f"{note_system_prompt}\n\n{update_note_instructions}\n\n"
            f"Previously Generated Clinic Note:\n{previous_note}\n\n"
            f"Updated Reasoning (in JSON format):\n{reasoning_json}"
        )
    
    start_time = time.time()
    try:
        response = ollama.generate(
            model=model,
            prompt=full_prompt,
        )
        duration = time.time() - start_time
        return {
            'note': response['response'].strip(),
            'duration_seconds': duration,
            'error': None
        }
    except Exception as e:
        duration = time.time() - start_time
        return {
            'note': previous_note if previous_note else "",
            'duration_seconds': duration,
            'error': str(e)
        }


def process_chunks_with_reasoning_and_notes(
    reasoning_system_prompt: str,
    note_system_prompt: str,
    chunks: List[str],
    model: str,
    update_reasoning_instructions: str,
    update_note_instructions: str
) -> Dict[str, Any]:
    """
    Process chunks incrementally: for each chunk, update JSON then update note.
    
    Args:
        reasoning_system_prompt: The reasoning system prompt
        note_system_prompt: The note generation system prompt
        chunks: List of transcript chunks
        model: Ollama model to use
        update_reasoning_instructions: Instructions for updating reasoning JSON with subsequent chunks
        update_note_instructions: Instructions for updating the note with subsequent chunks
    
    Returns:
        Dictionary with final JSON, final note, timing information, and chunk details
    """
    chunk_timings = []
    total_start_time = time.time()
    current_json = None
    current_note = None
    
    for chunk_idx, chunk in enumerate(chunks):
        print(f"    Processing chunk {chunk_idx + 1}/{len(chunks)}...")
        
        # Step 1: Generate/update reasoning JSON from chunk
        json_start_time = time.time()
        reasoning_result = generate_reasoning_json_from_chunk(
            reasoning_system_prompt,
            chunk,
            model,
            current_json,
            update_reasoning_instructions
        )
        json_duration = time.time() - json_start_time
        
        if reasoning_result['error']:
            print(f"      Error generating reasoning JSON: {reasoning_result['error']}")
            # Keep previous JSON if available, otherwise continue with empty
            if current_json is None:
                current_json = ""
        else:
            current_json = reasoning_result['json']
            print(f"      Generated/updated reasoning JSON ({len(current_json)} chars) in {json_duration:.2f}s")
        
        # Step 2: Generate/update clinic note from JSON
        note_start_time = time.time()
        note_result = generate_note_from_json(
            note_system_prompt,
            current_json,
            model,
            current_note,
            update_note_instructions
        )
        note_duration = time.time() - note_start_time
        
        if note_result['error']:
            print(f"      Error generating note: {note_result['error']}")
            # Keep previous note if available
            if current_note is None:
                current_note = ""
        else:
            current_note = note_result['note']
            print(f"      Generated/updated clinic note ({len(current_note)} chars) in {note_duration:.2f}s")
        
        chunk_timings.append({
            'chunk_index': chunk_idx,
            'reasoning_duration_seconds': json_duration,
            'note_duration_seconds': note_duration,
            'total_chunk_duration_seconds': json_duration + note_duration,
            'reasoning_attempts': reasoning_result.get('attempts'),
            'reasoning_error': reasoning_result.get('error'),
            'note_error': note_result.get('error')
        })
    
    total_duration = time.time() - total_start_time
    
    return {
        'json': current_json if current_json else "",
        'note': current_note if current_note else "",
        'raw': reasoning_result.get('raw', ''),
        'duration_seconds': total_duration,
        'num_chunks': len(chunks),
        'chunk_timings': chunk_timings,
        'error': None if all(t['reasoning_error'] is None and t['note_error'] is None for t in chunk_timings) else "Some chunks had errors"
    }


def run_chunking_reasoning_stage(
    excel_file: str = "data/Fake OSCEs.xlsx",
    reasoning_template_path: str = "methods/templates/reasoning_prompt.yaml",
    note_template_path: str = "methods/templates/baseline_prompt.yaml",
    chunking_config_path: str = "methods/chunking/chunking_config.yaml",
    output_dir: Path = None,
):
    """
    - Chunk transcripts and process each chunk: update reasoning JSON, then update clinic note.
    - Save final reasoning JSON and final clinic note after all chunks are processed.
    
    Args:
        excel_file: Path to Excel file with transcripts
        reasoning_template_path: Path to reasoning prompt template
        note_template_path: Path to baseline prompt template for note generation
        chunking_config_path: Path to chunking configuration file
        output_dir: Optional output directory. If provided, will resume/continue existing work.
                   If None, creates a new timestamped directory.
    """
    models = [
        'gemma3:27b',
        'qwen3:32b',
        'llama3.1:8b',
        'mistral-small3.1:24b',
        'qwen2.5:3b',
        'llama3.2:3b',
        'deepseek-r1:32b',
        'deepseek-r1:7b',
        'gemma3:4b'
    ]
    
    # Setup paths
    base_path = Path(__file__).parent
    excel_path = base_path / excel_file
    reasoning_path = base_path / reasoning_template_path
    note_path = base_path / note_template_path
    chunking_config_file = base_path / chunking_config_path
    
    if output_dir is None:
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        output_dir = base_path / "outputs" / "chunking_reasoning" / timestamp
        output_dir.mkdir(parents=True, exist_ok=True)
        print(f"Output directory: {output_dir}")
    else:
        output_dir = Path(output_dir)
        if not output_dir.is_absolute():
            output_dir = base_path / output_dir
        print(f"Using existing output directory: {output_dir}")
    
    # Load reasoning prompt template
    print("Loading reasoning prompt template...")
    reasoning_config = load_prompt_template(str(reasoning_path))
    reasoning_system_prompt = reasoning_config.get('system_prompt', '')
    reasoning_name = reasoning_config.get('name', '')
    reasoning_version = reasoning_config.get('version', '')
    
    print(f"Reasoning prompt: {reasoning_name} v{reasoning_version}")
    
    # Load note prompt template
    print("Loading note prompt template...")
    note_config = load_prompt_template(str(note_path))
    note_system_prompt = note_config['system_prompt'].format(input_type="reasoning JSON")
    note_name = note_config.get('name', '')
    note_version = note_config.get('version', '')
    
    print(f"Note prompt: {note_name} v{note_version}")
    
    # Load chunking config
    print("Loading chunking configuration...")
    chunking_config = load_chunking_config(str(chunking_config_file))
    chunk_config_name = chunking_config.get('name', '')
    chunk_config_version = chunking_config.get('version', '')
    chunk_size = chunking_config.get('chunk_size')
    overlap = chunking_config.get('overlap')
    update_reasoning_instructions = chunking_config.get('update_reasoning_instructions', '')
    update_note_instructions = chunking_config.get('update_note_instructions', '').format(chunk_type="updated JSON")
    
    print(f"Chunking config: {chunk_config_name} v{chunk_config_version}")
    print(f"  Chunk size: {chunk_size} words")
    print(f"  Overlap: {overlap} words")
    
    # Load Excel file
    print(f"Loading Excel file: {excel_path}")
    try:
        df = pd.read_excel(excel_path)
    except Exception as e:
        print(f"Error loading Excel file: {e}")
        return
    
    print(f"Found {len(df)} transcripts in Excel file")
    
    # Create model directories upfront
    model_dirs = {}
    for model in models:
        model_dir = output_dir / model.replace(':', '_')
        reasoning_dir = model_dir / "reasoning"
        notes_dir = model_dir / "notes"
        reasoning_dir.mkdir(parents=True, exist_ok=True)
        notes_dir.mkdir(parents=True, exist_ok=True)
        model_dirs[model] = {
            'reasoning': reasoning_dir,
            'notes': notes_dir
        }
    
    # Initialize or load results structure
    results_file = output_dir / "results.json"
    
    if results_file.exists():
        print(f"Loading existing results from {results_file}...")
        with open(results_file, 'r', encoding='utf-8') as f:
            results = json.load(f)
    else:
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        results = {
            'experiment_info': {
                'reasoning_prompt': {
                    'name': reasoning_name,
                    'version': reasoning_version
                },
                'note_prompt': {
                    'name': note_name,
                    'version': note_version
                },
                'chunking_prompt': {
                    'name': chunk_config_name,
                    'version': chunk_config_version
                },
                'timestamp': timestamp,
                'models': models,
                'total_transcripts': len(df)
            },
            'results': []
        }
        with open(results_file, 'w', encoding='utf-8') as f:
            json.dump(results, f, indent=2, ensure_ascii=False)
    
    for idx, row in df.iterrows():
        transcript_file = row.get('transcription_file', '')
        if pd.isna(transcript_file) or not transcript_file:
            print(f"Row {idx+1}: No transcription_file, skipping...")
            continue
        
        # Load transcript
        transcript_path = excel_path.parent / str(transcript_file)
        print(f"\nRow {idx+1}: Processing {transcript_file}...")
        try:
            transcript = load_transcript(str(transcript_path))
        except Exception as e:
            print(f"  Error loading transcript: {e}")
            continue
        
        # Chunk the transcript
        chunks = chunk_by_words(transcript, chunk_size, overlap)
        print(f"  Chunked transcript into {len(chunks)} chunks")
        
        # Load meta data
        pathology = row.get('pathology', '')
        visit_type = row.get('visit_type', '')
        
        # Check if reasoning JSON already exists for this transcript
        row_result = None
        for existing_result in results['results']:
            if existing_result.get('transcription_file') == str(transcript_file):
                row_result = existing_result
                break
        
        if row_result is None:  # If no row_result exists for this transcript, create a new one
            row_result = {
                'row_index': int(idx),
                'transcription_file': str(transcript_file),
                'transcript_path': str(transcript_path),
                'pathology': pathology,
                'visit_type': visit_type,
                'num_chunks': len(chunks),
                'models': {}
            }
            results['results'].append(row_result)
        
        for model in models:
            print(f"  Processing with {model}...")
            transcript_name = Path(transcript_file).stem
            
            # Check if both reasoning and note files already exist
            reasoning_file = model_dirs[model]['reasoning'] / f"{transcript_name}.json"
            note_file = model_dirs[model]['notes'] / f"{transcript_name}.txt"
            if reasoning_file.exists() and note_file.exists():
                print(f"  Reasoning JSON and note already exist for {model}, skipping...")
                continue
            
            # Process chunks: for each chunk, update JSON then update note
            print(f"    Processing chunks with reasoning and note generation...")
            result = process_chunks_with_reasoning_and_notes(
                reasoning_system_prompt,
                note_system_prompt,
                chunks,
                model,
                update_reasoning_instructions,
                update_note_instructions
            )
            
            if result['error']:
                print(f"    Error processing chunks: {result['error']}")
                model_entry = row_result['models'].get(model, {}).copy()
                # Preserve any existing data while adding new results
                if 'reasoning' not in model_entry:
                    model_entry['reasoning'] = {
                        'json': result['json'],
                        'raw': result['raw'],
                        'duration_seconds': result['duration_seconds'],
                        'error': result['error']
                    }
                if 'note' not in model_entry:
                    model_entry['note'] = {
                        'note': result['note'],
                        'duration_seconds': result['duration_seconds'],
                        'error': result['error']
                    }
                row_result['models'][model] = model_entry
                with open(results_file, 'w', encoding='utf-8') as f:
                    json.dump(results, f, indent=2, ensure_ascii=False)
                continue
            
            # Save final reasoning JSON
            with open(reasoning_file, 'w', encoding='utf-8') as f:
                f.write(result['json'])
            print(f"    Saved final reasoning JSON ({len(result['json'])} chars) to {reasoning_file}")
            
            # Save final clinic note
            with open(note_file, 'w', encoding='utf-8') as f:
                f.write(result['note'])
            print(f"    Saved final clinic note ({len(result['note'])} chars) to {note_file}")
            
            # Log chunk timings
            for chunk_timing in result['chunk_timings']:
                reasoning_err = chunk_timing.get('reasoning_error')
                note_err = chunk_timing.get('note_error')
                if reasoning_err or note_err:
                    print(f"      Chunk {chunk_timing['chunk_index']}: Reasoning error - {reasoning_err}, Note error - {note_err}")
                else:
                    print(
                        f"      Chunk {chunk_timing['chunk_index']}: "
                        f"JSON {chunk_timing['reasoning_duration_seconds']:.2f}s, "
                        f"Note {chunk_timing['note_duration_seconds']:.2f}s, "
                        f"Total {chunk_timing['total_chunk_duration_seconds']:.2f}s"
                    )
            
            # Store results
            model_entry = row_result['models'].get(model, {}).copy()
            # Preserve any existing data while adding new results
            if 'reasoning' not in model_entry:
                model_entry['reasoning'] = {
                    'json': result['json'],
                    'raw': result['raw'],
                    'duration_seconds': result['duration_seconds'],
                    'num_chunks': result['num_chunks'],
                    'chunk_timings': result['chunk_timings'],
                    'error': result['error']
                }
            if 'note' not in model_entry:
                model_entry['note'] = {
                    'note': result['note'],
                    'duration_seconds': result['duration_seconds'],
                    'num_chunks': result['num_chunks'],
                    'chunk_timings': result['chunk_timings'],
                    'error': result['error']
                }
            row_result['models'][model] = model_entry
            
            # Update results file after each model completes
            with open(results_file, 'w', encoding='utf-8') as f:
                json.dump(results, f, indent=2, ensure_ascii=False)
    
    print("\n=== Chunking Reasoning Stage Complete ===")
    print(f"Reasoning outputs saved to: {output_dir}/<model>/reasoning/")
    print(f"Clinic notes saved to: {output_dir}/<model>/notes/")
    
    return output_dir


if __name__ == "__main__":
    output_dir = run_chunking_reasoning_stage(
        excel_file="data/Fake OSCEs.xlsx",
        reasoning_template_path="methods/templates/reasoning_prompt.yaml",
        note_template_path="methods/templates/baseline_prompt.yaml",
        chunking_config_path="methods/chunking/chunking_config.yaml"
    )
    

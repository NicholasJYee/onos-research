"""
Run chunking experiment: Generate medical notes from chunked transcripts using multiple LLM models.

Loads transcripts from Fake OSCEs.xlsx, chunks them following the instructions in chunking_config.yaml,
uses baseline prompt template, runs inference with multiple Ollama models, and saves results.
"""

from pathlib import Path
import pandas as pd
import yaml
import json
import ollama
from typing import List, Dict, Any
from datetime import datetime
import time

from methods.chunking.chunk_transcript import chunk_by_words


def load_prompt_template(filepath: str) -> Dict[str, Any]:
    """Load prompt template from YAML file."""
    with open(filepath, 'r') as f:
        config = yaml.safe_load(f)
    return config


def load_chunking_config(filepath: str) -> Dict[str, Any]:
    """Load chunking configuration from YAML file."""
    with open(filepath, 'r') as f:
        config = yaml.safe_load(f)
    return config


def load_transcript(filepath: str) -> str:
    """Load transcript from file."""
    with open(filepath, 'r', encoding='utf-8') as f:
        return f.read().strip()


def generate_medical_note_from_chunks(
    prompt_template: str,
    chunks: List[str],
    model: str,
    update_instruction: str
) -> Dict[str, Any]:
    """
    Generate medical note from chunked transcript using Ollama LLM.
    Processes chunks sequentially, updating the clinic note with each chunk.
    
    Args:
        prompt_template: The prompt template
        chunks: List of transcript chunks
        model: Ollama model to use
        update_instruction: Instructions for updating the note with subsequent chunks
    
    Returns:
        Dictionary with generated note, timing information, and chunk details
    """
    chunk_timings = []
    total_start_time = time.time()
    current_note = ""
    
    for chunk_idx, chunk in enumerate(chunks):
        chunk_start_time = time.time()
        
        if chunk_idx == 0:
            # First chunk: use template + chunk text
            full_prompt = f"{prompt_template}\n\nConversation:\n{chunk}"
        else:
            # Subsequent chunks: use template + previous note + current chunk
            full_prompt = (
                f"{prompt_template}\n\n{update_instruction}\n\n"
                f"Previously Generated Clinic Note:\n{current_note}\n\n"
                f"New Conversation Chunk:\n{chunk}"
            )
        
        try:
            response = ollama.generate(
                model=model,
                prompt=full_prompt,
            )
            chunk_duration = time.time() - chunk_start_time
            current_note = response['response'].strip()
            chunk_timings.append({
                'chunk_index': chunk_idx,
                'duration_seconds': chunk_duration,
                'error': None
            })
        except Exception as e:
            chunk_duration = time.time() - chunk_start_time
            chunk_timings.append({
                'chunk_index': chunk_idx,
                'duration_seconds': chunk_duration,
                'error': str(e)
            })
            # If there's an error, keep the previous note (or empty if first chunk)
            if chunk_idx == 0:
                current_note = ""
    
    total_duration = time.time() - total_start_time
    
    return {
        'note': current_note,
        'duration_seconds': total_duration,
        'num_chunks': len(chunks),
        'chunk_timings': chunk_timings,
        'error': None if all(t['error'] is None for t in chunk_timings) else "Some chunks had errors"
    }


def run_chunking_experiment(
    excel_file: str = "data/Fake OSCEs.xlsx",
    prompt_template_path: str = "methods/templates/baseline_prompt.yaml",
    chunking_config_path: str = "methods/chunking/chunking_config.yaml",
):
    """
    Run chunking experiment with multiple LLM models.
    
    Args:
        excel_file: Path to Fake OSCEs.xlsx file
        prompt_template_path: Path to baseline prompt template
        chunking_config_path: Path to chunking configuration file
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
    prompt_path = base_path / prompt_template_path
    chunking_config_file = base_path / chunking_config_path
    
    # Create output directory with timestamp
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    output_dir = base_path / "outputs" / "chunking" / timestamp
    output_dir.mkdir(parents=True, exist_ok=True)
    
    print(f"Output directory: {output_dir}")
    
    # Load prompt template
    print("Loading prompt template...")
    prompt_config = load_prompt_template(str(prompt_path))
    prompt_template = prompt_config['prompt']
    prompt_name = prompt_config['name']
    prompt_version = prompt_config['version']
    
    print(f"Prompt: {prompt_name} v{prompt_version}")
    
    # Load chunking config
    print("Loading chunking configuration...")
    chunking_config = load_chunking_config(str(chunking_config_file))
    chunk_config_name = chunking_config['name']
    chunk_config_version = chunking_config['version']
    chunk_size = chunking_config['chunk_size']
    overlap = chunking_config['overlap']
    update_instruction = chunking_config.get('update_instruction', '')
    
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
        model_dir.mkdir(exist_ok=True)
        model_dirs[model] = model_dir
    
    # Initialize results structure
    results_file = output_dir / "results.json"
    results = {
        'experiment_info': {
            'name': prompt_name,
            'version': prompt_version,
            'chunking_config_name': chunk_config_name,
            'chunking_config_version': chunk_config_version,
            'timestamp': timestamp,
            'models': models,
            'total_transcripts': len(df)
        },
        'results': []
    }
    
    # Save initial results file
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
        
        # Generate notes for each model
        row_result = {
            'row_index': int(idx),
            'transcription_file': str(transcript_file),
            'transcript_path': str(transcript_path),
            'pathology': pathology,
            'visit_type': visit_type,
            'num_chunks': len(chunks),
            'models': {}
        }
        
        # Add row_result to results before processing models
        results['results'].append(row_result)
        
        for model in models:
            print(f"  Generating with {model}...")
            model_start_time = time.time()
            result = generate_medical_note_from_chunks(prompt_template, chunks, model, update_instruction)
            model_total_time = time.time() - model_start_time
            
            # Add total model time to result
            result['total_model_time_seconds'] = model_total_time
            
            row_result['models'][model] = result
            
            if result['error']:
                print(f"    Error: {result['error']}")
            else:
                print(f"    Generated note ({len(result['note'])} chars) from {result['num_chunks']} chunks")
                print(f"    Total time: {result['duration_seconds']:.2f}s (model time: {model_total_time:.2f}s)")
                
                # Log individual chunk timings
                for chunk_timing in result['chunk_timings']:
                    if chunk_timing['error']:
                        print(f"      Chunk {chunk_timing['chunk_index']}: Error - {chunk_timing['error']}")
                    else:
                        print(f"      Chunk {chunk_timing['chunk_index']}: {chunk_timing['duration_seconds']:.2f}s")
                
                # Save individual note file immediately
                transcript_name = Path(transcript_file).stem
                note_file = model_dirs[model] / f"{transcript_name}.txt"
                with open(note_file, 'w', encoding='utf-8') as f:
                    f.write(result['note'])
                print(f"    Saved note to {note_file}")
            
            # Update and save results.json after each note
            with open(results_file, 'w', encoding='utf-8') as f:
                json.dump(results, f, indent=2, ensure_ascii=False)
    
    print(f"\n=== Experiment Complete ===")
    print(f"Results saved to: {results_file}")
    print(f"Processed {len(results['results'])} transcripts")
    print(f"Models used: {', '.join(models)}")
    print(f"\nIndividual notes saved to: {output_dir}/<model>/")
    
    return results


if __name__ == "__main__":
    run_chunking_experiment(
        excel_file="data/Fake OSCEs.xlsx",
        prompt_template_path="methods/templates/baseline_prompt.yaml",
        chunking_config_path="methods/chunking/chunking_config.yaml"
    )


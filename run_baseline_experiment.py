"""
Run baseline experiment: Generate medical notes from transcripts using multiple LLM models.

Loads transcripts from Fake OSCEs.xlsx, uses baseline prompt template,
runs inference with multiple Ollama models, and saves results.
"""

from pathlib import Path
import pandas as pd
import yaml
import json
import ollama
from typing import List, Dict, Any
from datetime import datetime
import time


def load_prompt_template(filepath: str) -> Dict[str, Any]:
    """Load prompt template from YAML file."""
    with open(filepath, 'r') as f:
        config = yaml.safe_load(f)
    return config


def load_transcript(filepath: str) -> str:
    """Load transcript from file."""
    with open(filepath, 'r', encoding='utf-8') as f:
        return f.read().strip()


def generate_medical_note(prompt_template: str, transcript: str, model: str) -> Dict[str, Any]:
    """
    Generate medical note using Ollama LLM.
    
    Args:
        prompt_template: The prompt template
        transcript: The doctor-patient conversation transcript
        model: Ollama model to use
    
    Returns:
        Dictionary with generated note and timing information
    """
    full_prompt = f"{prompt_template}\n\nConversation:\n{transcript}"
    
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
            'note': "",
            'duration_seconds': duration,
            'error': str(e)
        }


def run_baseline_experiment(
    excel_file: str = "data/Fake OSCEs.xlsx",
    prompt_template_path: str = "methods/templates/baseline_prompt.yaml",
):
    """
    Run baseline experiment with multiple LLM models.
    
    Args:
        excel_file: Path to Fake OSCEs.xlsx file
        prompt_template_path: Path to baseline prompt template
    """
    models = [
        'gemma3:27b',
        'qwen3:32b',
        # 'llama3.1:8b',
        # 'mistral-small3.1:24b',
        # 'qwen2.5:3b',
        # 'llama3.2:3b',
        # 'deepseek-r1:32b',
        # 'deepseek-r1:7b',
        # 'gemma3:4b'
    ]
    
    # Setup paths
    base_path = Path(__file__).parent
    excel_path = base_path / excel_file
    prompt_path = base_path / prompt_template_path
    
    # Create output directory with timestamp
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    output_dir = base_path / "outputs" / "baseline" / timestamp
    output_dir.mkdir(parents=True, exist_ok=True)
    
    print(f"Output directory: {output_dir}")
    
    # Load prompt template
    print("Loading prompt template...")
    prompt_config = load_prompt_template(str(prompt_path))
    prompt_template = prompt_config['prompt']
    prompt_name = prompt_config['name']
    prompt_version = prompt_config['version']
    
    print(f"Prompt: {prompt_name} v{prompt_version}")
    
    # Load Excel file
    print(f"Loading Excel file: {excel_path}")
    try:
        df = pd.read_excel(excel_path)
    except Exception as e:
        print(f"Error loading Excel file: {e}")
        return
    
    print(f"Found {len(df)} transcripts in Excel file")
    
    # Process each row
    results = {
        'experiment_info': {
            'name': prompt_name,
            'version': prompt_version,
            'timestamp': timestamp,
            'models': models,
            'total_transcripts': len(df)
        },
        'results': []
    }
    
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
            'models': {}
        }
        
        for model in models:
            print(f"  Generating with {model}...")
            result = generate_medical_note(prompt_template, transcript, model)
            row_result['models'][model] = result
            
            if result['error']:
                print(f"    Error: {result['error']}")
            else:
                print(f"    Generated note ({len(result['note'])} chars) in {result['duration_seconds']:.2f}s")
        
        results['results'].append(row_result)
    
    # Save results
    results_file = output_dir / "results.json"
    with open(results_file, 'w', encoding='utf-8') as f:
        json.dump(results, f, indent=2, ensure_ascii=False)
    
    print(f"\n=== Experiment Complete ===")
    print(f"Results saved to: {results_file}")
    print(f"Processed {len(results['results'])} transcripts")
    print(f"Models used: {', '.join(models)}")
    
    # Save individual notes per model
    for model in models:
        model_dir = output_dir / model.replace(':', '_')
        model_dir.mkdir(exist_ok=True)
        
        for result in results['results']:
            if model in result['models']:
                note = result['models'][model]['note']
                if note:
                    # Save note to file
                    transcript_name = Path(result['transcription_file']).stem
                    note_file = model_dir / f"{transcript_name}.txt"
                    with open(note_file, 'w', encoding='utf-8') as f:
                        f.write(note)
    
    print(f"\nIndividual notes saved to: {output_dir}/<model>/")
    
    return results


if __name__ == "__main__":
    run_baseline_experiment(
        excel_file="data/Fake OSCEs.xlsx",
        prompt_template_path="methods/templates/baseline_prompt.yaml"
    )

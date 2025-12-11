"""
Run metrics evaluation on generated medical notes.

Loads notes from experiment output folder, compares them to transcripts,
and calculates BERTScore, ROUGE, and SummarizationMetric scores.
"""

from pathlib import Path
import pandas as pd
import json
import yaml
from dotenv import load_dotenv

# Load environment variables from .env.local (for DeepEval Ollama configuration)
load_dotenv(dotenv_path='.env.local')

from methods.metrics.rouge import calculate_rouge
from methods.metrics.bertscore import calculate_bertscore
from methods.metrics.summarization import calculate_summarization_metric


def load_transcript(filepath: str) -> str:
    """Load transcript from file."""
    with open(filepath, 'r', encoding='utf-8') as f:
        return f.read().strip()


def load_note(filepath: str) -> str:
    """Load generated note from file."""
    with open(filepath, 'r', encoding='utf-8') as f:
        return f.read().strip()


def run_metrics(
    output_folder: str,
    excel_file: str = "data/Fake OSCEs.xlsx"
):
    """
    Run metrics evaluation on generated notes.
    
    Args:
        output_folder: Path to experiment output folder (e.g., "outputs/baseline/20251203_195311")
        excel_file: Path to Fake OSCEs.xlsx file
    """
    base_path = Path(__file__).parent
    output_dir = base_path / output_folder
    excel_path = base_path / excel_file
    
    if not output_dir.exists():
        print(f"Error: Output folder not found: {output_dir}")
        return
    
    # Load results.json to get experiment_info
    results_file = output_dir / "results.json"
    if not results_file.exists():
        print(f"Error: results.json not found in {output_dir}")
        return
    
    with open(results_file, 'r', encoding='utf-8') as f:
        results_data = json.load(f)
    
    experiment_info = results_data['experiment_info']
    
    # Load Excel file
    print(f"Loading Excel file: {excel_path}")
    try:
        df = pd.read_excel(excel_path)
    except Exception as e:
        print(f"Error loading Excel file: {e}")
        return

    # Get list of models from experiment_info
    models = experiment_info['models']

    print(f"Found models: {models}")
    print(f"Found {len(df)} transcripts in Excel file")
    
    # Load or initialize metrics structure
    metrics_file = output_dir / "metrics.json"
    if metrics_file.exists():
        print(f"Loading existing metrics from: {metrics_file}")
        with open(metrics_file, 'r', encoding='utf-8') as f:
            metrics_data = json.load(f)
    else:
        metrics_data = {
            'experiment_info': experiment_info,
            'transcripts': {}
        }
    
    # Save initial metrics file
    with open(metrics_file, 'w', encoding='utf-8') as f:
        json.dump(metrics_data, f, indent=2, ensure_ascii=False)
    
    # Flag to track if we've extracted assessment questions info yet
    assessment_questions_extracted = False
    
    # Process each transcript
    for idx, row in df.iterrows():
        transcript_file = row.get('transcription_file', '')
        file_name = row.get('file_name', '')
        
        if pd.isna(transcript_file) or not transcript_file:
            print(f"Row {idx+1}: No transcription_file, skipping...")
            continue
        
        if pd.isna(file_name) or not file_name:
            print(f"Row {idx+1}: No file_name, skipping...")
            continue
        
        print(f"\nProcessing transcript {idx+1}: {file_name}...")
        
        # Load transcript
        transcript_path = excel_path.parent / str(transcript_file)
        
        if not transcript_path.exists():
            print(f"  Error: Transcript file not found: {transcript_file}")
            continue
        
        transcript = load_transcript(str(transcript_path))
        
        # Collect notes for all models
        notes_by_model = {}
        for model in models:
            model_dir_name = model.replace(':', '_')
            # If output folder is reasoning, notes are in a "notes" subdirectory
            if "reasoning" in str(output_dir):
                note_file = output_dir / model_dir_name / "notes" / f"{file_name}.txt"
            else:
                note_file = output_dir / model_dir_name / f"{file_name}.txt"
            
            if note_file.exists():
                note = load_note(str(note_file))
                notes_by_model[model] = note
            else:
                print(f"  Warning: Note file not found for {model}: {note_file}")
        
        # Initialize or load existing transcript entry
        if file_name in metrics_data['transcripts']:
            transcript_entry = metrics_data['transcripts'][file_name]
        else:
            transcript_entry = {
                'file_name': str(file_name),
                'metrics': {
                    'bertscore': {},
                    'rouge': {},
                    'summarization': {}
                }
            }
            # Add transcript entry to metrics data before processing models
            metrics_data['transcripts'][file_name] = transcript_entry
        
        # Calculate metrics for each model
        print(f"  Calculating metrics for {len(notes_by_model)} models...")
        
        for model, note in notes_by_model.items():
            print(f"    Processing {model}...")
            
            # BERTScore
            bertscore_entry = transcript_entry['metrics']['bertscore'].get(model, {})
            bertscore_exists = (
                bertscore_entry.get('f1') is not None and
                'error' not in bertscore_entry
            )
            if bertscore_exists:
                print(f"      BERTScore already calculated, skipping...")
            else:
                try:
                    bertscore_result = calculate_bertscore(
                        predictions=[note],
                        references=[transcript]
                    )
                    # Extract individual score (first one)
                    transcript_entry['metrics']['bertscore'][model] = {
                        'precision': bertscore_result['scores'][0]['precision'],
                        'recall': bertscore_result['scores'][0]['recall'],
                        'f1': bertscore_result['scores'][0]['f1']
                    }
                except Exception as e:
                    print(f"      Error calculating BERTScore: {e}")
                    transcript_entry['metrics']['bertscore'][model] = {'error': str(e)}
            
            # ROUGE
            rouge_entry = transcript_entry['metrics']['rouge'].get(model, {})
            rouge_exists = (
                rouge_entry.get('rouge1') is not None and
                'error' not in rouge_entry
            )
            if rouge_exists:
                print(f"      ROUGE already calculated, skipping...")
            else:
                try:
                    rouge_result = calculate_rouge(
                        predictions=[note],
                        references=[transcript]
                    )
                    # Extract individual score (first one)
                    transcript_entry['metrics']['rouge'][model] = rouge_result['scores'][0]
                except Exception as e:
                    print(f"      Error calculating ROUGE: {e}")
                    transcript_entry['metrics']['rouge'][model] = {'error': str(e)}
            
            # SummarizationMetric
            summarization_entry = transcript_entry['metrics']['summarization'].get(model, {})
            summarization_exists = (
                summarization_entry.get('score') is not None and
                'error' not in summarization_entry
            )
            if summarization_exists:
                print(f"      SummarizationMetric already calculated, skipping...")
            else:
                try:
                    summarization_result = calculate_summarization_metric(
                        predictions=[note],
                        references=[transcript]
                    )
                    # Extract individual score (first one)
                    transcript_entry['metrics']['summarization'][model] = {
                        'score': summarization_result['scores'][0],
                        'reason': summarization_result['reasons'][0]
                    }
                    
                    # Extract assessment questions info from first summarization result
                    if not assessment_questions_extracted:
                        experiment_info['assessment_questions'] = summarization_result['assessment_questions']
                        metrics_data['experiment_info'] = experiment_info
                        assessment_questions_extracted = True
                except Exception as e:
                    print(f"      Error calculating SummarizationMetric: {e}")
                    transcript_entry['metrics']['summarization'][model] = {'error': str(e)}
            
            # Update and save metrics.json after each model completes
            with open(metrics_file, 'w', encoding='utf-8') as f:
                json.dump(metrics_data, f, indent=2, ensure_ascii=False)
    
    print(f"\n=== Metrics Evaluation Complete ===")
    print(f"Metrics saved to: {metrics_file}")
    print(f"Processed {len(metrics_data['transcripts'])} transcripts")
    
    return metrics_data


if __name__ == "__main__":
    output_folder = "outputs/chunking/20251207_130615"
    
    run_metrics(output_folder)


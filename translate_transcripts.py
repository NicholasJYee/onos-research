"""
Translate transcripts from data/1interviews/transcripts to Hungarian.

Reads all transcript files, translates them to Hungarian using OpenAI API, and saves
the translated transcripts to data/1interviews/transcripts_hu.
"""

from pathlib import Path
import yaml
from openai import OpenAI
import time


def load_api_key(filepath: str) -> str:
    """Load OpenAI API key from YAML file."""
    with open(filepath, 'r') as f:
        config = yaml.safe_load(f)
    return config.get('openai', '')


def load_transcript(filepath: Path) -> str:
    """Load transcript from file."""
    with open(filepath, 'r', encoding='utf-8') as f:
        return f.read().strip()


def translate_text(client: OpenAI, text: str, language_code: str, model: str = "gpt-5-nano") -> str:
    """
    Translate text to target language using OpenAI API.
    
    Args:
        client: OpenAI client instance
        text: Text to translate
        language_code: 2-letter language code (e.g., 'hu' for Hungarian)
        model: OpenAI model to use
    
    Returns:
        Translated text
    """
    # Map language codes to language names
    language_names = {
        'hu': 'Hungarian',
        # Add more languages here as needed
    }
    
    target_language = language_names.get(language_code.lower(), f'language code {language_code}')
    
    prompt = f"Translate the following medical conversation transcript to {target_language}. Preserve the original formatting and medical terminology accuracy. Only return the translated text without any additional commentary:\n\n{text}"
    
    try:
        response = client.chat.completions.create(
            model=model,
            messages=[
                {"role": "system", "content": "You are a professional medical translator. Translate medical conversations accurately while preserving the original structure and formatting."},
                {"role": "user", "content": prompt}
            ],
        )
        return response.choices[0].message.content.strip()
    except Exception as e:
        print(f"    Error during translation: {e}")
        raise


def remove_line_breaks(transcripts_dir: str):
    """
    Remove line breaks from transcript files and save to a cleaned directory.
    
    Args:
        transcripts_dir: Directory containing transcript files to clean
    """
    # Setup paths
    base_path = Path(__file__).parent
    input_dir = base_path / transcripts_dir
    
    if not input_dir.exists():
        print(f"Error: Directory not found: {input_dir}")
        return
    
    # Create output directory
    output_dir = base_path / f"{transcripts_dir}_cleaned"
    output_dir.mkdir(parents=True, exist_ok=True)
    print(f"\nCleaning transcripts...")
    print(f"Input directory: {input_dir}")
    print(f"Output directory: {output_dir}")
    
    # Get all transcript files
    transcript_files = sorted([f for f in input_dir.iterdir() if f.is_file() and f.suffix == '.txt'])
    
    if not transcript_files:
        print(f"No transcript files found in {input_dir}")
        return
    
    print(f"Found {len(transcript_files)} transcript files to clean")
    print()
    
    # Process each transcript
    successful = 0
    failed = 0
    
    for idx, transcript_file in enumerate(transcript_files, 1):
        print(f"[{idx}/{len(transcript_files)}] Cleaning {transcript_file.name}...")
        
        try:
            # Load transcript
            with open(transcript_file, 'r', encoding='utf-8') as f:
                transcript_text = f.read()
            
            if not transcript_text.strip():
                print(f"    Warning: Empty transcript, skipping...")
                continue
            
            # Remove line breaks - replace newlines with spaces and clean up multiple spaces
            cleaned_text = ' '.join(transcript_text.split())
            
            # Save cleaned transcript
            output_file = output_dir / transcript_file.name
            with open(output_file, 'w', encoding='utf-8') as f:
                f.write(cleaned_text)
            
            print(f"    ✓ Cleaned ({len(cleaned_text)} chars)")
            print(f"    Saved to: {output_file}")
            successful += 1
            
        except Exception as e:
            print(f"    ✗ Error: {e}")
            failed += 1
        
        print()
    
    # Summary
    print("=" * 60)
    print(f"Cleaning Complete!")
    print(f"  Successful: {successful}")
    print(f"  Failed: {failed}")
    print(f"  Total: {len(transcript_files)}")
    print(f"  Output directory: {output_dir}")
    print("=" * 60)


def translate_transcripts(
    transcripts_dir: str = "data/1interviews/transcripts",
    language_code: str = "hu",
    model: str = "gpt-5-nano"
):
    """
    Translate all transcripts in the transcripts directory to target language.
    
    Args:
        transcripts_dir: Directory containing transcript files
        language_code: 2-letter language code (currently only 'hu' for Hungarian is supported)
        model: OpenAI model to use for translation
    """
    # Validate language code (only Hungarian supported for now)
    language_code = language_code.lower().strip()
    if language_code != 'hu':
        print(f"Error: Only Hungarian ('hu') is currently supported. You provided: {language_code}")
        return
    
    # Setup paths
    base_path = Path(__file__).parent
    transcripts_path = base_path / transcripts_dir
    api_keys_path = base_path / "API_KEYS.yaml"
    
    # Load API key
    print("Loading API key...")
    api_key = load_api_key(str(api_keys_path))
    if not api_key:
        print("Error: Could not load OpenAI API key from API_KEYS.yaml")
        return
    
    # Initialize OpenAI client
    client = OpenAI(api_key=api_key)
    
    print(f"Target language: Hungarian ({language_code})")
    
    # Create output directory
    output_dir = base_path / "data" / "1interviews" / f"transcripts_{language_code}"
    output_dir.mkdir(parents=True, exist_ok=True)
    print(f"Output directory: {output_dir}")
    
    # Get all transcript files
    transcript_files = sorted([f for f in transcripts_path.iterdir() if f.is_file() and f.suffix == '.txt'])
    
    if not transcript_files:
        print(f"No transcript files found in {transcripts_path}")
        return
    
    print(f"Found {len(transcript_files)} transcript files to translate")
    print(f"Using model: {model}")
    print()
    
    # Translate each transcript
    successful = 0
    failed = 0
    
    for idx, transcript_file in enumerate(transcript_files, 1):
        print(f"[{idx}/{len(transcript_files)}] Translating {transcript_file.name}...")
        
        try:
            # Load transcript
            transcript_text = load_transcript(transcript_file)
            
            if not transcript_text:
                print(f"    Warning: Empty transcript, skipping...")
                continue
            
            # Translate
            start_time = time.time()
            translated_text = translate_text(client, transcript_text, language_code, model)
            duration = time.time() - start_time
            
            # Save translated transcript
            output_file = output_dir / transcript_file.name
            with open(output_file, 'w', encoding='utf-8') as f:
                f.write(translated_text)
            
            print(f"    ✓ Translated ({len(translated_text)} chars) in {duration:.2f}s")
            print(f"    Saved to: {output_file}")
            successful += 1
            
        except Exception as e:
            print(f"    ✗ Error: {e}")
            failed += 1
        
        print()
    
    # Summary
    print("=" * 60)
    print(f"Translation Complete!")
    print(f"  Successful: {successful}")
    print(f"  Failed: {failed}")
    print(f"  Total: {len(transcript_files)}")
    print(f"  Output directory: {output_dir}")
    print("=" * 60)


if __name__ == "__main__":
    language_code = "hu"
    transcripts_dir = f"data/1interviews/transcripts"
    
    # translate_transcripts(
    #     transcripts_dir=transcripts_dir,
    #     language_code=language_code,
    #     model="gpt-5-nano"
    # )
    
    remove_line_breaks(transcripts_dir=f"{transcripts_dir}_{language_code}")


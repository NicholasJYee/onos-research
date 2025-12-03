"""
Transcript chunking utilities for LLM processing.

Splits transcripts into manageable chunks by word count while preserving
sentence boundaries for LLM processing with context window limitations.
"""

from typing import List


def chunk_by_words(
    text: str,
    chunk_size: int,
    overlap: int,
) -> List[str]:
    """
    Split text into chunks by word count. Preserves sentences.
    
    Args:
        text: The transcript text to chunk
        chunk_size: Maximum words per chunk
        overlap: Number of words to overlap between chunks
    
    Returns:
        List of text chunks
    """
    if not text:
        return []
    
    words = text.split()  # This keeps punctuation attached to words (i.e. "hello, world" becomes ['hello,', 'world'])
    
    if not words:
        return []
    
    chunks = []
    start_idx = 0
    total_words = len(words)
    
    while start_idx < total_words:
        end_idx = min(start_idx + chunk_size, total_words)
        
        # Try to find sentence boundary near the end
        if end_idx < total_words:
            # Look backwards from end_idx for sentence-ending punctuation
            for i in range(end_idx - 1, start_idx - 1, -1):
                if words[i].endswith(('.', '!', '?')):
                    end_idx = i + 1
                    break
        
        # Join words with spaces to form chunk
        chunk = ' '.join(words[start_idx:end_idx])
        if chunk:
            chunks.append(chunk)
        
        # Move start position with overlap
        # Ensure start always advances to prevent infinite loop
        new_start = end_idx - overlap if overlap > 0 else end_idx
        start_idx = max(new_start, start_idx + 1)  # Always advance by at least 1 word
    
    return chunks



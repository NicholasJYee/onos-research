"""
ROUGE metric evaluation for medical note generation.

ROUGE (Recall-Oriented Understudy for Gisting Evaluation) measures the overlap
of n-grams between generated and reference summaries.

Uses the rouge-score package from PyPI: https://pypi.org/project/rouge-score/

Using ROUGE metrics:
- ROUGE-1: Overlap of unigrams (each word)
- ROUGE-2: Overlap of bigrams
- ROUGE-L: Longest Common Subsequence (LCS) based statistics
- ROUGE-W: Weighted LCS-based statistics that favors consecutive LCSes
- ROUGE-S: Skip-bigram based co-occurrence statistics on any pair of words in their sentence order.
- ROUGE-SU: Skip-bigram plus unigram-based co-occurrence statistics
"""

from typing import List, Dict, Any
import statistics
from rouge_score import rouge_scorer


def calculate_rouge(
    predictions: List[str],
    references: List[str]
) -> Dict[str, Any]:
    """
    Calculate ROUGE scores for predictions against references.
    
    Uses the rouge-score package (https://pypi.org/project/rouge-score/).
    Computes all 6 ROUGE metrics: rouge1, rouge2, rougeL, rougeW, rougeS, rougeSU.
    
    Args:
        predictions: List of generated medical notes
        references: List of reference/ground truth medical notes
    
    Returns:
        Dictionary with individual scores and summary statistics:
        {
            'scores': [
                {
                    'rouge1': {'precision': float, 'recall': float, 'fmeasure': float},
                    'rouge2': {...},
                    'rougeL': {...},
                    'rougeW': {...},
                    'rougeS': {...},
                    'rougeSU': {...}
                },
                ...
            ],
            'summary': {
                'rouge1': {
                    'precision': {'mean': float, 'median': float, 'sd': float, 'p25': float, 'p75': float},
                    'recall': {...},
                    'fmeasure': {...}
                },
                ... (same for rouge2, rougeL, rougeW, rougeS, rougeSU)
            }
        }
    
    Example:
        predictions = [
            "Patient has ankle fracture.",
            "Patient reports pain."
        ]
        references = [
            "Patient presents with ankle fracture.",
            "Patient complains of pain."
        ]
        result = calculate_rouge(predictions, references)
    """
    rouge_types = ['rouge1', 'rouge2', 'rougeL', 'rougeW', 'rougeS', 'rougeSU']
    
    if len(predictions) != len(references):
        raise ValueError(f"Predictions ({len(predictions)}) and references ({len(references)}) must have the same length")
    
    scorer = rouge_scorer.RougeScorer(rouge_types, use_stemmer=True)
    
    # Compute scores for each pair
    all_scores = {rouge_type: {'precision': [], 'recall': [], 'fmeasure': []} 
                  for rouge_type in rouge_types}
    individual_scores = []
    
    for pred, ref in zip(predictions, references):
        scores = scorer.score(ref, pred)
        pair_scores = {}
        for rouge_type in rouge_types:
            precision = scores[rouge_type].precision
            recall = scores[rouge_type].recall
            fmeasure = scores[rouge_type].fmeasure
            
            all_scores[rouge_type]['precision'].append(precision)
            all_scores[rouge_type]['recall'].append(recall)
            all_scores[rouge_type]['fmeasure'].append(fmeasure)
            
            pair_scores[rouge_type] = {
                'precision': precision,
                'recall': recall,
                'fmeasure': fmeasure
            }
        individual_scores.append(pair_scores)
    
    # Calculate summary statistics for each metric
    def calculate_stats(values: List[float]) -> Dict[str, float]:
        mean = statistics.mean(values)
        median = statistics.median(values)
        sd = statistics.stdev(values) if len(values) > 1 else 0.0
        
        # Calculate percentiles using quantiles (handles sorting internally)
        quantiles = statistics.quantiles(values, n=4)  # Returns [p25, p50, p75]
        p25 = quantiles[0]
        p75 = quantiles[2]
        
        return {
            'mean': mean,
            'median': median,
            'sd': sd,
            'p25': p25,
            'p75': p75
        }
    
    # Build summary statistics
    summary = {}
    for rouge_type in rouge_types:
        summary[rouge_type] = {
            'precision': calculate_stats(all_scores[rouge_type]['precision']),
            'recall': calculate_stats(all_scores[rouge_type]['recall']),
            'fmeasure': calculate_stats(all_scores[rouge_type]['fmeasure'])
        }
    
    return {
        'scores': individual_scores,
        'summary': summary
    }


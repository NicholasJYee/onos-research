"""
BERTScore metric evaluation for medical note generation.

BERTScore evaluates text generation by computing contextual embeddings using BERT
and matching tokens in candidate and reference sentences.

Uses the bert-score package from PyPI: https://pypi.org/project/bert-score/
Recommended model: 'microsoft/deberta-xlarge-mnli' for best correlation with human evaluation.
Model	                        Best Layer	WMT16 To-English Pearson Correlation	Rank	Max Length
microsoft/deberta-xlarge-mnli	40	        0.7781	                                1	    510
"""

from typing import List, Dict, Optional, Any
import statistics
from bert_score import score


def calculate_bertscore(
    predictions: List[str],
    references: List[str],
    model_type: Optional[str] = 'microsoft/deberta-xlarge-mnli',
    verbose: bool = False
) -> Dict[str, Any]:
    """
    Calculate BERTScore for predictions against references.
    
    Uses the bert-score package (https://pypi.org/project/bert-score/).
    For best results, use model_type='microsoft/deberta-xlarge-mnli'.
    
    Args:
        predictions: List of generated medical notes
        references: List of reference/ground truth medical notes
        model_type: BERT model to use. Default/Recommended: 'microsoft/deberta-xlarge-mnli'
        verbose: Whether to print progress
    
    Returns:
        Dictionary with individual scores and summary statistics:
        {
            'scores': [
                {'precision': float, 'recall': float, 'f1': float},
                ...
            ],
            'summary': {
                'precision': {
                    'mean': float,
                    'median': float,
                    'sd': float,
                    'p25': float,
                    'p75': float
                },
                'recall': {
                    'mean': float,
                    'median': float,
                    'sd': float,
                    'p25': float,
                    'p75': float
                },
                'f1': {
                    'mean': float,
                    'median': float,
                    'sd': float,
                    'p25': float,
                    'p75': float
                }
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
        result = calculate_bertscore(predictions, references)
        # Returns summary with mean, median, sd, p25, p75 for each metric
    """
    if len(predictions) != len(references):
        raise ValueError(f"Predictions ({len(predictions)}) and references ({len(references)}) must have the same length")
    
    # Prepare kwargs for score function
    score_kwargs = {
        'lang': 'en',
        'verbose': verbose,
        'model_type': model_type
    }
    
    # Compute BERTScore using bert-score package
    P, R, F1 = score(
        predictions,
        references,
        **score_kwargs
    )
    
    # Convert to list of dictionaries
    scores = []
    precisions = []
    recalls = []
    f1s = []
    
    for i in range(len(predictions)):
        precision = float(P[i].item())
        recall = float(R[i].item())
        f1 = float(F1[i].item())
        
        scores.append({
            'precision': precision,
            'recall': recall,
            'f1': f1
        })
        precisions.append(precision)
        recalls.append(recall)
        f1s.append(f1)
    
    # Calculate summary statistics for each metric
    def calculate_stats(values: List[float]) -> Dict[str, float]:
        mean = statistics.mean(values)
        median = statistics.median(values)
        sd = statistics.stdev(values)
        
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
    
    summary = {
        'precision': calculate_stats(precisions),
        'recall': calculate_stats(recalls),
        'f1': calculate_stats(f1s)
    }
    
    return {
        'scores': scores,
        'summary': summary
    }


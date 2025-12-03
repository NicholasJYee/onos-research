"""
DeepEval SummarizationMetric for medical note generation evaluation.

Uses DeepEval's SummarizationMetric which combines multiple evaluation criteria
including relevance, coherence, and consistency.

Assessment questions are loaded from assessment_questions.yaml.
"""

from typing import List, Dict, Optional, Any
from pathlib import Path
import yaml
import statistics
from deepeval.metrics import SummarizationMetric
from deepeval.test_case import LLMTestCase


def load_assessment_questions(filepath: Optional[str] = None) -> List[str]:
    """
    Load assessment questions from YAML file.
    
    Args:
        filepath: Path to assessment questions YAML file.
                  Defaults to methods/metrics/assessment_questions.yaml
    
    Returns:
        List of assessment questions
    """
    if filepath is None:
        filepath = Path(__file__).parent / "assessment_questions.yaml"
    
    with open(filepath, 'r') as f:
        config = yaml.safe_load(f)
    
    return config['questions']


def calculate_summarization_metric(
    predictions: List[str],
    references: List[str],
    threshold: float = 0.5,
    assessment_questions_file: Optional[str] = None
) -> Dict[str, Any]:
    """
    Calculate DeepEval's SummarizationMetric for predictions against references. This is a LLM-As-A-Judge (LLM-AAJ) evaluation metric.
    
    Uses DeepEval's SummarizationMetric (https://deepeval.com/docs/metrics-summarization).
    Assessment questions are loaded from assessment_questions.yaml by default.
    
    Args:
        predictions: List of generated medical notes (actual_output)
        references: List of reference/ground truth medical notes (input)
        threshold: Score threshold for passing (default: 0.5)
        assessment_questions_file: Optional path to assessment questions YAML file
    
    Returns:
        Dictionary with evaluation results:
        {
            'scores': List[float],   # Individual scores for each example
            'summary': {
                'mean': float,       # Mean score
                'median': float,     # Median score
                'p25': float,       # 25th percentile
                'p75': float,       # 75th percentile
                'pass_rate': float  # Percentage of examples above threshold
            }
        }
    
    Example:
        predictions = ["Patient has ankle fracture."]
        references = ["Patient presents with ankle fracture."]
        results = calculate_summarization_metric(predictions, references)
    """
    if len(predictions) != len(references):
        raise ValueError(f"Predictions ({len(predictions)}) and references ({len(references)}) must have the same length")
    
    # Load assessment questions from YAML
    assessment_questions = load_assessment_questions(assessment_questions_file)
    
    # Initialize metric with assessment questions
    metric = SummarizationMetric(
        threshold=threshold,
        assessment_questions=assessment_questions
    )
    scores = []
    
    for pred, ref in zip(predictions, references):
        # Create test case: input=ground truth, actual_output=predictions
        test_case = LLMTestCase(
            input=ref,
            actual_output=pred
        )
        
        # Measure using the metric
        metric.measure(test_case)
        
        score = metric.score
        scores.append(score)
    
    # Calculate summary statistics
    mean_score = statistics.mean(scores)
    median_score = statistics.median(scores)
    quantiles = statistics.quantiles(scores, n=4)  # Returns [p25, p50, p75]
    p25 = quantiles[0]
    p75 = quantiles[2]
    pass_rate = sum(1 for s in scores if s >= threshold) / len(scores)

    
    return {
        'scores': scores,
        'summary': {
            'mean': mean_score,
            'median': median_score,
            'p25': p25,
            'p75': p75,
            'pass_rate': pass_rate
        }
    }

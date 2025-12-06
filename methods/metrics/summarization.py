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
import os
from deepeval.metrics import SummarizationMetric
from deepeval.test_case import LLMTestCase
from deepeval.models import OllamaModel


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
            'reasons': List[str],     # Reasoning for each example
            'assessment_questions': {
                'name': str,         # Name from assessment_questions.yaml
                'version': str       # Version from assessment_questions.yaml
            },
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
    
    # Load assessment questions config from YAML
    if assessment_questions_file is None:
        assessment_questions_path = Path(__file__).parent / "assessment_questions.yaml" 
    else:
        assessment_questions_path = Path(assessment_questions_file)
    
    with open(assessment_questions_path, 'r') as f:
        assessment_config = yaml.safe_load(f)
    
    assessment_questions = assessment_config['questions']
    assessment_name = assessment_config['name']
    assessment_version = assessment_config['version']
    
    # Set DeepEval timeout
    if 'DEEPEVAL_PER_ATTEMPT_TIMEOUT_SECONDS_OVERRIDE' not in os.environ:
        os.environ['DEEPEVAL_PER_ATTEMPT_TIMEOUT_SECONDS_OVERRIDE'] = '1200'  # 20 minutes timeout
    
    # Create OllamaModel instance (designed specifically for Ollama)
    ollama_model = OllamaModel(
        model='gpt-oss:20b',
        base_url='http://localhost:11434/'
    )
    
    # Initialize metric with assessment questions and Ollama model
    metric = SummarizationMetric(
        threshold=threshold,
        assessment_questions=assessment_questions,
        model=ollama_model
    )
    scores = []
    reasons = []
    
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
        reason = metric.reason
        reasons.append(reason)
    
    # Calculate summary statistics (only if more than 1 entry)
    if len(predictions) > 1:
        mean_score = statistics.mean(scores)
        median_score = statistics.median(scores)
        quantiles = statistics.quantiles(scores, n=4)  # Returns [p25, p50, p75]
        p25 = quantiles[0]
        p75 = quantiles[2]
        pass_rate = sum(1 for s in scores if s >= threshold) / len(scores)
        
        summary = {
            'mean': mean_score,
            'median': median_score,
            'p25': p25,
            'p75': p75,
            'pass_rate': pass_rate
        }
    else:
        summary = None
    
    return {
        'scores': scores,
        'reasons': reasons,
        'assessment_questions': {
            'name': assessment_name,
            'version': assessment_version
        },
        'summary': summary
    }


if __name__ == "__main__":
    predictions = ["Patient has ankle fracture."]
    references = ["Patient presents with ankle fracture."]
    results = calculate_summarization_metric(predictions, references)
    print(results)
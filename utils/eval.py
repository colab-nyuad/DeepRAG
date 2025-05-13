from typing import List, Dict, Optional
import evaluate
from sklearn.feature_extraction.text import CountVectorizer
import numpy as np
from utils import ModelManager
from utils import QA_EVAL_SYSTEM_PROMPT,RELEVANCE_USER_PROMPT,FAITHFULNESS_USER_PROMPT, relevance_json, faithfulness_json
import json
import time

class RetrieverMetrics:
    """Class to calculate retrieval evaluation metrics."""
    
    @staticmethod
    def calculate_metrics(retrieved_doc_ids: List[List[str]], 
                          ground_truth_doc_ids: List[List[str]],
                          k_values: List[int] = [1, 3, 5, 10]) -> Dict[str, Dict[str, float]]:
        """Calculate precision, recall and F1 at different k values."""
        metrics = {}
        
        for k in k_values:
            precision_sum = 0
            recall_sum = 0
            f1_sum = 0
            
            for retrieved, ground_truth in zip(retrieved_doc_ids, ground_truth_doc_ids):
                # Take only top-k retrieved docs
                retrieved_at_k = set(retrieved[:k])
                ground_truth_set = set(ground_truth)

                # Calculate intersection
                correct = len(retrieved_at_k.intersection(ground_truth_set))
                
                # Calculate metrics
                precision = correct / k if k > 0 else 0
                recall = correct / len(ground_truth_set) if ground_truth_set else 0
                f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0
                
                precision_sum += precision
                recall_sum += recall
                f1_sum += f1
            
            n_queries = len(retrieved_doc_ids)
            metrics[f"@{k}"] = {
                "precision": round(precision_sum / n_queries * 100, 2),
                "recall": round(recall_sum / n_queries * 100, 2),
                "f1": round(f1_sum / n_queries * 100, 2)
            }
        
        return metrics


class ReaderMetrics:
    """Class to evaluate reader responses against ground truth answers using 
    BLEU, ROUGE, and Jaccard similarity metrics using standard libraries."""
    
    # Load metrics once at class level to avoid reloading for each evaluation
    _bleu_metric = None
    _rouge_metric = None
    _model_manager = None
    
    @classmethod
    def _get_rouge_metric(cls):
        """Lazy loading of rouge metric to avoid loading it multiple times."""
        if cls._rouge_metric is None:
            cls._rouge_metric = evaluate.load('rouge')
        return cls._rouge_metric
    
    @classmethod
    def _get_bleu_metric(cls):
        """Lazy loading of bleu metric."""
        if cls._bleu_metric is None:
            cls._bleu_metric = evaluate.load('bleu')
        return cls._bleu_metric
    
    @classmethod
    def _get_model(cls):
        if cls._model_manager is None:
            cls._model_manager = ModelManager(provider="claude")
        return cls._model_manager
    
    # @staticmethod
    # def preprocess_text(text: str) -> List[str]:
    #     """Preprocess text by converting to lowercase and tokenizing."""
    #     # Convert to lowercase
    #     text = text.lower()
        
    #     # Tokenize using NLTK
    #     try:
    #         tokens = nltk.word_tokenize(text)
    #     except LookupError:
    #         # If NLTK tokenizer not available, fall back to simple splitting
    #         nltk.download('punkt')
    #         tokens = nltk.word_tokenize(text)
            
    #     return tokens

    @staticmethod
    def _retry_claude_call(call_func, max_retries=2, backoff=1):
        """Helper to retry Claude model call with exponential backoff."""
        for attempt in range(max_retries):
            try:
                return call_func()
            except Exception as e:
                print(f"Attempt {attempt + 1} failed: {e}")
                if attempt < max_retries - 1:
                    time.sleep(backoff ** attempt)
                else:
                    return call_func()  # Last attempt without delay
    
    @staticmethod
    def calculate_jaccard_similarity(response: str, ground_truth: str) -> float:
        """Calculate Jaccard similarity between response and ground truth using scikit-learn."""
        # Create a vectorizer to convert texts to binary vectors
        vectorizer = CountVectorizer(binary=True, lowercase=True)
        
        try:
            # Fit and transform the texts to get binary vectors
            vectors = vectorizer.fit_transform([response, ground_truth]).toarray()
        except ValueError:
            # Handle empty strings
            return 0.0
        
        # Calculate Jaccard similarity from binary vectors
        if np.sum(vectors[0]) == 0 or np.sum(vectors[1]) == 0:
            return 0.0
            
        intersection = np.sum(np.logical_and(vectors[0], vectors[1]))
        union = np.sum(np.logical_or(vectors[0], vectors[1]))
        
        return float(intersection / union if union > 0 else 0.0)
    
    @classmethod
    def calculate_bleu(cls, response: str, ground_truth: str) -> float:
        """Calculate BLEU score between response and ground truth."""
        if not response or not ground_truth:
            return 0.0
        
        # Get the bleu metric
        bleu = cls._get_bleu_metric()
            
        results = bleu.compute(predictions=[response], references=[[ground_truth]], smooth=True)
        
        return results['bleu']
    
    @classmethod
    def calculate_rouge(cls, response: str, ground_truth: str) -> Dict[str, float]:
        """Calculate ROUGE scores between response and ground truth using evaluate library."""
        if not response or not ground_truth:
            return {
                'rouge1': 0.0,
                'rouge2': 0.0,
                'rougeL': 0.0
            }
        
        # Get the rouge metric
        rouge = cls._get_rouge_metric()
        
        # Calculate ROUGE scores
        results = rouge.compute(predictions=[response], references=[ground_truth])
        
        # Return the scores
        return {
            'rouge1': results['rouge1'],
            'rouge2': results['rouge2'],
            'rougeL': results['rougeL']
        }
    
    @staticmethod
    def evaluate(response: str, ground_truth: str) -> Dict[str, float]:
        """Evaluate reader response against ground truth using multiple metrics."""
        # Calculate Jaccard similarity
        jaccard = ReaderMetrics.calculate_jaccard_similarity(response, ground_truth)
        
        # Calculate BLEU score
        bleu = ReaderMetrics.calculate_bleu(response, ground_truth)
        
        # Calculate ROUGE scores
        rouge_scores = ReaderMetrics.calculate_rouge(response, ground_truth)
        
        # Compile all metrics into a single dictionary
        metrics = {
            "jaccard": jaccard,
            "bleu": bleu,
            "rouge1": rouge_scores['rouge1'],
            "rouge2": rouge_scores['rouge2'],
            "rougeL": rouge_scores['rougeL']
        }
        
        return metrics
    
    @staticmethod
    def evaluate_batch(responses: List[str], ground_truths: List[str]) -> Dict[str, float]:
        """Evaluate a batch of reader responses against ground truth answers."""
        if len(responses) != len(ground_truths):
            raise ValueError("Number of responses must match number of ground truths")
            
        # Initialize counters for average metrics
        metrics_sum = {}
        
        # Calculate metrics for each pair
        for response, ground_truth in zip(responses, ground_truths):
            metrics = ReaderMetrics.evaluate(response, ground_truth)
            
            # Initialize metrics_sum with the first set of metrics
            if not metrics_sum:
                metrics_sum = {metric: 0.0 for metric in metrics}
            
            for metric, value in metrics.items():
                metrics_sum[metric] += value
        
        # Calculate averages
        batch_size = len(responses)
        metrics_avg = {metric: value / batch_size for metric, value in metrics_sum.items()}
        
        return metrics_avg
    
    @classmethod
    def evaluate_faithfulness_claude(cls, 
                                   question: str, 
                                   answer: str, 
                                   documents: str,
                                   claude_model=None) -> Dict[str, int]:
        """
        Evaluate the faithfulness of an answer using Claude.
        
        Args:
            question: The question being answered
            answer: The model's answer
            documents: Retrieved passages that the answer should be based on
            model_manager: An instance of ModelManager configured to use Claude
            
        Returns:
            Dictionary with faithfulness score
        """ 
        system_prompt = QA_EVAL_SYSTEM_PROMPT.format(schema=json.dumps(faithfulness_json,indent=2))

        # Format documents into a string
        documents_text = "\n\n".join([f"Document content: {text}" 
                                      for i, (doc_id, text) in enumerate(documents)])
        
        user_prompt = FAITHFULNESS_USER_PROMPT.format(
            question=question,
            answer=answer,
            documents=documents_text
        )
        
        # Call the Claude model through model_manager
        response = cls._retry_claude_call(lambda: claude_model.call_model(
            system_prompt=system_prompt,
            user_prompt=user_prompt
        ))
        
        try:
            # Parse the response as JSON
            result = json.loads(response.strip())
            # Ensure the score is an integer within the expected range
            faithfulness_score = int(result.get("faithfulness", -1))
            if faithfulness_score not in [-1, 0, 1]:
                print(f"Warning: Unexpected faithfulness score {faithfulness_score}. Clamping to valid range.")
                faithfulness_score = max(-1, min(faithfulness_score, 1))
                
            return {"faithfulness": faithfulness_score}
        except (json.JSONDecodeError, ValueError) as e:
            print(f"Error parsing Claude faithfulness evaluation: {e}")
            print(f"Raw response: {response}")
            return {"faithfulness": -1}  # Default to neutral score on error
    
    @classmethod
    def evaluate_relevance_claude(cls,
                                question: str,
                                answer: str,
                                claude_model=None) -> Dict[str, int]:
        """
        Evaluate the relevance of an answer using Claude.
        
        Args:
            question: The question being answered
            answer: The model's answer
            model_manager: An instance of ModelManager configured to use Claude
            
        Returns:
            Dictionary with relevance score
        """
            
        system_prompt = QA_EVAL_SYSTEM_PROMPT.format(schema=json.dumps(relevance_json,indent=2))
        
        user_prompt = RELEVANCE_USER_PROMPT.format(
            question=question,
            answer=answer
        )
        
        # Call the Claude model through model_manager
        response = cls._retry_claude_call(lambda: claude_model.call_model(
            system_prompt=system_prompt,
            user_prompt=user_prompt
        ))
        
        try:
            # Parse the response as JSON
            result = json.loads(response.strip())
            # Ensure the score is an integer within the expected range
            relevance_score = int(result.get("relevance", 0))
            if relevance_score not in [-1, 0, 1, 2]:
                print(f"Warning: Unexpected relevance score {relevance_score}. Clamping to valid range.")
                relevance_score = max(-1, min(relevance_score, 2))
                
            return {"relevance": relevance_score}
        except (json.JSONDecodeError, ValueError) as e:
            print(f"Error parsing Claude relevance evaluation: {e}")
            print(f"Raw response: {response}")
            return {"relevance": 0}  # Default to neutral score on error
    
    @staticmethod
    def evaluate(response: str, ground_truth: str) -> Dict[str, float]:
        """Evaluate reader response against ground truth using multiple metrics."""
        # Calculate Jaccard similarity
        jaccard = ReaderMetrics.calculate_jaccard_similarity(response, ground_truth)
        
        # Calculate BLEU score
        bleu = ReaderMetrics.calculate_bleu(response, ground_truth)
        
        # Calculate ROUGE scores
        rouge_scores = ReaderMetrics.calculate_rouge(response, ground_truth)
        
        # Compile all metrics into a single dictionary
        metrics = {
            "jaccard": jaccard,
            "bleu": bleu,
            "rouge1": rouge_scores['rouge1'],
            "rouge2": rouge_scores['rouge2'],
            "rougeL": rouge_scores['rougeL']
        }
        
        return metrics
    
    @classmethod
    def evaluate_with_claude(cls, 
                          question: str, 
                          answer: str, 
                          ground_truth: Optional[str] = None,
                          documents: Optional[str] = None,) -> Dict[str, float]:
        """
        Comprehensive evaluation of a response using both traditional metrics and Claude-based evaluation.
        
        Args:
            question: The question being answered
            answer: The model's answer
            ground_truth: Optional ground truth for traditional metrics calculation
            documents: Optional retrieved passages for faithfulness evaluation
            model_manager: A ModelManager instance configured to use Claude
            
        Returns:
            Dictionary with all evaluation metrics
        """
        
        # Initialize metrics dictionary
        metrics = {}

        claude_model = cls._get_model()
        
        # Traditional metrics if ground_truth is provided
        if ground_truth:
            trad_metrics = cls.evaluate(answer, ground_truth)
            metrics.update(trad_metrics)
        
        # Claude-based relevance evaluation
        relevance_metrics = cls.evaluate_relevance_claude(
            question=question, 
            answer=answer,
            claude_model=claude_model
        )
        metrics.update(relevance_metrics)
        
        # Claude-based faithfulness evaluation if documents are provided
        if documents:
            faithfulness_metrics = cls.evaluate_faithfulness_claude(
                question=question,
                answer=answer,
                documents=documents,
                claude_model=claude_model
            )
            metrics.update(faithfulness_metrics)
        
        return metrics
    
    @staticmethod
    def evaluate_batch(responses: List[str], ground_truths: List[str]) -> Dict[str, float]:
        """Evaluate a batch of reader responses against ground truth answers."""
        if len(responses) != len(ground_truths):
            raise ValueError("Number of responses must match number of ground truths")
            
        # Initialize counters for average metrics
        metrics_sum = {}
        
        # Calculate metrics for each pair
        for response, ground_truth in zip(responses, ground_truths):
            metrics = ReaderMetrics.evaluate(response, ground_truth)
            
            # Initialize metrics_sum with the first set of metrics
            if not metrics_sum:
                metrics_sum = {metric: 0.0 for metric in metrics}
            
            for metric, value in metrics.items():
                metrics_sum[metric] += value
        
        # Calculate averages
        batch_size = len(responses)
        metrics_avg = {metric: value / batch_size for metric, value in metrics_sum.items()}
        
        return metrics_avg
    
    @classmethod
    def evaluate_batch_with_claude(cls,
                                 questions: List[str],
                                 answers: List[str],
                                 ground_truths: Optional[List[str]] = None,
                                 documents_list: Optional[List[str]] = None) -> Dict[str, float]:
        """
        Evaluate a batch of reader responses using both traditional metrics and Claude-based evaluation.
        
        Args:
            questions: List of questions
            answers: List of model answers
            ground_truths: Optional list of ground truth answers
            documents_list: Optional list of retrieved passages for each question
            model_manager: A ModelManager instance configured to use Claude
            
        Returns:
            Dictionary with average scores across all evaluation metrics
        """
        
        if len(questions) != len(answers):
            raise ValueError("Number of questions must match number of answers")
            
        if ground_truths and len(ground_truths) != len(answers):
            raise ValueError("Number of ground truths must match number of answers")
            
        if documents_list and len(documents_list) != len(answers):
            raise ValueError("Number of documents sets must match number of answers")
        
        # Initialize metrics sum
        metrics_sum = {}
        
        # Evaluate each sample
        for i, (question, answer) in enumerate(zip(questions, answers)):
            # Get optional params if available
            ground_truth = ground_truths[i] if ground_truths else None
            documents = documents_list[i] if documents_list else None
            
            # Evaluate
            metrics = cls.evaluate_with_claude(
                question=question,
                answer=answer,
                ground_truth=ground_truth,
                documents=documents
            )
            
            # Initialize metrics_sum with the first set of metrics
            if not metrics_sum:
                metrics_sum = {metric: 0.0 for metric in metrics}
            
            # Add to running sum
            for metric, value in metrics.items():
                metrics_sum[metric] += value
        
        # Calculate averages
        batch_size = len(questions)
        metrics_avg = {metric: value / batch_size for metric, value in metrics_sum.items()}
        
        return metrics_avg
import json
from typing import Any, Dict, List, Tuple

class EvaluationDataset:
    """Class to load and manage test dataset from JSONL file."""
    
    def __init__(self, file_path: str):
        self.file_path = file_path
        self.data = self._load_data()
    
    def _load_data(self) -> List[Dict[str, Any]]:
        """Load data from JSONL file."""
        with open(self.file_path, 'r', encoding='utf-8') as f:
            return json.load(f)
        
    def get_data_for_single_doc_questions(self) -> Tuple[List[str], List[str], List[List[str]], List[str]]:
        """Get data for questions with a single document."""
        queries, answers, ground_truth_ids= [], [], []
        for item in self.data:
            if len(item["document_ids"]) == 1:
                queries.append(item.get("question", ""))
                answers.append(item.get("answer", ""))
                ground_truth_ids.append(item.get("document_ids", []))
        
        return queries, answers, ground_truth_ids
    
    def get_data_for_multiple_doc_questions(self) -> Tuple[List[str], List[str], List[List[str]], List[List[str]]]:
        """Get data for questions with multiple documents."""
        queries, answers, ground_truth_ids = [], [], []
        for item in self.data:
            if len(item["document_ids"]) > 1:
                queries.append(item.get("question", ""))
                answers.append(item.get("answer", ""))
                ground_truth_ids.append(item.get("document_ids", []))
        
        return queries, answers, ground_truth_ids

    def get_all_data(self) -> Tuple[List[str], List[str], List[List[str]], List[List[str]], List[List[str]]]:
        """Get all data."""
        queries, answers, ground_truth_ids, ground_truth_chunks = [], [], [], []
        for item in self.data:
            queries.append(item.get("question", ""))
            answers.append(item.get("answer", ""))
            ground_truth_ids.append(item.get("document_ids", []))
            ground_truth_chunks.append(item.get("chunks", []))
        return queries, answers, ground_truth_ids, ground_truth_chunks 
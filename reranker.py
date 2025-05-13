from typing import List, Tuple, Dict, Optional
from flashrank import Ranker, RerankRequest
from tqdm import tqdm

class DocumentReranker:
    """
    A class to rerank retrieved documents using FlashRank.
    """
    
    def __init__(self, 
                 model_name: str = None, 
                 max_length: int = 128, 
                 cache_dir: Optional[str] = None):
        """
        Initialize the reranker with a specific model or configuration.
        
        Args:
            model_name (str, optional): Name of the reranking model. 
                If None, uses the nano model.
            max_length (int, optional): Maximum sequence length. Defaults to 128.
            cache_dir (str, optional): Directory to cache model. Defaults to None.
        """
        self.ranker = Ranker(
            model_name=model_name, 
            cache_dir=cache_dir or "/opt"
        )
    
    def rerank(self, query: str, documents: List[Tuple[str, str]]) -> List[Tuple[str, str]]:
        """
        Rerank documents for a given query.
        
        Args:
            query (str): The original query
            documents (List[Tuple[str, str]]): List of (doc_id, text) tuples
            top_k (int, optional): Number of top documents to return. Defaults to 10.
        
        Returns:
            List[Tuple[str, str]]: Reranked list of (doc_id, text) tuples
        """
        # Prepare passages for reranking
        passages = [
            {
                "id": doc_id, 
                "text": text,
            } for doc_id, text in documents
        ]
        
        # Create rerank request
        rerank_request = RerankRequest(
            query=query, 
            passages=passages
        )
        
        # Rerank documents
        reranked_results = self.ranker.rerank(rerank_request)
        
        # Map reranked results back to original documents
        reranked_docs = [
            (str(result['id']), result['text']) 
            for result in reranked_results
        ]
        
        return reranked_docs
    
    def batch_rerank(self, 
                     queries: List[str], 
                     documents_list: List[List[Tuple[str, str]]]) -> List[List[Tuple[str, str]]]:
        """
        Batch rerank documents for multiple queries.
        
        Args:
            queries (List[str]): List of queries
            documents_list (List[List[Tuple[str, str]]]): List of document lists
        
        Returns:
            List[List[Tuple[str, str]]]: List of reranked document lists
        """
        reranked_results = []
        for query, documents in tqdm(zip(queries, documents_list), total=len(queries), desc="Reranking docs"):
            reranked_docs = self.rerank(query, documents)
            reranked_results.append(reranked_docs)
        
        return reranked_results
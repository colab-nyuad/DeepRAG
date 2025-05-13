import json
from typing import List, Optional, Any, Tuple
from dataclasses import dataclass
from tqdm import tqdm
from utils import ModelManager
from utils.prompts import SUBQUERY_ANSWER_SYSTEM_PROMPT, SUBQUERY_ANSWER_FROM_DOCS_USER_PROMPT

class SubqueryAnswerer:
    """
    A class to generate document answers for subqueries using a language model.
    Each subquery gets a separate "document" answer based on retrieved documents
    that will later be used by the reader.
    """
    
    def __init__(self, 
                 provider: str = 'ollama', 
                 model_name: str = None, 
                 temperature: float = 0.7,
                 max_workers: Optional[int] = None,
                 batch_size: int = 10,
                 verbose: bool = False):
        """
        Initialize the subquery answerer.
        
        Args:
            provider (str, optional): Model provider name. Defaults to 'ollama'.
            model_name (str, optional): Specific model to use.
            temperature (float, optional): Sampling temperature for model generation.
            max_workers (int, optional): Number of parallel workers for processing.
            batch_size (int, optional): Number of queries to process in a batch.
            verbose (bool, optional): Whether to print verbose output.
        """
        self.model_manager = ModelManager(provider, model_name, temperature)
        self.max_workers = max_workers
        self.batch_size = batch_size
        self.verbose = verbose
    
    def _parse_response(self, response: Any) -> str:
        """
        Parse the JSON response from the model.
        
        Args:
            response (Any): Raw model response
        
        Returns:
            str: Parsed document answer
        
        Raises:
            ValueError: If response cannot be parsed
        """
        try:
            parsed_response = json.loads(response)
            
            # Validate parsed response
            if not isinstance(parsed_response, dict):
                raise ValueError("Response is not a dictionary")
            
            # Ensure required keys exist
            if 'document_answer' not in parsed_response:
                raise ValueError("No 'document_answer' key in response")
            
            # Ensure 'document_answer' is a str
            if not isinstance(parsed_response['document_answer'], str):
                raise ValueError('document_answer not a string')
            
            return parsed_response['document_answer']
        except Exception as e:
            # If we can't parse as JSON, just return the raw text as the document answer
            if isinstance(response, str) and len(response.strip()) > 0:
                if self.verbose:
                    print(f"Couldn't parse response as JSON, using raw text: {e}")
                return response.strip()
            else:
                raise ValueError(f"Error parsing response: {e}")
    
    def generate_answer_from_docs(self, 
                                 query_docs: Tuple[str, List[Tuple[str, str]]], 
                                 max_retries: int = 2) -> str:
        """
        Generate a document answer for a subquery based on retrieved documents.
        
        Args:
            query_docs (Tuple[str, List[Tuple[str, str]]]): Tuple of (subquery, list of (doc_id, text))
            max_retries (int, optional): Number of retry attempts
        
        Returns:
            str: Generated document answer
        """
        subquery, documents = query_docs
        
        # Format documents into a string
        documents_text = "\n\n".join([f"Document {i+1}: {text}" 
                                      for i, (doc_id, text) in enumerate(documents)])
        
        # Prepare formatted prompt
        formatted_user_prompt = SUBQUERY_ANSWER_FROM_DOCS_USER_PROMPT.format(
            subquery=subquery,
            documents=documents_text
        )
        
        if self.verbose:
            print(f"Generating document for subquery from retrieved docs: {subquery}")

        for attempt in range(max_retries + 1):
            try:
                # Generate answer using the model
                response = self.model_manager.call_model(
                    system_prompt=SUBQUERY_ANSWER_SYSTEM_PROMPT,
                    user_prompt=formatted_user_prompt
                )
                
                # Parse response
                document_answer = self._parse_response(response)
                
                if self.verbose:
                    print(f"Generated document from docs: {document_answer[:100]}...")
                
                return document_answer
            
            except Exception as e:
                print(f"Attempt {attempt + 1} failed: {e}")
        
        # If all attempts fail, return a default answer
        default_answer = f"No information could be found in the documents for the subquery: {subquery}"
        return default_answer
    
    def batch_generate_answers_from_docs(self, 
                                       subqueries: List[str],
                                       documents_list: List[List[Tuple[str, str]]],
                                       max_retries: int = 1) -> List[str]:
        """
        Generate document answers for multiple subqueries based on retrieved documents.
        
        Args:
            subqueries (List[str]): List of subqueries
            documents_list (List[List[Tuple[str, str]]]): List of document lists for each subquery
            max_retries (int, optional): Number of retry attempts
        
        Returns:
            List[str]: List of generated document answers
        """
        all_results = []
        
        # Ensure input lists have the same length
        if len(subqueries) != len(documents_list):
            raise ValueError("Number of subqueries and document lists must match")
        
        num_batches = (len(subqueries) + self.batch_size - 1) // self.batch_size
        
        for i in tqdm(range(0, len(subqueries), self.batch_size), 
                      total=num_batches, 
                      desc="Generating Document Answers from Retrieved Docs"):
            batch_subqueries = subqueries[i:i + self.batch_size]
            batch_documents = documents_list[i:i + self.batch_size]
            
            # Prepare input for processing
            query_documents = list(zip(batch_subqueries, batch_documents))
            
            # Process query-document pairs sequentially
            for qd in query_documents:
                result = self.generate_answer_from_docs(qd, max_retries)
                all_results.append(result)
        
        return all_results 
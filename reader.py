import json
from typing import List, Optional, Any, Tuple
from dataclasses import dataclass
from tqdm import tqdm
from utils import ModelManager
from utils import QA_SYSTEM_PROMPT, QA_USER_PROMPT

@dataclass
class Answer:
    """
    Dataclass to represent an answer to a query.
    
    Attributes:
        original_query (str): The original user query
        answer (str): Generated answer text
        document_id (Optional[str]): ID of the source document
        document_content (Optional[str]): Full content of the source document
    """
    original_query: str
    answer: str
    document_id: Optional[str] = None
    document_content: Optional[str] = None

class DocumentReader:
    """
    A class to generate answers from documents using a language model.
    """
    
    def __init__(self, 
                 provider: str = 'ollama', 
                 model_name: str = None, 
                 temperature: float = 0.8,
                 max_workers: Optional[int] = None,
                 batch_size: int = 10,
                 verbose: bool = True):
        """
        Initialize the document reader.
        
        Args:
            provider (str, optional): Model provider name. Defaults to 'ollama'.
            model_name (str, optional): Specific model to use.
            temperature (float, optional): Sampling temperature for model generation.
            batch_size (int, optional): Number of queries to process in a batch.
        """
        self.model_manager = ModelManager(provider, model_name, temperature)
        self.max_workers = max_workers
        self.batch_size = batch_size
        self.verbose = verbose
    
    def _parse_response(self, response: Any) -> dict:
        """
        Parse the JSON response from the model.
        
        Args:
            response (Any): Raw model response
        
        Returns:
            dict: Parsed answer data
        
        Raises:
            ValueError: If response cannot be parsed
        """
        try:
            parsed_response = json.loads(response)
            
            # Validate parsed response
            if not isinstance(parsed_response, dict):
                raise ValueError("Response is not a dictionary")
            
            # Ensure required keys exist
            if 'answer' not in parsed_response:
                raise ValueError("No 'answer' key in response")
            
            # Ensure 'answer' is a str
            if type(parsed_response['answer']) != str:
                raise ValueError('Answer not a string')
            
            return parsed_response
        except Exception as e:
            raise ValueError(f"Error parsing response: {e}")
    
    def generate_answer(self, 
                        query_documents: Tuple[str, List[Tuple[str, str]]], 
                        max_retries: int = 2) -> Answer:
        """
        Generate an answer for a single query.
        
        Args:
            query_documents (Tuple[str, List[Tuple[str, str]]]): Tuple of query and list of (doc_id, text)
            max_retries (int, optional): Number of retry attempts
        
        Returns:
            Answer: Generated answer object
        """
        query, documents = query_documents
        expanded = query.expanded_queries
        query = query.original_query
        
        # Format documents into a string
        documents_text = "\n\n".join([f"Document content: {text}" 
                                      for i, (doc_id, text) in enumerate(documents)])
        
        # Prepare formatted prompt
        formatted_user_prompt = QA_USER_PROMPT.format(
            query=query, 
            expanded = expanded,
            documents=documents_text
        )
        
        if self.verbose:
            print("Prompt :" , formatted_user_prompt)

        for attempt in range(max_retries + 1):
            try:
                # Generate answer using the model
                response = self.model_manager.call_model(
                    system_prompt=QA_SYSTEM_PROMPT,
                    user_prompt=formatted_user_prompt
                )
                
                # Parse response
                parsed_response = self._parse_response(response)
                
                # Find the document with the matching document_id if specified
                document_content = None
                document_id = parsed_response.get('document_id')
                if document_id:
                    for doc_id, text in documents:
                        if str(doc_id) == str(document_id):
                            document_content = text
                            break
                
                return Answer(
                    original_query=query,
                    answer=parsed_response.get('answer', ''),
                    document_id=document_id,
                    document_content=document_content
                )
            
            except Exception as e:
                print(f"Attempt {attempt + 1} failed: {e}")
        
        # If all attempts fail, return a default answer
        return Answer(
            original_query=query,
            answer="Sorry, I couldn't generate an answer based on the provided documents.",
            document_id=None,
            document_content=None
        )
    
    def batch_generate(self, 
                       queries: List[str], 
                       documents_list: List[List[Tuple[str, str]]], 
                       max_retries: int = 1) -> List[Answer]:
        """
        Generate answers for multiple queries sequentially.
        
        Args:
            queries (List[str]): List of user queries
            documents_list (List[List[Tuple[str, str]]]): List of document lists
            max_retries (int, optional): Number of retry attempts
        
        Returns:
            List[Answer]: List of generated answers
        """
        all_results = []
        num_batches = (len(queries) + self.batch_size - 1) // self.batch_size
        
        for i in tqdm(range(0, len(queries), self.batch_size), 
                      total=num_batches, 
                      desc="Generating Answers"):
            batch_queries = queries[i:i + self.batch_size]
            batch_documents = documents_list[i:i + self.batch_size]
            
            # Prepare input for sequential processing
            query_documents = list(zip(batch_queries, batch_documents))
            
            # Process queries sequentially
            for qd in query_documents:
                result = self.generate_answer(qd, max_retries)
                all_results.append(result)
        
        return all_results
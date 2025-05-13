import json
from typing import List, Optional, Any, Tuple, Set
from dataclasses import dataclass
from tqdm import tqdm
from utils import ModelManager
from utils.prompts import (LLM_RERANKER_SYSTEM_PROMPT, 
                          LLM_RERANKER_SELECT_BEST_PROMPT, 
                          LLM_RERANKER_SELECT_TOP_K_PROMPT,
                          LLM_SUBQUERY_VALIDATOR_SYSTEM_PROMPT,
                          LLM_SUBQUERY_VALIDATOR_USER_PROMPT,
                          LLM_DOC_SUMMARY_SYSTEM_PROMPT,
                          LLM_DOC_SUMMARY_USER_PROMPT)

class LLMReranker:
    """
    A class that uses an LLM to select the most relevant documents for queries and subqueries.
    """
    
    def __init__(self, 
                 provider: str = 'ollama', 
                 model_name: str = None, 
                 temperature: float = 0.2,  # Lower temperature for more deterministic selections
                 max_workers: Optional[int] = None,
                 batch_size: int = 10,
                 verbose: bool = False):
        """
        Initialize the LLM reranker.
        
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
    
    def _parse_best_doc_response(self, response: Any) -> Optional[int]:
        """
        Parse the JSON response from the model to get the best document index.
        
        Args:
            response (Any): Raw model response
        
        Returns:
            Optional[int]: Index of the best document or None if parsing failed
        """
        try:
            if isinstance(response, str):
                # Try to extract JSON if it's embedded in text
                import re
                json_match = re.search(r'\{.*\}', response, re.DOTALL)
                if json_match:
                    response = json_match.group(0)
            
            parsed_response = json.loads(response)
            
            # Validate parsed response
            if not isinstance(parsed_response, dict):
                raise ValueError("Response is not a dictionary")
            
            # Ensure required keys exist
            if 'best_document_index' not in parsed_response:
                # Check for alternative keys
                if 'document_index' in parsed_response:
                    return parsed_response['document_index']
                else:
                    raise ValueError("No index key in response")
            
            return parsed_response['best_document_index']
        except Exception as e:
            if self.verbose:
                print(f"Error parsing response: {e}")
                print(f"Raw response: {response}")
            
            # Try to extract just the number if JSON parsing failed
            if isinstance(response, str):
                import re
                number_match = re.search(r'\b\d+\b', response)
                if number_match:
                    try:
                        index = int(number_match.group(0))
                        if self.verbose:
                            print(f"Extracted index from text: {index}")
                        return index
                    except:
                        pass
            
            return None
    
    def _parse_top_k_response(self, response: Any) -> List[int]:
        """
        Parse the JSON response from the model to get the indices of the top k documents.
        
        Args:
            response (Any): Raw model response
        
        Returns:
            List[int]: List of indices of the top k documents
        """
        try:
            if isinstance(response, str):
                # Try to extract JSON if it's embedded in text
                import re
                json_match = re.search(r'\{.*\}', response, re.DOTALL)
                if json_match:
                    response = json_match.group(0)
            
            parsed_response = json.loads(response)
            
            # Validate parsed response
            if not isinstance(parsed_response, dict):
                raise ValueError("Response is not a dictionary")
            
            # Ensure required keys exist
            if 'top_document_indices' not in parsed_response:
                # Check for alternative keys
                if 'document_indices' in parsed_response:
                    indices = parsed_response['document_indices']
                else:
                    raise ValueError("No document indices key in response")
            else:
                indices = parsed_response['top_document_indices']
            
            # Validate that indices are a list of integers
            if not isinstance(indices, list):
                raise ValueError("Document indices is not a list")
            
            # Convert to integers if they're strings
            indices = [int(idx) if isinstance(idx, str) else idx for idx in indices]
            
            return indices
        except Exception as e:
            if self.verbose:
                print(f"Error parsing response: {e}")
                print(f"Raw response: {response}")
            
            # Try to extract just the numbers if JSON parsing failed
            if isinstance(response, str):
                import re
                numbers = re.findall(r'\b\d+\b', response)
                if numbers:
                    try:
                        indices = [int(num) for num in numbers]
                        if self.verbose:
                            print(f"Extracted indices from text: {indices}")
                        return indices
                    except:
                        pass
            
            return []
    
    def select_best_document(self, query: str, documents: List[Tuple[str, str]], max_retries: int = 2) -> Optional[Tuple[str, str]]:
        """
        Select the best document for a query from a list of retrieved documents.
        
        Args:
            query (str): The query to evaluate documents against
            documents (List[Tuple[str, str]]): List of (doc_id, text) tuples
            max_retries (int, optional): Number of retry attempts
        
        Returns:
            Optional[Tuple[str, str]]: The selected (doc_id, text) tuple or None if selection failed
        """
        if not documents:
            return None
        
        # Format documents into a string with indexes
        formatted_docs = "\n\n".join([f"Document {i}:\n{text}" 
                                     for i, (doc_id, text) in enumerate(documents)])
        
        # Calculate max index
        max_index = len(documents) - 1
        
        # Prepare formatted prompt
        formatted_user_prompt = LLM_RERANKER_SELECT_BEST_PROMPT.format(
            query=query,
            documents=formatted_docs,
            num_documents=len(documents),
            max_index=max_index
        )
        
        if self.verbose:
            print(f"Selecting best document for query: {query}")

        for attempt in range(max_retries + 1):
            try:
                # Generate answer using the model
                response = self.model_manager.call_model(
                    system_prompt=LLM_RERANKER_SYSTEM_PROMPT,
                    user_prompt=formatted_user_prompt
                )
                
                # Parse response to get the best document index
                best_index = self._parse_best_doc_response(response)
                
                if best_index is not None and 0 <= best_index < len(documents):
                    if self.verbose:
                        print(f"Selected document index: {best_index}")
                    return documents[best_index]
                else:
                    print(f"Invalid document index: {best_index}. Expected 0-{max_index}")
            
            except Exception as e:
                print(f"Attempt {attempt + 1} failed: {e}")
        
        # If all attempts fail, return the first document as a fallback
        print(f"All attempts to select the best document failed. Returning first document as fallback.")
        return documents[0] if documents else None
    
    def select_top_r_documents(self, query: str, documents: List[Tuple[str, str, int]], top_r: int, max_retries: int = 2) -> List[Tuple[str, str]]:
        """
        Select the top r most relevant document summaries for a query.
        
        Args:
            query (str): The query to evaluate document summaries against
            documents (List[Tuple[str, str, int]]): List of (doc_id, summary, frequency) tuples
            top_r (int): Number of summaries to select
            max_retries (int, optional): Number of retry attempts
        
        Returns:
            List[Tuple[str, str]]: List of selected (doc_id, summary) tuples
        """
        if not documents:
            return []
        
        # If asking for more documents than available, return all (formatted correctly)
        if top_r >= len(documents):
            # Return (doc_id, summary) tuples
            return [(doc_id, summary) for doc_id, summary, freq in documents]
        
        # Format documents into a string with indexes, frequency, and SUMMARY
        formatted_docs = "\n\n".join([f"Document {i} (Frequency: {freq}):\n{summary}" 
                                     for i, (doc_id, summary, freq) in enumerate(documents)])
        
        # Calculate max index
        max_index = len(documents) - 1
        
        # Prepare formatted prompt
        formatted_user_prompt = LLM_RERANKER_SELECT_TOP_K_PROMPT.format(
            query=query,
            documents=formatted_docs, # Now contains summaries
            num_documents=len(documents),
            top_r=top_r,
            max_index=max_index
        )
        
        if self.verbose:
            print(f"Selecting top {top_r} document summaries for query: {query}")

        for attempt in range(max_retries + 1):
            try:
                # Generate answer using the model
                response = self.model_manager.call_model(
                    system_prompt=LLM_RERANKER_SYSTEM_PROMPT,
                    user_prompt=formatted_user_prompt
                )
                
                # Parse response to get the top k document indices
                selected_indices = self._parse_top_k_response(response)
                
                # Validate indices and remove duplicates while preserving order
                valid_indices = [idx for idx in selected_indices if 0 <= idx < len(documents)]
                unique_indices = list(dict.fromkeys(valid_indices))  # Preserve order while removing duplicates
                
                if len(unique_indices) > 0:
                    # If we got fewer than requested, add more documents (based on original order)
                    if len(unique_indices) < top_r:
                        # Add documents not already selected, up to top_r
                        remaining = [i for i in range(len(documents)) if i not in unique_indices]
                        unique_indices.extend(remaining[:top_r - len(unique_indices)])
                    
                    # Limit to top_r
                    unique_indices = unique_indices[:top_r]
                    
                    # Get the selected documents, returning (doc_id, summary) tuples
                    selected_docs = [(documents[idx][0], documents[idx][1]) for idx in unique_indices]
                    
                    if self.verbose:
                        print(f"Selected document indices: {unique_indices}")
                    
                    return selected_docs
                else:
                    print(f"No valid document indices returned")
            
            except Exception as e:
                print(f"Attempt {attempt + 1} failed: {e}")
        
        # If all attempts fail, return the first top_r documents as a fallback
        print(f"All attempts to select the top {top_r} document summaries failed. Returning first {top_r} summaries as fallback.")
        # Return (doc_id, summary) tuples for the fallback
        return [(doc_id, summary) for doc_id, summary, freq in documents[:top_r]]
    
    # Maintain backwards compatibility
    def select_best_documents(self, query: str, documents: List[Tuple[str, str]], top_r: int, max_retries: int = 2) -> List[Tuple[str, str]]:
        """
        Backward compatibility method that calls select_top_r_documents.
        
        Args:
            query (str): The query to evaluate documents against
            documents (List[Tuple[str, str]]): List of (doc_id, text) tuples
            top_r (int): Number of documents to select
            max_retries (int, optional): Number of retry attempts
        
        Returns:
            List[Tuple[str, str]]: List of selected (doc_id, text) tuples
        """
        return self.select_top_r_documents(query, documents, top_r, max_retries)
    
    def batch_select_best_documents(self, 
                                   queries: List[str],
                                   documents_list: List[List[Tuple[str, str]]],
                                   max_retries: int = 1) -> List[Optional[Tuple[str, str]]]:
        """
        Batch select the best document for each query.
        
        Args:
            queries (List[str]): List of queries
            documents_list (List[List[Tuple[str, str]]]): List of document lists for each query
            max_retries (int, optional): Number of retry attempts
        
        Returns:
            List[Optional[Tuple[str, str]]]: List of selected documents (or None for failures)
        """
        all_results = []
        
        # Ensure input lists have the same length
        if len(queries) != len(documents_list):
            raise ValueError("Number of queries and document lists must match")
        
        num_batches = (len(queries) + self.batch_size - 1) // self.batch_size
        
        for i in tqdm(range(0, len(queries), self.batch_size), 
                      total=num_batches, 
                      desc="Selecting Best Documents"):
            batch_queries = queries[i:i + self.batch_size]
            batch_documents = documents_list[i:i + self.batch_size]
            
            # Process query-document pairs sequentially
            for query, docs in zip(batch_queries, batch_documents):
                result = self.select_best_document(query, docs, max_retries)
                all_results.append(result)
        
        return all_results 
    
    def does_document_answer_subquery(self, subquery: str, document_text: str, max_retries: int = 1) -> bool:
        """
        Use LLM to check if a document unambiguously answers a specific subquery.
        
        Args:
            subquery (str): The specific subquery.
            document_text (str): The text content of the document.
            max_retries (int, optional): Number of retry attempts.
        
        Returns:
            bool: True if the document is judged to answer the subquery, False otherwise.
        """
        formatted_user_prompt = LLM_SUBQUERY_VALIDATOR_USER_PROMPT.format(
            subquery=subquery,
            document_text=document_text
        )

        if self.verbose:
            print(f"Validating if document answers subquery: '{subquery[:50]}...'")

        for attempt in range(max_retries + 1):
            try:
                response = self.model_manager.call_model(
                    system_prompt=LLM_SUBQUERY_VALIDATOR_SYSTEM_PROMPT,
                    user_prompt=formatted_user_prompt
                )
                
                # Parse response (basic JSON parsing, expecting {"answers_subquery": true/false})
                parsed_response = json.loads(response)
                if isinstance(parsed_response, dict) and 'answers_subquery' in parsed_response:
                    result = parsed_response['answers_subquery']
                    if isinstance(result, bool):
                        if self.verbose:
                            print(f"  -> Validation result: {result}")
                        return result
                    else:
                         print(f"  -> Unexpected value type for 'answers_subquery': {type(result)}")
                else:
                    print(f"  -> Unexpected response format: {response}")

            except Exception as e:
                print(f"  -> Attempt {attempt + 1} failed during validation: {e}")
                if attempt == max_retries:
                    print(f"  -> Validation failed after {max_retries + 1} attempts. Assuming False.")
                    return False # Default to False on repeated errors
        
        return False # Should not be reached if retry logic is correct, but default to False 

    def summarize_document(self, query: str, document_text: str, max_retries: int = 1) -> str:
        """
        Generate a concise, query-focused, one-sentence summary of a document.
        
        Args:
            query (str): The query to focus the summary on.
            document_text (str): The text content of the document.
            max_retries (int, optional): Number of retry attempts.
        
        Returns:
            str: The one-sentence summary, or an empty string if no relevant info or on failure.
        """
        formatted_user_prompt = LLM_DOC_SUMMARY_USER_PROMPT.format(
            query=query,
            document_text=document_text
        )

        if self.verbose:
            print(f"Summarizing document for query: '{query[:50]}...'")

        for attempt in range(max_retries + 1):
            try:
                response = self.model_manager.call_model(
                    system_prompt=LLM_DOC_SUMMARY_SYSTEM_PROMPT,
                    user_prompt=formatted_user_prompt
                )
                
                # Parse response (expecting {"summary": "..."})
                parsed_response = json.loads(response)
                if isinstance(parsed_response, dict) and 'summary' in parsed_response:
                    summary = parsed_response['summary']
                    if isinstance(summary, str):
                        if self.verbose:
                            print(f"  -> Summary generated: '{summary[:100]}...'")
                        return summary
                    else:
                        print(f"  -> Unexpected value type for 'summary': {type(summary)}")
                else:
                    print(f"  -> Unexpected response format for summary: {response}")

            except Exception as e:
                print(f"  -> Attempt {attempt + 1} failed during summarization: {e}")
                if attempt == max_retries:
                    print(f"  -> Summarization failed after {max_retries + 1} attempts. Returning empty summary.")
                    return "" # Default to empty string on repeated errors
        
        return "" # Default to empty string 
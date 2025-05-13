import json
from typing import List, Optional,Any
from dataclasses import dataclass
from tqdm import tqdm
from multiprocessing.pool import ThreadPool
from multiprocessing import TimeoutError
from utils import QUERY_EXPANSION_USER_PROMPT, QUERY_EXPANSION_SYSTEM_PROMPT,QUERY_EXPANSION_USER_PROMPT_DECOMPOSE,ModelManager
import logging

@dataclass
class Query:
    original_query: str
    expanded_queries: List[str]

class QueryExpander:
    def __init__(self, 
                 provider: str = 'ollama', 
                 model_name: str = None, 
                 temperature: float = 0.5,
                 max_workers: Optional[int] = None,
                 batch_size: int = 10,
                 user_prompt: str = QUERY_EXPANSION_USER_PROMPT_DECOMPOSE,
                 timeout: Optional[float] = None):
        self.model_manager = ModelManager(provider, model_name, temperature)
        self.max_workers = max_workers
        self.batch_size = batch_size
        self.user_prompt = user_prompt
        self.timeout = timeout
    
    def _parse_response(self, response: Any) -> list:
        try:
            # Attempt to parse the JSON
            parsed_data = json.loads(response)
            parsed_response = parsed_data.get('queries', [])
            if not isinstance(parsed_response, list):
                # Silently raise error if format is wrong
                raise ValueError("'queries' field is not a list")
            return parsed_response
        except json.JSONDecodeError as json_err:
            # Remove logging, just raise error for retry mechanism
            raise ValueError(f"Error parsing JSON response: {json_err}") 
        except Exception as e:
            # Remove logging, just raise error for retry mechanism
            raise ValueError(f"Error parsing response: {e}")
    
    def expand_query(self, query: str, number: int = 3, max_retries: int = 1) -> Query:
        formatted_prompt = self.user_prompt.format(query=query, number=number)
        for attempt in range(max_retries + 1):
            try:
                response = self.model_manager.call_model(
                    system_prompt=QUERY_EXPANSION_SYSTEM_PROMPT,
                    user_prompt=formatted_prompt
                )
                
                # Check if response is empty/None before parsing
                if not response:
                    # Don't log, just raise error to trigger retry/failure
                    raise ValueError("Received empty response from LLM")
                    
                expanded_queries = self._parse_response(response)
                
                # If parsing succeeds and we get queries, return immediately
                if expanded_queries:
                    return Query(original_query=query, expanded_queries=expanded_queries)
                
                # If parsing succeeded but list is empty, consider it a failure for this attempt
                # (or we could return [] here - depends on desired behavior)
                # For now, treat empty list as needing retry 
                # (remove the old print statement)
                pass # Continue to except block or next iteration

            except Exception as e:
                # Silence the error print, just pass to allow retry
                # Log for debugging if needed, but remove for production
                # logging.debug(f"Attempt {attempt + 1} failed silently: {e}") 
                pass # Go to the next attempt

        # If loop finishes without returning, all attempts failed. Return original query with empty list.
        # Remove the print statement here
        return Query(original_query=query, expanded_queries=[])

    def _batch_expand(self, queries: List[str], number: int, max_retries: int, timeout: Optional[float]) -> List[Query]:
        with ThreadPool(self.max_workers) as pool:
            # Use map_async to allow for timeout
            async_results = pool.map_async(
                lambda query: self.expand_query(query, number, max_retries),
                queries
            )
            try:
                # Get results with the specified timeout
                results = async_results.get(timeout=timeout) 
            except TimeoutError:
                logging.warning(f"Query expansion batch timed out after {timeout} seconds.")
                # Return default Query objects for the entire batch on timeout
                results = [Query(original_query=q, expanded_queries=[]) for q in queries]
            except Exception as e:
                logging.error(f"Error during batch query expansion: {e}", exc_info=True)
                # Return default Query objects on other errors as well
                results = [Query(original_query=q, expanded_queries=[]) for q in queries]
        return results

    def batch_query_expansion(self, queries: List[str], number: int = 3, max_retries: int = 1, timeout: Optional[float] = None) -> List[Query]:
        all_results = []
        # Use the instance timeout if no specific timeout is passed to this method
        effective_timeout = timeout if timeout is not None else self.timeout 
        num_batches = (len(queries) + self.batch_size - 1) // self.batch_size
        for i in tqdm(range(0, len(queries), self.batch_size), total=num_batches, desc="Query Expansion", disable=True):
            batch = queries[i:i + self.batch_size]
            # Pass the effective timeout to _batch_expand
            results = self._batch_expand(batch, number, max_retries, effective_timeout)
            all_results.extend(results)
        return all_results
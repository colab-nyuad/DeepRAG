import json
from typing import List, Optional,Any
from dataclasses import dataclass
from tqdm import tqdm
from multiprocessing.pool import ThreadPool
from utils import QUERY_EXPANSION_USER_PROMPT, QUERY_EXPANSION_SYSTEM_PROMPT, ModelManager

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
                 batch_size: int = 10):
        self.model_manager = ModelManager(provider, model_name, temperature)
        self.max_workers = max_workers
        self.batch_size = batch_size
    
    def _parse_response(self, response: Any) -> list:
        try:
            parsed_response = json.loads(response).get('queries', [])
            if not isinstance(parsed_response, list):
                raise ValueError("Not a list")
            return parsed_response
        except Exception as e:
            raise ValueError(f"Error parsing response: {e}")
    
    def expand_query(self, query: str, number: int = 3, max_retries: int = 1) -> Query:
        formatted_prompt = QUERY_EXPANSION_USER_PROMPT.format(query=query, number=number)
        for attempt in range(max_retries + 1):
            try:
                response = self.model_manager.call_model(
                    system_prompt=QUERY_EXPANSION_SYSTEM_PROMPT,
                    user_prompt=formatted_prompt
                )
                expanded_queries = self._parse_response(response)
                if expanded_queries:
                    return Query(original_query=query, expanded_queries=expanded_queries)
                print(f"Attempt {attempt + 1}: No queries generated")

            except Exception as e:
                print(f"Attempt {attempt + 1} failed: {e}")

        print("All query expansion attempts failed. Returning original query.")
        return Query(original_query=query, expanded_queries=[])

    def _batch_expand(self, queries: List[str], number: int, max_retries: int) -> List[Query]:
        with ThreadPool(self.max_workers) as pool:
            results = pool.map(
                lambda query: self.expand_query(query, number, max_retries),
                queries
            )
        return results

    def batch_query_expansion(self, queries: List[str], number: int = 3, max_retries: int = 1) -> List[Query]:
        all_results = []
        num_batches = (len(queries) + self.batch_size - 1) // self.batch_size
        for i in tqdm(range(0, len(queries), self.batch_size), total=num_batches, desc="Query Expansion"):
            batch = queries[i:i + self.batch_size]
            results = self._batch_expand(batch, number, max_retries)
            all_results.extend(results)
        return all_results
import warnings
from typing import List, Literal, Dict, Any, Optional
from multiprocessing.pool import ThreadPool
from dataclasses import dataclass
from abc import ABC, abstractmethod

import boto3
import torch
import numpy as np
from pinecone import Pinecone
from transformers import AutoModel, AutoTokenizer
from opensearchpy import OpenSearch, AWSV4SignerAuth, RequestsHttpConnection
from tqdm import tqdm
from utils import RetrieverMetrics
from query_expansion import Query
from collections import defaultdict

warnings.filterwarnings('ignore')
warnings.filterwarnings('ignore', category=DeprecationWarning)

AWS_PROFILE_NAME = "sigir-participant"
AWS_REGION_NAME = "us-east-1"
OPENSEARCH_INDEX_NAME = "fineweb10bt-512-0w-e5-base-v2"
PINECONE_INDEX_NAME = "fineweb10bt-512-0w-e5-base-v2"
PINECONE_NAMESPACE = "default"

@dataclass
class Doc:
    """Container for a single retrieved document."""
    doc_id: str
    score: float
    text: str
    chunk_id: Optional[str] = None
    metadata: Dict[str, Any] = None

class AWSUtils:
    """Utility class for AWS operations."""
    
    @staticmethod
    def get_ssm_value(key: str, profile: str = AWS_PROFILE_NAME, region: str = AWS_REGION_NAME) -> str:
        """Get a cleartext value from AWS SSM."""
        session = boto3.Session(profile_name=profile, region_name=region)
        ssm = session.client("ssm")
        return ssm.get_parameter(Name=key)["Parameter"]["Value"]

    @staticmethod
    def get_ssm_secret(key: str, profile: str = AWS_PROFILE_NAME, region: str = AWS_REGION_NAME):
        """Get an encrypted value from AWS SSM."""
        session = boto3.Session(profile_name=profile, region_name=region)
        ssm = session.client("ssm")
        return ssm.get_parameter(Name=key, WithDecryption=True)["Parameter"]["Value"]


class EmbeddingModel:
    """Class for embedding text using transformer models."""
    
    def __init__(self, model_name: str = "intfloat/e5-base-v2"):
        self.model_name = model_name
        self.tokenizer = self._get_tokenizer()
        self.model = self._get_model()
    
    @property
    def device(self):
        if torch.backends.mps.is_available():
            return "mps"
        elif torch.cuda.is_available():
            return "cuda"
        else:
            return "cpu"
    
    def _get_tokenizer(self):
        """Load the tokenizer."""
        return AutoTokenizer.from_pretrained(self.model_name)
    
    def _get_model(self):
        """Load the model and move to appropriate device."""
        model = AutoModel.from_pretrained(self.model_name, trust_remote_code=True)
        return model.to(self.device)
    
    def _average_pool(self, last_hidden_states: torch.Tensor, attention_mask: torch.Tensor) -> torch.Tensor:
        """Apply average pooling on transformer outputs."""
        last_hidden = last_hidden_states.masked_fill(~attention_mask[..., None].bool(), 0.0)
        return last_hidden.sum(dim=1) / attention_mask.sum(dim=1)[..., None]
    
    def embed_query(self, query: str, query_prefix: str = "query: ", 
                    pooling: Literal["cls", "avg"] = "avg", normalize: bool = True) -> List[float]:
        """Embed a single query."""
        return self.batch_embed_queries([query], query_prefix, pooling, normalize)[0]
    
    def batch_embed_queries(self, queries: List[str], query_prefix: str = "query: ", 
                            pooling: Literal["cls", "avg"] = "avg", normalize: bool = True) -> List[List[float]]:
        """Embed a batch of queries."""
        with_prefixes = [" ".join([query_prefix, query]) for query in queries]
        
        with torch.no_grad():
            encoded = self.tokenizer(with_prefixes, padding=True, return_tensors="pt", truncation="longest_first")
            encoded = encoded.to(self.model.device)
            model_out = self.model(**encoded)
            
            if pooling == "cls":
                embeddings = model_out.last_hidden_state[:, 0]
            elif pooling == "avg":
                embeddings = self._average_pool(model_out.last_hidden_state, encoded["attention_mask"])
            else:
                raise ValueError(f"Unsupported pooling method: {pooling}")
                
            if normalize:
                embeddings = torch.nn.functional.normalize(embeddings, p=2, dim=1)
                
        return embeddings.tolist()

class BaseRetriever(ABC):
    """
    Abstract base class for retrievers
    
    Defines the interface and provides common methods for document retrieval
    """

    @abstractmethod
    def retrieve(self, query: str, top_k: int = 3) -> List[Doc]:
        """
        Abstract method to retrieve documents for a single query.
        
        Args:
            query (str): Input query string
            top_k (int, optional): Number of documents to retrieve. Defaults to 3.
        
        Returns:
            List[Doc]: Retrieved documents
        """
        pass
    
    @abstractmethod
    def batch_retrieve(self, queries: List[str], top_k: int = 3, n_parallel: int = 3) -> List[List[Doc]]:
        """
        Abstract method to retrieve documents for multiple queries.
        
        Args:
            queries (List[str]): List of input query strings
            top_k (int, optional): Number of documents to retrieve per query. Defaults to 3.
            n_parallel (int, optional): Number of parallel threads. Defaults to 3.
        
        Returns:
            List[List[Doc]]: Retrieved documents for each query
        """
        pass
    
    def filter_unique_doc_ids(self, docs: List[Doc], top_n: int = 100) -> List[str]:
        """
        Filter and return unique document IDs from a list of retrieved documents.
        
        Args:
            docs (List[Doc]): List of retrieved documents
            top_n (int, optional): Maximum number of unique IDs to return. Defaults to 100.
        
        Returns:
            List[str]: List of unique document IDs
        """
        unique_ids_set , unique_ids_list = set(), []
        for doc in docs:
            if doc.doc_id not in unique_ids_set:
                unique_ids_set.add(doc.doc_id)
                unique_ids_list.append(doc.doc_id)
            # if len(unique_ids_list) == top_n:
            #     break
        
        # Return top_n unique IDs or all if less than top_n
        return unique_ids_list
    
    def retrieve_unique_ids(self, query: str, top_k: int = 3) -> List[str]:
        """
        Retrieve documents and return unique document IDs.
        
        Args:
            query (str): Input query string
            top_k (int, optional): Number of documents to retrieve. Defaults to 3.
        
        Returns:
            List[str]: List of unique document IDs
        """
        docs = self.retrieve(query, top_k)
        return self.filter_unique_doc_ids(docs)
    
    def batch_retrieve_unique_ids(self, queries: List[str], top_k: int = 3) -> List[List[str]]:
        """
        Batch retrieve documents and return unique document IDs for each query.
        
        Args:
            queries (List[str]): List of input query strings
            top_k (int, optional): Number of documents to retrieve per query. Defaults to 3.
            n_parallel (int, optional): Number of parallel threads. Defaults to 3.
        
        Returns:
            List[List[str]]: List of unique document IDs for each query
        """
        batch_docs = self.batch_retrieve(queries, top_k)
        return [self.filter_unique_doc_ids(docs) for docs in batch_docs]
    
    def batch_retrieve_with_expansion(self, queries: List[Query], top_k: int =3):
        # Prepare all queries (original + expanded)
        all_batch_queries = []
        len_of_each_query = []

        for query in queries:
            # Combine original and expanded queries
            combined_queries = [query.original_query] + query.expanded_queries
            all_batch_queries.extend(combined_queries)
            len_of_each_query.append(len(combined_queries))  # Track how many sub-queries per original query

        # Batch retrieve for all queries
        batch_docs = self.batch_retrieve(all_batch_queries, top_k)

        # Group retrieved docs for each original query (flattened per query)
        grouped_docs = []
        idx = 0
        for count in len_of_each_query:
            # Combine all docs (from original + expansions) for one original query
            docs_for_query = []
            for subquery_docs in batch_docs[idx:idx + count]:
                docs_for_query.extend(subquery_docs)
            grouped_docs.append(docs_for_query)
            idx += count

        return grouped_docs  # List[List[Doc]], one list of docs per original query

    def batch_retrieve_unique_ids_with_expansion(self, queries: List[Query], top_k: int = 3) -> List[List[str]]:
        """
        Batch retrieve documents with expansion and return unique document IDs for each query.
        
        Args:
            queries (List[Query]): List of queries with original and expanded queries
            top_k (int, optional): Number of documents to retrieve per query. Defaults to 3.
        
        Returns:
            List[List[str]]: List of unique document IDs for each query
        """
        # Retrieve documents with expansion
        batch_docs = self.batch_retrieve_with_expansion(queries, top_k)
        
        # Filter unique document IDs for each query
        return [self.filter_unique_doc_ids(docs) for docs in batch_docs]
    
        
    
    def group_and_concatenate_by_doc_id(self, docs: List[Doc]) -> List[Doc]:
        """
        Group documents by doc_id, sort by chunk_id, and concatenate their text.
        
        Args:
            docs (List[Doc]): List of retrieved documents
        
        Returns:
            List[Doc]: List of unique documents with concatenated text
        """
        # Group documents by doc_id
        grouped_docs = defaultdict(list)
        for doc in docs:
            grouped_docs[doc.doc_id].append(doc)
        
        # Sort each group by chunk_id and concatenate text
        result = []
        for doc_id, doc_group in grouped_docs.items(): 
            # Sort by chunk_id (assuming chunk_id contains ordering information)
            sorted_docs = sorted(doc_group, key=lambda x: x.chunk_id)
            
            # Concatenate text from all chunks
            combined_text = "".join(doc.text for doc in sorted_docs)
            
            # Use the highest score among all chunks for this document
            max_score = max(doc.score for doc in sorted_docs)
            
            # Create a new Doc with combined information
            combined_doc = Doc(
                doc_id=doc_id,
                score=max_score,
                text=combined_text,
                # chunk_id=f"{doc_id}-combined",  # Create a new identifier for the combined document
                # metadata=sorted_docs[0].metadata  # Keep metadata from first chunk (or could merge)
            )
            
            result.append(combined_doc)
        
        # Sort results by score in descending order
        return sorted(result, key=lambda x: x.score, reverse=True)

    
    def batch_retrieve_grouped_with_expansion(self, queries: List[Query], top_k: int = 3) -> List[List[Doc]]:
        """
        Batch retrieve documents with query expansion, then group by doc_id and concatenate text.
        
        Args:
            queries (List[Query]): List of queries with original and expanded queries
            top_k (int, optional): Number of documents to retrieve per query. Defaults to 3.
        
        Returns:
            List[List[Doc]]: List of grouped and concatenated documents for each query
        """
        # First retrieve documents with expansion
        batch_docs = self.batch_retrieve_with_expansion(queries, top_k)
        
        # Then group and concatenate results
        return [self.group_and_concatenate_by_doc_id(docs) for docs in batch_docs]
    
    def get_predicted_ids(self, predicted_docs: List[List[Doc]]) -> List[List[str]]:
        """
        Extract predicted document IDs from retrieved documents.
        
        Args:
            predicted_docs (List[List[Doc]]): List of retrieved documents for each query
        
        Returns:
            List[List[str]]: List of predicted document IDs for each query
        """
        return [[doc.doc_id for doc in docs] for docs in predicted_docs]
            
    
    def evaluate(self, predicted_ids: List[List[Doc]], ground_truth_ids: List[List[str]], k_values: List[int]=[1,3,5,10]):
        """
        Evaluate the retriever on a dataset.

        Args:
            predicted_ids (List[List[Doc]]): List of retrieved documents for each query
            ground_truth_ids (List[List[str]]): List of ground truth document IDs for each query
        """
        metrics = RetrieverMetrics.calculate_metrics(predicted_ids, ground_truth_ids, k_values)
        return  metrics
        
class PineconeRetriever(BaseRetriever):
    """Retriever using Pinecone vector database."""
    
    def __init__(self, 
                 embedding_model: EmbeddingModel = EmbeddingModel(),
                 index_name: str = PINECONE_INDEX_NAME, 
                 namespace: str = PINECONE_NAMESPACE,
                 max_workers: int = 10,
                 batch_size: int = 10):
        self.embedding_model = embedding_model
        self.index_name = index_name
        self.namespace = namespace
        self.index = self._get_index()
        self.max_workers = max_workers
        self.batch_size = batch_size
        self.doc_id_to_text = {}
    
    def _get_index(self):
        """Connect to Pinecone index."""
        pc = Pinecone(api_key=AWSUtils.get_ssm_secret("/pinecone/ro_token"))
        return pc.Index(name=self.index_name)
    
    def get_vector_count(self):
        return self.index.describe_index_stats().get('total_vector_count', 0)

    def retrieve_chunks_using_data(self, data:List[Dict[str, Any]]):
        """Retrieve documents by their IDs."""
        for doc in data:
            chunks = []
            doc_id = doc["document_ids"][0]
            id_data =  sorted([ids for ids in self.index.list(prefix=f'doc-{doc_id}', namespace=self.namespace)][0])
            fetch_data = self.index.fetch(ids= id_data, namespace=self.namespace)['vectors']
            for id in id_data:
                chunks.append(fetch_data[id]['metadata']['text'])

            # update data with keys chunks_id and chunks
            doc['chunk_ids'] = id_data
            doc['chunks'] = chunks

        return data
    
    def retrieve_chunks_by_id(self, query_ids:List[List[str]], first_k:int = 100) -> List[List[Doc]]:
        """Retrieve all documents chunks by their IDs."""
        # Flatten the query_ids list and remove duplicates
        unique_ids = set(id for sublist in query_ids for id in sublist)
        for query_id in tqdm(unique_ids,total=len(unique_ids), desc="Retrieving chunks by ID"):
            if query_id in self.doc_id_to_text:
                continue
            chunks = []
            id_lists = [ids for ids in self.index.list(prefix=f'doc-{query_id}', namespace=self.namespace)]
            flattened_ids = sorted([item for sublist in id_lists for item in sublist])
            if len(flattened_ids) == 0:
                continue
            fetch_data = self.index.fetch(ids= flattened_ids, namespace=self.namespace)['vectors']
            for id in flattened_ids:
                chunks.append(fetch_data[id]['metadata']['text'])
            self.doc_id_to_text[query_id]= "".join(chunks)
        
        all_docs = []
        for sublist in query_ids:
            docs = [(id, self.doc_id_to_text.get(id, "")) for id in sublist]
            all_docs.append(docs)

        return all_docs

    def retrieve(self, query: str, top_k: int = 3) -> List[Doc]:
        """Retrieve documents for a single query."""
        vector = self.embedding_model.embed_query(query)
        results = self.index.query(
            vector=vector,
            top_k=top_k,
            include_values=False,
            namespace=self.namespace,
            include_metadata=True
        )
        
        docs = []
        for match in results["matches"]:
            docs.append(Doc(
                doc_id=match["metadata"].get("doc_id", ""),
                chunk_id=match["id"],
                score=match["score"],
                text=match["metadata"].get("text", ""),
                # metadata=match["metadata"]
            ))
        
        return docs

    def batch_retrieve(self, queries: List[str], top_k: int = 3) -> List[List[Doc]]:
        """
        Retrieve documents for a batch of queries using concurrent processing.
        
        Args:
            queries (List[str]): List of queries to retrieve documents for
            top_k (int, optional): Number of top documents to retrieve per query. Defaults to 3.
        Returns:
            List[List[Doc]]: List of retrieved documents for each query
        """
        # Initialize results list
        all_batch_docs = []
        
        # Create tqdm progress bar for the entire process
        progress_bar = tqdm(total=len(queries), desc="Retrieving documents(Using Pinecone)")
        
        # Process queries in batches
        for i in range(0, len(queries), self.batch_size):
            # Get current batch of queries
            batch_queries = queries[i:i+self.batch_size]
            embeddings = self.embedding_model.batch_embed_queries(batch_queries)
            
            # Use ThreadPool for parallel querying
            with ThreadPool(self.max_workers) as pool:
                results = pool.map(
                    lambda embed: self.index.query(
                        vector=embed, 
                        top_k=top_k, 
                        include_values=False,
                        namespace=self.namespace,
                        include_metadata=True
                    ),
                    embeddings
                )

            all_docs = [[
                Doc(
                    doc_id=match["metadata"].get("doc_id", ""),
                    chunk_id=match["id"],
                    score=match["score"],
                    text=match["metadata"].get("text", ""),
                    # metadata=match["metadata"]
                )
                for match in result["matches"]
            ] for result in results]

            progress_bar.update(len(batch_queries))
            all_batch_docs.extend(all_docs)
        
        # Close the progress bar
        progress_bar.close()
        
        return all_batch_docs

class OpenSearchRetriever(BaseRetriever):
    """Retriever using OpenSearch."""
    
    def __init__(self, index_name: str = OPENSEARCH_INDEX_NAME, batch_size: int = 10):
        self.index_name = index_name
        self.client = self._get_client()
        self.batch_size = batch_size
    
    def _get_client(self, profile: str = AWS_PROFILE_NAME, region: str = AWS_REGION_NAME):
        """Initialize OpenSearch client."""
        credentials = boto3.Session(profile_name=profile).get_credentials()
        auth = AWSV4SignerAuth(credentials, region=region)
        host_name = AWSUtils.get_ssm_value("/opensearch/endpoint", profile=profile, region=region)
        aos_client = OpenSearch(
            hosts=[{"host": host_name, "port": 443}],
            http_auth=auth,
            use_ssl=True,
            verify_certs=True,
            connection_class=RequestsHttpConnection,
        )
        return aos_client

    def retrieve(self, query: str, top_k: int = 3) -> List[Doc]:
        """Retrieve documents for a single query."""
        results = self.client.search(
            index=self.index_name, 
            body={"query": {"match": {"text": query}}, "size": top_k}
        )
        
        hits = results["hits"]["hits"]
        docs = []
        for hit in hits:
            docs.append(Doc(
                doc_id=hit["_source"].get("doc_id", ""),
                chunk_id=hit["_id"],
                score=hit["_score"],
                text=hit["_source"].get("text", ""),
                # metadata=hit["_source"]
            ))
        
        return docs

    def batch_retrieve(self, queries: List[str], top_k: int = 3) -> List[List[Doc]]:
        """
        Retrieve documents for a batch of queries.
        
        Args:
            queries (List[str]): List of queries to retrieve documents for
            top_k (int, optional): Number of top documents to retrieve per query. Defaults to 3.
        
        Returns:
            List[List[Doc]]: List of retrieved documents for each query
        """
        # Initialize results list
        all_batch_docs = []
        
        # Create tqdm progress bar for the entire process
        progress_bar = tqdm(total=len(queries), desc="Retrieving documents(Using OpenSearch)")
        
        # Process queries in batches
        for i in range(0, len(queries), self.batch_size):
            # Get current batch of queries
            batch_queries = queries[i:i+self.batch_size]
            
            # Build msearch request body
            request = []
            for query in batch_queries:
                req_head = {"index": self.index_name}
                req_body = {
                    "query": {
                        "multi_match": {
                            "query": query,
                            "fields": ["text"],
                        }
                    },
                    "size": top_k,
                }
                request.extend([req_head, req_body])
            
            # Execute batch query
            results = self.client.msearch(body=request)
            
            # Process results
            batch_docs = []
            for response in results["responses"]:
                hits = response["hits"]["hits"]
                docs = []
                for hit in hits:
                    docs.append(Doc(
                        doc_id=hit["_source"].get("doc_id", ""),
                        chunk_id=hit["_id"],
                        score=hit["_score"],
                        text=hit["_source"].get("text", ""),
                        # metadata=hit["_source"]
                    ))
                batch_docs.append(docs)
                
                # Update progress bar
                progress_bar.update(1)
            
            # Extend the overall results
            all_batch_docs.extend(batch_docs)
        
        # Close the progress bar
        progress_bar.close()
        
        return all_batch_docs

class HybridRetriever(BaseRetriever):
    """Hybrid retriever combining vector and keyword search."""
    
    def __init__(self, 
                 vector_retriever: PineconeRetriever, 
                 keyword_retriever: OpenSearchRetriever,
                 vector_weight: float = 0.7):
        self.vector_retriever = vector_retriever
        self.keyword_retriever = keyword_retriever
        self.vector_weight = vector_weight
    
    def retrieve(self, query: str, top_k: int = 3) -> List[Doc]:
        """Hybrid retrieval combining vector and keyword search results."""
        # Get results from both retrievers
        vector_results = self.vector_retriever.retrieve(query, top_k)
        keyword_results = self.keyword_retriever.retrieve(query, top_k)
        
        # Combine and rerank results
        combined_docs = self._combine_results(
            vector_results, keyword_results, 
            self.vector_weight, top_k
        )
        
        return combined_docs
    
    def batch_retrieve(self, queries: List[str], top_k: int = 3) -> List[List[Doc]]:
        """Hybrid batch retrieval."""
       # Get results from both retrievers in parallel
        vector_results = self.vector_retriever.batch_retrieve(queries, top_k)
        keyword_results = self.keyword_retriever.batch_retrieve(queries, top_k)
        
        # Combine and rerank with progress tracking
        combined_results = []
        progress_bar = tqdm(total=len(queries), desc="Hybrid Retrieval(Re-ranking)")
        
        for ind, (v_result, k_result) in enumerate(zip(vector_results, keyword_results)):
            combined_result = self._combine_results(
                v_result, k_result, 
                self.vector_weight, top_k
            )
            combined_results.append(combined_result)
            
            # Update progress bar
            progress_bar.update(1)
        
        # Close the progress bar
        progress_bar.close()
        
        return combined_results

    def batch_retrieve_with_expansion(self, queries: List[Query], top_k: int = 3) -> List[List[Doc]]:
        """
        Batch retrieve documents with query expansion using both vector and keyword retrievers.
        
        Args:
            queries (List[Query]): List of queries with original and expanded queries
            top_k (int, optional): Number of top documents to retrieve per query. Defaults to 3.
        
        Returns:
            List[List[Doc]]: List of retrieved and reranked documents for each query
        """
        # Retrieve documents from both vector and keyword retrievers using expansion
        vector_results = self.vector_retriever.batch_retrieve_with_expansion(queries, top_k)
        keyword_results = self.keyword_retriever.batch_retrieve_with_expansion(queries, top_k)
        
        # Combine and rerank with progress tracking
        combined_results = []
        progress_bar = tqdm(total=len(queries), desc="Hybrid Retrieval(Re-ranking)")

        for v_result, k_result in zip(vector_results, keyword_results):
            combined_result = self._combine_results(
                v_result, k_result, 
                self.vector_weight, top_k
            )
            combined_results.append(combined_result)
            # Update progress bar
            progress_bar.update(1)

        # Close the progress bar
        progress_bar.close()
        
        return combined_results
    
    
    def _combine_results(self, 
                         vector_results: List[Doc], 
                         keyword_results: List[Doc], 
                         vector_weight: float, 
                         top_k: int) -> List[Doc]:
        """Combine and rerank results from vector and keyword retrievers."""
        # Create a dictionary for all documents with combined scores
        doc_scores = {}
        
        # Normalize vector scores to [0, 1] range (they're already in this range)
        max_keyword_score = max([doc.score for doc in keyword_results]) if keyword_results else 1.0
       
        # Add vector scores (with weight)
        for doc in vector_results:
            if  doc.chunk_id in doc_scores:
                doc_scores[doc.chunk_id]["score"] += doc.score * vector_weight
            else:
                doc_scores[doc.chunk_id] = {
                    "score": doc.score * vector_weight,
                    "doc": doc
                }
        
        # Add keyword scores (with weight)
        keyword_weight = 1.0 - vector_weight
        for doc in keyword_results:
            # Normalize keyword score
            normalized_score = doc.score / max_keyword_score
            
            if doc.chunk_id in doc_scores:
                # Document exists in both results, add scores
                doc_scores[doc.chunk_id]["score"] += normalized_score * keyword_weight
            else:
                # Document only in keyword results
                doc_scores[doc.chunk_id] = {
                    "score": normalized_score * keyword_weight,
                    "doc": doc
                }
        
        # Sort combined results by score and take top-k
        sorted_docs = sorted(
            [(chunk_id, info["score"], info["doc"]) 
             for chunk_id, info in doc_scores.items()],
            key=lambda x: x[1],
            reverse=True
        )[:top_k]
        
        # Unpack the results
        return [doc for _, _, doc in sorted_docs]
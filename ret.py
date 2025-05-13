"""Retriever module for LiveRAG with evaluation capabilities."""
import argparse
import json
import warnings
import os
from typing import List, Literal, Dict, Any
from multiprocessing.pool import ThreadPool
from functools import cache
from dataclasses import dataclass

import boto3
import torch
import numpy as np
from pinecone import Pinecone
from transformers import AutoModel, AutoTokenizer
from opensearchpy import OpenSearch, AWSV4SignerAuth, RequestsHttpConnection
from tqdm import tqdm

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
    chunk_id: str
    score: float
    text: str
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

class PineconeRetriever:
    """Retriever using Pinecone vector database."""
    
    def __init__(self, 
                 embedding_model: EmbeddingModel,
                 index_name: str = PINECONE_INDEX_NAME, 
                 namespace: str = PINECONE_NAMESPACE):
        self.embedding_model = embedding_model
        self.index_name = index_name
        self.namespace = namespace
        self.index = self._get_index()
    
    def _get_index(self):
        """Connect to Pinecone index."""
        pc = Pinecone(api_key=AWSUtils.get_ssm_secret("/pinecone/ro_token"))
        return pc.Index(name=self.index_name)
    
    def get_vector_count(self):
        return self.index.describe_index_stats().get('total_vector_count', 0)

    def retrieve_by_id(self, data:List[Dict[str, Any]]):
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
                metadata=match["metadata"]
            ))
        
        return docs
    
    def batch_retrieve(self, queries: List[str], top_k: int = 3, n_parallel: int = 3) -> List[List[Doc]]:
        """Retrieve documents for a batch of queries."""
        embeddings = self.embedding_model.batch_embed_queries(queries)
        
        # Use ThreadPool for parallel querying
        with ThreadPool(n_parallel) as pool:
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
     
        # Convert raw results to Doc objects
        batch_docs = []
        for result in results:
            docs = []
            for match in result["matches"]:
                docs.append(Doc(
                    doc_id=match["metadata"].get("doc_id", ""),
                    chunk_id=match["id"],
                    score=match["score"],
                    text=match["metadata"].get("text", ""),
                    metadata=match["metadata"]
                ))
            batch_docs.append(docs)
            
        return batch_docs

class OpenSearchRetriever:
    """Retriever using OpenSearch."""
    
    def __init__(self, index_name: str = OPENSEARCH_INDEX_NAME):
        self.index_name = index_name
        self.client = self._get_client()
    
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
                metadata=hit["_source"]
            ))
        
        return docs
    
    def batch_retrieve(self, queries: List[str], top_k: int = 3, n_parallel: int = 3) -> List[List[Doc]]:
        """Retrieve documents for a batch of queries."""
        # Build msearch request body
        request = []
        for query in queries:
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
                    metadata=hit["_source"]
                ))
            batch_docs.append(docs)
        
        return batch_docs

class HybridRetriever:
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
    
    def batch_retrieve(self, queries: List[str], top_k: int = 3, n_parallel: int = 3) -> List[List[Doc]]:
        """Hybrid batch retrieval."""
        # Get results from both retrievers in parallel
        vector_results = self.vector_retriever.batch_retrieve(queries, top_k, n_parallel)
        keyword_results = self.keyword_retriever.batch_retrieve(queries, top_k)
        
        # Combine and rerank for each query
        combined_results = []
        for v_result, k_result in zip(vector_results, keyword_results):
            combined_result = self._combine_results(
                v_result, k_result, 
                self.vector_weight, top_k
            )
            combined_results.append(combined_result)
        
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
        max_keyword_score = max(doc.score for doc in keyword_results) if keyword_results else 1.0
        
        # Add vector scores (with weight)
        for doc in vector_results:
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

class EvaluationDataset:
    """Class to load and manage test dataset from JSONL file."""
    
    def __init__(self, file_path: str):
        self.file_path = file_path
        self.data = self._load_data()
    
    def _load_data(self) -> List[Dict[str, Any]]:
        """Load data from JSONL file."""
        with open(self.file_path, 'r', encoding='utf-8') as f:
            return json.load(f)
    
    def get_queries(self) -> List[str]:
        """Get all questions/queries from the dataset."""
        return [item["question"] for item in self.data]
    
    def get_ground_truth_ids(self) -> List[List[str]]:
        """Get ground truth document IDs for each query."""
        return [item["document_ids"] for item in self.data]
    
    def get_contexts(self) -> List[str]:
        """Get contexts (if available) for each query."""
        return [item.get("context", "") for item in self.data]

class MetricsCalculator:
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

# Update the run_evaluation function to work with new Doc class
def run_evaluation(retriever, dataset: EvaluationDataset, top_k: int = 10, batch_size: int = 16, n_parallel: int = 3) -> Dict[str, Dict[str, float]]:
    """Run evaluation of retriever on a dataset."""
    queries = dataset.get_queries()
    ground_truth_ids = dataset.get_ground_truth_ids()
    
    # Process queries in batches
    all_retrieved_ids = []
    for i in tqdm(range(0, len(queries), batch_size), desc="Retrieving documents"):
        batch_queries = queries[i:i+batch_size]
        batch_results = retriever.batch_retrieve(batch_queries, top_k=top_k, n_parallel=n_parallel)
        all_retrieved_ids.extend([[doc.doc_id for doc in result] for result in batch_results])
    
    # Calculate metrics
    k_values = [1, 3, 5, 10] if top_k <= 10 else [1, 3, 5, 10]
    metrics = MetricsCalculator.calculate_metrics(all_retrieved_ids, ground_truth_ids, k_values)
    
    return metrics

# The main function and rest of the code remains the same
def main():
    """Main entry point with argument parsing."""
    parser = argparse.ArgumentParser(description="LiveRAG Retrieval Evaluation")
    parser.add_argument("--data", type=str, default="./data/test.json", help="Path to evaluation data")
    parser.add_argument("--top-k", type=int, default=10, help="Number of documents to retrieve")
    parser.add_argument("--batch-size", type=int, default=10, help="Batch size for processing")
    parser.add_argument("--vector-weight", type=float, default=0.7, help="Weight for vector retrieval in hybrid mode")
    parser.add_argument("--retriever", type=str, choices=["vector", "keyword", "hybrid"], default="hybrid", 
                        help="Retriever type to use")
    parser.add_argument("--model", type=str, default="intfloat/e5-base-v2", help="Embedding model to use")
    parser.add_argument("--n-parallel", type=int, default=10, help="Number of parallel threads for batch retrieval")
    args = parser.parse_args()
    
    # Initialize embedding model
    embedding_model = EmbeddingModel(model_name=args.model)
    
    # Initialize retrievers
    vector_retriever = PineconeRetriever(embedding_model=embedding_model)
    keyword_retriever = OpenSearchRetriever()
    hybrid_retriever = HybridRetriever(
        vector_retriever=vector_retriever,
        keyword_retriever=keyword_retriever,
        vector_weight=args.vector_weight
    )
    
    # Select retriever based on arguments
    if args.retriever == "vector":
        retriever = vector_retriever
    elif args.retriever == "keyword":
        retriever = keyword_retriever
    else:  # hybrid
        retriever = hybrid_retriever
    
    # Load dataset
    dataset = EvaluationDataset(file_path=args.data)
    
    # Run evaluation with the provided n_parallel setting
    metrics = run_evaluation(
        retriever=retriever,
        dataset=dataset,
        top_k=args.top_k,
        batch_size=args.batch_size,
        n_parallel=args.n_parallel
    )
    
    # Print results
    print(f"\nEvaluation Results for {args.retriever.capitalize()} Retriever:")
    for k, values in metrics.items():
        if k == "overall":
            print(f"\nOverall Metrics:")
        else:
            print(f"\nMetrics {k}:")
        for metric, value in values.items():
            print(f"  {metric.capitalize()}: {value:.2f}")


if __name__ == "__main__":
    main()
"""Run DeepRAG inference pipeline for challenge."""
import argparse
import json
import re
import os
import time
import jsonlines
from typing import List, Dict, Any
import random
from opensearchpy.exceptions import ConnectionTimeout, ConnectionError

from query_expansion_ola import QueryExpander, Query
from retriever import PineconeRetriever, OpenSearchRetriever, HybridRetriever, Doc
from reader import DocumentReader
import torch
from transformers import AutoTokenizer, AutoModelForMaskedLM
import numpy as np
from tqdm import tqdm
# Import the QA prompt from utils
from utils.prompts import QA_USER_PROMPT

def retry_with_backoff(func, max_retries=5, initial_delay=1, backoff_factor=2, exceptions=(ConnectionTimeout, ConnectionError)):
    """
    Retry a function with exponential backoff
    
    Args:
        func: Function to retry
        max_retries: Maximum number of retries
        initial_delay: Initial delay between retries in seconds
        backoff_factor: Factor to increase delay for each retry
        exceptions: Tuple of exceptions to catch and retry
        
    Returns:
        Result of the function call
    """
    delay = initial_delay
    for retry in range(max_retries):
        try:
            return func()
        except exceptions as e:
            if retry == max_retries - 1:
                raise
            
            # Add some jitter to the delay
            jitter = random.uniform(0.8, 1.2)
            sleep_time = delay * jitter
            
            print(f"Connection error: {e}. Retrying in {sleep_time:.2f} seconds... (Attempt {retry+1}/{max_retries})")
            time.sleep(sleep_time)
            
            # Increase delay for next retry
            delay *= backoff_factor

def is_noisy_document(text, threshold=0.5):
    """
    Filter out noisy documents like navigation menus, lists, etc.
    
    Args:
        text (str): Document text
        threshold (float): Threshold for determining noise
        
    Returns:
        bool: True if document is likely noise, False otherwise
    """
    # Check for noisy patterns
    indicators = [
        # Navigation menu patterns
        re.search(r'(home|about|contact|search|login|sign in|register|menu|navigation)', text.lower()) is not None,
        # Lists of links or bullet points
        text.count('•') > 5 or text.count('|') > 10 or text.count('›') > 5,
        # Short texts that are likely headers or navigation
        len(text.split()) < 20 and any(x in text.lower() for x in ['click', 'here', 'next', 'previous']),
        # Boilerplate text
        any(x in text.lower() for x in ['copyright ©', 'all rights reserved', 'terms of use', 'privacy policy']),
        # Docs with too many non-alphanumeric characters
        sum(not c.isalnum() and not c.isspace() for c in text) / len(text) > 0.3 if len(text) > 0 else False
    ]
    
    # Calculate the noise score
    noise_score = sum(indicators) / len(indicators)
    
    return noise_score > threshold

# Import SPLADE functions
def get_splade_vectors(texts: List[str], model, tokenizer, device: str = 'cpu', batch_size: int = 16) -> np.ndarray:
    """Generates SPLADE sparse vectors for a list of texts."""
    all_vectors = []
    model.eval()
    with torch.no_grad():
        for i in tqdm(range(0, len(texts), batch_size), desc="SPLADE Embed", leave=False, disable=True):
            batch_texts = texts[i:i+batch_size]
            try:
                tokens = tokenizer(batch_texts, return_tensors='pt', padding=True, truncation=True, max_length=512).to(device)
                output = model(**tokens).logits
                vectors = torch.log(1 + torch.relu(output)) * tokens['attention_mask'].unsqueeze(-1)
                vectors = torch.max(vectors, dim=1)[0]
                all_vectors.append(vectors.cpu().numpy())
            except Exception as e:
                print(f"Error embedding batch starting at index {i}: {e}")
                print(f"Skipping batch due to error. Output array shape might be affected.")

    return np.vstack(all_vectors) if all_vectors else np.array([[]])

def sparse_dot_product(query_vector: np.ndarray, doc_vectors: np.ndarray) -> np.ndarray:
    """Calculates dot product between a single query vector and multiple doc vectors."""
    # Ensure query_vector is 1D
    if query_vector.ndim > 1:
         query_vector = query_vector.flatten()
    # Handle empty doc_vectors case
    if doc_vectors.shape[0] == 0:
        return np.array([])
    return np.dot(doc_vectors, query_vector)

def get_top_docs_by_splade(query: str, docs: List[tuple], splade_model, splade_tokenizer, device: str, top_k: int = 5) -> List[tuple]:
    """Rerank documents using SPLADE scores and return the top-k."""
    if not docs:
        return []
    
    # Extract document texts and IDs
    doc_ids = [doc[0] for doc in docs]
    doc_texts = [doc[1] for doc in docs]
    
    # Prepare texts for SPLADE embedding
    all_texts_to_embed = [query] + doc_texts
    
    # Get SPLADE vectors
    all_vectors = get_splade_vectors(all_texts_to_embed, splade_model, splade_tokenizer, device)
    
    if all_vectors.shape[0] != len(all_texts_to_embed):
        print("Error: SPLADE embedding failed or returned unexpected number of vectors.")
        return docs[:top_k]  # Fallback to original order
    
    # Separate vectors
    query_vector = all_vectors[0]
    doc_vectors = all_vectors[1:]
    
    # Calculate similarity scores
    similarity_scores = sparse_dot_product(query_vector, doc_vectors)
    
    # Sort documents by similarity score
    doc_with_scores = list(zip(docs, similarity_scores))
    ranked_docs = sorted(doc_with_scores, key=lambda x: x[1], reverse=True)
    
    # Return top-k documents
    return [doc for doc, score in ranked_docs[:top_k]]

def create_final_prompt(question: str, passages: List[Dict]) -> str:
    """Create a final prompt using the retrieved passages and QA_USER_PROMPT."""
    # Format the documents as expected by the QA prompt
    document_texts = [p["passage"] for p in passages]
    documents_formatted = "\n\n".join(document_texts)
    
    # Use the standard QA prompt from utils/prompts.py
    prompt = QA_USER_PROMPT.format(
        documents=documents_formatted,
        query=question
    )
    
    return prompt

def main():
    """Main entry point with argument parsing."""
    parser = argparse.ArgumentParser(description="DeepRAG Challenge Inference")
    parser.add_argument("--input", type=str, required=True, help="Path to input JSONL file")
    parser.add_argument("--output", type=str, required=True, help="Path to output JSON file")
    parser.add_argument("--top-k", type=int, default=50, help="Number of documents to retrieve")
    parser.add_argument("--batch-size", type=int, default=10, help="Batch size for processing")
    parser.add_argument("--vector-weight", type=float, default=0.5, help="Weight for vector retrieval in hybrid mode")
    parser.add_argument("--retriever", type=str, choices=["vector", "keyword", "hybrid"], default="hybrid", 
                        help="Retriever type to use")
    parser.add_argument("--model", type=str, default="intfloat/e5-base-v2", help="Embedding model to use")
    parser.add_argument("--n-parallel", type=int, default=10, help="Number of parallel threads for batch retrieval")
    parser.add_argument("--top-r", type=int, default=5, help="Number of documents to provide to reader")
    parser.add_argument("--verbose", action='store_true', help='Print out info')
    parser.add_argument("--splade-model", type=str, default="naver/splade-cocondenser-ensembledistil",
                        help="Hugging Face model name/path for SPLADE.")
    # Add query expansion flag
    parser.add_argument("--expand", action='store_true', help='Enable query expansion with a single subquestion')

    args = parser.parse_args()

    # Initialize SPLADE model
    print(f"Initializing SPLADE model: {args.splade_model}")
    device = "cuda:3" if torch.cuda.is_available() else "cpu"
    
    splade_tokenizer = AutoTokenizer.from_pretrained(args.splade_model)
    splade_model = AutoModelForMaskedLM.from_pretrained(args.splade_model).to(device)
    splade_model.eval()
    print(f"SPLADE model initialized on {device}")

    # Load input data
    input_data = []
    with jsonlines.open(args.input) as reader:
        for item in reader:
            input_data.append(item)
    
    print(f"Loaded {len(input_data)} questions from {args.input}")

    # Initialize query expander if expansion is enabled
    if args.expand:
        query_expander = QueryExpander(max_workers=args.n_parallel, batch_size=args.batch_size)
        print("Query expansion enabled")

    # initialize retrievers
    vector_retriever = PineconeRetriever(max_workers=args.n_parallel, batch_size=args.batch_size)
    keyword_retriever = OpenSearchRetriever(batch_size=args.batch_size)
    hybrid_retriever = HybridRetriever(vector_retriever=vector_retriever, keyword_retriever=keyword_retriever, 
                                    vector_weight=args.vector_weight)

    retriever = {"vector": vector_retriever, "keyword": keyword_retriever, "hybrid": hybrid_retriever}[args.retriever]

    # initialize reader
    reader = DocumentReader(max_workers=args.n_parallel, batch_size=args.batch_size, verbose=args.verbose)

    # Prepare output data
    output_data = []

    # Total time 
    total_time = 0

    for i, item in enumerate(input_data):
        question_id = item.get("id", i)
        question = item.get("question", "")
        
        # Start time
        start_time = time.time()

        print(f"[{i+1}/{len(input_data)}] Processing question {question_id}: {question}")
        
        # Retrieve documents with retry mechanism
        print("Retrieving documents...")
        try:
            retrieved_docs_raw = retry_with_backoff(
                lambda: retriever.retrieve(question, args.top_k),
                max_retries=5,
                initial_delay=2,
                backoff_factor=2
            )
            retrieved_docs = [(doc.doc_id, doc.text) for doc in retrieved_docs_raw]
        except Exception as e:
            print(f"Error retrieving documents after multiple retries: {e}")
            print("Skipping this question due to persistent retrieval errors.")
            continue
        
        if args.verbose:
            print(f"Retrieved {len(retrieved_docs)} documents")
        
        # Filter out noisy documents
        clean_docs = []
        for doc_id, text in retrieved_docs:
            if not is_noisy_document(text):
                clean_docs.append((doc_id, text)) 
        
        if len(clean_docs) < len(retrieved_docs):
            print(f"Filtered out {len(retrieved_docs) - len(clean_docs)} noisy documents")
        
        # Create a list of documents for SPLADE reranking
        docs_to_rerank = clean_docs
        
        # Track expansion information
        subquestion = None
        expansion_status = "Not attempted"
        expansion_added = False
        
        # Handle query expansion if enabled - BEFORE RERANKING
        if args.expand:
            print("\nGenerating subquestion for query expansion...")
            # Generate a single subquestion
            expanded_query_result = query_expander.batch_query_expansion([question])[0]
            
            if expanded_query_result.expanded_queries:
                subquestion = expanded_query_result.expanded_queries[0]  # Get the first expanded query
                print(f"Subquestion: {subquestion}")
                expansion_status = "Subquestion generated"
                
                # Retrieve top document for the subquestion using vector retriever for precision
                print(f"Retrieving document for subquestion...")
                try:
                    subq_docs_raw = retry_with_backoff(
                        lambda: vector_retriever.retrieve(subquestion, top_k=1),
                        max_retries=3,
                        initial_delay=1,
                        backoff_factor=2
                    )
                    
                    if subq_docs_raw:
                        subq_doc = subq_docs_raw[0]
                        subq_doc_id = subq_doc.doc_id
                        subq_doc_text = subq_doc.text
                        expansion_status = "Document retrieved"
                        
                        # Check if document ID is already in the retrieved docs
                        retrieved_ids_set = set([doc_id for doc_id, _ in retrieved_docs])
                        
                        # Only add if the document is not noisy and not already in the retrieved docs
                        if not is_noisy_document(subq_doc_text):
                            if subq_doc_id not in retrieved_ids_set:
                                docs_to_rerank.append((subq_doc_id, subq_doc_text))
                                expansion_added = True
                                print(f"Added document for subquestion: {subq_doc_id}")
                                expansion_status = "Document added for reranking"
                                if args.verbose:
                                    print(f"Subquestion doc: {subq_doc_text[:100]}...")
                            else:
                                print(f"Expansion document was already retrieved in initial results")
                                expansion_status = "Document already retrieved"
                        else:
                            print(f"Expansion document was filtered as noisy")
                            expansion_status = "Document was noisy"
                    else:
                        print(f"No document retrieved for subquestion")
                        expansion_status = "No document retrieved"
                        
                except Exception as e:
                    print(f"Error retrieving expansion document: {e}")
                    expansion_status = "Retrieval error"
            else:
                print("No subquestion generated")
                expansion_status = "No subquestion generated"
        
        # Rerank with SPLADE
        print(f"\nReranking {len(docs_to_rerank)} documents using SPLADE...")
        reranked_docs = get_top_docs_by_splade(
            query=question,
            docs=docs_to_rerank,
            splade_model=splade_model,
            splade_tokenizer=splade_tokenizer,
            device=device,
            top_k=args.top_r
        )
        
        print(f"Selected {len(reranked_docs)} documents using SPLADE reranking")
        
        # Format documents for reader
        reader_docs = [(doc_id, text) for doc_id, text in reranked_docs]
        reader_documents = [reader_docs]
        
        # Format passages for output
        passages = []
        for doc_id, text in reader_docs:
            # For the challenge format, we need to group passages by text
            # First, check if this text already exists
            existing_passage = next((p for p in passages if p["passage"] == text), None)
            
            if existing_passage:
                # Add doc_id to existing passage
                existing_passage["doc_IDs"].append(doc_id)
            else:
                # Create new passage entry
                passages.append({
                    "passage": text,
                    "doc_IDs": [doc_id]
                })
        
        # Create final prompt
        final_prompt = create_final_prompt(question, passages)
        
        # Generate query object for reader with or without the subquestion
        expanded_queries = []
        if args.expand and subquestion:
            expanded_queries = [subquestion]
            print(f"Including subquestion in reader context: {subquestion}")
            
        query_obj = Query(original_query=question, expanded_queries=expanded_queries)
        
        # Generate answer
        answers = reader.batch_generate(
            queries=[query_obj],
            documents_list=reader_documents
        )
        
        answer_text = answers[0].answer if answers else ""
        print(f"Answer: {answer_text}")
        
        # Create output entry with expansion info if available
        output_entry = {
            "id": question_id,
            "question": question,
            "passages": passages,
            "final_prompt": final_prompt,
            "answer": answer_text
        }
        
        # Add expansion information to output if enabled
        if args.expand:
            output_entry.update({
                "expansion_info": {
                    "status": expansion_status,
                    "subquestion": subquestion if subquestion else "",
                    "document_added": expansion_added
                }
            })
        
        output_data.append(output_entry)
        
        # End time
        end_time = time.time()
        elapsed_time = end_time - start_time
        total_time += elapsed_time
        
        print(f"Time taken: {elapsed_time:.2f} seconds")
        print("-------------------------------------------------------")
    
    # Save output data
    with open(args.output, 'w') as f:
        json.dump(output_data, f, indent=2)
    
    print(f"Output saved to {args.output}")
    print(f"Total time: {total_time:.2f} seconds, Average time per question: {total_time/len(input_data):.2f} seconds")


if __name__ == "__main__":
    main() 
"""Script focused on evaluating retrieval and reranking using FlashRank."""

import argparse
import json
import time
from tqdm import tqdm
from typing import List, Dict, Tuple, Set

# Assuming utils and retriever components are in the same structure
from query_expansion_ola import QueryExpander
from retriever import PineconeRetriever, OpenSearchRetriever, HybridRetriever, Doc
from utils import EvaluationDataset

# Import LLM Reranker components for validation/summarization
from llm_reranker import LLMReranker
from utils.prompts import (
    LLM_SUBQUERY_VALIDATOR_SYSTEM_PROMPT,
    LLM_SUBQUERY_VALIDATOR_USER_PROMPT,
    LLM_DOC_SUMMARY_SYSTEM_PROMPT,
    LLM_DOC_SUMMARY_USER_PROMPT
)

# Import flashrank
try:
    from flashrank import Ranker, RerankRequest
except ImportError:
    print("Error: flashrank is not installed. Please install it using 'pip install flashrank'")
    exit(1)

# Import MMR dependencies if needed
try:
    import sentence_transformers # Import the main package
    from sentence_transformers import SentenceTransformer 
    from sentence_transformers.util import cos_sim # Try importing cos_sim directly
    SENTENCE_TRANSFORMERS_AVAILABLE = True
except ImportError:
    SENTENCE_TRANSFORMERS_AVAILABLE = False
    # Don't exit immediately, only fail if MMR mode is selected later

# Configure logging to suppress INFO messages from opensearch-py
import logging
logging.getLogger('opensearch').setLevel(logging.WARNING)
# Suppress INFO messages from httpx (used by ollama client)
logging.getLogger('httpx').setLevel(logging.WARNING)

def calculate_metrics(retrieved_ids: List[str], ground_truth_ids: Set[str]) -> Dict[str, float]:
    """Calculate precision, recall, and F1 for a single query result."""
    if not ground_truth_ids: # Handle cases with no ground truth
        return {"precision": 0.0, "recall": 0.0, "f1": 0.0}
    
    retrieved_set = set(retrieved_ids)
    true_positives = len(retrieved_set.intersection(ground_truth_ids))
    
    precision = true_positives / len(retrieved_set) if retrieved_set else 0.0
    recall = true_positives / len(ground_truth_ids) if ground_truth_ids else 0.0
    f1 = 2 * (precision * recall) / (precision + recall) if (precision + recall) > 0 else 0.0
    
    # Return metrics as fractions (0.0 to 1.0)
    return {"precision": precision, "recall": recall, "f1": f1}

def main():
    """Main entry point with argument parsing."""
    parser = argparse.ArgumentParser(description="DeepRetriever Evaluation with FlashRank")
    parser.add_argument("--dataset", type=str, default="./data/big.json", help="Path to data")
    parser.add_argument("--datatype", type=str, default="all", choices=["all", "single", "multiple"],
                        help="Type of data to evaluate")
    parser.add_argument("--top-k", type=int, default=5, help="Number of documents to retrieve per query/subquery")
    parser.add_argument("--top-r", type=int, default=3, help="Number of documents to return after reranking")
    parser.add_argument("--batch-size", type=int, default=10, help="Batch size for processing (used by some components)")
    parser.add_argument("--vector-weight", type=float, default=0.7, help="Weight for vector retrieval in hybrid mode")
    parser.add_argument("--retriever", type=str, choices=["vector", "keyword", "hybrid"], default="hybrid",
                        help="Retriever type to use")
    parser.add_argument("--n-parallel", type=int, default=10, help="Number of parallel threads for batch retrieval")
    parser.add_argument("--reranker-mode", type=str, default="flashrank", choices=["flashrank", "mmr", "llm"], help="Reranking strategy to use.")
    parser.add_argument("--flashrank-model", type=str, default="ms-marco-MiniLM-L-12-v2", help="FlashRank model name (used if mode is flashrank)")
    parser.add_argument("--flashrank-cache-dir", type=str, default="/tmp/flashrank_cache", help="Cache directory for FlashRank models")
    parser.add_argument("--mmr-embedding-model", type=str, default="all-MiniLM-L6-v2", help="Sentence Transformer model for embeddings (used if mode is mmr)")
    parser.add_argument("--mmr-lambda", type=float, default=0.5, help="Lambda parameter for MMR (0=max diversity, 1=max relevance)")
    parser.add_argument("--verbose", action='store_true', help='Print out detailed info')

    # Add args for LLM Validator/Summarizer
    parser.add_argument("--llm-provider", type=str, default="ollama", help="LLM provider for validation/summarization (e.g., ollama, claude)")
    parser.add_argument("--llm-model-name", type=str, default=None, help="Specific LLM model name for validation/summarization")
    parser.add_argument("--llm-temperature", type=float, default=0.1, help="Temperature for LLM validation/summarization")
    parser.add_argument("--oversample-factor", type=int, default=3, help="Factor to multiply top_k by for initial subquery retrieval (>=1)")

    args = parser.parse_args()

    # Validate oversample factor
    if args.oversample_factor < 1:
        print("Warning: --oversample-factor must be 1 or greater. Setting to 1.")
        args.oversample_factor = 1

    # Load dataset
    dataset = EvaluationDataset(args.dataset)

    # Get queries, answers, and ground truth ids based on datatype
    if args.datatype == "single":
        queries, _, ground_truth_ids_list = dataset.get_data_for_single_doc_questions()
    elif args.datatype == "multiple":
        queries, _, ground_truth_ids_list = dataset.get_data_for_multiple_doc_questions()
    else: # "all"
        queries, _, ground_truth_ids_list, _ = dataset.get_all_data()

    # Initialize query expander
    query_expander = QueryExpander(max_workers=args.n_parallel, batch_size=args.batch_size)

    # Initialize retrievers
    vector_retriever = PineconeRetriever(max_workers=args.n_parallel, batch_size=args.batch_size)
    keyword_retriever = OpenSearchRetriever(batch_size=args.batch_size)
    hybrid_retriever = HybridRetriever(vector_retriever=vector_retriever, keyword_retriever=keyword_retriever,
                                       vector_weight=args.vector_weight)

    retriever = {"vector": vector_retriever, "keyword": keyword_retriever, "hybrid": hybrid_retriever}[args.retriever]

    # --- Initialize Reranker based on mode ---
    reranker_model = None # This variable is mainly for flashrank/mmr models
    if args.reranker_mode == 'flashrank':
        try:
            reranker_model = Ranker(model_name=args.flashrank_model, cache_dir=args.flashrank_cache_dir)
            print(f"FlashRank reranker initialized with model: {args.flashrank_model}")
        except Exception as e:
            print(f"Error initializing FlashRank Ranker '{args.flashrank_model}': {e}")
            exit(1)
    elif args.reranker_mode == 'mmr':
        if not SENTENCE_TRANSFORMERS_AVAILABLE:
            print("Error: --reranker-mode 'mmr' selected, but sentence-transformers library is not installed.")
            print("Please install it using 'pip install sentence-transformers'")
            exit(1)
        try:
            reranker_model = SentenceTransformer(args.mmr_embedding_model)
            print(f"MMR mode selected. Initialized Sentence Transformer model: {args.mmr_embedding_model}")
        except Exception as e:
            print(f"Error initializing Sentence Transformer model '{args.mmr_embedding_model}': {e}")
            exit(1)
    elif args.reranker_mode == 'llm':
        # No specific model initialization needed here, 
        # we will use the llm_validator_summarizer instance later.
        print("LLM reranker mode selected. Will use the initialized LLM validator/summarizer.")
        pass # Placeholder, initialization happens later
    else:
        # This case should not happen due to argparse choices, but good practice
        print(f"Error: Unknown reranker mode '{args.reranker_mode}'")
        exit(1)

    # Initialize LLM for validation/summarization (Used by SubQ validation AND LLM Reranker mode)
    try:
        llm_validator_summarizer = LLMReranker(
            provider=args.llm_provider,
            model_name=args.llm_model_name,
            temperature=args.llm_temperature,
            verbose=args.verbose # Use the same verbose flag
        )
        print(f"LLM Validator/Summarizer initialized with provider: {args.llm_provider}, model: {args.llm_model_name or 'default'}")
    except Exception as e:
        print(f"Error initializing LLM Validator/Summarizer: {e}")
        exit(1)

    # Evaluation dictionary
    eval_results = {}
    total_time = 0

    for i, orig_query in enumerate(tqdm(queries, desc="Processing Queries")):
        start_time = time.time()
        ground_truth_ids = set(ground_truth_ids_list[i]) # Use a set for faster lookups
        eval_results[i] = {}

        if args.verbose:
            print(f"\nProcessing Query {i}: {orig_query}")
            print(f"Ground Truth IDs: {ground_truth_ids}")

        # 1. Expand Queries
        try:
            expanded_queries = query_expander.batch_query_expansion([orig_query])[0]
            expanded_query_list = expanded_queries.expanded_queries
            all_queries = [orig_query] + expanded_query_list
            if args.verbose:
                 print(f"Expanded Queries: {expanded_query_list}")
        except Exception as e:
            print(f"Warning: Query expansion failed for query {i}: {e}. Using original query only.")
            all_queries = [orig_query]
            expanded_query_list = []


        # 2. Retrieve Documents for all queries
        pooled_docs_dict = {} # Use dict {doc_id: text} for deduplication
        # Track which queries retrieved which docs for frequency count
        doc_retrieval_sources = {} # {doc_id: set(query_indices)} 
        retrieved_ids_original_query = []
        
        # Use tqdm for inner loop if not verbose
        # print(f"Retrieving top-{args.top_k} docs for {len(all_queries)} queries (original + expanded)...")
        query_iterator = tqdm(all_queries, desc=f"Retrieving (Q{i})", leave=False) if not args.verbose else all_queries

        for idx, current_query in enumerate(query_iterator):
            try:
                retrieved_docs_raw: List[Doc] = retriever.retrieve(current_query, args.top_k)
                if args.verbose:
                    query_type = "Original Query" if idx == 0 else f"Subquery '{current_query[:50]}...'"
                    print(f"  Retrieved {len(retrieved_docs_raw)} docs for {query_type}")
                    # Log details for each retrieved doc
                    # for doc in retrieved_docs_raw:
                    #     first_five_words = " ".join(doc.text.split()[:5])
                    #     print(f"    - ID: {doc.doc_id}, Start: '{first_five_words}...'") # Keep this commented or modify as needed
                
                # Store IDs for original query baseline evaluation
                if idx == 0:
                    retrieved_ids_original_query = [doc.doc_id for doc in retrieved_docs_raw]
                    # Add original query results directly to pool (no validation)
                    for doc in retrieved_docs_raw:
                        if doc.doc_id not in pooled_docs_dict:
                            pooled_docs_dict[doc.doc_id] = doc.text
                            doc_retrieval_sources[doc.doc_id] = {idx} # Track source query index
                        else:
                            doc_retrieval_sources[doc.doc_id].add(idx)
                            
                        if args.verbose:
                             first_five_words = " ".join(doc.text.split()[:5])
                             print(f"    - [Original Query] Adding ID: {doc.doc_id}, Start: '{first_five_words}...'")
                            
                # Add SUBQUERY results to pool ONLY IF they pass validation, up to top_k
                else: 
                    validated_for_this_subquery = 0
                    docs_processed_for_subquery = 0
                    # Calculate how many docs to retrieve initially for this subquery
                    k_to_retrieve = args.top_k * args.oversample_factor
                    if args.verbose:
                        print(f"  Attempting to retrieve up to {k_to_retrieve} docs for subquery to find {args.top_k} validated ones.")
                    
                    try:
                         retrieved_docs_raw: List[Doc] = retriever.retrieve(current_query, k_to_retrieve)
                         if args.verbose:
                              print(f"  Retrieved {len(retrieved_docs_raw)} raw docs for subquery validation.")
                    except Exception as e:
                         print(f"  Warning: Initial retrieval failed for subquery '{current_query}': {e}. Skipping validation for this subquery.")
                         retrieved_docs_raw = [] # Ensure loop doesn't run

                    for doc in retrieved_docs_raw:
                        docs_processed_for_subquery += 1
                        # Stop if we've found enough validated docs for this subquery
                        if validated_for_this_subquery >= args.top_k:
                             if args.verbose:
                                 print(f"    Reached target of {args.top_k} validated docs for this subquery. Stopping validation.")
                             break 

                        # Avoid re-validating docs already added by original query or previous subqueries
                        if doc.doc_id not in pooled_docs_dict: 
                            # 1. Generate summary based on SUBQUERY (REMOVED)
                            # summary = llm_validator_summarizer.summarize_document(query=current_query, document_text=doc.text)
                            
                            # 2. Validate using the SUBQUERY and the FULL DOCUMENT TEXT
                            # if summary and llm_validator_summarizer.does_document_answer_subquery(subquery=current_query, document_text=summary):
                            if llm_validator_summarizer.does_document_answer_subquery(subquery=current_query, document_text=doc.text):
                                # 3. Add ORIGINAL document text to pool if validation passes
                                pooled_docs_dict[doc.doc_id] = doc.text
                                doc_retrieval_sources[doc.doc_id] = {idx} # Track source query index
                                validated_for_this_subquery += 1
                                if args.verbose:
                                    first_five_words = " ".join(doc.text.split()[:5])
                                    print(f"    - [Validated SubQ {validated_for_this_subquery}/{args.top_k}] Added ID: {doc.doc_id}, Start: '{first_five_words}...'")
                            elif args.verbose:
                                # print(f"    - [Filtered SubQ] Discarding ID: {doc.doc_id} (summary: '{summary[:30]}...')")
                                print(f"    - [Filtered SubQ] Discarding ID: {doc.doc_id} (Full text did not validate)")
                        # else: # Optional: log if doc was already in pool
                        #    if args.verbose:
                        #        print(f"    - [Skipped SubQ] ID: {doc.doc_id} already in pool.")
                    
                    if args.verbose:
                         print(f"  Subquery processing finished: Added {validated_for_this_subquery} validated documents from this subquery (processed {docs_processed_for_subquery}/{len(retrieved_docs_raw)} retrieved). Target was {args.top_k}.")

            except Exception as e:
                 print(f"  Warning: Retrieval/Validation failed for query '{current_query}': {e}")

        # Calculate final frequency from sources
        doc_frequency = {doc_id: len(sources) for doc_id, sources in doc_retrieval_sources.items()}

        # Prepare pooled documents in FlashRank format
        pooled_docs_flashrank = [{"id": doc_id, "text": text} for doc_id, text in pooled_docs_dict.items()]
        
        if not pooled_docs_flashrank:
            print(f"Warning: No documents retrieved/kept for query {i}. Skipping metrics calculation.")
            eval_results[i]["baseline"] = {"precision": 0.0, "recall": 0.0, "f1": 0.0}
            eval_results[i]["reranked"] = {"precision": 0.0, "recall": 0.0, "f1": 0.0}
            continue # Skip to next query

        if args.verbose:
            print(f"Total unique documents pooled for reranking: {len(pooled_docs_flashrank)}")

        # 3. Baseline Evaluation (Entire Pooled Document Set)
        # Extract IDs from the final pool
        all_pooled_ids = [doc['id'] for doc in pooled_docs_flashrank]
        # Take the first top_r IDs from the constructed pool (order based on dict iteration)
        # baseline_ids_at_r = all_pooled_ids[:args.top_r] # REMOVE THIS SLICING
        baseline_metrics = calculate_metrics(all_pooled_ids, ground_truth_ids) # Use all pooled IDs
        eval_results[i]["baseline"] = baseline_metrics
        if args.verbose:
             # print(f"Baseline Metrics (Pooled Docs @{args.top_r}): ") # Old description
             print(f"Baseline Metrics (Entire Pool [{len(all_pooled_ids)} docs]): " # New description
                   f"P={baseline_metrics['precision']:.3f}, "
                   f"R={baseline_metrics['recall']:.3f}, "
                   f"F1={baseline_metrics['f1']:.3f}")

        # 4. Rerank using selected mode
        reranked_ids = []
        try:
            if args.reranker_mode == 'flashrank':
                if args.verbose:
                    print(f"Reranking {len(pooled_docs_flashrank)} documents with FlashRank...")
                request = RerankRequest(query=orig_query, passages=pooled_docs_flashrank)
                full_reranked_results = reranker_model.rerank(request) # Use reranker_model
                reranked_results = full_reranked_results[:args.top_r]
                reranked_ids = [result['id'] for result in reranked_results]
                if args.verbose:
                    print(f"  FlashRank selected IDs: {reranked_ids}")
                    
            elif args.reranker_mode == 'mmr':
                # Add NumPy import if MMR is selected
                import numpy as np
                if args.verbose:
                    print(f"Reranking {len(pooled_docs_flashrank)} documents with Manual MMR (lambda={args.mmr_lambda})...")
                
                if pooled_docs_flashrank:
                    # Extract texts for embedding
                    doc_texts = [doc['text'] for doc in pooled_docs_flashrank]
                    doc_ids = [doc['id'] for doc in pooled_docs_flashrank]
                    
                    # Embed the query and documents
                    query_embedding = reranker_model.encode(orig_query, convert_to_tensor=True)
                    doc_embeddings = reranker_model.encode(doc_texts, convert_to_tensor=True)
                    
                    # Calculate Cosine Similarities
                    query_doc_similarity = cos_sim(query_embedding, doc_embeddings)[0] # Shape (1, N) -> (N,)
                    doc_doc_similarity = cos_sim(doc_embeddings, doc_embeddings)
                    
                    # --- Manual MMR Implementation ---
                    selected_indices = []
                    candidates_indices = list(range(len(doc_ids)))
                    
                    # Ensure query_doc_similarity is usable for indexing/ranking
                    if hasattr(query_doc_similarity, 'cpu'): # Handle potential GPU tensor
                        query_doc_similarity = query_doc_similarity.cpu()
                    if hasattr(doc_doc_similarity, 'cpu'):
                        doc_doc_similarity = doc_doc_similarity.cpu()
                        
                    query_doc_similarity_numpy = query_doc_similarity.numpy() # Convert to NumPy if needed
                    doc_doc_similarity_numpy = doc_doc_similarity.numpy()

                    # Select the first document (most relevant to query)
                    if candidates_indices:
                        first_selection_idx = candidates_indices[query_doc_similarity_numpy[candidates_indices].argmax()]
                        selected_indices.append(first_selection_idx)
                        candidates_indices.remove(first_selection_idx)

                    # Iteratively select remaining documents
                    while len(selected_indices) < args.top_r and candidates_indices:
                        mmr_scores = []
                        for idx in candidates_indices:
                            relevance_score = query_doc_similarity_numpy[idx]
                            # Similarity to already selected documents
                            similarity_to_selected = doc_doc_similarity_numpy[idx][selected_indices].max()
                            
                            mmr_score = args.mmr_lambda * relevance_score - (1 - args.mmr_lambda) * similarity_to_selected
                            mmr_scores.append(mmr_score)
                        
                        # Select the document with the highest MMR score
                        best_candidate_local_idx = np.argmax(mmr_scores)
                        best_candidate_global_idx = candidates_indices[best_candidate_local_idx]
                        
                        selected_indices.append(best_candidate_global_idx)
                        candidates_indices.remove(best_candidate_global_idx)
                    # --- End Manual MMR Implementation ---
                    
                    # Get the corresponding document IDs
                    reranked_ids = [doc_ids[idx] for idx in selected_indices]
                    if args.verbose:
                        print(f"  MMR selected IDs: {reranked_ids}")
                else:
                     if args.verbose:
                         print("  No documents to rerank with MMR.")
                     reranked_ids = []
            
            elif args.reranker_mode == 'llm':
                if args.verbose:
                    print(f"Reranking {len(pooled_docs_flashrank)} documents with LLM Summaries...")
                 
                if pooled_docs_flashrank:
                    # Prepare list with frequency for summarization
                    docs_with_frequency = [
                        (doc['id'], doc['text'], doc_frequency.get(doc['id'], 1)) 
                        for doc in pooled_docs_flashrank
                    ]
                    
                    # Generate summaries focused on the original query
                    summaries_with_freq = []
                    print(f"  Generating summaries for {len(docs_with_frequency)} documents (using original query focus)...")
                    for doc_id, text, freq in docs_with_frequency:
                        summary = llm_validator_summarizer.summarize_document(query=orig_query, document_text=text)
                        if summary: # Only keep documents with non-empty summaries
                            summaries_with_freq.append((doc_id, summary, freq))
                        elif args.verbose:
                            print(f"    -> Discarding document {doc_id} due to empty summary for original query.")
                    print(f"  Kept {len(summaries_with_freq)} summaries for LLM reranking.")
                    
                    if summaries_with_freq:
                         # Call LLMReranker's select_top_r_documents
                         selected_summaries = llm_validator_summarizer.select_top_r_documents(
                            query=orig_query,
                            documents=summaries_with_freq, # Pass (id, summary, freq)
                            top_r=args.top_r
                         )
                         # Extract IDs from the result (which is list of (id, summary))
                         reranked_ids = [doc_id for doc_id, summary in selected_summaries]
                         if args.verbose:
                             print(f"  LLM Reranker selected IDs: {reranked_ids}")
                    else:
                         if args.verbose:
                            print("  No non-empty summaries generated for LLM reranking.")
                         reranked_ids = []
                else:
                    if args.verbose:
                        print("  No documents to rerank with LLM.")
                    reranked_ids = []

        except Exception as e:
            print(f"  Warning: Reranking failed for query {i} using mode '{args.reranker_mode}': {e}. Setting reranked metrics to 0.")
            reranked_ids = []

        # 5. Reranked Evaluation
        reranked_metrics = calculate_metrics(reranked_ids, ground_truth_ids)
        eval_results[i]["reranked"] = reranked_metrics
        if args.verbose:
            print(f"Reranked Metrics (@{args.top_r}):              "
                  f"P={reranked_metrics['precision']:.3f}, "
                  f"R={reranked_metrics['recall']:.3f}, "
                  f"F1={reranked_metrics['f1']:.3f}")
                  
        end_time = time.time()
        query_time = end_time - start_time
        total_time += query_time
        if args.verbose:
            print(f"Time taken for query {i}: {query_time:.2f} seconds")
            print("--------------------")

    # --- Overall Metrics Calculation ---
    print("\n========== OVERALL METRICS SUMMARY ==========")
    
    avg_baseline_p, avg_baseline_r, avg_baseline_f1 = 0.0, 0.0, 0.0
    avg_reranked_p, avg_reranked_r, avg_reranked_f1 = 0.0, 0.0, 0.0
    num_queries_evaluated = len(eval_results)

    if num_queries_evaluated > 0:
        for i in eval_results:
            avg_baseline_p += eval_results[i]["baseline"]["precision"]
            avg_baseline_r += eval_results[i]["baseline"]["recall"]
            avg_baseline_f1 += eval_results[i]["baseline"]["f1"]
            avg_reranked_p += eval_results[i]["reranked"]["precision"]
            avg_reranked_r += eval_results[i]["reranked"]["recall"]
            avg_reranked_f1 += eval_results[i]["reranked"]["f1"]

        avg_baseline_p /= num_queries_evaluated
        avg_baseline_r /= num_queries_evaluated
        avg_baseline_f1 /= num_queries_evaluated
        avg_reranked_p /= num_queries_evaluated
        avg_reranked_r /= num_queries_evaluated
        avg_reranked_f1 /= num_queries_evaluated

        print(f"Evaluated {num_queries_evaluated} queries.")
        print("\nAverage Baseline Metrics (Entire Pool):") # Update description
        # Multiply by 100 for percentage display
        print(f"  Precision: {avg_baseline_p * 100:.2f}%")
        print(f"  Recall:    {avg_baseline_r * 100:.2f}%")
        print(f"  F1-Score:  {avg_baseline_f1 * 100:.2f}%")

        print("\nAverage Reranked Metrics (FlashRank/MMR/LLM @ top_r):") # Update description
        # Multiply by 100 for percentage display
        print(f"  Precision: {avg_reranked_p * 100:.2f}%")
        print(f"  Recall:    {avg_reranked_r * 100:.2f}%")
        print(f"  F1-Score:  {avg_reranked_f1 * 100:.2f}%")
        
        # Improvement Calculation
        precision_diff = avg_reranked_p - avg_baseline_p
        recall_diff = avg_reranked_r - avg_baseline_r
        f1_diff = avg_reranked_f1 - avg_baseline_f1
        
        # Use absolute diff * 100 for % point difference
        # Use relative diff for percentage change
        precision_pct = (precision_diff / avg_baseline_p) * 100 if avg_baseline_p > 0 else float('inf')
        recall_pct = (recall_diff / avg_baseline_r) * 100 if avg_baseline_r > 0 else float('inf')
        f1_pct = (f1_diff / avg_baseline_f1) * 100 if avg_baseline_f1 > 0 else float('inf')

        print("\nImprovement over Baseline:")
        print(f"  Precision: {precision_diff * 100:.2f}% points ({precision_pct:+.1f}%)")
        print(f"  Recall:    {recall_diff * 100:.2f}% points ({recall_pct:+.1f}%)")
        print(f"  F1-Score:  {f1_diff * 100:.2f}% points ({f1_pct:+.1f}%)")
        
    else:
        print("No queries were successfully evaluated.")

    print(f"\nTotal execution time: {total_time:.2f} seconds")
    if num_queries_evaluated > 0:
        print(f"Average time per query: {total_time / num_queries_evaluated:.2f} seconds")
    print("===========================================")


if __name__ == "__main__":
    main() 
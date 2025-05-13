"""Reranks retrieved chunks using a complementarity graph approach and evaluates recall."""
import random
import argparse
import json
import time
from typing import List, Set, Dict, Any, Tuple
from tqdm import tqdm
import numpy as np
import networkx as nx
from sentence_transformers import SentenceTransformer
from sentence_transformers.util import cos_sim

# Imports from the project structure
from query_expansion_ola import QueryExpander # Assuming this is the correct location

# Configure logging
import logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

# Attempt to suppress HTTP client logs more forcefully
logging.getLogger("httpx").setLevel(logging.WARNING)
logging.getLogger("httpcore").setLevel(logging.WARNING) # httpcore is often used by httpx
logging.getLogger("ollama").setLevel(logging.WARNING) # Add ollama just in case

# Reuse recall calculation function
def calculate_recall_at_k(retrieved_ids: List[str], ground_truth_ids: Set[str], k_values: List[int]) -> Dict[int, float]:
    """Calculates recall at different values of k."""
    recall_results = {}
    num_ground_truth = len(ground_truth_ids)

    if num_ground_truth == 0:
        # Return 0 recall if no ground truth
        for k in k_values:
            recall_results[k] = 0.0
        return recall_results

    retrieved_set = set(retrieved_ids) # Use set for efficient lookup if needed later
    for k in k_values:
        # Ensure k does not exceed the number of retrieved documents
        actual_k = min(k, len(retrieved_ids))
        if actual_k == 0:
            recall_results[k] = 0.0
            continue

        # Get the top-k retrieved IDs *from the ordered list*
        top_k_ids_set = set(retrieved_ids[:actual_k])

        # Calculate true positives for this k
        true_positives_at_k = len(top_k_ids_set.intersection(ground_truth_ids))

        # Calculate recall@k
        recall_at_k = true_positives_at_k / num_ground_truth
        recall_results[k] = recall_at_k

    return recall_results

# --- Graph Building based on Complementarity to Different Subqueries --- 
def build_complementarity_graph(
    chunks: List[Dict[str, Any]], 
    chunk_sq_sims: np.ndarray, # Precomputed [num_chunks x num_subqueries]
    min_best_sq_relevance_threshold: float
) -> nx.Graph:
    """Builds a weighted graph connecting chunks relevant to *different* subqueries."""
    graph = nx.Graph()
    num_chunks, num_subqueries = chunk_sq_sims.shape

    if num_chunks < 2 or num_subqueries == 0:
        # Add nodes even if no edges can be formed
        for i, chunk in enumerate(chunks):
            graph.add_node(i, chunk_id=chunk.get('chunk_id'), doc_id=chunk.get('doc_id'))
        return graph

    # Find the best subquery index and score for each chunk
    best_sq_indices = np.argmax(chunk_sq_sims, axis=1)
    max_sim_scores = np.max(chunk_sq_sims, axis=1)

    # Add nodes and store relevance info
    node_relevance_info = {}
    for i, chunk in enumerate(chunks):
        is_relevant = max_sim_scores[i] > min_best_sq_relevance_threshold
        graph.add_node(i, 
                       chunk_id=chunk.get('chunk_id'), 
                       doc_id=chunk.get('doc_id'),
                       best_sq_idx=best_sq_indices[i],
                       max_sim=max_sim_scores[i],
                       is_relevant=is_relevant)
        if is_relevant:
             node_relevance_info[i] = {'best_sq_idx': best_sq_indices[i], 'max_sim': max_sim_scores[i]}

    # Add weighted edges based on complementarity
    relevant_indices = list(node_relevance_info.keys())
    for idx1 in range(len(relevant_indices)):
        i = relevant_indices[idx1]
        info_i = node_relevance_info[i]
        for idx2 in range(idx1 + 1, len(relevant_indices)):
            j = relevant_indices[idx2]
            info_j = node_relevance_info[j]
            
            # Condition 1: Both relevant (implicit by iterating relevant_indices)
            # Condition 2: Best subqueries are different
            if info_i['best_sq_idx'] != info_j['best_sq_idx']:
                # Edge weight: Average of their max similarities to their *respective* best subqueries
                edge_weight = (info_i['max_sim'] + info_j['max_sim']) / 2.0
                graph.add_edge(i, j, weight=edge_weight)

    return graph

# --- Reranking using Personalized PageRank on the Weighted Graph --- 
def rerank_docs_by_pagerank(
    graph: nx.Graph, 
    chunks: List[Dict[str, Any]], 
    chunk_orig_query_sim: np.ndarray # Similarity of each chunk to the ORIGINAL query
) -> List[str]:
    """Reranks parent docs based on PPR scores, personalized by original query sim."""
    if not graph.nodes or len(chunk_orig_query_sim) != len(chunks):
        logging.warning("Cannot compute PPR. Empty graph or missing/mismatched similarity data.")
        # Fallback: Return documents based on original chunk order
        fallback_ids = [] # ... (fallback logic remains the same) ...
        seen_fallback = set()
        for ch in chunks:
            p_id = ch.get('doc_id')
            if p_id and p_id not in seen_fallback:
                seen_fallback.add(p_id)
                fallback_ids.append(p_id)
        return fallback_ids

    num_chunks = len(chunks)
    
    # Create personalization vector based on similarity to ORIGINAL query
    personalization_raw = {i: chunk_orig_query_sim[i] if i < num_chunks else 0.0 for i in graph.nodes()} 
    
    # Normalize the personalization vector
    norm_factor = sum(personalization_raw.values())
    if norm_factor > 0:
        personalization_dict = {node: score / norm_factor for node, score in personalization_raw.items()}
    else:
        logging.warning("All original query similarities are zero/negative. Falling back to uniform teleportation.")
        personalization_dict = None 

    # Calculate Personalized PageRank on the WEIGHTED graph
    try:
        pagerank_scores = nx.pagerank(graph, alpha=0.85, personalization=personalization_dict, weight='weight')
    except nx.PowerIterationFailedConvergence:
         logging.warning("Personalized PageRank failed to converge, using degree centrality fallback.")
         # Fallback: Use weighted degree centrality if graph has weights
         if nx.is_weighted(graph):
             degrees = graph.degree(weight='weight')
         else:
             degrees = graph.degree()
         total_degree = sum(dict(degrees).values())
         pagerank_scores = {node: degree / total_degree for node, degree in degrees} if total_degree > 0 else {node: 0 for node in graph.nodes()}

    # Aggregate scores per parent document (max score)
    doc_scores = {}
    for node_index, score in pagerank_scores.items():
        if node_index < num_chunks:
            parent_doc_id = chunks[node_index].get('doc_id')
            if parent_doc_id:
                doc_scores[parent_doc_id] = max(doc_scores.get(parent_doc_id, 0), score)

    # Sort documents by aggregated score
    sorted_docs = sorted(doc_scores.items(), key=lambda item: item[1], reverse=True)
    reranked_doc_ids = [doc_id for doc_id, score in sorted_docs]
    return reranked_doc_ids

# --- Helper function for processing queries with specific params --- 
def evaluate_params(
    query_indices: List[str], 
    retrieved_data: Dict[str, Dict],
    query_expander: QueryExpander,
    embed_model: SentenceTransformer,
    max_subqueries: int,
    # Renamed relevance threshold
    min_best_sq_relevance: float, 
    k_values_to_calculate: List[int],
    verbose: bool = False
) -> Tuple[Dict[int, float], float, int, int]:
    """Processes queries using the new graph logic and returns average recall."""
    
    all_recall_results = {k: [] for k in k_values_to_calculate}
    total_processing_time = 0.0
    queries_processed_count = 0
    queries_with_errors = 0

    # Removed max_chunk_similarity from description
    progress_bar = tqdm(query_indices, desc=f"Eval (MinBestSQRel={min_best_sq_relevance:.2f})", unit="query", leave=False)

    for query_idx_str in progress_bar:
        query_data = retrieved_data.get(query_idx_str)
        if not query_data:
            continue

        start_time = time.time()
        try:
            original_query = query_data.get("query", "")
            chunks = query_data.get("retrieved_chunks", [])
            ground_truth_ids = set(query_data.get("ground_truth_ids", []))
            recall_results = {} 

            if not chunks or not original_query:
                recall_results = calculate_recall_at_k([], ground_truth_ids, k_values_to_calculate)
            else:
                try:
                    # --- Query Expansion --- 
                    expansion_result = query_expander.batch_query_expansion([original_query])[0]
                    subqueries = expansion_result.expanded_queries[:max_subqueries]
                    if verbose: 
                        print(f"  Query {query_idx_str}: Original = '{original_query}'")
                        print(f"  Query {query_idx_str}: Subqueries = {subqueries}")
                        
                    # --- Fallback if no subqueries --- 
                    if not subqueries:
                         logging.warning(f"Query {query_idx_str}: No subqueries. Using original retrieval order.")
                         # ... (Fallback logic remains the same) ...
                         chunk_ids_for_recall = []; seen_ids_fallback = set()
                         for ch in chunks: 
                             r_id = ch.get('chunk_id') if ch.get('chunk_id') is not None else ch.get('doc_id'); p_id = ch.get('doc_id')
                             if r_id not in seen_ids_fallback: 
                                 seen_ids_fallback.add(r_id)
                                 if p_id and p_id not in chunk_ids_for_recall: chunk_ids_for_recall.append(p_id)
                         recall_results = calculate_recall_at_k(chunk_ids_for_recall, ground_truth_ids, k_values_to_calculate)
                    
                    # --- Proceed with Graph Reranking --- 
                    else:
                        # Embed chunks, original query, and subqueries
                        chunk_texts = [chunk.get("text", "") for chunk in chunks]
                        all_embeddings = embed_model.encode([original_query] + subqueries + chunk_texts, show_progress_bar=False)
                        
                        # Separate embeddings
                        original_query_embedding = all_embeddings[0:1]
                        subquery_embeddings = all_embeddings[1:1+len(subqueries)]
                        chunk_embeddings = all_embeddings[1+len(subqueries):]
                        
                        # Calculate necessary similarities ONCE
                        chunk_orig_query_sim = cos_sim(chunk_embeddings, original_query_embedding).numpy().flatten()
                        chunk_sq_sims = cos_sim(chunk_embeddings, subquery_embeddings).numpy()
                        
                        # Build complementarity graph (passes chunk-sq similarities)
                        graph = build_complementarity_graph(chunks, chunk_sq_sims, min_best_sq_relevance)
                                                            
                        # Rerank parent documents using Personalized PageRank (passes original query sims)
                        reranked_parent_doc_ids = rerank_docs_by_pagerank(graph, chunks, chunk_orig_query_sim)
                        
                        # Calculate recall based on reranked parent docs
                        recall_results = calculate_recall_at_k(reranked_parent_doc_ids, ground_truth_ids, k_values_to_calculate)

                # --- Handle Query Expansion Errors --- 
                except Exception as qe_err:
                     logging.error(f"Query {query_idx_str}: Query expansion failed: {qe_err}. Using original retrieval order.")
                     # ... (Fallback logic remains the same) ...
                     chunk_ids_for_recall = []; seen_ids_fallback = set()
                     for ch in chunks: 
                          r_id = ch.get('chunk_id') if ch.get('chunk_id') is not None else ch.get('doc_id'); p_id = ch.get('doc_id')
                          if r_id not in seen_ids_fallback: 
                              seen_ids_fallback.add(r_id)
                              if p_id and p_id not in chunk_ids_for_recall: chunk_ids_for_recall.append(p_id)
                     recall_results = calculate_recall_at_k(chunk_ids_for_recall, ground_truth_ids, k_values_to_calculate)
            
            # Store results for averaging (only if ground truth exists and results calculated)
            if ground_truth_ids and recall_results:
                for k in k_values_to_calculate:
                    if k in recall_results:
                         all_recall_results[k].append(recall_results[k])
            queries_processed_count += 1 
            total_processing_time += (time.time() - start_time)

        except Exception as e:
            queries_with_errors += 1
            logging.error(f"Error processing query index {query_idx_str}: {e}", exc_info=True)
            continue
            
    progress_bar.close() 

    # Calculate final average recall
    final_avg_recall = {}
    valid_k_values = sorted([k for k in k_values_to_calculate if all_recall_results.get(k)])
    if valid_k_values:
        for k in valid_k_values:
             final_avg_recall[k] = np.mean(all_recall_results[k]) if all_recall_results[k] else 0.0
    else: # Handle case where no results were collected (e.g., all GT empty)
        for k in k_values_to_calculate: final_avg_recall[k] = 0.0
        logging.warning("No valid recall results were collected for this parameter set.")
        
    avg_time = total_processing_time / queries_processed_count if queries_processed_count > 0 else 0

    return final_avg_recall, avg_time, queries_processed_count, queries_with_errors

def main():
    parser = argparse.ArgumentParser(description="Rerank chunks using complementarity graph and calculate recall.")
    parser.add_argument("--input-file", type=str, default="retrieved_full_data.json",
                        help="Path to the input JSON file containing retrieved data.")
    # Re-add embedding model argument
    parser.add_argument("--embedding-model", type=str, default="all-MiniLM-L6-v2",
                        help="Sentence Transformer model for embedding chunks and subqueries.")
    # Updated argument name and help text
    parser.add_argument("--min-best-sq-relevance", type=float, default=0.6,
                        help="Min similarity for a chunk to its BEST subquery to be included in graph construction.")
    parser.add_argument("--max-subqueries", type=int, default=5,
                        help="Maximum number of subqueries to generate and use.")
    parser.add_argument("--sample", type=int, default=None,
                        help="Number of queries to randomly sample from the input file.")
    parser.add_argument("--verbose", action='store_true',
                        help="Print generated subqueries for each query.")
    # QueryExpander specific args (adjust if needed based on its __init__)
    parser.add_argument("--qe-provider", type=str, default='ollama', help="Provider for QueryExpander LLM.")
    parser.add_argument("--qe-model", type=str, default=None, help="Model name for QueryExpander LLM.")
    parser.add_argument("--qe-temp", type=float, default=0.5, help="Temperature for QueryExpander LLM.")
    parser.add_argument("--qe-max-workers", type=int, default=5, help="Max workers for QueryExpander.") # Smaller default? 

    args = parser.parse_args()

    # 1. Load Data
    logging.info(f"Loading data from: {args.input_file}")
    try:
        with open(args.input_file, 'r') as f:
            retrieved_data = json.load(f)
        logging.info(f"Loaded data for {len(retrieved_data)} queries.")
    except FileNotFoundError:
        logging.error(f"Error: Input file not found at {args.input_file}")
        exit(1)
    except json.JSONDecodeError:
        logging.error(f"Error: Could not decode JSON from {args.input_file}")
        exit(1)
    except Exception as e:
        logging.error(f"Error loading data: {e}")
        exit(1)

    # 2. Initialize Models (SentenceTransformer and QueryExpander)
    logging.info(f"Initializing embedding model: {args.embedding_model}")
    try:
        embed_model = SentenceTransformer(args.embedding_model)
    except Exception as e:
        logging.error(f"Error initializing embedding model '{args.embedding_model}': {e}")
        exit(1)
    
    logging.info(f"Initializing Query Expander (Provider: {args.qe_provider}, Model: {args.qe_model or 'default'})" )
    try:
        query_expander = QueryExpander(
            provider=args.qe_provider, 
            model_name=args.qe_model, 
            temperature=args.qe_temp,
            max_workers=args.qe_max_workers
            # Add batch_size if QueryExpander accepts it 
        )
    except Exception as e:
        logging.error(f"Error initializing QueryExpander: {e}")
        exit(1)

    # Determine indices to process
    query_indices = list(retrieved_data.keys())
    if args.sample is not None:
        if args.sample <= 0:
             logging.error("Error: --sample must be a positive integer.")
             exit(1)
        if args.sample > len(query_indices):
             logging.warning(f"Sample size ({args.sample}) is larger than total queries ({len(query_indices)}). Running on all queries.")
        else:
             query_indices = random.sample(query_indices, args.sample)
             logging.info(f"Randomly sampling {args.sample} queries.")
    else:
        logging.info(f"Processing all {len(query_indices)} queries.")

    k_values_to_calculate = [5, 10, 50, 100, 200] # Define k values for evaluation

    # --- Single Run --- 
    logging.info("Starting single run with provided parameters...")
    avg_recall_results, avg_time, count, errors = evaluate_params(
        query_indices=query_indices,
        retrieved_data=retrieved_data,
        query_expander=query_expander,
        embed_model=embed_model,
        max_subqueries=args.max_subqueries,
        min_best_sq_relevance=args.min_best_sq_relevance, # Use updated arg name
        k_values_to_calculate=k_values_to_calculate,
        verbose=args.verbose
    )
    
    logging.info("--- Final Average Recall Results (Subquery Complementarity Graph Reranking) ---")
    if count > 0:
        valid_k_values = sorted(avg_recall_results.keys())
        for k in valid_k_values:
             logging.info(f"Avg Recall@{k:<3}: {avg_recall_results[k]:.4f}")
        logging.info(f"\nSuccessfully processed {count} queries.")
        if errors > 0: logging.warning(f"Encountered errors in {errors} queries...")
        logging.info(f"Average processing time per successful query: {avg_time:.2f} seconds")
    else:
        logging.warning("No queries were successfully processed.")
        if errors > 0: logging.warning(f"Encountered errors in {errors} queries.")
    logging.info("--------------------------------------------------------")


if __name__ == "__main__":
    main()

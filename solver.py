"""Reranks retrieved documents using a subquery coverage optimization approach and evaluates recall."""
import random
import argparse
import json
import time
from typing import List, Set, Dict, Any, Tuple
from tqdm import tqdm
import numpy as np
import torch
from transformers import AutoTokenizer, AutoModelForMaskedLM

# Import QueryExpander from the correct file
try:
    from query_expansion_ola import QueryExpander # Corrected filename
except ImportError:
    print("Error: Could not import QueryExpander. Ensure query_expansion_ola.py is available.")
    exit(1)

# Import PuLP for optimization
import pulp

# Configure logging
import logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logging.getLogger("transformers.tokenization_utils_base").setLevel(logging.ERROR) # Suppress tokenizer warnings
logging.getLogger("httpx").setLevel(logging.WARNING)
logging.getLogger("httpcore").setLevel(logging.WARNING)
logging.getLogger("ollama").setLevel(logging.WARNING)

# Reuse recall calculation function
def calculate_recall_at_k(retrieved_ids: List[str], ground_truth_ids: Set[str], k_values: List[int]) -> Dict[int, float]:
    """Calculates recall at different values of k."""
    recall_results = {}
    num_ground_truth = len(ground_truth_ids)

    if num_ground_truth == 0:
        for k in k_values:
            recall_results[k] = 0.0
        return recall_results

    retrieved_set = set(retrieved_ids)
    for k in k_values:
        actual_k = min(k, len(retrieved_ids))
        if actual_k == 0:
            recall_results[k] = 0.0
            continue
        top_k_ids_set = set(retrieved_ids[:actual_k])
        true_positives_at_k = len(top_k_ids_set.intersection(ground_truth_ids))
        recall_at_k = true_positives_at_k / num_ground_truth
        recall_results[k] = recall_at_k
    return recall_results

# --- SPLADE Helper Function (Copied from splade.py) ---
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
                logging.error(f"Error embedding batch starting at index {i}: {e}")
                logging.warning(f"Skipping batch due to error. Output array shape might be affected.")

    return np.vstack(all_vectors) if all_vectors else np.array([[]])

# --- Sparse Similarity Matrix (Copied from graph.py / splade.py) ---
def sparse_similarity_matrix(vectors1: np.ndarray, vectors2: np.ndarray) -> np.ndarray:
    """Calculates dot product similarity matrix between two sets of vectors."""
    if vectors1.ndim < 2 or vectors2.ndim < 2 or vectors1.shape[0] == 0 or vectors2.shape[0] == 0 or vectors1.shape[1] != vectors2.shape[1]:
        logging.warning(f"Invalid shapes for similarity matrix: {vectors1.shape}, {vectors2.shape}")
        return np.empty((vectors1.shape[0], vectors2.shape[0]))
    return np.dot(vectors1, vectors2.T)

# --- Document Scoring and Coverage Calculation ---
def calculate_doc_scores_and_coverage(
    chunks: List[Dict[str, Any]], 
    all_queries: List[str], # Renamed from subqueries
    chunk_all_query_sims: np.ndarray, # Renamed from chunk_sq_sims
    query_relevance_threshold: float # Renamed from subquery_relevance_threshold
) -> Tuple[Dict[str, float], Dict[str, List[int]]]:
    """Calculates relevance score and query coverage for each parent document.
    
    Args:
        chunks: List of chunk dictionaries.
        all_queries: List of query strings (original + subqueries).
        chunk_all_query_sims: Numpy array of similarities (chunks x all_queries).
        query_relevance_threshold: Min similarity for a chunk to be considered relevant to a query.
        
    Returns:
        Tuple containing:
            - doc_relevance_scores (Dict[str, float]): Max SPLADE score for each parent doc to any query.
            - doc_query_coverage (Dict[str, List[int]]): List of covered query indices for each parent doc.
    """
    doc_relevance_scores = {}
    doc_query_coverage = {}
    num_all_queries = len(all_queries)

    if chunk_all_query_sims.ndim < 2 or chunk_all_query_sims.shape[0] != len(chunks) or chunk_all_query_sims.shape[1] != num_all_queries:
        # Handle case where embedding might have failed upstream, return empty dicts
        # Or just calculate relevance based on chunk data if sims are empty
        if chunk_all_query_sims.ndim < 2 or chunk_all_query_sims.size == 0:
             for chunk in chunks: # Calculate placeholder relevance if needed
                  parent_doc_id = chunk.get('doc_id')
                  if parent_doc_id:
                       doc_relevance_scores[parent_doc_id] = doc_relevance_scores.get(parent_doc_id, 0) + 1 # Example placeholder score
             return doc_relevance_scores, {}
        else: # Shape mismatch is a real error
            logging.error(f"Mismatch between chunks/queries ({len(chunks)}/{num_all_queries}) and similarity matrix shape ({chunk_all_query_sims.shape}).")
            return {}, {}

    for i, chunk in enumerate(chunks):
        parent_doc_id = chunk.get('doc_id')
        if not parent_doc_id:
            continue

        # Calculate relevance (max score to any query for this chunk)
        max_chunk_score = np.max(chunk_all_query_sims[i, :]) if chunk_all_query_sims.shape[1] > 0 else -float('inf')
        doc_relevance_scores[parent_doc_id] = max(doc_relevance_scores.get(parent_doc_id, -float('inf')), max_chunk_score)

        # Calculate coverage
        covered_indices = set(doc_query_coverage.get(parent_doc_id, [])) # Use set for efficient adding
        for q_idx in range(num_all_queries):
            if chunk_all_query_sims[i, q_idx] > query_relevance_threshold:
                covered_indices.add(q_idx)
        
        if covered_indices:
             doc_query_coverage[parent_doc_id] = list(covered_indices) # Convert back to list for consistency

    return doc_relevance_scores, doc_query_coverage

# --- Top-K Selection using Integer Programming ---
def select_top_5_optimized(
    doc_relevance_scores: Dict[str, float], 
    doc_query_coverage: Dict[str, List[int]], # Renamed from doc_subquery_coverage
    num_all_queries: int, # Renamed from num_subqueries
    relevance_weight: float = 0.1, # Weight for relevance score in objective
    must_include_doc_id: str | None = None, # <-- Add this parameter
    verbose: bool = False # Added verbose parameter
) -> List[str]:
    """Selects top 5 documents using PuLP to maximize coverage + relevance.
    
    Returns:
        List of selected document IDs.
    """
    candidate_doc_ids = list(doc_relevance_scores.keys())
    if not candidate_doc_ids:
        return []
    if len(candidate_doc_ids) <= 5:
         # If 5 or fewer docs, return them sorted by relevance (simple heuristic)
         return sorted(candidate_doc_ids, key=lambda doc_id: doc_relevance_scores.get(doc_id, -float('inf')), reverse=True)

    # --- Define the ILP Problem --- 
    prob = pulp.LpProblem("DocumentSelection", pulp.LpMaximize)

    # --- Decision Variables --- 
    # x_d = 1 if document d is selected, 0 otherwise
    doc_vars = pulp.LpVariable.dicts("Doc", candidate_doc_ids, 0, 1, pulp.LpBinary)
    
    # y_q = 1 if query q (original or subquery) is covered by at least one selected document, 0 otherwise
    query_vars = pulp.LpVariable.dicts("QueryCovered", range(num_all_queries), 0, 1, pulp.LpBinary) # Use num_all_queries

    # --- Objective Function --- 
    # Maximize: (Sum over q of y_q) + relevance_weight * (Sum over d of x_d * relevance_d)
    prob += (pulp.lpSum(query_vars[q_idx] for q_idx in range(num_all_queries)) + # Sum over num_all_queries
             relevance_weight * pulp.lpSum(doc_vars[doc_id] * doc_relevance_scores.get(doc_id, 0) 
                                           for doc_id in candidate_doc_ids)), "TotalScore"

    # --- Constraints --- 
    # 1. Select exactly 5 documents
    prob += pulp.lpSum(doc_vars[doc_id] for doc_id in candidate_doc_ids) == 5, "SelectExactly5"

    # 1.b. Force include the top document if specified and valid
    if must_include_doc_id and must_include_doc_id in doc_vars:
         prob += doc_vars[must_include_doc_id] == 1, f"ForceInclude_{must_include_doc_id}"
         if verbose:
             print(f"    -> Forcing inclusion of doc: {must_include_doc_id}") # Optional verbose print

    # 2. Link query coverage variables (y_q) to document selection (x_d)
    # y_q <= Sum(x_d for d covering q)
    for q_idx in range(num_all_queries): # Iterate over num_all_queries
        docs_covering_q = [doc_id for doc_id, covered_qs in doc_query_coverage.items() if q_idx in covered_qs] # Use doc_query_coverage
        if docs_covering_q:
            prob += query_vars[q_idx] <= pulp.lpSum(doc_vars[doc_id] for doc_id in docs_covering_q), f"CoverageLink_Q{q_idx}"
        else:
            # If no document covers this query, its coverage variable must be 0
            prob += query_vars[q_idx] == 0, f"CoverageLink_Q{q_idx}_NoDocs"

    # --- Solve the problem --- 
    # Suppress solver messages regardless of script verbosity
    solver = pulp.PULP_CBC_CMD(msg=False) 
    prob.solve(solver)

    # --- Extract Results --- 
    selected_doc_ids = []
    if pulp.LpStatus[prob.status] == 'Optimal':
        for doc_id in candidate_doc_ids:
            if doc_vars[doc_id].varValue > 0.5: # Check if binary variable is selected
                selected_doc_ids.append(doc_id)
        # Optional: Sort the final 5 by original relevance for stable ordering?
        selected_doc_ids.sort(key=lambda doc_id: doc_relevance_scores.get(doc_id, -float('inf')), reverse=True)
        
        if len(selected_doc_ids) != 5:
             logging.warning(f"Solver found optimal solution but selected {len(selected_doc_ids)} != 5 documents. Falling back to top 5 by relevance.")
             selected_doc_ids = sorted(candidate_doc_ids, key=lambda doc_id: doc_relevance_scores.get(doc_id, -float('inf')), reverse=True)[:5]
             
    else:
        logging.warning(f"Optimization problem status: {pulp.LpStatus[prob.status]}. Falling back to top 5 by relevance.")
        # Fallback if solver fails
        selected_doc_ids = sorted(candidate_doc_ids, key=lambda doc_id: doc_relevance_scores.get(doc_id, -float('inf')), reverse=True)[:5]
        
    # Ensure exactly 5 are returned in fallback cases too
    return selected_doc_ids[:5] 


# --- Evaluation Helper Function ---
def evaluate_solver_reranking(
    query_indices: List[str],
    retrieved_data: Dict[str, Dict],
    query_expander: QueryExpander,
    splade_model,
    splade_tokenizer,
    device: str,
    max_subqueries: int,
    subquery_relevance_threshold: float,
    relevance_weight: float,
    k_values_to_calculate: List[int],
    verbose: bool = False,
    debug: bool = False # Add debug flag
) -> Tuple[Dict[int, float], float, int, int, List[Dict[str, Any]], float, int, float, int, float, float, float]:
    """Processes queries using the solver reranking strategy."""

    all_recall_results = {k: [] for k in k_values_to_calculate}
    detailed_results_list = []
    total_processing_time = 0.0
    queries_processed_count = 0
    queries_with_errors = 0
    running_recall_at_5_sum = 0.0 # Keep track of sum for running average
    single_doc_recall_5_sum = 0.0
    single_doc_count = 0
    multi_doc_recall_5_sum = 0.0
    multi_doc_count = 0
    # Add tracking for initial recall
    initial_recall_at_5_sum = 0.0
    initial_single_doc_recall_5_sum = 0.0
    initial_multi_doc_recall_5_sum = 0.0
    # counts are the same (single_doc_count, multi_doc_count, queries_processed_count)

    progress_bar = tqdm(query_indices, desc="Eval (Solver Rerank)", unit="query", leave=False)

    for query_idx_str in progress_bar:
        query_data = retrieved_data.get(query_idx_str)
        if not query_data:
            continue

        start_time = time.time()
        query_detail = {
            "query_id": query_idx_str,
            "query": "",
            "ground_truth_ids": [],
            "subqueries_generated": [],
            "selected_doc_ids": [], # IDs selected by the solver
            # We could add doc scores and coverage here if needed for analysis
        }
        try:
            original_query = query_data.get("query", "")
            chunks = query_data.get("retrieved_chunks", [])
            ground_truth_ids = set(query_data.get("ground_truth_ids", []))
            recall_results = {}

            query_detail["query"] = original_query
            query_detail["ground_truth_ids"] = list(ground_truth_ids)

            # --- Calculate Initial Recall@5 (Before Solver) --- 
            initial_top_5_doc_ids = []
            seen_initial_doc_ids = set()
            if chunks: # Only calculate if chunks exist
                for ch in chunks:
                    p_id = ch.get('doc_id')
                    if p_id and p_id not in seen_initial_doc_ids:
                        seen_initial_doc_ids.add(p_id)
                        initial_top_5_doc_ids.append(p_id)
                        if len(initial_top_5_doc_ids) == 5: break
            initial_recall_results = calculate_recall_at_k(initial_top_5_doc_ids, ground_truth_ids, [5]) # Only need k=5
            initial_recall_5 = initial_recall_results.get(5, 0.0)
            # Update initial sums
            initial_recall_at_5_sum += initial_recall_5
            if len(ground_truth_ids) == 1:
                initial_single_doc_recall_5_sum += initial_recall_5
            elif len(ground_truth_ids) > 1:
                initial_multi_doc_recall_5_sum += initial_recall_5
            # --- End Initial Recall Calculation ---

            if not chunks or not original_query:
                recall_results = calculate_recall_at_k([], ground_truth_ids, k_values_to_calculate)
                query_detail["selected_doc_ids"] = []
            else:
                # --- Query Expansion --- 
                expansion_result = query_expander.batch_query_expansion([original_query])[0]
                subqueries = expansion_result.expanded_queries[:max_subqueries]
                query_detail["subqueries_generated"] = subqueries
                num_subqueries = len(subqueries)
                
                # Print subqueries if verbose
                if verbose:
                    print(f"  Query {query_idx_str}: Original = '{original_query}'")
                    print(f"  Query {query_idx_str}: Subqueries = {subqueries}")

                if not subqueries:
                    logging.warning(f"Query {query_idx_str}: No subqueries generated. Cannot use solver. Falling back to original chunk order (top 5 docs).")
                    # Fallback: Top 5 unique parent docs from original chunk list
                    fallback_ids = []
                    seen_fallback = set()
                    for ch in chunks:
                        p_id = ch.get('doc_id')
                        if p_id and p_id not in seen_fallback:
                            seen_fallback.add(p_id)
                            fallback_ids.append(p_id)
                            if len(fallback_ids) == 5: break
                    recall_results = calculate_recall_at_k(fallback_ids, ground_truth_ids, k_values_to_calculate)
                    query_detail["selected_doc_ids"] = fallback_ids
                else:
                    # Combine original query with subqueries
                    all_queries = [original_query] + subqueries
                    num_all_queries = len(all_queries)
                    
                    # --- SPLADE Embeddings --- 
                    chunk_texts = [chunk.get("text", "") for chunk in chunks]
                    all_texts_to_embed = all_queries + chunk_texts # Embed all queries + chunks
                    all_vectors = get_splade_vectors(all_texts_to_embed, splade_model, splade_tokenizer, device)
                    
                    if all_vectors.shape[0] != len(all_texts_to_embed):
                         # Log error and fallback to relevance
                         logging.error("SPLADE embedding returned incorrect number of vectors. Falling back to top 5 by relevance.")
                         doc_relevance_scores, _ = calculate_doc_scores_and_coverage(chunks, all_queries, np.array([[]]), subquery_relevance_threshold) # Call with empty sims to get scores
                         selected_doc_ids = sorted(doc_relevance_scores.keys(), key=lambda doc_id: doc_relevance_scores.get(doc_id, -float('inf')), reverse=True)[:5]
                         query_detail["selected_doc_ids"] = selected_doc_ids
                         recall_results = calculate_recall_at_k(selected_doc_ids, ground_truth_ids, k_values_to_calculate)
                         # Skip the rest of the loop for this query if embedding failed
                    else:
                        all_query_vectors = all_vectors[:num_all_queries] # Vectors for original + subqueries
                        chunk_vectors = all_vectors[num_all_queries:]

                        # --- Calculate Similarities (Chunks vs. All Queries) --- 
                        chunk_all_query_sims = sparse_similarity_matrix(chunk_vectors, all_query_vectors)

                        # --- Identify Top Document for Original Query --- 
                        top_orig_doc_id = None
                        max_orig_score = -float('inf')
                        if chunk_all_query_sims.shape[0] > 0: # Check if there are chunks/scores
                            chunk_orig_query_sims = chunk_all_query_sims[:, 0] # Similarities vs original query
                            for chunk_idx, chunk_dict in enumerate(chunks):
                                parent_doc_id = chunk_dict.get('doc_id')
                                if parent_doc_id and chunk_orig_query_sims[chunk_idx] > max_orig_score:
                                    max_orig_score = chunk_orig_query_sims[chunk_idx]
                                    top_orig_doc_id = parent_doc_id
                        # --- End Identification --- 

                        # --- Debug Print Logic --- 
                        if debug:
                            print(f"--- DEBUG Scores for Query {query_idx_str} ---")
                            ground_truth_set = set(ground_truth_ids) # Use set for faster lookup
                            for chunk_idx, chunk_dict in enumerate(chunks):
                                parent_doc_id = chunk_dict.get('doc_id')
                                if parent_doc_id in ground_truth_set:
                                    print(f"  Chunk {chunk_idx} (Parent: {parent_doc_id}):")
                                    # Query 0 is the original query
                                    print(f"    vs Orig Query (0): {chunk_all_query_sims[chunk_idx, 0]:.2f}")
                                    # Print scores for subqueries (indices 1 to num_all_queries-1)
                                    for q_idx in range(1, num_all_queries):
                                        print(f"    vs Subquery   ({q_idx}): {chunk_all_query_sims[chunk_idx, q_idx]:.2f} (Text: '{all_queries[q_idx]}')")
                            print("--- END DEBUG --- \n")
                        # --- End Debug --- 

                        # --- Calculate Doc Scores & Coverage (using All Queries) --- 
                        doc_relevance_scores, doc_query_coverage = calculate_doc_scores_and_coverage(
                            chunks, all_queries, chunk_all_query_sims, subquery_relevance_threshold # Pass all queries & sims
                        )

                        # --- Select Top 5 using Solver (using All Queries info) --- 
                        selected_doc_ids = select_top_5_optimized(
                            doc_relevance_scores, doc_query_coverage, num_all_queries, relevance_weight, top_orig_doc_id, verbose # Pass num_all_queries and top_orig_doc_id
                        )
                        query_detail["selected_doc_ids"] = selected_doc_ids

                        # --- Calculate Recall --- 
                        recall_results = calculate_recall_at_k(selected_doc_ids, ground_truth_ids, k_values_to_calculate)

            # Store results
            if ground_truth_ids and recall_results:
                for k in k_values_to_calculate:
                    if k in recall_results:
                        all_recall_results[k].append(recall_results[k])
                        # Update running Recall@5 if k is 5
                        if k == 5:
                             current_recall_5 = recall_results[k]
                             running_recall_at_5_sum += current_recall_5
                             # Update single/multi doc sums
                             if len(ground_truth_ids) == 1:
                                 single_doc_recall_5_sum += current_recall_5
                             elif len(ground_truth_ids) > 1:
                                 multi_doc_recall_5_sum += current_recall_5
            queries_processed_count += 1
            # Update single/multi doc counts
            if len(ground_truth_ids) == 1:
                single_doc_count += 1
                prefix = "(*) "
            elif len(ground_truth_ids) > 1:
                multi_doc_count += 1
                prefix = "(**) "
            else: # Should not happen if ground_truth_ids is always populated
                prefix = "(0) "
            total_processing_time += (time.time() - start_time)
            detailed_results_list.append(query_detail)

            # Display running Recall@5 with prefix
            if queries_processed_count > 0:
                current_avg_recall_5 = running_recall_at_5_sum / queries_processed_count
                current_avg_initial_recall_5 = initial_recall_at_5_sum / queries_processed_count # Calculate running avg for initial
                # Calculate type-specific running average
                type_avg_recall_5 = 0.0
                type_avg_initial_recall_5 = 0.0
                if prefix == "(*) " and single_doc_count > 0:
                    type_avg_recall_5 = single_doc_recall_5_sum / single_doc_count
                    type_avg_initial_recall_5 = initial_single_doc_recall_5_sum / single_doc_count
                elif prefix == "(**) " and multi_doc_count > 0:
                    type_avg_recall_5 = multi_doc_recall_5_sum / multi_doc_count
                    type_avg_initial_recall_5 = initial_multi_doc_recall_5_sum / multi_doc_count
                
                print(f"  {prefix}Query {query_idx_str}: R@5={recall_results.get(5, 0.0):.4f} (Init:{initial_recall_5:.4f}) | Avg R@5(Type)={type_avg_recall_5:.4f} (Init:{type_avg_initial_recall_5:.4f}) | Avg R@5(All)={current_avg_recall_5:.4f} (Init:{current_avg_initial_recall_5:.4f})")
            else: # First query
                 print(f"  {prefix}Query {query_idx_str}: Recall@5 = {recall_results.get(5, 0.0):.4f} (Initial: {initial_recall_5:.4f})")

        except Exception as e:
            queries_with_errors += 1
            logging.error(f"Error processing query index {query_idx_str}: {e}", exc_info=True)
            detailed_results_list.append(query_detail) # Store partial info on error
            continue

    progress_bar.close()

    # Calculate final average recall
    final_avg_recall = {}
    valid_k_values = sorted([k for k in k_values_to_calculate if all_recall_results.get(k)])
    if valid_k_values:
        for k in valid_k_values:
             final_avg_recall[k] = np.mean(all_recall_results[k]) if all_recall_results[k] else 0.0
    else:
        for k in k_values_to_calculate: final_avg_recall[k] = 0.0
        logging.warning("No valid recall results were collected for this run.")

    avg_time = total_processing_time / queries_processed_count if queries_processed_count > 0 else 0

    # Return the separate sums and counts as well, including initial recall sums
    return final_avg_recall, avg_time, queries_processed_count, queries_with_errors, detailed_results_list, \
           single_doc_recall_5_sum, single_doc_count, multi_doc_recall_5_sum, multi_doc_count, \
           initial_recall_at_5_sum, initial_single_doc_recall_5_sum, initial_multi_doc_recall_5_sum

def main():
    parser = argparse.ArgumentParser(description="Rerank using subquery coverage optimization (solver) and evaluate recall.")
    parser.add_argument("--input-file", type=str, default="retrieved_full_data_50.json", # Expects 500 chunks
                        help="Path to the input JSON file containing retrieved data.")
    # SPLADE Model argument
    parser.add_argument("--splade-model", type=str, default="naver/splade-cocondenser-ensembledistil", 
                        help="Hugging Face model name/path for SPLADE.")
    # Query Expansion arguments
    parser.add_argument("--qe-provider", type=str, default='ollama', help="Provider for QueryExpander LLM.")
    parser.add_argument("--qe-model", type=str, default=None, help="Model name for QueryExpander LLM.")
    parser.add_argument("--qe-temp", type=float, default=0.5, help="Temperature for QueryExpander LLM.")
    parser.add_argument("--qe-max-workers", type=int, default=5, help="Max workers for QueryExpander.")
    # Solver / Reranking parameters
    parser.add_argument("--max-subqueries", type=int, default=2,
                        help="Maximum number of subqueries to generate and use.")
    parser.add_argument("--query-relevance-threshold", type=float, default=15.0, # Needs tuning for SPLADE scores!
                        help="Min SPLADE score for a chunk to be considered relevant to ANY query (original or subquery) for coverage.")
    parser.add_argument("--relevance-weight", type=float, default=0.1,
                        help="Weight factor for document relevance score in the optimization objective.")
    # Other args
    parser.add_argument("--sample", type=int, default=None,
                        help="Number of queries to randomly sample from the input file.")
    parser.add_argument("--verbose", action='store_true',
                        help="Enable detailed logging and PuLP solver messages.")
    parser.add_argument("--output-json", type=str, default=None,
                        help="Path to save detailed reranking results as a JSON file.")
    # Add debug flag
    parser.add_argument("--debug", action='store_true',
                        help="Print SPLADE scores for ground truth document chunks against all queries.")

    args = parser.parse_args()

    # --- Setup Device ---
    device = 'cuda' if torch.cuda.is_available() else 'cpu' # Auto-detect CUDA
    logging.info(f"Using device: {device}")

    # --- Load Data ---
    logging.info(f"Loading data from: {args.input_file}")
    try:
        with open(args.input_file, 'r') as f:
            retrieved_data = json.load(f)
        logging.info(f"Loaded data for {len(retrieved_data)} queries.")
    except Exception as e:
        logging.error(f"Error loading data: {e}")
        return

    # --- Initialize Models --- 
    logging.info(f"Initializing SPLADE model: {args.splade_model}")
    try:
        splade_tokenizer = AutoTokenizer.from_pretrained(args.splade_model)
        splade_model = AutoModelForMaskedLM.from_pretrained(args.splade_model).to(device)
        splade_model.eval()
    except Exception as e:
        logging.error(f"Error initializing SPLADE model '{args.splade_model}': {e}")
        return
        
    logging.info(f"Initializing Query Expander (Provider: {args.qe_provider}, Model: {args.qe_model or 'default'})" )
    try:
        query_expander = QueryExpander(
            provider=args.qe_provider,
            model_name=args.qe_model,
            temperature=args.qe_temp,
            max_workers=args.qe_max_workers
        )
    except Exception as e:
        logging.error(f"Error initializing QueryExpander: {e}")
        return
        
    # --- Determine indices to process ---
    query_indices = list(retrieved_data.keys())
    if args.sample is not None:
        # ... (sampling logic as before) ...
        if args.sample <= 0:
             logging.error("Error: --sample must be a positive integer.")
             return
        if args.sample > len(query_indices):
             logging.warning(f"Sample size ({args.sample}) is larger than total queries ({len(query_indices)}). Running on all queries.")
        else:
             query_indices = random.sample(query_indices, args.sample)
             logging.info(f"Randomly sampling {args.sample} queries.")
    else:
        logging.info(f"Processing all {len(query_indices)} queries.")
        
    # k values for recall (we only select 5 docs, so only calculate up to k=5)
    k_values_to_calculate = [5] 

    # --- Run Evaluation --- 
    logging.info("Starting run with solver-based reranking...")
    avg_recall_results, avg_time, count, errors, detailed_results, \
    single_doc_recall_5_sum, single_doc_count, multi_doc_recall_5_sum, multi_doc_count, \
    initial_recall_at_5_sum, initial_single_doc_recall_5_sum, initial_multi_doc_recall_5_sum = evaluate_solver_reranking(
        query_indices=query_indices,
        retrieved_data=retrieved_data,
        query_expander=query_expander,
        splade_model=splade_model,
        splade_tokenizer=splade_tokenizer,
        device=device,
        max_subqueries=args.max_subqueries,
        subquery_relevance_threshold=args.query_relevance_threshold,
        relevance_weight=args.relevance_weight,
        k_values_to_calculate=k_values_to_calculate,
        verbose=args.verbose,
        debug=args.debug
    )

    # --- Results Output --- 
    logging.info(f"--- Final Average Recall Results (Solver Reranking: MaxSQ={args.max_subqueries}, SQThresh={args.query_relevance_threshold}, RelWgt={args.relevance_weight}) ---")
    if count > 0:
        # Calculate initial averages
        avg_initial_recall_5 = initial_recall_at_5_sum / count if count > 0 else 0.0
        avg_initial_single_doc_recall_5 = initial_single_doc_recall_5_sum / single_doc_count if single_doc_count > 0 else 0.0
        avg_initial_multi_doc_recall_5 = initial_multi_doc_recall_5_sum / multi_doc_count if multi_doc_count > 0 else 0.0
        
        # Calculate solver averages (already have overall from avg_recall_results)
        avg_solver_single_doc_recall_5 = single_doc_recall_5_sum / single_doc_count if single_doc_count > 0 else 0.0
        avg_solver_multi_doc_recall_5 = multi_doc_recall_5_sum / multi_doc_count if multi_doc_count > 0 else 0.0
        avg_solver_overall_recall_5 = avg_recall_results.get(5, 0.0)

        # Print comparison
        logging.info(f"Overall Avg Recall@5:       {avg_solver_overall_recall_5:.4f} (Initial: {avg_initial_recall_5:.4f})")
        logging.info(f"Single-Doc (*) Avg Recall@5: {avg_solver_single_doc_recall_5:.4f} (Initial: {avg_initial_single_doc_recall_5:.4f}) ({single_doc_count} queries)")
        logging.info(f"Multi-Doc (**) Avg Recall@5: {avg_solver_multi_doc_recall_5:.4f} (Initial: {avg_initial_multi_doc_recall_5:.4f}) ({multi_doc_count} queries)")
             
        # Print other details
        logging.info(f"\nSuccessfully processed {count} queries.")
        if errors > 0: logging.warning(f"Encountered errors in {errors} queries...")
        logging.info(f"Average processing time per successful query: {avg_time:.2f} seconds")
    else:
        logging.warning("No queries were successfully processed.")
        if errors > 0: logging.warning(f"Encountered errors in {errors} queries...")
    logging.info("--------------------------------------------------------")

    # --- Save detailed results --- 
    if args.output_json:
        if detailed_results:
            logging.info(f"Saving detailed results for {len(detailed_results)} queries to: {args.output_json}")
            try:
                with open(args.output_json, 'w') as f:
                    json.dump(detailed_results, f, indent=4)
                logging.info("Detailed results saved successfully.")
            except Exception as e:
                logging.error(f"Failed to save detailed results to {args.output_json}: {e}")
        else:
            logging.warning(f"No detailed results collected to save to {args.output_json}.")

if __name__ == "__main__":
    main() 
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

# --- Reranking by Average SPLADE Score --- 
def rerank_by_avg_splade_score(
    chunks: List[Dict[str, Any]],
    chunk_all_query_sims: np.ndarray # Similarity matrix: chunks x all_queries
) -> List[str]:
    """Reranks parent documents based on their average SPLADE score across all queries."""
    if chunk_all_query_sims.ndim < 2 or chunk_all_query_sims.shape[0] != len(chunks):
        logging.warning("Cannot rerank by average SPLADE score due to mismatched inputs.")
        # Fallback: Return top 5 unique parent docs from original chunk order
        fallback_ids = []
        seen_fallback = set()
        for ch in chunks:
            p_id = ch.get('doc_id')
            if p_id and p_id not in seen_fallback:
                seen_fallback.add(p_id)
                fallback_ids.append(p_id)
                if len(fallback_ids) == 5: break
        return fallback_ids

    # Aggregate scores per parent document
    doc_scores_sum = {} 
    doc_chunk_count = {} # Count chunks per doc to average later

    for i, chunk in enumerate(chunks):
        parent_doc_id = chunk.get('doc_id')
        if not parent_doc_id:
            continue
        
        # Sum scores for this chunk across all queries
        chunk_score_sum = np.sum(chunk_all_query_sims[i, :])
        
        doc_scores_sum[parent_doc_id] = doc_scores_sum.get(parent_doc_id, 0.0) + chunk_score_sum
        doc_chunk_count[parent_doc_id] = doc_chunk_count.get(parent_doc_id, 0) + 1

    # Calculate average score per document
    doc_avg_scores = {
        doc_id: doc_scores_sum[doc_id] / doc_chunk_count[doc_id]
        for doc_id in doc_scores_sum if doc_chunk_count[doc_id] > 0
    }

    # Sort documents by average score
    sorted_docs = sorted(doc_avg_scores.items(), key=lambda item: item[1], reverse=True)
    
    # Get top 5 document IDs
    reranked_doc_ids = [doc_id for doc_id, score in sorted_docs[:5]]
    
    return reranked_doc_ids

# --- Evaluation Helper Function ---
def evaluate_expander_reranking( # Renamed function
    query_indices: List[str],
    retrieved_data: Dict[str, Dict],
    query_expander: QueryExpander,
    splade_model,
    splade_tokenizer,
    device: str,
    max_subqueries: int,
    k_values_to_calculate: List[int],
    verbose: bool = False,
    debug: bool = False # Add debug flag - keep for now, might be useful
) -> Tuple[Dict[int, float], float, int, int, List[Dict[str, Any]], float, int, float, int, float, float, float]: # Signature unchanged
    """Processes queries using the expander + average SPLADE score reranking strategy."""

    # Initialize metrics (same as before)
    all_recall_results = {k: [] for k in k_values_to_calculate}
    detailed_results_list = []
    total_processing_time = 0.0
    queries_processed_count = 0
    queries_with_errors = 0
    running_recall_at_5_sum = 0.0
    single_doc_recall_5_sum = 0.0
    single_doc_count = 0
    multi_doc_recall_5_sum = 0.0
    multi_doc_count = 0
    initial_recall_at_5_sum = 0.0
    initial_single_doc_recall_5_sum = 0.0
    initial_multi_doc_recall_5_sum = 0.0

    progress_bar = tqdm(query_indices, desc="Eval (Expander Rerank)", unit="query", leave=False) # Updated desc

    for query_idx_str in progress_bar:
        # --- Load Data and Calc Initial Recall (same as before) ---
        query_data = retrieved_data.get(query_idx_str)
        if not query_data:
            continue

        start_time = time.time()
        query_detail = {
            "query_id": query_idx_str,
            "query": "",
            "ground_truth_ids": [],
            "subqueries_generated": [],
            "selected_doc_ids": [], # IDs selected by average score
        }
        try:
            original_query = query_data.get("query", "")
            chunks = query_data.get("retrieved_chunks", [])
            ground_truth_ids = set(query_data.get("ground_truth_ids", []))
            recall_results = {}

            query_detail["query"] = original_query
            query_detail["ground_truth_ids"] = list(ground_truth_ids)

            # --- Calculate Initial Recall@5 (Before Rerank) --- 
            initial_top_5_doc_ids = []
            seen_initial_doc_ids = set()
            if chunks: 
                for ch in chunks:
                    p_id = ch.get('doc_id')
                    if p_id and p_id not in seen_initial_doc_ids:
                        seen_initial_doc_ids.add(p_id)
                        initial_top_5_doc_ids.append(p_id)
                        if len(initial_top_5_doc_ids) == 5: break
            initial_recall_results = calculate_recall_at_k(initial_top_5_doc_ids, ground_truth_ids, [5]) 
            initial_recall_5 = initial_recall_results.get(5, 0.0)
            initial_recall_at_5_sum += initial_recall_5
            if len(ground_truth_ids) == 1:
                initial_single_doc_recall_5_sum += initial_recall_5
            elif len(ground_truth_ids) > 1:
                initial_multi_doc_recall_5_sum += initial_recall_5
            # --- End Initial Recall ---

            if not chunks or not original_query:
                recall_results = calculate_recall_at_k([], ground_truth_ids, k_values_to_calculate)
                query_detail["selected_doc_ids"] = []
            else:
                # --- Query Expansion (same as solver.py) ---
                expansion_result = query_expander.batch_query_expansion([original_query])[0]
                subqueries = expansion_result.expanded_queries[:max_subqueries]
                query_detail["subqueries_generated"] = subqueries
                # num_subqueries = len(subqueries) # Not needed directly
                
                if verbose:
                    print(f"  Query {query_idx_str}: Original = '{original_query}'")
                    print(f"  Query {query_idx_str}: Subqueries = {subqueries}")

                # Combine original query with subqueries
                all_queries = [original_query] + subqueries
                num_all_queries = len(all_queries)
                
                # --- SPLADE Embeddings (same as solver.py) ---
                chunk_texts = [chunk.get("text", "") for chunk in chunks]
                all_texts_to_embed = all_queries + chunk_texts
                all_vectors = get_splade_vectors(all_texts_to_embed, splade_model, splade_tokenizer, device)
                    
                if all_vectors.shape[0] != len(all_texts_to_embed):
                     logging.error("SPLADE embedding failed. Falling back to initial order.")
                     selected_doc_ids = initial_top_5_doc_ids # Use initial order as fallback
                     query_detail["selected_doc_ids"] = selected_doc_ids
                     recall_results = calculate_recall_at_k(selected_doc_ids, ground_truth_ids, k_values_to_calculate)
                     # Skip the rest of the calculation if embedding fails
                else:
                    all_query_vectors = all_vectors[:num_all_queries]
                    chunk_vectors = all_vectors[num_all_queries:]

                    # --- Calculate Similarities (same as solver.py) ---
                    chunk_all_query_sims = sparse_similarity_matrix(chunk_vectors, all_query_vectors)

                    # --- Debug Print Logic (same as solver.py) ---
                    if debug:
                        print(f"--- DEBUG Scores for Query {query_idx_str} ---")
                        ground_truth_set = set(ground_truth_ids) 
                        for chunk_idx, chunk_dict in enumerate(chunks):
                            parent_doc_id = chunk_dict.get('doc_id')
                            if parent_doc_id in ground_truth_set:
                                print(f"  Chunk {chunk_idx} (Parent: {parent_doc_id}):")
                                for q_idx in range(num_all_queries):
                                    q_type = "Orig Query" if q_idx == 0 else "Subquery  "
                                    print(f"    vs {q_type} ({q_idx}): {chunk_all_query_sims[chunk_idx, q_idx]:.2f} (Text: '{all_queries[q_idx]}')")
                        print("--- END DEBUG --- \n")
                    # --- End Debug --- 

                    # --- Rerank using Average SPLADE Score --- 
                    selected_doc_ids = rerank_by_avg_splade_score(
                        chunks, chunk_all_query_sims
                    )
                    query_detail["selected_doc_ids"] = selected_doc_ids

                    # --- Calculate Recall --- 
                    recall_results = calculate_recall_at_k(selected_doc_ids, ground_truth_ids, k_values_to_calculate)

            # --- Store results and Print Running Averages (same as solver.py) ---
            if ground_truth_ids and recall_results:
                for k in k_values_to_calculate:
                    if k in recall_results:
                        all_recall_results[k].append(recall_results[k])
                        if k == 5:
                             current_recall_5 = recall_results[k]
                             running_recall_at_5_sum += current_recall_5
                             if len(ground_truth_ids) == 1:
                                 single_doc_recall_5_sum += current_recall_5
                             elif len(ground_truth_ids) > 1:
                                 multi_doc_recall_5_sum += current_recall_5
            queries_processed_count += 1
            if len(ground_truth_ids) == 1:
                single_doc_count += 1
                prefix = "(*) "
            elif len(ground_truth_ids) > 1:
                multi_doc_count += 1
                prefix = "(**) "
            else: 
                prefix = "(0) "
            total_processing_time += (time.time() - start_time)
            detailed_results_list.append(query_detail)

            # Display running Recall@5 
            if queries_processed_count > 0:
                current_avg_recall_5 = running_recall_at_5_sum / queries_processed_count
                current_avg_initial_recall_5 = initial_recall_at_5_sum / queries_processed_count 
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
            detailed_results_list.append(query_detail) 
            continue

    progress_bar.close()

    # Calculate final average recall (same as solver.py)
    final_avg_recall = {}
    valid_k_values = sorted([k for k in k_values_to_calculate if all_recall_results.get(k)])
    if valid_k_values:
        for k in valid_k_values:
             final_avg_recall[k] = np.mean(all_recall_results[k]) if all_recall_results[k] else 0.0
    else:
        for k in k_values_to_calculate: final_avg_recall[k] = 0.0
        logging.warning("No valid recall results were collected for this run.")

    avg_time = total_processing_time / queries_processed_count if queries_processed_count > 0 else 0

    # Return metrics (same as solver.py)
    return final_avg_recall, avg_time, queries_processed_count, queries_with_errors, detailed_results_list, \
           single_doc_recall_5_sum, single_doc_count, multi_doc_recall_5_sum, multi_doc_count, \
           initial_recall_at_5_sum, initial_single_doc_recall_5_sum, initial_multi_doc_recall_5_sum

def main():
    parser = argparse.ArgumentParser(description="Rerank using subquery coverage optimization (solver) and evaluate recall.")
    parser.add_argument("--input-file", type=str, default="retrieved_full_data_50.json", 
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
    logging.info("Starting run with expander-based reranking...") # Updated log
    avg_recall_results, avg_time, count, errors, detailed_results, \
    single_doc_recall_5_sum, single_doc_count, multi_doc_recall_5_sum, multi_doc_count, \
    initial_recall_at_5_sum, initial_single_doc_recall_5_sum, initial_multi_doc_recall_5_sum = evaluate_expander_reranking( # Renamed function call
        query_indices=query_indices,
        retrieved_data=retrieved_data,
        query_expander=query_expander,
        splade_model=splade_model,
        splade_tokenizer=splade_tokenizer,
        device=device,
        max_subqueries=args.max_subqueries,
        # subquery_relevance_threshold=args.query_relevance_threshold, # Removed
        # relevance_weight=args.relevance_weight, # Removed
        k_values_to_calculate=k_values_to_calculate,
        verbose=args.verbose,
        debug=args.debug
    )

    # --- Results Output --- 
    logging.info(f"--- Final Average Recall Results (Expander Reranking: MaxSQ={args.max_subqueries}) ---") # Updated log
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
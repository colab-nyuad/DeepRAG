"""Reranks retrieved chunks using a custom DocumentReranker and evaluates recall."""
import random
import argparse
import json
import time
from typing import List, Set, Dict, Any, Tuple
from tqdm import tqdm
import numpy as np

# Import torch for device detection
import torch

# Import the custom DocumentReranker
try:
    from reranker import DocumentReranker # Assuming reranker.py is in the path
except ImportError:
    print("Error: Could not import DocumentReranker from reranker.py.")
    print("Please ensure reranker.py is in the same directory or accessible via PYTHONPATH.")
    exit(1)

# Configure logging
import logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

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

# --- Helper function for processing queries with Custom Reranker ---
def evaluate_custom_reranking(
    query_indices: List[str],
    retrieved_data: Dict[str, Dict],
    custom_reranker: DocumentReranker, # Pass the initialized reranker instance
    k_values_to_calculate: List[int],
    verbose: bool = False
) -> Tuple[Dict[int, float], float, int, int]:
    """Processes queries using the custom DocumentReranker and returns average recall."""

    all_recall_results = {k: [] for k in k_values_to_calculate}
    total_processing_time = 0.0
    queries_processed_count = 0
    queries_with_errors = 0

    progress_bar = tqdm(query_indices, desc="Eval (Custom Reranker)", unit="query", leave=False)

    for query_idx_str in progress_bar:
        query_data = retrieved_data.get(query_idx_str)
        if not query_data:
            logging.warning(f"No data found for query index {query_idx_str}. Skipping.")
            continue

        start_time = time.time()
        try:
            original_query = query_data.get("query", "")
            chunks = query_data.get("retrieved_chunks", [])
            ground_truth_ids_list = query_data.get("ground_truth_ids", [])
            ground_truth_ids = set(ground_truth_ids_list) if ground_truth_ids_list is not None else set()
            recall_results = {}

            if not chunks or not original_query:
                logging.warning(f"Query {query_idx_str}: No chunks or original query found. Calculating recall on empty set.")
                recall_results = calculate_recall_at_k([], ground_truth_ids, k_values_to_calculate)
            else:
                # --- Rerank using Custom DocumentReranker --- 
                # Prepare list of (doc_id, text) tuples for the custom reranker
                documents_to_rerank = [
                    (chunk.get("doc_id"), chunk.get("text", "")) 
                    for chunk in chunks if chunk.get("doc_id") # Ensure doc_id exists
                ]
                
                if not documents_to_rerank:
                    logging.warning(f"Query {query_idx_str}: No valid (doc_id, text) pairs prepared for custom reranking.")
                    recall_results = calculate_recall_at_k([], ground_truth_ids, k_values_to_calculate)
                else:
                    # Pass the list of (doc_id, text) tuples
                    # This now returns a sorted list of (doc_id, text) tuples
                    reranked_docs_tuples = custom_reranker.rerank(query=original_query, documents=documents_to_rerank)
                    
                    # Extract the sorted doc_ids directly from the result
                    reranked_parent_doc_ids = [doc_id for doc_id, text in reranked_docs_tuples]
                    
                    # Calculate recall based on the directly reranked parent docs
                    recall_results = calculate_recall_at_k(reranked_parent_doc_ids, ground_truth_ids, k_values_to_calculate)

            # Store results for averaging
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
    else:
        for k in k_values_to_calculate: final_avg_recall[k] = 0.0
        logging.warning("No valid recall results were collected for this run.")

    avg_time = total_processing_time / queries_processed_count if queries_processed_count > 0 else 0

    return final_avg_recall, avg_time, queries_processed_count, queries_with_errors

def main():
    parser = argparse.ArgumentParser(description="Rerank chunks using a custom DocumentReranker and calculate recall.")
    parser.add_argument("--input-file", type=str, default="retrieved_full_data_50.json",
                        help="Path to the input JSON file containing retrieved data (expects 500 chunks).")
    # Custom Reranker arguments
    parser.add_argument("--reranker-model", type=str, default="rank-T5-flan",
                        help="Model name for the custom DocumentReranker.")
    parser.add_argument("--cache-dir", type=str, default="/storage/prince/flashrank_models",
                        help="Cache directory for the custom DocumentReranker models.")
    # Sampling/Verbosity
    parser.add_argument("--sample", type=int, default=None,
                        help="Number of queries to randomly sample from the input file.")
    parser.add_argument("--verbose", action='store_true',
                        help="Enable potentially more detailed logging.")

    args = parser.parse_args()

    # 1. Load Data
    logging.info(f"Loading data from: {args.input_file}")
    try:
        with open(args.input_file, 'r') as f:
            retrieved_data = json.load(f)
        logging.info(f"Loaded data for {len(retrieved_data)} queries.")
    except FileNotFoundError:
        logging.error(f"Error: Input file not found at {args.input_file}")
        return
    except json.JSONDecodeError:
        logging.error(f"Error: Could not decode JSON from {args.input_file}")
        return
    except Exception as e:
        logging.error(f"Error loading data: {e}")
        return

    # 2. Initialize Custom Reranker Model
    logging.info(f"Initializing custom DocumentReranker model: {args.reranker_model} (Cache: {args.cache_dir})")
    try:
        # Pass the determined device to the initializer
        custom_reranker = DocumentReranker(model_name=args.reranker_model, cache_dir=args.cache_dir)
    except NameError:
        # This case is handled by the import check at the top, but good to be safe
        logging.error("DocumentReranker class not found. Ensure reranker.py is available.")
        return
    except Exception as e:
        logging.error(f"Error initializing custom DocumentReranker model '{args.reranker_model}': {e}", exc_info=True)
        return

    # Determine indices to process
    query_indices = list(retrieved_data.keys())
    if args.sample is not None:
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

    k_values_to_calculate = [5, 10, 50] # Recall@k values

    # --- Single Run using Custom Reranking ---
    logging.info("Starting single run with custom DocumentReranker...")
    avg_recall_results, avg_time, count, errors = evaluate_custom_reranking(
        query_indices=query_indices,
        retrieved_data=retrieved_data,
        custom_reranker=custom_reranker,
        k_values_to_calculate=k_values_to_calculate,
        verbose=args.verbose
    )

    # --- Results Output ---
    logging.info(f"--- Final Average Recall Results (Custom Reranker: {args.reranker_model}) ---")
    if count > 0:
        valid_k_values = sorted(avg_recall_results.keys())
        for k in valid_k_values:
             logging.info(f"Avg Recall@{k:<3}: {avg_recall_results[k]:.4f}")
        logging.info(f"Successfully processed {count} queries.")
        if errors > 0: logging.warning(f"Encountered errors in {errors} queries...")
        logging.info(f"Average processing time per successful query: {avg_time:.2f} seconds")
    else:
        logging.warning("No queries were successfully processed.")
        if errors > 0: logging.warning(f"Encountered errors in {errors} queries.")
    logging.info("--------------------------------------------------------")


if __name__ == "__main__":
    main() 

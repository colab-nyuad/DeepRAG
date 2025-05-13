"""Reranks retrieved chunks using direct SPLADE similarity and evaluates recall."""
import random
import argparse
import json
import time
from typing import List, Set, Dict, Any, Tuple
from tqdm import tqdm
import numpy as np
import torch
from transformers import AutoTokenizer, AutoModelForMaskedLM
# Removed networkx import

# Configure logging
import logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logging.getLogger("transformers.tokenization_utils_base").setLevel(logging.ERROR) # Suppress tokenizer warnings
# Removed other logging suppressions if not needed

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

# --- SPLADE Helper Function ---
def get_splade_vectors(texts: List[str], model, tokenizer, device: str = 'cpu', batch_size: int = 16) -> np.ndarray:
    """Generates SPLADE sparse vectors for a list of texts."""
    all_vectors = []
    model.eval()
    with torch.no_grad():
        # Disable the embedding progress bar
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
                # Add placeholder zeros for failed batch to maintain array shape if needed downstream,
                # or handle more gracefully depending on requirements.
                # Here, we'll just skip the batch, which might lead to errors later if not handled.
                # A better approach might be needed depending on how failures should be treated.
                logging.warning(f"Skipping batch due to error. Output array shape might be affected.")

    return np.vstack(all_vectors) if all_vectors else np.array([[]]) # Return 2D empty array if nothing was embedded

# --- Dot Product Similarity ---
def sparse_dot_product(query_vector: np.ndarray, doc_vectors: np.ndarray) -> np.ndarray:
    """Calculates dot product between a single query vector and multiple doc vectors."""
    if query_vector.ndim == 0 or doc_vectors.ndim < 2 or doc_vectors.shape[0] == 0 or doc_vectors.shape[1] == 0:
         logging.warning(f"Skipping dot product due to empty or invalid vectors. Query shape: {query_vector.shape}, Docs shape: {doc_vectors.shape}")
         return np.array([]) # Return empty array if vectors are invalid/empty

    # Ensure query_vector is 1D
    if query_vector.ndim > 1:
        query_vector = query_vector.flatten()

    # Check for dimension mismatch after flattening
    if query_vector.shape[0] != doc_vectors.shape[1]:
        logging.warning(f"Dimension mismatch in sparse_dot_product. Query dim: {query_vector.shape[0]}, Doc dim: {doc_vectors.shape[1]}. Returning empty array.")
        return np.array([])

    return np.dot(doc_vectors, query_vector)


# --- Get Top 5 Chunks based on SPLADE score ---
def get_top_5_splade_chunks(
    chunks: List[Dict[str, Any]],
    chunk_splade_scores: np.ndarray # Similarity of each chunk to the ORIGINAL query
) -> List[Dict[str, Any]]:
    """Returns the top 5 chunk dictionaries based on SPLADE score."""
    if len(chunk_splade_scores) != len(chunks) or chunk_splade_scores.ndim == 0:
        logging.warning("Cannot get top chunks by SPLADE score. Missing/mismatched score data.")
        return [] # Return empty list

    # Combine scores with original chunk dictionaries
    scored_chunks = []
    for i, chunk in enumerate(chunks):
        score = chunk_splade_scores[i]
        scored_chunks.append((score, chunk)) # Tuple of (score, chunk_dict)

    # Sort chunks by score in descending order
    scored_chunks.sort(key=lambda item: item[0], reverse=True)

    # Get the top 5 chunk dictionaries
    top_5_chunks = [chunk_dict for score, chunk_dict in scored_chunks[:5]]
    
    return top_5_chunks

# --- Helper function for processing queries with specific params ---
def evaluate_splade_reranking(
    query_indices: List[str],
    retrieved_data: Dict[str, Dict],
    # SPLADE arguments
    splade_model,
    splade_tokenizer,
    device: str,
    k_values_to_calculate: List[int],
    # Removed query_expander, max_subqueries, min_best_sq_relevance
    verbose: bool = False # Keep verbose for potential future print statements
) -> Tuple[Dict[int, float], float, int, int, List[Dict[str, Any]], float, int, float, int, float, float, float]: # Added initial recall sums
    """Processes queries using direct SPLADE reranking and returns average recall + detailed results."""

    all_recall_results = {k: [] for k in k_values_to_calculate}
    detailed_results_list = [] # List to store detailed results for each query
    total_processing_time = 0.0
    queries_processed_count = 0
    queries_with_errors = 0
    # SPLADE Recall Tracking
    running_recall_at_5_sum = 0.0
    single_doc_recall_5_sum = 0.0
    single_doc_count = 0
    multi_doc_recall_5_sum = 0.0
    multi_doc_count = 0
    # Initial Recall Tracking
    initial_recall_at_5_sum = 0.0
    initial_single_doc_recall_5_sum = 0.0
    initial_multi_doc_recall_5_sum = 0.0

    # Disable the main query progress bar
    progress_bar = tqdm(query_indices, desc="Eval (SPLADE Rerank)", unit="query", leave=False, disable=True)

    for query_idx_str in progress_bar:
        query_data = retrieved_data.get(query_idx_str)
        if not query_data:
            continue

        start_time = time.time()
        query_detail = { # Initialize details for this query
            "query_id": query_idx_str,
            "query": "",
            "ground_truth_ids": [],
            "top_5_splade_chunks_details": [] # Changed field name to store dicts
        }
        try:
            original_query = query_data.get("query", "")
            chunks = query_data.get("retrieved_chunks", [])
            ground_truth_ids = set(query_data.get("ground_truth_ids", []))
            recall_results = {}

            # Store query and ground truth info
            query_detail["query"] = original_query
            query_detail["ground_truth_ids"] = list(ground_truth_ids) # Convert set to list for JSON

            # --- Calculate Initial Recall@5 (Before SPLADE Rerank) --- 
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
                # Handle case with no chunks or query
                recall_results = calculate_recall_at_k([], ground_truth_ids, k_values_to_calculate)
                # Store empty list for top chunks details
                query_detail["top_5_splade_chunks_details"] = [] 
            else:
                # --- Rerank using direct SPLADE similarity ---
                # Prepare texts for SPLADE
                chunk_texts = [chunk.get("text", "") for chunk in chunks]
                all_texts_to_embed = [original_query] + chunk_texts

                # Get SPLADE vectors
                all_vectors = get_splade_vectors(all_texts_to_embed, splade_model, splade_tokenizer, device)

                if all_vectors.shape[0] != len(all_texts_to_embed) or all_vectors.ndim < 2:
                     logging.error(f"Query {query_idx_str}: SPLADE embedding failed or returned unexpected shape {all_vectors.shape}. Using original retrieval order.")
                     # Fallback: Use original retrieval order - Get top 5 from original chunks if possible
                     top_5_fallback_chunks = chunks[:5]
                     # Get parent doc IDs of fallback chunks
                     fallback_doc_ids = []
                     seen_fallback_doc_ids = set()
                     for ch in top_5_fallback_chunks:
                          p_id = ch.get('doc_id')
                          if p_id and p_id not in seen_fallback_doc_ids:
                              seen_fallback_doc_ids.add(p_id)
                              fallback_doc_ids.append(p_id)
                     recall_results = calculate_recall_at_k(fallback_doc_ids, ground_truth_ids, k_values_to_calculate)
                     # Store fallback details (doc_id and text)
                     query_detail["top_5_splade_chunks_details"] = [
                         {"doc_id": ch.get('doc_id'), "text": ch.get("text", "")}
                         for ch in top_5_fallback_chunks
                     ]
                else:
                    # Separate vectors
                    original_query_vector = all_vectors[0:1] # Keep as 2D for consistency, dot product handles flattening
                    chunk_vectors = all_vectors[1:]

                    # Calculate SPARSE similarities (dot product) between query and chunks
                    chunk_orig_query_sim_sparse = sparse_dot_product(original_query_vector, chunk_vectors)

                    # Get the top 5 chunk dictionaries based on score
                    top_5_splade_chunks = get_top_5_splade_chunks(chunks, chunk_orig_query_sim_sparse)
                    
                    # Store the DETAILS (doc_id and text) of the top 5 chunks
                    query_detail["top_5_splade_chunks_details"] = [
                        {"doc_id": chunk.get('doc_id'), "text": chunk.get("text", "")}
                        for chunk in top_5_splade_chunks
                    ]

                    # Get the unique parent doc IDs from the top 5 chunks for recall calculation
                    top_5_parent_doc_ids = []
                    seen_top_5_doc_ids = set()
                    for chunk in top_5_splade_chunks:
                        parent_id = chunk.get('doc_id')
                        if parent_id and parent_id not in seen_top_5_doc_ids:
                            seen_top_5_doc_ids.add(parent_id)
                            top_5_parent_doc_ids.append(parent_id)
                    
                    # Calculate recall based on the parent docs of the top 5 chunks
                    recall_results = calculate_recall_at_k(top_5_parent_doc_ids, ground_truth_ids, k_values_to_calculate)

            # Store results for averaging
            if ground_truth_ids and recall_results:
                for k in k_values_to_calculate:
                    if k in recall_results:
                         all_recall_results[k].append(recall_results[k])
                         # Update running sums for Recall@5 if k is 5
                         if k == 5:
                             current_recall_5 = recall_results[k]
                             running_recall_at_5_sum += current_recall_5
                             if len(ground_truth_ids) == 1:
                                 single_doc_recall_5_sum += current_recall_5
                             elif len(ground_truth_ids) > 1:
                                 multi_doc_recall_5_sum += current_recall_5
            queries_processed_count += 1
            # Determine query type prefix and update counts
            if len(ground_truth_ids) == 1:
                single_doc_count += 1
                prefix = "(*) "
            elif len(ground_truth_ids) > 1:
                multi_doc_count += 1
                prefix = "(**) "
            else: # Should not happen if ground_truth_ids is always populated
                prefix = "(0) "
            total_processing_time += (time.time() - start_time)
            detailed_results_list.append(query_detail) # Add collected details to the list

            # Display running Recall@5 with type-specific avg
            if queries_processed_count > 0:
                current_avg_recall_5 = running_recall_at_5_sum / queries_processed_count
                current_avg_initial_recall_5 = initial_recall_at_5_sum / queries_processed_count
                # Calculate type-specific running average
                type_avg_recall_5 = 0.0
                type_avg_initial_recall_5 = 0.0
                if prefix == "(*) " and single_doc_count > 0:
                    type_avg_recall_5 = single_doc_recall_5_sum / single_doc_count
                    type_avg_initial_recall_5 = initial_single_doc_recall_5_sum / single_doc_count
                elif prefix == "(**) " and multi_doc_count > 0:
                    type_avg_recall_5 = multi_doc_recall_5_sum / multi_doc_count
                    type_avg_initial_recall_5 = initial_multi_doc_recall_5_sum / multi_doc_count
                
                # Print the detailed running averages
                print(f"  {prefix}Query {query_idx_str}: R@5={recall_results.get(5, 0.0):.4f} (Init:{initial_recall_5:.4f}) | Avg R@5(Type)={type_avg_recall_5:.4f} (Init:{type_avg_initial_recall_5:.4f}) | Avg R@5(All)={current_avg_recall_5:.4f} (Init:{current_avg_initial_recall_5:.4f})")
            else: # First query
                 print(f"  {prefix}Query {query_idx_str}: Recall@5 = {recall_results.get(5, 0.0):.4f} (Initial: {initial_recall_5:.4f})")

        except Exception as e:
            queries_with_errors += 1
            logging.error(f"Error processing query index {query_idx_str}: {e}", exc_info=True)
            # Store minimal info on error
            detailed_results_list.append(query_detail) 
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

    # Calculate running recall at 5
    running_recall_at_5 = single_doc_recall_5_sum / single_doc_count if single_doc_count > 0 else 0.0

    # Calculate initial recall at 5
    initial_recall_at_5 = initial_single_doc_recall_5_sum / single_doc_count if single_doc_count > 0 else 0.0

    # Return initial recall sums as well
    return final_avg_recall, avg_time, queries_processed_count, queries_with_errors, detailed_results_list, \
           single_doc_recall_5_sum, single_doc_count, multi_doc_recall_5_sum, multi_doc_count, \
           initial_recall_at_5_sum, initial_single_doc_recall_5_sum, initial_multi_doc_recall_5_sum

def main():
    parser = argparse.ArgumentParser(description="Rerank chunks using direct SPLADE similarity and calculate recall.")
    parser.add_argument("--input-file", type=str, default="retrieved_expanded_data_200.json", #retrieved_full_data_50.json",
                        help="Path to the input JSON file containing retrieved data.")
    # SPLADE Model argument
    parser.add_argument("--splade-model", type=str, default="naver/splade-cocondenser-ensembledistil", 
                        help="Hugging Face model name/path for SPLADE.")
    # Removed graph/subquery parameters
    # Sampling/Verbosity
    parser.add_argument("--sample", type=int, default=None,
                        help="Number of queries to randomly sample from the input file.")
    parser.add_argument("--verbose", action='store_true',
                        help="Enable potentially more detailed logging (currently unused).")
    # Removed QueryExpander arguments
    parser.add_argument("--output-json", type=str, default=None,
                        help="Path to save detailed reranking results as a JSON file.")

    args = parser.parse_args()

    # --- Setup Device ---
    device = 'cuda:3' if torch.cuda.is_available() else 'cpu'
    logging.info(f"Using device: {device}")

    # 1. Load Data
    logging.info(f"Loading data from: {args.input_file}")
    try:
        with open(args.input_file, 'r') as f:
            retrieved_data = json.load(f)
        logging.info(f"Loaded data for {len(retrieved_data)} queries.")
    except FileNotFoundError:
        logging.error(f"Error: Input file not found at {args.input_file}")
        return # Use return instead of exit(1) for cleaner exit
    except json.JSONDecodeError:
        logging.error(f"Error: Could not decode JSON from {args.input_file}")
        return
    except Exception as e:
        logging.error(f"Error loading data: {e}")
        return

    # 2. Initialize SPLADE Model
    logging.info(f"Initializing SPLADE model: {args.splade_model}")
    try:
        splade_tokenizer = AutoTokenizer.from_pretrained(args.splade_model)
        splade_model = AutoModelForMaskedLM.from_pretrained(args.splade_model).to(device)
        splade_model.eval() # Set to eval mode
    except Exception as e:
        logging.error(f"Error initializing SPLADE model '{args.splade_model}': {e}")
        return

    # Removed Query Expander initialization

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

    k_values_to_calculate = [5, 10, 50] # Define k values for evaluation

    # --- Single Run using SPLADE Reranking ---
    logging.info("Starting single run with direct SPLADE reranking...")
    avg_recall_results, avg_time, count, errors, detailed_results, single_doc_recall_5_sum, single_doc_count, multi_doc_recall_5_sum, multi_doc_count, \
    initial_recall_at_5_sum, initial_single_doc_recall_5_sum, initial_multi_doc_recall_5_sum = evaluate_splade_reranking(
        query_indices=query_indices,
        retrieved_data=retrieved_data,
        # Pass SPLADE components
        splade_model=splade_model,
        splade_tokenizer=splade_tokenizer,
        device=device,
        # Other parameters
        k_values_to_calculate=k_values_to_calculate,
        verbose=args.verbose
    )

    # --- Results Output ---
    logging.info("--- Final Average Recall Results (Direct SPLADE Reranking) ---")
    if count > 0:
        # Calculate initial averages
        avg_initial_recall_5 = initial_recall_at_5_sum / count if count > 0 else 0.0
        avg_initial_single_doc_recall_5 = initial_single_doc_recall_5_sum / single_doc_count if single_doc_count > 0 else 0.0
        avg_initial_multi_doc_recall_5 = initial_multi_doc_recall_5_sum / multi_doc_count if multi_doc_count > 0 else 0.0
        
        # Calculate SPLADE averages
        avg_splade_single_doc_recall_5 = single_doc_recall_5_sum / single_doc_count if single_doc_count > 0 else 0.0
        avg_splade_multi_doc_recall_5 = multi_doc_recall_5_sum / multi_doc_count if multi_doc_count > 0 else 0.0
        # Find overall SPLADE Recall@5 from avg_recall_results (assuming 5 is in k_values_to_calculate)
        avg_splade_overall_recall_5 = avg_recall_results.get(5, 0.0)

        # Print comparison
        logging.info(f"Overall Avg Recall@5:       {avg_splade_overall_recall_5:.4f} (Initial: {avg_initial_recall_5:.4f})")
        logging.info(f"Single-Doc (*) Avg Recall@5: {avg_splade_single_doc_recall_5:.4f} (Initial: {avg_initial_single_doc_recall_5:.4f}) ({single_doc_count} queries)")
        logging.info(f"Multi-Doc (**) Avg Recall@5: {avg_splade_multi_doc_recall_5:.4f} (Initial: {avg_initial_multi_doc_recall_5:.4f}) ({multi_doc_count} queries)")

        # Print other details (like higher K values if calculated)
        other_k_values = sorted([k for k in avg_recall_results if k != 5])
        if other_k_values:
             logging.info("--- Other Overall Avg Recall Values --- ")
             for k in other_k_values:
                 logging.info(f"Avg Recall@{k:<3}: {avg_recall_results[k]:.4f}")
        
        logging.info(f"\nSuccessfully processed {count} queries.")
        if errors > 0: logging.warning(f"Encountered errors in {errors} queries...")
        logging.info(f"Average processing time per successful query: {avg_time:.2f} seconds")
    else:
        logging.warning("No queries were successfully processed.")
        if errors > 0: logging.warning(f"Encountered errors in {errors} queries...")
    logging.info("--------------------------------------------------------")

    # Save detailed results to file if specified
    if args.output_json:
        if detailed_results:
            logging.info(f"Saving detailed results for {len(detailed_results)} queries to: {args.output_json}")
            try:
                with open(args.output_json, 'w') as f:
                    json.dump(detailed_results, f, indent=4) # Use indent for readability
                logging.info("Detailed results saved successfully.")
            except Exception as e:
                logging.error(f"Failed to save detailed results to {args.output_json}: {e}")
        else:
            logging.warning(f"No detailed results collected to save to {args.output_json}.")

if __name__ == "__main__":
    main() 

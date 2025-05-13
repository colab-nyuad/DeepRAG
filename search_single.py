"""Script to perform hybrid search for multiple queries and calculate average recall@k."""

import argparse
import time
import random
import json # Added JSON import
from typing import List, Set, Dict
from tqdm import tqdm
import numpy as np # For calculating averages safely

# Imports from the existing codebase
from utils.dataset import EvaluationDataset
from retriever import PineconeRetriever, OpenSearchRetriever, HybridRetriever, Doc
# Import EmbeddingModel directly from ret.py where it is defined
from ret import EmbeddingModel
# Removed QueryExpander import

# Configure logging (optional but good practice, copied from deepret.py)
import logging
logging.getLogger('opensearch').setLevel(logging.WARNING)
logging.getLogger('httpx').setLevel(logging.WARNING) 

def calculate_recall_at_k(retrieved_ids: List[str], ground_truth_ids: Set[str], k_values: List[int]) -> Dict[int, float]:
    """Calculates recall at different values of k."""
    recall_results = {}
    num_ground_truth = len(ground_truth_ids)

    if num_ground_truth == 0:
        for k in k_values:
            recall_results[k] = 0.0 
        return recall_results

    for k in k_values:
        actual_k = min(k, len(retrieved_ids))
        if actual_k == 0:
            recall_results[k] = 0.0
            continue
        # Need to compare against the chunk_ids from the retrieved_info list
        # This requires modifying how calculate_recall_at_k is called or structured
        # For now, assuming retrieved_ids contains the correct IDs (e.g., chunk IDs)
        top_k_ids = set(retrieved_ids[:actual_k]) 
        true_positives_at_k = len(top_k_ids.intersection(ground_truth_ids))
        recall_at_k = true_positives_at_k / num_ground_truth
        recall_results[k] = recall_at_k
        
    return recall_results

def main():
    """Main execution function."""
    parser = argparse.ArgumentParser(description="Perform hybrid search for queries and calculate average recall@k.")
    parser.add_argument("--dataset", type=str, default="data/all.json", 
                        help="Path to the dataset file.")
    parser.add_argument("--sample", type=int, default=None, 
                        help="Number of queries to randomly sample. If not set, runs on all.")
    parser.add_argument("--top-k-retrieve", type=int, default=200, 
                        help="Total number of documents to retrieve per query.")
    parser.add_argument("--vector-weight", type=float, default=0.5, 
                        help="Weight for vector retrieval in hybrid mode.")
    parser.add_argument("--n-parallel", type=int, default=10, 
                        help="Number of parallel threads for retriever components.") 
    parser.add_argument("--model", type=str, default="intfloat/e5-base-v2", 
                        help="Embedding model to use for PineconeRetriever.")

    args = parser.parse_args()
    
    # 1. Load Dataset - Capture gt_answers now
    print(f"Loading dataset from: {args.dataset}")
    try:
        dataset = EvaluationDataset(args.dataset)
        # Ensure gt_answers is captured
        queries, gt_answers, ground_truth_ids_list, _ = dataset.get_all_data()
    except FileNotFoundError:
        print(f"Error: Dataset file not found at {args.dataset}")
        exit(1)
    except Exception as e:
        print(f"Error loading dataset: {e}")
        exit(1)

    # Check for None in ground_truth_ids_list (logic kept, prints removed)
    none_indices = [i for i, gt_list in enumerate(ground_truth_ids_list) if gt_list is None]
    if none_indices:
        logging.warning(f"Found {len(none_indices)} entries where ground_truth_ids_list is None. Indices: {none_indices[:10]}...")

    num_total_queries = len(queries)
    if num_total_queries == 0:
        print("Error: No queries found in the dataset.")
        exit(1)

    # Check if gt_answers length matches queries length (optional sanity check)
    if len(gt_answers) != num_total_queries:
        logging.warning(f"Number of queries ({num_total_queries}) does not match number of ground truth answers ({len(gt_answers)}). Ground truth answers might be missing for some queries.")
        # Pad gt_answers with None if necessary, or handle appropriately
        # For simplicity, we'll proceed assuming indices align, but beware of IndexErrors if they don't.

    # Determine indices to process (remains the same)
    if args.sample is not None:
        if args.sample <= 0:
            print("Error: --sample must be a positive integer.")
            exit(1)
        if args.sample > num_total_queries:
            print(f"Warning: Sample size ({args.sample}) is larger than total queries ({num_total_queries}). Running on all queries.")
            indices_to_process = list(range(num_total_queries))
        else:
            indices_to_process = random.sample(range(num_total_queries), args.sample)
            print(f"Randomly sampling {args.sample} queries.")
    else:
        indices_to_process = list(range(num_total_queries))
        print(f"Running on all {num_total_queries} queries.")

    # 2. Initialize Retriever ONLY
    try:
        embedding_model = EmbeddingModel(model_name=args.model)
        vector_retriever = PineconeRetriever(embedding_model=embedding_model)
        keyword_retriever = OpenSearchRetriever()
        hybrid_retriever = HybridRetriever(vector_retriever=vector_retriever, 
                                           keyword_retriever=keyword_retriever,
                                           vector_weight=args.vector_weight)
    except NameError as e:
        if "EmbeddingModel" in str(e):
             print("Error: Could not import EmbeddingModel from ret.py.")
        else:
            print(f"Error during initialization: {e}")
        exit(1)
    except Exception as e:
        print(f"Error initializing retriever components: {e}")
        exit(1)

    # 3. Process Queries, Calculate Metrics, and Store Comprehensive Info
    k_values_to_calculate = [5, 10, 50, 100, 200]
    k_values_to_calculate = [k for k in k_values_to_calculate if k <= args.top_k_retrieve] 
    
    all_recall_results = {k: [] for k in k_values_to_calculate}
    retrieved_data_store = {} # Initialize dictionary to store comprehensive info
    total_processing_time = 0.0 
    queries_processed_count = 0
    queries_with_errors = 0
    
    # --- Add tracking for single/two doc recall ---
    single_doc_recall_results = {k: [] for k in k_values_to_calculate}
    two_doc_recall_results = {k: [] for k in k_values_to_calculate}
    # --- End tracking addition ---

    print(f"\nProcessing {len(indices_to_process)} queries (Retrieving Top {args.top_k_retrieve})...")
    progress_bar = tqdm(indices_to_process, desc="Queries", unit="query")

    for i in progress_bar:
        query = queries[i]
        # Get ground truth answer, handle potential index mismatch if lists aren't aligned
        try:
            ground_truth_answer = gt_answers[i] if i < len(gt_answers) else None 
        except IndexError:
             ground_truth_answer = None # Or log a warning
             logging.warning(f"Index mismatch accessing ground truth answer for query index {i}. Setting to None.")

        gt_ids_list = ground_truth_ids_list[i]
        ground_truth_ids_for_recall = set(gt_ids_list) if gt_ids_list is not None else set()
        # Ensure ground truth IDs are stored as a list (even if empty)
        ground_truth_ids_to_store = gt_ids_list if gt_ids_list is not None else [] 
        
        start_time = time.time()
        
        try:
            # Direct retrieval using args.top_k_retrieve
            retrieved_docs_raw: List[Doc] = hybrid_retriever.retrieve(query, args.top_k_retrieve)
            
            # Extract detailed info for storage AND collect unique parent doc_ids for recall
            retrieved_info_list = []
            final_retrieved_parent_doc_ids = [] # List for unique parent doc IDs for recall
            seen_parent_doc_ids = set() # Track seen parent doc IDs for recall list
            # We still need seen_chunk_ids if hybrid retrieval might return duplicate chunks
            seen_chunk_ids_for_storage = set()
            
            for doc in retrieved_docs_raw:
                # Use chunk_id if available for deduplicating stored info, but prioritize doc_id for recall
                storage_dedup_id = doc.chunk_id if doc.chunk_id is not None else doc.doc_id
                
                # Store detailed info if chunk hasn't been seen
                if storage_dedup_id not in seen_chunk_ids_for_storage:
                    seen_chunk_ids_for_storage.add(storage_dedup_id)
                    retrieved_info = {
                        "text": doc.text,
                        "chunk_id": doc.chunk_id,
                        "doc_id": doc.doc_id
                    }
                    retrieved_info_list.append(retrieved_info)

                # Add parent doc_id to the recall list if not already seen
                parent_doc_id = doc.doc_id
                if parent_doc_id not in seen_parent_doc_ids:
                    seen_parent_doc_ids.add(parent_doc_id)
                    final_retrieved_parent_doc_ids.append(parent_doc_id)
            
            end_time = time.time()
            total_processing_time += (end_time - start_time)

            # *** Store comprehensive data for this query index ***
            query_output_data = {
                "query": query,
                "ground_truth_answer": ground_truth_answer,
                "ground_truth_ids": ground_truth_ids_to_store,
                "retrieved_chunks": retrieved_info_list
            }
            retrieved_data_store[i] = query_output_data
            
            # Calculate recall using the unique PARENT document IDs
            recall_results = calculate_recall_at_k(final_retrieved_parent_doc_ids, ground_truth_ids_for_recall, k_values_to_calculate)
            
            # Store recall results for averaging
            num_ground_truth = len(ground_truth_ids_for_recall) # Get number of ground truth docs
            if num_ground_truth > 0: 
                for k in k_values_to_calculate:
                    recall_score = recall_results[k]
                    all_recall_results[k].append(recall_score) # Append to overall
                    # --- Append to specific category ---
                    if num_ground_truth == 1:
                        single_doc_recall_results[k].append(recall_score)
                    elif num_ground_truth == 2:
                        two_doc_recall_results[k].append(recall_score)
                    # --- End specific category append ---
            
            queries_processed_count += 1 # Increment for each successfully processed query (after try block)

        except Exception as e:
            queries_with_errors += 1
            logging.error(f"Error processing query index {i} ('{query[:50]}...'): {e}", exc_info=True) 
            continue 

        # Update progress bar postfix (remains the same logic, add single/two doc R@5)
        if queries_processed_count > 0:
            running_avg_recall = {}
            for k in k_values_to_calculate:
                running_avg_recall[f'R@{k}'] = np.mean(all_recall_results[k]) if all_recall_results[k] else 0.0
            postfix_metrics = {key: f"{value:.3f}" for key, value in running_avg_recall.items()}
            # --- Add single/two doc R@5 to postfix ---
            if 5 in k_values_to_calculate:
                 running_avg_recall_single_5 = np.mean(single_doc_recall_results[5]) if single_doc_recall_results.get(5) else 0.0
                 running_avg_recall_two_5 = np.mean(two_doc_recall_results[5]) if two_doc_recall_results.get(5) else 0.0
                 postfix_metrics['R@5(S)'] = f"{running_avg_recall_single_5:.3f}" # 'S' for Single
                 postfix_metrics['R@5(T)'] = f"{running_avg_recall_two_5:.3f}" # 'T' for Two
            # --- End postfix addition ---
            # --- Replace R@5(S/T) with R@max_k(*/**) ---
            if k_values_to_calculate: # Ensure there are k values
                max_k = max(k_values_to_calculate)
                running_avg_single_max_k = np.mean(single_doc_recall_results[max_k]) if single_doc_recall_results.get(max_k) else 0.0
                running_avg_two_max_k = np.mean(two_doc_recall_results[max_k]) if two_doc_recall_results.get(max_k) else 0.0
                postfix_metrics[f'R@{max_k}(*)'] = f"{running_avg_single_max_k:.3f}"
                postfix_metrics[f'R@{max_k}(**)'] = f"{running_avg_two_max_k:.3f}"
                # Remove the old R@5(S/T) if they exist
                postfix_metrics.pop('R@5(S)', None)
                postfix_metrics.pop('R@5(T)', None)
            # --- End replacement ---
            progress_bar.set_postfix(postfix_metrics)

    # 4. Print Final Recall Results (updated for breakdown)
    print("\n--- Final Average Recall Results (Hybrid Search) ---") 
    # Use count based on actual results collected, not just processed count
    overall_count = len(all_recall_results[k_values_to_calculate[0]]) if k_values_to_calculate and all_recall_results[k_values_to_calculate[0]] else 0
    single_doc_count = len(single_doc_recall_results[k_values_to_calculate[0]]) if k_values_to_calculate and single_doc_recall_results[k_values_to_calculate[0]] else 0
    two_doc_count = len(two_doc_recall_results[k_values_to_calculate[0]]) if k_values_to_calculate and two_doc_recall_results[k_values_to_calculate[0]] else 0

    if overall_count > 0: # Check if any recall results were actually calculated
        for k in sorted(k_values_to_calculate):
             avg_overall = np.mean(all_recall_results[k]) # Already checked list is non-empty
             avg_single = np.mean(single_doc_recall_results[k]) if single_doc_recall_results[k] else 0.0
             avg_two = np.mean(two_doc_recall_results[k]) if two_doc_recall_results[k] else 0.0
             
             # Updated print format
             print(f"Avg Recall@{k:<3}: Overall={avg_overall:.4f} (n={overall_count}) | SingleDoc={avg_single:.4f} (n={single_doc_count}) | TwoDoc={avg_two:.4f} (n={two_doc_count})")
            
        avg_time = total_processing_time / queries_processed_count if queries_processed_count else 0 # Keep avg time based on processed count
        print(f"\nSuccessfully processed {queries_processed_count} queries (with {overall_count} having ground truth).")
        if queries_with_errors > 0:
             print(f"Encountered errors in {queries_with_errors} queries (see logs for details).")
        print(f"Average processing time per successful query: {avg_time:.2f} seconds")
    else:
        print("No queries with ground truth were successfully processed to calculate recall.")
        if queries_with_errors > 0:
             print(f"Encountered errors in {queries_with_errors} queries.")
            
    print("--------------------------------------------------------")

    # 5. Save Comprehensive Retrieved Data to JSON
    output_filename = "retrieved_full_data_200.json" # Changed filename
    print(f"\nSaving comprehensive retrieved data for {len(retrieved_data_store)} queries to {output_filename}...")
    try:
        retrieved_data_store_str_keys = {str(k): v for k, v in retrieved_data_store.items()}
        with open(output_filename, 'w') as f:
            json.dump(retrieved_data_store_str_keys, f, indent=2) 
        print(f"Successfully saved retrieved data to {output_filename}.")
    except Exception as e:
        print(f"Error saving retrieved data to {output_filename}: {e}")

if __name__ == "__main__":
    main() 

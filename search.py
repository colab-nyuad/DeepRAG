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
# --- Import QueryExpander and the specific prompt ---
try:
    from query_expansion_ola import QueryExpander, QUERY_EXPANSION_USER_PROMPT_DECOMPOSE # Import the DECOMPOSE prompt instead of SINGLE
except ImportError:
    print("Error: Could not import QueryExpander or QUERY_EXPANSION_USER_PROMPT_DECOMPOSE from query_expansion_ola.py.")
    exit(1)
# --- End Import ---

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
    # --- Add Query Expansion arguments ---
    parser.add_argument("--qe-provider", type=str, default='ollama', help="Provider for QueryExpander LLM.")
    parser.add_argument("--qe-model", type=str, default=None, help="Model name for QueryExpander LLM.")
    parser.add_argument("--qe-temp", type=float, default=0.0, help="Temperature for QueryExpander LLM (0.0 for single query).") # Default 0.0 temp
    parser.add_argument("--qe-max-workers", type=int, default=5, help="Max workers for QueryExpander.")
    parser.add_argument("--qe-timeout", type=float, default=30.0, help="Timeout in seconds for query expansion batch.") # Add timeout argument
    # --- End Query Expansion arguments ---

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

    # 2. Initialize Retrievers AND Query Expander
    try:
        embedding_model = EmbeddingModel(model_name=args.model)
        vector_retriever = PineconeRetriever(embedding_model=embedding_model)
        keyword_retriever = OpenSearchRetriever()
        hybrid_retriever = HybridRetriever(vector_retriever=vector_retriever, 
                                           keyword_retriever=keyword_retriever,
                                           vector_weight=args.vector_weight)
        # --- Initialize Query Expander ---
        print(f"Initializing Query Expander (Provider: {args.qe_provider}, Model: {args.qe_model or 'default'}, Temp: {args.qe_temp})")
        query_expander = QueryExpander(
            provider=args.qe_provider,
            model_name=args.qe_model,
            temperature=args.qe_temp,
            max_workers=args.qe_max_workers,
            # Use the DECOMPOSE prompt (QueryExpander uses its default)
            # user_prompt=QUERY_EXPANSION_USER_PROMPT_DECOMPOSE # Removed as class doesn't accept it
        )
        # --- End Initializing Query Expander ---
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
    
    # --- Rename existing for combined results ---
    combined_recall_results = {k: [] for k in k_values_to_calculate}
    combined_single_doc_recall_results = {k: [] for k in k_values_to_calculate}
    combined_two_doc_recall_results = {k: [] for k in k_values_to_calculate}
    # --- Add new dicts for independent recall ---
    orig_recall_results = {k: [] for k in k_values_to_calculate}
    orig_single_doc_recall_results = {k: [] for k in k_values_to_calculate}
    orig_two_doc_recall_results = {k: [] for k in k_values_to_calculate}
    exp_recall_results = {k: [] for k in k_values_to_calculate}
    exp_single_doc_recall_results = {k: [] for k in k_values_to_calculate}
    exp_two_doc_recall_results = {k: [] for k in k_values_to_calculate}
    # --- End tracking addition ---

    retrieved_data_store = {} # Initialize dictionary to store comprehensive info
    total_processing_time = 0.0 
    queries_processed_count = 0
    queries_with_errors = 0
    
    print(f"\nProcessing {len(indices_to_process)} queries (Retrieving Top {args.top_k_retrieve} with 1 expansion)...")
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
            # --- Query Expansion ---
            expanded_query = None
            try:
                # Pass the timeout from args
                expansion_result = query_expander.batch_query_expansion([query], timeout=args.qe_timeout)[0] 
                # Expecting only one expansion from the prompt
                if expansion_result.expanded_queries:
                    expanded_query = expansion_result.expanded_queries[0] 
                    # print(f"  Expanded Query: {expanded_query}") # Optional: print expanded query
                else:
                    logging.warning(f"Query index {i}: No expansion generated for '{query[:50]}...'. Using original query only.")
            except Exception as qe_err:
                logging.error(f"Query index {i}: Error during query expansion for '{query[:50]}...': {qe_err}", exc_info=True)
                # Proceed without expansion
            # --- End Query Expansion ---

            # --- Split k and Retrieve ---
            k_total = args.top_k_retrieve
            k_orig = k_total // 2
            k_exp = k_total - k_orig
            
            # Retrieve for original query
            original_docs_raw: List[Doc] = hybrid_retriever.retrieve(query, k_orig)
            
            # --- Calculate Recall for Original Query --- 
            orig_parent_doc_ids = []
            seen_orig_parent_ids = set()
            for doc in original_docs_raw:
                if doc.doc_id not in seen_orig_parent_ids:
                    orig_parent_doc_ids.append(doc.doc_id)
                    seen_orig_parent_ids.add(doc.doc_id)
            # Calculate recall (use all k values for now, filter later if needed)
            orig_query_recall = calculate_recall_at_k(orig_parent_doc_ids, ground_truth_ids_for_recall, k_values_to_calculate)
            # --- End Original Recall Calculation ---

            # Retrieve for expanded query (if available)
            expanded_docs_raw: List[Doc] = []
            exp_query_recall = {k: 0.0 for k in k_values_to_calculate} # Default to 0 if no expansion
            if expanded_query:
                expanded_docs_raw = hybrid_retriever.retrieve(expanded_query, k_exp)
                # --- Calculate Recall for Expanded Query --- 
                exp_parent_doc_ids = []
                seen_exp_parent_ids = set()
                for doc in expanded_docs_raw:
                    if doc.doc_id not in seen_exp_parent_ids:
                        exp_parent_doc_ids.append(doc.doc_id)
                        seen_exp_parent_ids.add(doc.doc_id)
                exp_query_recall = calculate_recall_at_k(exp_parent_doc_ids, ground_truth_ids_for_recall, k_values_to_calculate)
                # --- End Expanded Recall Calculation ---
            else:
                # If no expansion, retrieve remaining k with original query
                # Note: This might slightly inflate original_docs_raw beyond k_orig for the combined calc later
                original_docs_raw.extend(hybrid_retriever.retrieve(query, k_exp))
            
            # --- Combine and Deduplicate ---
            combined_docs_raw = original_docs_raw + expanded_docs_raw
            final_retrieved_docs: List[Doc] = []
            seen_doc_ids_in_query = set()
            for doc in combined_docs_raw:
                if doc.doc_id not in seen_doc_ids_in_query:
                    final_retrieved_docs.append(doc)
                    seen_doc_ids_in_query.add(doc.doc_id)
            # --- End Combine and Deduplicate ---

            # Direct retrieval using args.top_k_retrieve (REPLACED by expansion logic above)
            # retrieved_docs_raw: List[Doc] = hybrid_retriever.retrieve(query, args.top_k_retrieve) 
            
            # --- Process final_retrieved_docs --- (Replaces processing of retrieved_docs_raw)
            retrieved_info_list = []
            final_retrieved_parent_doc_ids = [] # List for unique parent doc IDs for recall
            # Use final_retrieved_docs which are already deduplicated by doc_id for recall list
            for doc in final_retrieved_docs:
                # Store detailed info (chunk details might still be useful)
                retrieved_info = {
                    "text": doc.text,
                    "chunk_id": doc.chunk_id,
                    "doc_id": doc.doc_id
                }
                retrieved_info_list.append(retrieved_info)
                # Parent doc IDs are already unique in final_retrieved_docs
                final_retrieved_parent_doc_ids.append(doc.doc_id) 
            
            # No need for separate seen_parent_doc_ids or seen_chunk_ids_for_storage if we use final_retrieved_docs
            # Remove the old loop processing retrieved_docs_raw:
            # for doc in retrieved_docs_raw: ... (removed)
            # --- End Processing final_retrieved_docs ---
            
            end_time = time.time()
            total_processing_time += (end_time - start_time)

            # *** Store comprehensive data for this query index ***
            query_output_data = {
                "query": query,
                "expanded_query": expanded_query, # Store the expanded query
                "ground_truth_answer": ground_truth_answer,
                "ground_truth_ids": ground_truth_ids_to_store,
                "retrieved_chunks": retrieved_info_list, # Store info from combined/deduplicated list
                # --- Add independent recall results --- 
                "orig_recall": orig_query_recall,      # Recall results for original query
                "exp_recall": exp_query_recall,        # Recall results for expanded query
                "combined_recall": calculate_recall_at_k(final_retrieved_parent_doc_ids, ground_truth_ids_for_recall, k_values_to_calculate) # Recall results for combined list
                # --- End independent recall results --- 
            }
            retrieved_data_store[i] = query_output_data
            
            # Calculate recall using the unique PARENT document IDs from the combined list
            combined_query_recall = calculate_recall_at_k(final_retrieved_parent_doc_ids, ground_truth_ids_for_recall, k_values_to_calculate)
            
            # --- Store recall results for averaging --- 
            num_ground_truth = len(ground_truth_ids_for_recall)
            if num_ground_truth > 0: 
                for k in k_values_to_calculate:
                    # Store Combined Recall
                    combined_recall_score = combined_query_recall.get(k, 0.0)
                    combined_recall_results[k].append(combined_recall_score)
                    # Store Orig Recall
                    orig_recall_score = orig_query_recall.get(k, 0.0)
                    orig_recall_results[k].append(orig_recall_score)
                    # Store Exp Recall
                    exp_recall_score = exp_query_recall.get(k, 0.0)
                    exp_recall_results[k].append(exp_recall_score)
                    
                    # --- Append to specific category --- 
                    if num_ground_truth == 1:
                        combined_single_doc_recall_results[k].append(combined_recall_score)
                        orig_single_doc_recall_results[k].append(orig_recall_score)
                        exp_single_doc_recall_results[k].append(exp_recall_score)
                    elif num_ground_truth == 2:
                        combined_two_doc_recall_results[k].append(combined_recall_score)
                        orig_two_doc_recall_results[k].append(orig_recall_score)
                        exp_two_doc_recall_results[k].append(exp_recall_score)
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
                running_avg_recall[f'R@{k}'] = np.mean(combined_recall_results[k]) if combined_recall_results[k] else 0.0
            postfix_metrics = {key: f"{value:.3f}" for key, value in running_avg_recall.items()}
            # --- Add single/two doc R@5 to postfix ---
            if 5 in k_values_to_calculate:
                 running_avg_recall_single_5 = np.mean(combined_single_doc_recall_results[5]) if combined_single_doc_recall_results.get(5) else 0.0
                 running_avg_recall_two_5 = np.mean(combined_two_doc_recall_results[5]) if combined_two_doc_recall_results.get(5) else 0.0
                 postfix_metrics['R@5(S)'] = f"{running_avg_recall_single_5:.3f}" # 'S' for Single
                 postfix_metrics['R@5(T)'] = f"{running_avg_recall_two_5:.3f}" # 'T' for Two
            # --- End postfix addition ---
            # --- Modify to show R@100(*/**) specifically --- 
            target_k = 100 # Define the specific k we want to show
            if k_values_to_calculate and target_k in k_values_to_calculate: # Ensure 100 is being calculated
                # Calculate running averages for k=100 using ORIGINAL query results
                running_avg_single_100 = np.mean(orig_single_doc_recall_results[target_k]) if orig_single_doc_recall_results.get(target_k) else 0.0
                running_avg_two_100 = np.mean(orig_two_doc_recall_results[target_k]) if orig_two_doc_recall_results.get(target_k) else 0.0
                # Set postfix for R@100 based on ORIGINAL query recall
                postfix_metrics[f'R@{target_k}(Orig*)'] = f"{running_avg_single_100:.3f}" # Changed label
                postfix_metrics[f'R@{target_k}(Orig**)'] = f"{running_avg_two_100:.3f}" # Changed label
                # Remove R@max_k if it's different and was previously calculated (optional cleanup)
                max_k = max(k_values_to_calculate)
                if max_k != target_k:
                     postfix_metrics.pop(f'R@{max_k}(*)', None)
                     postfix_metrics.pop(f'R@{max_k}(**)', None)
                # Remove the old R@5(S/T) if they exist (already done in previous logic, keep for safety)
                postfix_metrics.pop('R@5(S)', None)
                postfix_metrics.pop('R@5(T)', None)
            # --- End R@100 modification ---
            progress_bar.set_postfix(postfix_metrics)

    # 4. Print Final Recall Results (updated for breakdown)
    print("\n--- Final Average Recall Results (Hybrid Search) ---") 
    # Use count based on actual results collected, not just processed count
    # Note: Counts should be consistent across orig/exp/combined if ground truth exists
    # Use combined_count as the reference for queries with ground truth
    combined_count = len(combined_recall_results[k_values_to_calculate[0]]) if k_values_to_calculate and combined_recall_results.get(k_values_to_calculate[0]) else 0
    single_doc_count = len(combined_single_doc_recall_results[k_values_to_calculate[0]]) if k_values_to_calculate and combined_single_doc_recall_results.get(k_values_to_calculate[0]) else 0
    two_doc_count = len(combined_two_doc_recall_results[k_values_to_calculate[0]]) if k_values_to_calculate and combined_two_doc_recall_results.get(k_values_to_calculate[0]) else 0

    if combined_count > 0: # Check if any recall results were actually calculated
        print(f"Metrics based on {combined_count} queries with ground truth ({single_doc_count} single-doc, {two_doc_count} two-doc).")
        print("Avg Recall@k: Orig=Original Query | Exp=Expanded Query | Comb=Combined List")
        print("--------------------------------------------------------------------------")
        for k in sorted(k_values_to_calculate):
            # Calculate overall averages
            avg_orig = np.mean(orig_recall_results[k]) if orig_recall_results.get(k) else 0.0
            avg_exp = np.mean(exp_recall_results[k]) if exp_recall_results.get(k) else 0.0 # Exp might have fewer entries if expansion failed often
            avg_comb = np.mean(combined_recall_results[k]) # Assumes list is non-empty based on combined_count check
            
            # Calculate single/two doc averages
            avg_orig_single = np.mean(orig_single_doc_recall_results[k]) if orig_single_doc_recall_results.get(k) else 0.0
            avg_exp_single = np.mean(exp_single_doc_recall_results[k]) if exp_single_doc_recall_results.get(k) else 0.0
            avg_comb_single = np.mean(combined_single_doc_recall_results[k]) if combined_single_doc_recall_results.get(k) else 0.0

            avg_orig_two = np.mean(orig_two_doc_recall_results[k]) if orig_two_doc_recall_results.get(k) else 0.0
            avg_exp_two = np.mean(exp_two_doc_recall_results[k]) if exp_two_doc_recall_results.get(k) else 0.0
            avg_comb_two = np.mean(combined_two_doc_recall_results[k]) if combined_two_doc_recall_results.get(k) else 0.0

            # Print overall averages for the current k
            print(f"Recall@{k:<3}: Orig={avg_orig:.4f} | Exp={avg_exp:.4f} | Comb={avg_comb:.4f}")
            # Print single/two doc breakdown for the current k
            print(f"  Single(*): Orig={avg_orig_single:.4f} | Exp={avg_exp_single:.4f} | Comb={avg_comb_single:.4f} (n={single_doc_count})")
            print(f"  TwoDoc(**): Orig={avg_orig_two:.4f} | Exp={avg_exp_two:.4f} | Comb={avg_comb_two:.4f} (n={two_doc_count})")
            print("--") # Separator for each k
            
        avg_time = total_processing_time / queries_processed_count if queries_processed_count else 0 # Keep avg time based on processed count
        print(f"\nSuccessfully processed {queries_processed_count} queries.")
        if queries_with_errors > 0:
             print(f"Encountered errors in {queries_with_errors} queries (see logs for details).")
        print(f"Average processing time per successful query: {avg_time:.2f} seconds")
    else:
        print("No queries with ground truth were successfully processed to calculate recall.")
        if queries_with_errors > 0:
             print(f"Encountered errors in {queries_with_errors} queries.")
            
    print("--------------------------------------------------------")

    # 5. Save Comprehensive Retrieved Data to JSON
    # --- Modify output filename to reflect expansion ---
    output_filename = f"retrieved_expanded_data_{args.top_k_retrieve}.json" # Dynamic filename
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

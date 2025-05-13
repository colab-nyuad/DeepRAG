"""Run entire DeepRAG pipeline."""
import argparse
import json
import re
import os
import time
from typing import List, Dict, Any

#from query_expansion import QueryExpander
from query_expansion_ola import QueryExpander, Query
from retriever import PineconeRetriever, OpenSearchRetriever, HybridRetriever
from utils import EvaluationDataset
from reranker import DocumentReranker
from reader import DocumentReader
import time
from utils import ReaderMetrics
from llm_reranker import LLMReranker  # Import new component
import torch
from transformers import AutoTokenizer, AutoModelForMaskedLM
import numpy as np
from tqdm import tqdm

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

def main():
    """Main entry point with argument parsing."""
    parser = argparse.ArgumentParser(description="DeepRAG")
    parser.add_argument("--dataset", type=str, default="./data/test.json", help="Path to data")
    parser.add_argument("--datatype", type=str, default="all", choices=["all", "single", "multiple"], 
                        help="Type of data to evaluate")
    parser.add_argument("--top-k", type=int, default=50, help="Number of documents to retrieve")
    parser.add_argument("--batch-size", type=int, default=10, help="Batch size for processing")
    parser.add_argument("--vector-weight", type=float, default=0.5, help="Weight for vector retrieval in hybrid mode")
    parser.add_argument("--retriever", type=str, choices=["vector", "keyword", "hybrid"], default="hybrid", 
                        help="Retriever type to use")
    parser.add_argument("--model", type=str, default="intfloat/e5-base-v2", help="Embedding model to use")
    parser.add_argument("--n-parallel", type=int, default=10, help="Number of parallel threads for batch retrieval")
    parser.add_argument("--top-r", type=int, default=5, help="Number of documents to provide to reader")
    parser.add_argument("--verbose", action='store_true', help='Print out info')
    parser.add_argument("--gt", action='store_true', help='use GT')
    # Add new SPLADE model argument
    parser.add_argument("--splade-model", type=str, default="naver/splade-cocondenser-ensembledistil",
                        help="Hugging Face model name/path for SPLADE.")
    # Add inference mode flag
    parser.add_argument("--inf", action='store_true', help='Run in inference mode without evaluation')
    # Add query expansion flag
    parser.add_argument("--expand", action='store_true', help='Enable query expansion with a single subquestion')

    args = parser.parse_args()

    # Initialize SPLADE model
    print(f"Initializing SPLADE model: {args.splade_model}")
    device = "cuda" if torch.cuda.is_available() else "cpu"
    
    # Remove try/except and just initialize the model directly
    splade_tokenizer = AutoTokenizer.from_pretrained(args.splade_model)
    splade_model = AutoModelForMaskedLM.from_pretrained(args.splade_model).to(device)
    splade_model.eval()
    print(f"SPLADE model initialized on {device}")

    # load dataset
    dataset = EvaluationDataset(args.dataset)

    # get queries, answers and ground truth ids
    if args.datatype == "single":
        queries, gt_answers, ground_truth_ids = dataset.get_data_for_single_doc_questions()
    elif args.datatype == "multiple":
        queries, gt_answers, ground_truth_ids = dataset.get_data_for_multiple_doc_questions()
    else:
        queries, gt_answers, ground_truth_ids, chunks = dataset.get_all_data()

    # Initialize query expander if expansion is enabled
    if args.expand:
        query_expander = QueryExpander(max_workers=args.n_parallel, batch_size=args.batch_size)

    # initialize retrievers
    vector_retriever = PineconeRetriever(max_workers=args.n_parallel, batch_size=args.batch_size)
    keyword_retriever = OpenSearchRetriever(batch_size=args.batch_size)
    hybrid_retriever = HybridRetriever(vector_retriever=vector_retriever, keyword_retriever=keyword_retriever, 
                                    vector_weight=args.vector_weight)

    retriever = {"vector": vector_retriever, "keyword": keyword_retriever, "hybrid": hybrid_retriever}[args.retriever]

    # initialize reader
    reader = DocumentReader(max_workers=args.n_parallel, batch_size=args.batch_size, verbose=args.verbose)

    # initialize reader evaluator
    reader_evaluator = ReaderMetrics()

    # total time 
    total_time = 0

    # evaluation dictionary
    retriever_eval_dict = {}
    reader_eval_dict = {}

    for i, orig_query in enumerate(queries):
        # start time
        start_time = time.time()

        print(f"Query: {orig_query}")
        if not args.inf:
            print(f"Ground truth IDs: {ground_truth_ids[i]}")
        
        # Final documents to provide to the reader
        reader_docs = []
        
        if args.gt and not args.inf:
            # If using ground truth, use the provided chunks as before
            gt_docs = []
            for j, text in enumerate(chunks[i]):
                gt_docs.append(("id_"+str(j), text))
            reader_docs = gt_docs
            print("GT Docs ++++++++++++++++++++", len(chunks), len(chunks[i]))
            print(chunks[i])
        else:
            # Simplified pipeline: Just retrieve and rerank - no subqueries
            
            print("Retrieving documents for query...")
            # Retrieve documents for the original query
            retrieved_docs_raw = retriever.retrieve(orig_query, args.top_k)
            retrieved_docs = [(doc.doc_id, doc.text) for doc in retrieved_docs_raw]
            retrieved_ids = [doc.doc_id for doc in retrieved_docs_raw]
            
            if args.verbose:
                print(f"Original Query: {orig_query}")
                print(f"Retrieved {len(retrieved_docs)} documents")
                # Print retrieved documents
                for doc_id, text in retrieved_docs:
                    print(f"Doc {doc_id}: {text[:100]}...")
                print("---")
            
            # Evaluate retrieval only if not in inference mode
            if not args.inf and retrieved_ids:
                retriever_metrics = retriever.evaluate([retrieved_ids], [ground_truth_ids[i]])
                retriever_eval_dict[(i, len(ground_truth_ids[i]))] = retriever_metrics
            
            # Filtered out noisy documents
            clean_docs = []
            for doc_id, text in retrieved_docs:
                if not is_noisy_document(text):
                    clean_docs.append((doc_id, text)) 
            
            if len(clean_docs) < len(retrieved_docs):
                print(f"Filtered out {len(retrieved_docs) - len(clean_docs)} noisy documents")
            
            # Create a list of documents for SPLADE reranking
            docs_to_rerank = clean_docs
            
            # Handle query expansion if enabled - MOVED BEFORE RERANKING
            expansion_status = "Not attempted"
            expansion_added = False
            expansion_in_ground_truth = False
            
            if args.expand:
                print("\nGenerating subquestion for query expansion...")
                # Generate a single subquestion
                expanded_query_result = query_expander.batch_query_expansion([orig_query])[0]
                
                # Evaluate metrics before adding expansion document (for all expansion attempts)
                if not args.inf:
                    pre_expansion_doc_ids = [doc_id for doc_id, _ in clean_docs]
                    gt_ids = ground_truth_ids[i]
                    
                    pre_expansion_true_positives = len(set(pre_expansion_doc_ids).intersection(set(gt_ids)))
                    pre_expansion_precision = pre_expansion_true_positives / len(pre_expansion_doc_ids) if len(pre_expansion_doc_ids) > 0 else 0
                    pre_expansion_recall = pre_expansion_true_positives / len(gt_ids) if len(gt_ids) > 0 else 0
                    pre_expansion_f1 = 2 * (pre_expansion_precision * pre_expansion_recall) / (pre_expansion_precision + pre_expansion_recall) if (pre_expansion_precision + pre_expansion_recall) > 0 else 0
                
                if expanded_query_result.expanded_queries:
                    subquestion = expanded_query_result.expanded_queries[0]  # Get the first expanded query
                    print(f"Subquestion: {subquestion}")
                    expansion_status = "Subquestion generated"
                    
                    # Retrieve top document for the subquestion using vector retriever for precision
                    print(f"Retrieving document for subquestion...")
                    subq_docs_raw = vector_retriever.retrieve(subquestion, top_k=1)
                    
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
                                
                                # Check if expansion document is in ground truth
                                if not args.inf and subq_doc_id in ground_truth_ids[i]:
                                    expansion_in_ground_truth = True
                                    
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
                else:
                    print("No subquestion generated")
                    expansion_status = "No subquestion generated"
            
            print(f"\nReranking {len(docs_to_rerank)} documents using SPLADE...")
            
            # Direct SPLADE reranking with original query
            reranked_docs = get_top_docs_by_splade(
                query=orig_query,
                docs=docs_to_rerank,
                splade_model=splade_model,
                splade_tokenizer=splade_tokenizer,
                device=device,
                top_k=args.top_r
            )
            
            # Map reranked docs to reader format
            reader_docs = [(doc_id, text) for doc_id, text in reranked_docs]
            print(f"Selected {len(reader_docs)} documents using SPLADE reranking")

            # Calculate reranking metrics only if not in inference mode
            if not args.inf:
                # Get the current reranked doc IDs (including any from expansion)
                reranked_doc_ids = [doc_id for doc_id, _ in reader_docs]
                gt_ids = ground_truth_ids[i]
                
                true_positives = len(set(reranked_doc_ids).intersection(set(gt_ids)))
                precision = true_positives / len(reranked_doc_ids) if len(reranked_doc_ids) > 0 else 0
                recall = true_positives / len(gt_ids) if len(gt_ids) > 0 else 0
                f1 = 2 * (precision * recall) / (precision + recall) if (precision + recall) > 0 else 0
                
                # Add expansion info to metric label if expansion was used
                metric_label = "Reranking"
                if args.expand and expansion_added:
                    metric_label += " + Expansion"
                
                print(f"{metric_label} Metrics - Precision: {precision:.2f}, Recall: {recall:.2f}, F1: {f1:.2f}")
                
                # Report expansion usefulness for all expansion attempts
                if args.expand:
                    print("\n========== EXPANSION USEFULNESS ==========")
                    print(f"Expansion status: {expansion_status}")
                    print(f"Metrics before expansion: Precision: {pre_expansion_precision:.2f}, Recall: {pre_expansion_recall:.2f}, F1: {pre_expansion_f1:.2f}")
                    print(f"Metrics after expansion:  Precision: {precision:.2f}, Recall: {recall:.2f}, F1: {f1:.2f}")
                    
                    # Calculate absolute improvement
                    precision_change = precision - pre_expansion_precision
                    recall_change = recall - pre_expansion_recall
                    f1_change = f1 - pre_expansion_f1
                    
                    # Determine if expansion was useful
                    if expansion_added:
                        if expansion_in_ground_truth:
                            print(f"✓ Expansion document was in ground truth!")
                        else:
                            print(f"✗ Expansion document was not in ground truth")
                            
                        if f1_change > 0:
                            print(f"✓ Expansion improved F1 by {f1_change:.2f}")
                        elif f1_change == 0:
                            print(f"○ Expansion had no effect on F1")
                        else:
                            print(f"✗ Expansion decreased F1 by {abs(f1_change):.2f}")
                    else:
                        print(f"○ No expansion document was added ({expansion_status})")
                    print("==========================================")
                
                # Store reranking metrics for comparison (as percentages)
                if (i, len(ground_truth_ids[i])) in retriever_eval_dict:
                    retriever_eval_dict[(i, len(ground_truth_ids[i]))]["reranked"] = {
                        "precision": precision * 100,
                        "recall": recall * 100,
                        "f1": f1 * 100
                    }
            
            # Format documents for reader
            reader_documents = [reader_docs]  # Format expected by batch_generate

        # Generate query object for reader with or without the subquestion
        expanded_queries = []
        if args.expand and 'subquestion' in locals() and subquestion:
            expanded_queries = [subquestion]
            
        query_obj = Query(original_query=orig_query, expanded_queries=expanded_queries)
        
        # Generate answers
        answers = reader.batch_generate(
            queries=[query_obj],
            documents_list=reader_documents
        )

        # save or print results
        for answer in answers:
            print(f"Answer: {answer.answer}")
            
            # Skip evaluation if in inference mode
            if not args.inf:
                # Evaluate the answer
                reader_metrics = reader_evaluator.evaluate_with_claude(
                    question=orig_query,
                    answer=answer.answer, 
                    ground_truth=gt_answers[i], 
                    documents=reader_documents[0]
                )
                print("Ground truth:", gt_answers[i])
                reader_eval_dict[(i, len(ground_truth_ids[i]))] = reader_metrics
                    
                # Print QA metrics for this query in a clear format
                print("\n========== QA METRICS ==========")
                print(f"Jaccard: {reader_metrics['jaccard']:.2f}")
                print(f"BLEU: {reader_metrics['bleu']:.2f}")
                print(f"ROUGE-1: {reader_metrics['rouge1']:.2f}")
                print(f"ROUGE-2: {reader_metrics['rouge2']:.2f}")
                print(f"ROUGE-L: {reader_metrics['rougeL']:.2f}")
                print(f"Relevance: {reader_metrics['relevance']:.2f}")
                print(f"Faithfulness: {reader_metrics['faithfulness']:.2f}")
                print("===============================\n")

                # Print a clear comparison of retrieval metrics before and after reranking
                if not args.gt and (i, len(ground_truth_ids[i])) in retriever_eval_dict:
                    metrics_key = (i, len(ground_truth_ids[i]))
                    print("\n========== RETRIEVAL METRICS COMPARISON ==========")
                    
                    # Original retrieval metrics (top-k)
                    print("Original Retrieval (top-k):")
                    for k in sorted([int(k.replace("@", "")) for k in retriever_eval_dict[metrics_key].keys() if k.startswith("@")]):
                        k_key = f"@{k}"
                        if k_key in retriever_eval_dict[metrics_key]:
                            p = retriever_eval_dict[metrics_key][k_key]["precision"]
                            r = retriever_eval_dict[metrics_key][k_key]["recall"]
                            f1 = retriever_eval_dict[metrics_key][k_key]["f1"]
                            print(f"  top-{k}: Precision={p:.2f}, Recall={r:.2f}, F1={f1:.2f}")
                    
                    # Reranking metrics (top-r)
                    if "reranked" in retriever_eval_dict[metrics_key]:
                        rerank_metrics = retriever_eval_dict[metrics_key]["reranked"]
                        print(f"\nAfter Reranking (top-{args.top_r}):")
                        print(f"  Precision={rerank_metrics['precision']:.2f}, Recall={rerank_metrics['recall']:.2f}, F1={rerank_metrics['f1']:.2f}")
                    
                    print("================================================\n")

        # end time
        end_time = time.time()

        # calculate total time
        total_time += end_time - start_time

        print(f"Time taken for query '{orig_query}': {end_time - start_time:.2f} seconds\n")
        print("-------------------------------------------------------\n")

    # Only calculate and print metrics breakdown if not in inference mode
    if not args.inf:
        get_breakdown_of_metrics(retriever_eval_dict, reader_eval_dict, args.datatype)

        # Add reranking improvement summary
        print("\n========== RERANKING IMPROVEMENT SUMMARY ==========")
        
        # Calculate average improvement from reranking
        orig_precision = 0
        orig_recall = 0
        orig_f1 = 0
        rerank_precision = 0
        rerank_recall = 0
        rerank_f1 = 0
        count = 0
        
        for key, metrics in retriever_eval_dict.items():
            if "reranked" in metrics:
                # Find the closest top-k to the top-r value
                top_r = args.top_r  # Use the command line argument instead
                closest_k = min([int(k.replace("@", "")) for k in metrics.keys() if k.startswith("@")], 
                            key=lambda x: abs(x - top_r))
                k_key = f"@{closest_k}"
                
                if k_key in metrics:
                    # Ensure consistent format - store all as decimals (0.0-1.0)
                    orig_precision += metrics[k_key]["precision"] 
                    orig_recall += metrics[k_key]["recall"] 
                    orig_f1 += metrics[k_key]["f1"] 
                    
                    rerank_precision += metrics["reranked"]["precision"]
                    rerank_recall += metrics["reranked"]["recall"]
                    rerank_f1 += metrics["reranked"]["f1"]
                    
                    count += 1
        
        if count > 0:
            print(f"Number of queries: {count}")
            
            # Metrics are already in percentage format
            print(f"\nOriginal Retrieval Average Metrics (closest top-k to top-r):")
            print(f"  Precision: {(orig_precision/count):.2f}%")
            print(f"  Recall: {(orig_recall/count):.2f}%")
            print(f"  F1: {(orig_f1/count):.2f}%")
            
            print(f"\nReranking Average Metrics:")
            print(f"  Precision: {(rerank_precision/count):.2f}%")
            print(f"  Recall: {(rerank_recall/count):.2f}%")
            print(f"  F1: {(rerank_f1/count):.2f}%")
            
            # Calculate absolute and percentage differences
            precision_diff = rerank_precision/count - orig_precision/count
            recall_diff = rerank_recall/count - orig_recall/count 
            f1_diff = rerank_f1/count - orig_f1/count
            
            # Calculate percentage changes
            precision_pct = (precision_diff / (orig_precision/count)) * 100 if orig_precision/count > 0 else float('inf')
            recall_pct = (recall_diff / (orig_recall/count)) * 100 if orig_recall/count > 0 else float('inf')
            f1_pct = (f1_diff / (orig_f1/count)) * 100 if orig_f1/count > 0 else float('inf')
            
            print(f"\nImprovement:")
            print(f"  Precision: {precision_diff:.2f}% ({precision_pct:.1f}%)")
            print(f"  Recall: {recall_diff:.2f}% ({recall_pct:.1f}%)")
            print(f"  F1: {f1_diff:.2f}% ({f1_pct:.1f}%)")
        else:
            print("No reranking metrics available for comparison.")
        
        print("====================================================\n")

    # print total time
    print(f"Total time taken for all queries: {total_time:.2f} seconds\n")
    # print average time per query
    print(f"Average time per query: {total_time / len(queries):.2f} seconds\n")

        

    # # retrieve documents
    # retrieved_ids = retriever.batch_retrieve_unique_ids(queries, args.top_k)
    
    # k_values = [1 ,3 ,5, 10] if args.top_k <=10 else [1, 3, 5, 10, 20, 50, 100]
    # # evaluate retriever
    # eval_results = retriever.evaluate(retrieved_ids, ground_truth_ids, k_values)
    # print(json.dumps(eval_results, indent=2))


def get_breakdown_of_metrics(retriever_eval_dict, reader_eval_dict, datatype):
    """
    Calculate and print breakdown of metrics based on datatype.
    
    Args:
        retriever_eval_dict: Dictionary with (query_index, num_ground_truths) as keys and retrieval metrics as values
        reader_eval_dict: Dictionary with (query_index, num_ground_truths) as keys and reader metrics as values
        datatype: Type of data being evaluated ("all", "single", or "multiple")
    """
    # Separate single and multiple document questions
    single_doc_retriever = {k: v for k, v in retriever_eval_dict.items() if k[1] == 1}
    multiple_doc_retriever = {k: v for k, v in retriever_eval_dict.items() if k[1] > 1}
    
    single_doc_reader = {k: v for k, v in reader_eval_dict.items() if k[1] == 1}
    multiple_doc_reader = {k: v for k, v in reader_eval_dict.items() if k[1] > 1}
    
    # Calculate average retriever metrics
    def average_retriever_metrics(metrics_dict):
        if not metrics_dict:
            return {}
        
        # Get all k values from the first item
        first_key = next(iter(metrics_dict))
        first_metrics = metrics_dict[first_key]
        k_values = [int(k.replace("@", "")) for k in first_metrics.keys() if k.startswith("@")]
        
        # Initialize result dictionary
        result = {}
        for k in k_values:
            result[f"@{k}"] = {"precision": 0, "recall": 0, "f1": 0}
        
        # Add reranking metrics if they exist
        result["reranked"] = {"precision": 0, "recall": 0, "f1": 0}
        reranking_count = 0
        
        # Sum all metrics
        for item_metrics in metrics_dict.values():
            for k in k_values:
                k_key = f"@{k}"
                if k_key in item_metrics:
                    result[k_key]["precision"] += item_metrics[k_key]["precision"]
                    result[k_key]["recall"] += item_metrics[k_key]["recall"]
                    result[k_key]["f1"] += item_metrics[k_key]["f1"]
            
            # Sum reranking metrics if they exist
            if "reranked" in item_metrics:
                result["reranked"]["precision"] += item_metrics["reranked"]["precision"]
                result["reranked"]["recall"] += item_metrics["reranked"]["recall"]
                result["reranked"]["f1"] += item_metrics["reranked"]["f1"]
                reranking_count += 1
        
        # Calculate averages
        num_items = len(metrics_dict)
        for k in k_values:
            k_key = f"@{k}"
            result[k_key]["precision"] = round(result[k_key]["precision"] / num_items, 2)
            result[k_key]["recall"] = round(result[k_key]["recall"] / num_items, 2)
            result[k_key]["f1"] = round(result[k_key]["f1"] / num_items, 2)
        
        # Calculate reranking averages if they exist
        if reranking_count > 0:
            result["reranked"]["precision"] = round(result["reranked"]["precision"] / reranking_count, 2)
            result["reranked"]["recall"] = round(result["reranked"]["recall"] / reranking_count, 2)
            result["reranked"]["f1"] = round(result["reranked"]["f1"] / reranking_count, 2)
        else:
            # Remove reranked key if no data
            result.pop("reranked", None)
        
        return result
    
    # Calculate average reader metrics
    def average_reader_metrics(metrics_dict):
        if not metrics_dict:
            return {}
        
        # Initialize result dictionary
        result = {
            "jaccard": 0,
            "bleu": 0,
            "rouge1": 0,
            "rouge2": 0,
            "rougeL": 0,
            "relevance": 0,
            "faithfulness": 0
        }
        
        # Sum all metrics
        for item_metrics in metrics_dict.values():
            result["jaccard"] += item_metrics["jaccard"]
            result["bleu"] += item_metrics["bleu"]
            result["rouge1"] += item_metrics["rouge1"]
            result["rouge2"] += item_metrics["rouge2"]
            result["rougeL"] += item_metrics["rougeL"]
            result["relevance"] += item_metrics["relevance"]
            result["faithfulness"] += item_metrics["faithfulness"]
        
        # Calculate averages
        num_items = len(metrics_dict)
        for key in result:
            result[key] = round(result[key] / num_items, 2)
        
        return result
    
    # Print results based on datatype
    if datatype == "single":
        print("\n===== SINGLE DOCUMENT QUESTIONS METRICS =====")
        print("\nRetriever Metrics:")
        print(json.dumps(average_retriever_metrics(single_doc_retriever), indent=2))
        print("\nReader Metrics:")
        print(json.dumps(average_reader_metrics(single_doc_reader), indent=2))
        
    elif datatype == "multiple":
        print("\n===== MULTIPLE DOCUMENT QUESTIONS METRICS =====")
        print("\nRetriever Metrics:")
        print(json.dumps(average_retriever_metrics(multiple_doc_retriever), indent=2))
        print("\nReader Metrics:")
        print(json.dumps(average_reader_metrics(multiple_doc_reader), indent=2))
        
    else:  # "all"
        # Single doc metrics
        print("\n===== SINGLE DOCUMENT QUESTIONS METRICS =====")
        print("\nRetriever Metrics:")
        print(json.dumps(average_retriever_metrics(single_doc_retriever), indent=2))
        print("\nReader Metrics:")
        print(json.dumps(average_reader_metrics(single_doc_reader), indent=2))
        
        # Multiple doc metrics
        print("\n===== MULTIPLE DOCUMENT QUESTIONS METRICS =====")
        print("\nRetriever Metrics:")
        print(json.dumps(average_retriever_metrics(multiple_doc_retriever), indent=2))
        print("\nReader Metrics:")
        print(json.dumps(average_reader_metrics(multiple_doc_reader), indent=2))
        
        # Overall metrics (all questions)
        print("\n===== OVERALL METRICS (ALL QUESTIONS) =====")
        print("\nRetriever Metrics:")
        print(json.dumps(average_retriever_metrics(retriever_eval_dict), indent=2))
        print("\nReader Metrics:")
        print(json.dumps(average_reader_metrics(reader_eval_dict), indent=2))
    


if __name__ == "__main__":
    main()
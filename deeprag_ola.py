"""Run entire DeepRAG pipeline."""
import argparse
import json

from query_expansion_ola import QueryExpander
from retriever import PineconeRetriever, OpenSearchRetriever, HybridRetriever
from utils import EvaluationDataset
from reranker import DocumentReranker
from reader import DocumentReader
import time
from utils import ReaderMetrics

def main():
    """Main entry point with argument parsing."""
    parser = argparse.ArgumentParser(description="DeepRAG")
    parser.add_argument("--dataset", type=str, default="./data/test.json", help="Path to data")
    parser.add_argument("--datatype", type=str, default="all", choices=["all", "single", "multiple"], 
                        help="Type of data to evaluate")
    parser.add_argument("--top-k", type=int, default=5, help="Number of documents to retrieve")
    parser.add_argument("--batch-size", type=int, default=10, help="Batch size for processing")
    parser.add_argument("--vector-weight", type=float, default=0.7, help="Weight for vector retrieval in hybrid mode")
    parser.add_argument("--retriever", type=str, choices=["vector", "keyword", "hybrid"], default="hybrid", 
                        help="Retriever type to use")
    parser.add_argument("--model", type=str, default="intfloat/e5-base-v2", help="Embedding model to use")
    parser.add_argument("--n-parallel", type=int, default=10, help="Number of parallel threads for batch retrieval")
    parser.add_argument("--top-r", type=int, default=5, help="Number of documents to provide to reader")
    parser.add_argument("--verbose", action='store_true', help='Print out info')
    args = parser.parse_args()

    # load dataset
    dataset = EvaluationDataset(args.dataset)

    # get queries , answers and ground truth ids
    if args.datatype == "single":
        queries, gt_answers, ground_truth_ids = dataset.get_data_for_single_doc_questions()
    elif args.datatype == "multiple":
        queries, gt_answers, ground_truth_ids = dataset.get_data_for_multiple_doc_questions()
    else:
        queries, gt_answers, ground_truth_ids = dataset.get_all_data()

    # initialize query expander
    query_expander = QueryExpander(max_workers=args.n_parallel, batch_size=args.batch_size)

     # initialize retrievers
    vector_retriever = PineconeRetriever(max_workers=args.n_parallel, batch_size=args.batch_size)
    keyword_retriever = OpenSearchRetriever(batch_size=args.batch_size)
    hybrid_retriever = HybridRetriever(vector_retriever=vector_retriever, keyword_retriever=keyword_retriever, 
                                    vector_weight=args.vector_weight)

    retriever = {"vector": vector_retriever, "keyword": keyword_retriever, "hybrid": hybrid_retriever}[args.retriever]

    # initialize reranker
    reranker = DocumentReranker(model_name="rank-T5-flan", cache_dir="/storage/prince/flashrank_models")

    # initialize reader
    reader = DocumentReader(max_workers=args.n_parallel, batch_size=args.batch_size, verbose=args.verbose)

    # initialize reader evaluator
    reader_evaluator = ReaderMetrics()

    # total time 
    total_time = 0

    # evaluation dictionary
    retriever_eval_dict = {}
    reader_eval_dict = {}
    single_doc_retriever_eval_dict = {}
    multiple_doc_retriever_eval_dict = {}
    single_doc_reader_eval_dict = {}
    multiple_doc_reader_eval_dict = {}

    for i, orig_query in enumerate(queries):
        # start time
        start_time = time.time()

        print(f"Query: {orig_query}")
        # expand queries
        expanded_queries = query_expander.batch_query_expansion([orig_query])

        # retrieve docs using expanded queries
        retrieved_docs = retriever.batch_retrieve_grouped_with_expansion(expanded_queries,args.top_k)

        # get predicted document IDs
        retrieved_ids = retriever.get_predicted_ids(retrieved_docs)

        # calculate recall
        metrics = retriever.evaluate(retrieved_ids, [ground_truth_ids[i]])

        
        # store evaluation results to compute average later(each is a dict so flatten it)
        for k, v in metrics.items():
            if k not in retriever_eval_dict:
                retriever_eval_dict[k] = []
            retriever_eval_dict[k].append(v)

        # get unique ids
        # retrieved_ids_with_text = vector_retriever.retrieve_chunks_by_id(retrieved_ids)

        # use documents for reranking
        retrieved_ids_with_text = [[(doc.doc_id, doc.text) for doc in docs] for docs in retrieved_docs]

        print("Number of Documents retrieved:", len(retrieved_ids_with_text[0]))

        # reranker
        reranked_docs = reranker.batch_rerank(
            queries=[query.original_query for query in expanded_queries], 
            documents_list=retrieved_ids_with_text
        ) if len(retrieved_ids_with_text[0]) > args.top_r else retrieved_ids_with_text


        # use top 10 docs for reader
        reranked_docs = [docs[:args.top_r] for docs in reranked_docs]

        print("Number of documents after re-ranking:", len(reranked_docs[0]))

        # generate answers
        answers = reader.batch_generate(
            queries=[query.original_query for query in expanded_queries],
            documents_list=reranked_docs
        )

        # save or print results (you can modify this part as needed)
        for answer in answers:
            print(f"Answer: {answer.answer}")
            reader_metrics = reader_evaluator.evaluate_with_claude(question=orig_query,answer=answer.answer, ground_truth=gt_answers[i], documents=reranked_docs[0])
            print("Ground truth :", gt_answers[i])
            print(json.dumps(reader_metrics))
            print(json.dumps(metrics))
            # store evaluation results to compute average later

            for k, v in reader_metrics.items():
                if k not in reader_eval_dict:
                    reader_eval_dict[k] = []
                reader_eval_dict[k].append(v)

        # end time
        end_time = time.time()

        # calculate total time
        total_time += end_time - start_time

        print(f"Time taken for query '{orig_query}': {end_time - start_time:.2f} seconds\n")
        print("-------------------------------------------------------\n")
    # calculate average metrics for retriever ( "@1" : [{precision, recall, f1}, {}], "@5" :[{}] ...)
    avg_retriever_metrics = {}
    for k, metrics_list in retriever_eval_dict.items():
        total = {"precision": 0.0, "recall": 0.0, "f1": 0.0}
        n = len(metrics_list)
        
        for metric in metrics_list:
            for m in ["precision", "recall", "f1"]:
                total[m] += metric.get(m, 0.0)
        
        avg_retriever_metrics[k] = {m: total[m] / n if n > 0 else 0.0 for m in total}

    print("Average Retriever Metrics:")

    print(json.dumps(avg_retriever_metrics, indent=2))
    print("\n-------------------------------------------------------\n")
    # calculate average metrics for reader
    avg_reader_metrics = {k: sum(v) / len(v) for k, v in reader_eval_dict.items()}
    print("Average Reader Metrics:")
    print(json.dumps(avg_reader_metrics, indent=2))
    print("\n-------------------------------------------------------\n")

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

if __name__ == "__main__":
    main()
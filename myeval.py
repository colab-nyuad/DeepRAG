import json
import sys
from utils.eval import ReaderMetrics

def extract_documents(passages):
    # Each passage is a dict with "passage" and "doc_IDs"
    # Return a list of (doc_id, text) tuples
    docs = []
    for p in passages:
        text = p.get("passage", "")
        for doc_id in p.get("doc_IDs", []):
            docs.append((doc_id, text))
    return docs

def main():
    import argparse
    parser = argparse.ArgumentParser(description="Evaluate average faithfulness and relevance using Claude")
    parser.add_argument("--input", type=str, required=True, help="Input results JSON file")
    args = parser.parse_args()

    with open(args.input, "r") as f:
        data = json.load(f)

    faithfulness_scores = []
    relevance_scores = []
    n = 0
    for entry in data:
        question = entry.get("question", "")
        answer = entry.get("answer", "")
        passages = entry.get("passages", [])
        docs = extract_documents(passages)
        metrics = ReaderMetrics.evaluate_with_claude(
            question=question,
            answer=answer,
            documents=docs
        )
        faith = metrics.get("faithfulness")
        rel = metrics.get("relevance")
        if faith is not None:
            faithfulness_scores.append(faith)
        if rel is not None:
            relevance_scores.append(rel)
        n += 1
        print(f"[{n}/{len(data)}] Faithfulness: {faith:.3f}  Relevance: {rel:.3f}")

    avg_faithfulness = sum(faithfulness_scores) / len(faithfulness_scores) if faithfulness_scores else 0
    avg_relevance = sum(relevance_scores) / len(relevance_scores) if relevance_scores else 0
    print(f"\nProcessed {n} questions")
    print(f"Average Faithfulness: {avg_faithfulness:.3f}")
    print(f"Average Relevance: {avg_relevance:.3f}")

if __name__ == "__main__":
    main() 
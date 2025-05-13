# DeepRAG: Deep Retrieval-Augmented Generation

DeepRAG is a question-answering system that leverages hybrid retrieval and reranking techniques to provide accurate answers to complex questions using a corpus of documents.

## Overview

DeepRAG implements a retrieval-augmented generation pipeline:
1. **Hybrid Retrieval**: Combines vector-based (Pinecone) and keyword-based (OpenSearch) retrieval
2. **Query Expansion**: Improves retrieval by generating subquestions 
3. **Reranking**: Uses SPLADE to optimize document relevance
4. **LLM-based Answer Generation**: Produces coherent answers based on retrieved documents

### Arguments

```
--input          Path to input JSONL file containing questions
--output         Path to output JSON file for results
--top-k          Number of documents to retrieve initially (default: 50)
--batch-size     Batch size for processing (default: 10)
--vector-weight  Weight for vector retrieval in hybrid mode (default: 0.5)
--retriever      Retriever type: "vector", "keyword", or "hybrid" (default: "hybrid")
--model          Embedding model name (default: "intfloat/e5-base-v2")
--n-parallel     Number of parallel threads (default: 10)
--top-r          Number of documents after reranking to provide to reader (default: 5)
--verbose        Enable verbose output
--splade-model   HuggingFace model for SPLADE (default: "naver/splade-cocondenser-ensembledistil")
--expand         Enable query expansion with a single subquestion
```

### Query Expansion

When the `--expand` flag is used, the system:
1. Generates a single subquestion related to the original query
2. Retrieves an additional document specifically for this subquestion
3. Adds this document to the candidate pool before SPLADE reranking
4. Includes expansion information in the output

### Output Format

The script produces a JSON file with entries for each question:
```json
{
  "id": "question_id",
  "question": "original question text",
  "passages": [
    {
      "passage": "document text",
      "doc_IDs": ["doc1", "doc2"]
    }
  ],
  "final_prompt": "prompt used for answer generation",
  "answer": "generated answer",
  "expansion_info": {
    "status": "expansion status",
    "subquestion": "generated subquestion",
    "document_added": true/false
  }
}
```

### Usage Example

```bash
python challenge-expand.py --input input.jsonl --output output.json --top-k 50 --top-r 5 --retriever hybrid --expand
```

Default QA model is 'falcon3:10b'
import json

query_expansion_json = {"queries": "List of specified number of expanded queries."}
qa_json = {"answer": "Answer to the question."}
faithfulness_json = {"faithfulness": "Choose among -1, 0 and 1"}
relevance_json = {"relevance": "Choose among -1, 0, 1 and 2"}
subquery_answer_json = {"document_answer": "An informative document that answers the subquery in detail."}
reranker_best_json = {"best_document_index": "Index of the most relevant document (integer)."}
reranker_topk_json = {"top_document_indices": "List of indices of the most relevant documents (list of integers)."}
subquery_validation_json = {"answers_subquery": "Boolean indicating if the document context unambiguously answers the subquery."}
doc_summary_json = {"summary": "Concise one-sentence summary of the document relevant to the query."}

CRITERIA_DESCRIPTIONS = {
    "relevance": "Measures the correctness and relevance of the answer to the question on a four-point scale.",
    "faithfulness": "Assesses whether the response is grounded in the retrieved passages on a three-point scale.",
}

QUERY_EXPANSION_SYSTEM_PROMPT = f"""You are an expert query expansion system that responds with JSON. The JSON object must always use the schema:\n{json.dumps(query_expansion_json, indent=2)}"""

QUERY_EXPANSION_USER_PROMPT = """You expand a given query into {number} queries that are similar in meaning.
          
--- Examples ---
Query: climate change effects
Output: {{"queries" : ["impact of climate change", "consequences of global warming", "effects of environmental changes"]}}

Query: machine learning algorithms
Output: {{"queries" : ["neural networks", "clustering", "supervised learning", "deep learning"]}}

--- Your Task ---
Query: {query}
Output:"""

QUERY_EXPANSION_USER_PROMPT_DECOMPOSE_old = """
Decompose the following question into the minimal set of semantically distinct subobjectives, each targeting an independent facet of the original query.
--- Examples ---
Query: Which of the countries in the Caribbean has the smallest country calling code?
{{"queries" : [
    "Countries in the Caribbean",
    "Country calling codes of Caribbean nations",
    "Smallest country calling code among those nations"
]}}

Query: What is the scientific explanation for why people who live at high altitudes often have larger lung capacities than those who live at sea level?
{{"queries" : [
    "Air pressure and oxygen levels at high altitudes",
    "Adaptations to lower oxygen availability over time",
    "Role of lung capacity in low-oxygen environments",
    "Genetic or environmental factors contributing to increased lung capacity",
    "Scientific explanations of high-altitude adaptation in human populations"
]}}
--- Your Task ---
Query: {query}
Output:
"""

QUERY_EXPANSION_USER_PROMPT_DECOMPOSE = """
Expand the following question by generating exactly one additional query that captures a necessary but distinct facet of the original question. 
The expansion should stay semantically close and help in retrieving supporting information.
Keep it simple, precise, and directly relevant to the original question.

--- Examples ---
Query: How do different approaches compare when it comes to the materials and construction methods used in modern drones and UAVs?
Expanded Query: {{"queries" : ["What are the main materials and construction methods used in modern drones and UAVs?"]}}

Query: How would asset pricing be affected if a market had both experienced and inexperienced traders using discounted cash flow analysis?
Expanded Query: {{"queries" : ["How does the experience level of traders impact the accuracy of discounted cash flow-based asset valuations?"]}}

--- Your Task ---
Query: {query}
Expanded Query:
"""



QUERY_EXPANSION_USER_PROMPT_DECOMPOSE2 = """
Please break down the process of answering the question into as few subobjectives as possible based on semantic analysis.
Here is an example:

Query: Which of the countries in the Caribbean has the smallest country calling code?
{{"queries" : [
    "Countries in the Caribbean",
    "Country calling codes of Caribbean nations",
    "Smallest country calling code among those nations"
]}}

Query: What is the scientific explanation for why people who live at high altitudes often have larger lung capacities than those who live at sea level?
{{"queries" : [
    "Air pressure and oxygen levels at high altitudes",
    "Adaptations to lower oxygen availability over time",
    "Role of lung capacity in low-oxygen environments",
    "Genetic or environmental factors contributing to increased lung capacity",
    "Scientific explanations of high-altitude adaptation in human populations"
]}}

Now you need to directly output subobjectives of the following question in list format without other information or notes.

Query: {query}

Output:
"""




#### Reader Prompts

QA_SYSTEM_PROMPT = f"""You are a question answering system that responds with JSON. The JSON object must always use the schema:\n{json.dumps(qa_json, indent=2)}"""

QA_USER_PROMPT = """
Below you will find several documents that provide context for answering the upcoming question. 
Please review the following texts carefully. 
While the context may include extraneous information, concentrate on details that are directly relevant to the question.

---Begin Context---
{documents}
---End Context---

--Question--
{query}

Instructions:
- Provide a short paragraph directly answering the question.
- Ignore any parts of the context that are not pertinent to the question.

--Response Rules--
- Response should be a json with key 'answer'"""


### Evaluation Prompts

QA_EVAL_SYSTEM_PROMPT = """You are an expert evaluator that responds with JSON. The JSON object must always use the schema:\n{schema}"""

RELEVANCE_USER_PROMPT= """Given a question and a corresponding answer, rate the relevance of the answer on a 4-point scale:

2: Correctly answers the question with no irrelevant content
1: Answers the question but includes some irrelevant content
0: No answer is provided (e.g., "I don't know")
-1: Does not answer the question at all

---Input---
Question: {question}
Answer: {answer}

--Response Rules--
- Response should be a JSON with key 'relevance'
- Do not include any other information other than JSON in the response"""

FAITHFULNESS_USER_PROMPT= """Given a question, an answer, and the corresponding retrieved passages, rate the faithfulness of the answer on a 3-point scale:

1: Fully supported — all parts of the answer are grounded in the retrieved passages
0: Partially supported — some parts of the answer are grounded, others are not
-1: Not supported — no parts of the answer are grounded in the retrieved passages

---Input---
Question: {question}
Answer: {answer}
Retrieved Passages: {documents}

--Response Rules--
- Response should be a JSON with key 'faithfulness'
- Do not include any other information other than JSON in the response"""

##### LLM Reranking Prompts


LLM_RERANKER_SYSTEM_PROMPT = f"""You are a document ranking system that responds with JSON. You evaluate the relevance of documents to a given query and select the most relevant ones. For single document selection, use the schema:\n{json.dumps(reranker_best_json, indent=2)}\n\nFor multiple document selection, use the schema:\n{json.dumps(reranker_topk_json, indent=2)}"""

LLM_RERANKER_SELECT_BEST_PROMPT = """Your task is to select the single most relevant document for answering this query:

---Query---
{query}

I'll provide you with {num_documents} documents. Read each one carefully and determine which is most relevant to the query.

---Documents---
{documents}

Instructions:
- Analyze each document's relevance to the query
- Select the single most relevant document that best addresses the query
- Consider factors like information completeness, directness of answer, and query-document match
- Respond with the index of the best document (as an integer from 0 to {max_index})

--Response Rules--
- Response must be a JSON with key 'best_document_index'
- The value must be an integer between 0 and {max_index}, inclusive"""

LLM_RERANKER_SELECT_TOP_K_PROMPT = """Select the {top_r} most relevant documents based on their summaries, which were generated specifically for this query:

---Query---
{query}

The document summaries are below. Each summary includes a frequency score `(Frequency: N)` indicating how many related subqueries retrieved the original document. Focus on finding summaries that indicate the document CONTAINS THE ACTUAL ANSWER to the query.

---Document Summaries---
{documents}

Guidelines:
- Evaluate the relevance of each document *summary* to the query.
- PRIORITIZE summaries that clearly indicate the document answers the query.
- CONSIDER DOCUMENT FREQUENCY: A higher frequency score *may* indicate higher relevance, but prioritize the summary's relevance to the query.
- SELECT the summaries that best correspond to documents providing FACTUAL INFORMATION relevant to the query.

--Response Format--
Provide a JSON object with key 'top_document_indices' containing an array of {top_r} integers (0 to {max_index}) corresponding to the document summaries.
Example: {{"top_document_indices": [5, 2, 7]}}
"""

##### LLM Subquery Document Validation Prompt

LLM_SUBQUERY_VALIDATOR_SYSTEM_PROMPT = f"""You are an expert evaluator that responds with JSON. The JSON object must always use the schema:\n{json.dumps(subquery_validation_json, indent=2)}"""

LLM_SUBQUERY_VALIDATOR_USER_PROMPT = """Assess if the provided document context unambiguously covers the subquery.

---Subquery---
{subquery}

---Document Context---
{document_text}

Instructions:
- Read the subquery and the document context carefully.
- Determine if the document contains information that directly provides information about the subquery.
- Your answer should be a simple boolean (true/false).

--Response Rules--
- Response must be a JSON with key 'answers_subquery' and a boolean value (true or false).
- Example: {{"answers_subquery": true}}
"""

##### LLM Document Summarization Prompt

LLM_DOC_SUMMARY_SYSTEM_PROMPT = f"""You are an expert summarizer that responds with JSON. The JSON object must always use the schema:\n{json.dumps(doc_summary_json, indent=2)}"""

LLM_DOC_SUMMARY_USER_PROMPT = """Generate a concise, one-sentence summary of the following document, focusing *only* on information directly relevant to the query provided.

---Query---
{query}

---Document Context---
{document_text}

Instructions:
- Read the query and the document context carefully.
- Extract the single most important piece of information from the document that helps answer the query.
- Condense this information into a single, fluent sentence.
- If the document contains no information relevant to the query, respond with an empty summary.

--Response Rules--
- Response must be a JSON with key 'summary' and a string value.
- The summary should be a single sentence.
- Example: {{"summary": "The program primarily serves financially disadvantaged youth from specific geographic areas."}}
""" 
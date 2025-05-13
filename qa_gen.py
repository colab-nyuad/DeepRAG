"""Generate Questions and Answers using DataMorgana."""
import json
import os
import time
from typing import Dict, List
from dotenv import load_dotenv
load_dotenv()

import requests

BASE_URL = "https://api.ai71.ai/v1/"

def get_api_key() -> str:
    if "AI71_KEY" not in os.environ:
        raise ValueError("AI71_KEY is not set")
    return os.environ["AI71_KEY"]

print(get_api_key())

def check_budget():
    resp = requests.get(
        f"{BASE_URL}check_budget",
        headers={"Authorization": f"Bearer {get_api_key()}"},
    )
    resp.raise_for_status()
    print(json.dumps(resp.json(), indent=4))
check_budget()

def bulk_generate(n_questions: int, question_categorizations: List[Dict], user_categorizations: List[Dict]):
    resp = requests.post(
        f"{BASE_URL}bulk_generation",
        headers={"Authorization": f"Bearer {get_api_key()}"},
        json={
                "n_questions": n_questions,
                "question_categorizations": question_categorizations,
                "user_categorizations": user_categorizations
            }
    )
    resp.raise_for_status()
    request_id = resp.json()["request_id"]
    print(json.dumps(resp.json(), indent=4))

    result = wait_for_generation_to_finish(request_id)
    return result


def wait_for_generation_to_finish(request_id: str):
    while True:
        resp = requests.get(
            f"{BASE_URL}fetch_generation_results",
            headers={"Authorization": f"Bearer {get_api_key()}"},
            params={"request_id": request_id},
        )
        resp.raise_for_status()
        if resp.json()["status"] == "completed":
            print(json.dumps(resp.json(), indent=4))
            return resp.json()
        else:
            print("Waiting for generation to finish...")
            time.sleep(5)

def get_files_using_id(request_id: str):
    result = wait_for_generation_to_finish(request_id)
    return result


question_length_categorization = {
    "categorization_name": "question_length",
    "categories": [
        {
            "name": "short",
            "description": "a short question with no more than 6 words.",
            "probability": 0.4
        },
        {
            "name": "long",
            "description": "a long question with at least 7 words.",
            "probability": 0.6
        }
    ]
}

question_formulation_categorization = {
    "categorization_name": "question_formulation",
    "categories": [
        {
            "name": "natural",
            "description": "phrased in the way people typically speak, reflecting everyday language use, without formal or artificial structure.",
            "probability": 0.8
        },
        {
            "name": "search query",
            "description": "phrased as a typed web query for search engines (only keywords, without punctuation and without a natural-sounding structure).",
            "probability": 0.2
        }
    ]
}

user_expertise_categorization = {
    "categorization_name": "user_expertise",
    "categories": [
        {
            "name": "expert",
            "description": "an expert of the subject discussed in the document, therefore he asks complex questions.",
            "probability": 0.8
        },
        {
        "name": "common person",
            "description": "a common person who is not expert of the subject discussed in the document, therefore he asks basic questions.",
            "probability": 0.2
        }
    ]
}

multi_doc_categorization = {
    "categorization_name": "multi-doc",
    "categories": [
        {
            "name": "comparison",
            "description": "a comparison question that requires comparing two related concepts or entities. The comparison must be natural and reasonable, i.e., comparing two entities by a common attribute which is meaningful and relevant to both entities. For example: 'Who is older, Glenn Hughes or Ross Lynch?', 'Are Pizhou and Jiujiang in the same province?', 'Pyotr Ilyich Tchaikovsky and Giuseppe Verdi have this profession in common'. The information required to answer the question needs to come from two documents, specifically, the first document must provide information about the first entity/concept, while the second must provide information about the second entity/concept.",
            "probability": 0.1,
            "is_multi_doc": True
        },
        {
            "name": "multi-aspect",
            "description": "A question about two different aspects of the same entity/concept. For example: 'What are the advantages of AI-powered diagnostics, and what are the associated risks of bias in medical decision-making?', 'How do cryptocurrencies enable financial inclusion, and what are the security risks associated with them?'. The information required to answer the question needs to come from two documents, specifically, the first document must provide information about the first aspect, while the second must provide information about the second aspect.",
            "probability": 0.1,
            "is_multi_doc": True
        },
        {
            "name": "causal",
            "description": "A question that asks about causes or effects related to an event, phenomenon, or concept. For example: 'What caused the fall of the Roman Empire, and what were its long-term effects?', 'How did the invention of the printing press influence literacy rates?'. The first document provides information about the cause, and the second document provides the effect (or vice versa).",
            "probability": 0.1,
            "is_multi_doc": True
        },
        {
            "name": "temporal",
            "description": "A question that involves understanding a sequence of events or changes over time. For example: 'What events led up to the Cuban Missile Crisis, and what were the immediate actions taken afterwards?', 'How has climate change policy evolved from the 1990s to the 2020s?'. The documents cover different time periods or event phases.",
            "probability": 0.1,
            "is_multi_doc": True
        },
        {
            "name": "counterfactual",
            "description": "A question exploring 'what if' scenarios or alternate outcomes. For example: 'What might have happened if the Allies lost WWII?', 'How would global trade be affected if the internet was never invented?'. One document provides factual background, the other speculates or analyzes the hypothetical scenario.",
            "probability": 0.1,
            "is_multi_doc": True
        },
        {
            "name": "definition-context",
            "description": "A question that requires both a definition of a term and its contextual application. For example: 'What is blockchain technology, and how is it used in supply chain management?'. One document provides a definition or technical explanation, and the other offers real-world usage or implications.",
            "probability": 0.1,
            "is_multi_doc": True
        },
        {
            "name": "process-outcome",
            "description": "A question requiring an explanation of a process and its outcome. For example: 'How does photosynthesis work, and what are its benefits to the environment?', 'What are the steps in vaccine development, and how effective are the final products?'.",
            "probability": 0.1,
            "is_multi_doc": True
        },
        {
            "name": "perspective",
            "description": "A question that asks for contrasting opinions or viewpoints about the same topic. For example: 'What do proponents and critics say about nuclear energy?', 'How do different political ideologies view universal basic income?'. Each document provides a distinct viewpoint.",
            "probability": 0.1,
            "is_multi_doc": True
        },
        {
            "name": "statistic-interpretation",
            "description": "A question that requires understanding a numerical or statistical fact and its implications. For example: 'What percentage of global carbon emissions come from transportation, and what does this imply for climate policy?'. One document provides the data, and the other offers interpretation or consequences.",
            "probability": 0.1,
            "is_multi_doc": True
        },
        {
            "name": "technical-societal",
            "description": "A question that bridges technical explanation and social impact. For example: 'What is gene editing, and what ethical concerns does it raise?', 'How does autonomous vehicle technology work, and how might it impact employment?'.",
            "probability": 0.1,
            "is_multi_doc": True
        }
    ]
}

# def get_all_requests():
#     resp = requests.get(
#         f"{BASE_URL}get_all_requests",
#         headers={"Authorization": f"Bearer {get_api_key()}"},
#     )
#     resp.raise_for_status()
#     print(json.dumps(resp.json(), indent=4))
# get_all_requests()

# try:
#     results = bulk_generate(n_questions=2,
#                         question_categorizations=[question_length_categorization, multi_doc_categorization],
#                         user_categorizations=[user_expertise_categorization])
# except requests.exceptions.HTTPError as e:
#     print("Status Code:", e.response.status_code)
#     print("Error details:", e.response.text)  # This should give more context

# results = get_files_using_id("75e649bc-099d-4482-8fd9-f9f87e072ba2")
# response = requests.get(results["file"])
# qas = [json.loads(line) for line in response.text.splitlines()]

# # save to ./data/test.json
# with open("./data/multidoc.json", "w") as f:
#     json.dump(qas, f, indent=4)
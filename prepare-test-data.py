from typing import Any, Dict, List, cast
import json

import datasets
from langchain.chat_models import init_chat_model

dataset =  cast(List[Dict[str, Any]], datasets.load_dataset("allenai/scifact", "claims", split="train"))

llm = init_chat_model(
    model="llama3.2:1b",
    model_provider="ollama"
)

baseQuery = "Please formulate the following fact into a question, make it more into a general question then picking up everything too detailed. Also just answer straight with the question, nothing else.\n\n "

supportedFacts = []
contradictedFacts = []

for claim in dataset:
    if not claim.get("evidence_label"):
        continue
    response = llm.invoke(baseQuery + claim.get("claim"))
    data = {
        "query": response.text(),
        "referenced_doc_id": claim.get("evidence_doc_id")
    }
    supportedFacts.append(data) if claim.get("evidence_label") == "SUPPORT" else contradictedFacts.append(data)
    print(data)


with open("data/tests/support-data.json", "w") as file:
    json.dump(supportedFacts, file, indent=4)

with open("data/tests/contradict-data.json", "w") as file:
    json.dump(contradictedFacts, file, indent=4)

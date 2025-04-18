from typing import TypedDict, List
import json

from models.baseline import BaselineModel
from models.basic import BasicModel


class TestData(TypedDict):
    query: str
    referenced_doc_id: str

def load_test_data(filename: str) -> List[TestData]:
    with open(filename, 'r') as f:
        data: List[TestData] = json.load(f)
    return data

test_data = load_test_data('./data/tests/support-data.json')

baseline_model = BaselineModel()
basic_model = BasicModel()

tests_total = len(test_data)
basic_correct = 0
baseline_correct = 0

for test in test_data:
    _, basic_retrieval_result = basic_model.retrieve(test['query'])
    _, baseline_retrieval_result = baseline_model.retrieve(test['query'])
    basic_doc_id = basic_retrieval_result[0].dict().get('metadata')['source']
    baseline_doc_id = baseline_retrieval_result[0].dict().get('metadata')['source']

    if str(basic_doc_id) == test['referenced_doc_id']:
        basic_correct += 1
    if str(baseline_doc_id) == test['referenced_doc_id']:
        baseline_correct += 1

basic_correct_retrieval_rate = basic_correct / tests_total
baseline_correct_retrieval_rate = baseline_correct / tests_total

print("PERFORMANCE BASELINE: ", baseline_correct_retrieval_rate)
print("PERFORMANCE BASIC: ", basic_correct_retrieval_rate)

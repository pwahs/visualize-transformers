from algorithms.smarter_greedy import smarter_greedy
from data.datasets import get_datasets


def test_smarter_greedy():
    for data in get_datasets():
        result = smarter_greedy(data["include"], data["exclude"])
        assert len(result) >= data["optimal"]

from algorithms.greedy import greedy
from data.datasets import get_datasets


def test_greedys():
    for data in get_datasets():
        result = greedy(data["include"], data["exclude"])
        assert len(result) >= data["optimal"]

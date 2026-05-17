from typing import List, Optional
from datasets import load_dataset, Dataset
from pydantic import BaseModel
import random

class PythonScenario(BaseModel):
    source_file: str
    task_id: int
    prompt: str
    code: str
    test_imports: list
    test_list: list
    plan: Optional[str] = None
    before_runs: bool = False
    before_pass: bool = False
    before_partial: float = 0.0

python_id = "google-research-datasets/mbpp"

def load_coding_problem(
    limit: Optional[int] = None,
    shuffle: bool = False,
) -> List[PythonScenario]:
    dataset: Dataset = load_dataset(python_id, "sanitized",split="train")

    if shuffle:
        dataset = dataset.shuffle()

    tasks = [PythonScenario(**row) for row in dataset]

    if shuffle:
        random.shuffle(tasks)

    if limit is not None:
        return tasks[:limit]
    else:
        return tasks

def load_test(
    limit: Optional[int] = None,
    shuffle: bool = False,
) -> List[PythonScenario]:
    dataset: Dataset = load_dataset(python_id, "sanitized", split="test")

    if shuffle:
        dataset = dataset.shuffle()

    tasks = [PythonScenario(**row) for row in dataset]

    if shuffle:
        random.shuffle(tasks)

    if limit is not None:
        return tasks[:limit]
    else:
        return tasks
from typing import List, Optional
from datasets import load_dataset, Dataset
from pydantic import BaseModel
import random


class PythonScenario(BaseModel):
    source_file: str = "mbpp_full"
    task_id: int
    prompt: str
    code: str
    test_imports: list = []
    test_list: list
    plan: Optional[str] = None
    before_runs: bool = False
    before_pass: bool = False
    before_partial: float = 0.0
    passing_codes: Optional[List[str]] = None
    buggy_codes: Optional[List[dict]] = None 


python_id = "google-research-datasets/mbpp"


def _row_to_scenario(row: dict) -> PythonScenario:
    setup = row.get("test_setup_code", "") or ""
    return PythonScenario(
        source_file="mbpp_full",
        task_id=row["task_id"],
        prompt=row["text"],
        code=row.get("code", ""),
        test_imports=[setup] if setup else [],
        test_list=list(row.get("test_list", [])),
    )


def load_coding_problem(
    limit: Optional[int] = None,
    shuffle: bool = False,
) -> List[PythonScenario]:
    dataset: Dataset = load_dataset(python_id, split="train")

    if shuffle:
        dataset = dataset.shuffle()

    tasks = [_row_to_scenario(row) for row in dataset]

    if shuffle:
        random.shuffle(tasks)

    if limit is not None:
        return tasks[:limit]
    return tasks


def load_test(
    limit: Optional[int] = None,
    shuffle: bool = False,
) -> List[PythonScenario]:
    dataset: Dataset = load_dataset(python_id, split="test")

    if shuffle:
        dataset = dataset.shuffle()

    tasks = [_row_to_scenario(row) for row in dataset]

    if shuffle:
        random.shuffle(tasks)

    if limit is not None:
        return tasks[:limit]
    return tasks


if __name__ == "__main__":
    train = load_coding_problem()
    test = load_test()
    print(f"train: {len(train)}")
    print(f"test:  {len(test)}")
    s = train[0]
    print(f"task_id: {s.task_id}")
    print(f"prompt:  {s.prompt}")
    print(f"test[0]: {s.test_list[0] if s.test_list else '(empty)'}")
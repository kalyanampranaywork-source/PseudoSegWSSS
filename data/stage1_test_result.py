from dataclasses import dataclass
from datetime import datetime
from pathlib import Path


@dataclass
class Stage1TestResult:
    """
    Stores the results of Stage-1 classifier evaluation.
    """

    checkpoint_path: Path

    test_dataset: str

    inference_results: dict | float | object

    evaluation_time: datetime
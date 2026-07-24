from dataclasses import dataclass
from datetime import datetime
from pathlib import Path


@dataclass
class PseudoMaskGenerationResult:
    """
    Result returned after Stage-1 pseudo-mask generation.
    """

    checkpoint_path: Path
    dataset: str
    data_root: str
    output_paths: dict[str, Path]
    generation_time: datetime
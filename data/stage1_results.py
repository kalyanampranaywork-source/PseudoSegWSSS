"""
Stage-1 Result

This module defines the result object returned after completing one
Stage-1 curriculum iteration.
"""

from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Any


@dataclass(slots=True)
class Stage1Result:
    """
    Result produced after one Stage-1 training iteration.

    Parameters
    ----------
    iteration : int
        Current curriculum iteration.

    dataset : str
        Dataset used for training.

    network : str
        Classification backbone.

    checkpoint_path : Path
        Path to the trained classifier checkpoint.

    output_directory : Path
        Root output directory for this iteration.

    tensorboard_directory : Path
        TensorBoard log directory.

    log_directory : Path
        Runtime log directory.

    training_history : Any
        Training statistics collected during optimization.

    runtime_seconds : float
        Total training runtime in seconds.
    """

    # ------------------------------------------------------------
    # Experiment Information
    # ------------------------------------------------------------

    iteration: int

    dataset: str

    network: str

    # ------------------------------------------------------------
    # Output Artifacts
    # ------------------------------------------------------------

    checkpoint_path: Path

    output_directory: Path

    tensorboard_directory: Path

    log_directory: Path

    # ------------------------------------------------------------
    # Training Information
    # ------------------------------------------------------------

    training_history: Any
    
    # ------------------------------------------------------------
    # Test Information
    # ------------------------------------------------------------
    
    inference_score: Any


    runtime_seconds: float
    
    start_time: datetime
    end_time: datetime
    
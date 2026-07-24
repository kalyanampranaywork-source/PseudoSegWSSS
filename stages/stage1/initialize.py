"""
Runtime initialization for Stage-1 classification training.

This module prepares the execution environment for one curriculum
iteration. It does not build the model, datasets, or optimizer.
"""

import random
from dataclasses import dataclass
from datetime import datetime

import numpy as np
import torch
from torch.backends import cudnn
from torch.utils.tensorboard import SummaryWriter

from tool import pyutils

import logging
from pathlib import Path


# ============================================================
# Runtime Context
# ============================================================

@dataclass
class RuntimeContext:
    """
    Stores runtime resources used throughout Stage-1 training.
    """

    device: torch.device

    writer: SummaryWriter

    logger: object

    timer: pyutils.Timer

    iteration: int

    start_time: datetime


# ============================================================
# Runtime Initialization
# ============================================================

def _initialize_runtime(
    config,
    iteration,
    iteration_manager,
):
    """
    Initialize the runtime environment for one Stage-1 curriculum iteration.

    Responsibilities
    ----------------
    1. Select computation device.
    2. Configure random seeds.
    3. Configure cuDNN.
    4. Initialize logging.
    5. Initialize TensorBoard.
    6. Initialize timer.
    7. Record experiment configuration.

    Parameters
    ----------
    config : Stage1Config

    iteration : int

    iteration_manager : IterationManager

    Returns
    -------
    RuntimeContext
    """

    # ==========================================================
    # Configure Device
    # ==========================================================

    device = torch.device(
        "cuda" if torch.cuda.is_available() else "cpu"
    )

    # ==========================================================
    # Configure Random Seeds
    # ==========================================================

    if hasattr(config, "seed"):

        random.seed(config.seed)

        np.random.seed(config.seed)

        torch.manual_seed(config.seed)

        if torch.cuda.is_available():
            torch.cuda.manual_seed(config.seed)
            torch.cuda.manual_seed_all(config.seed)

    # ==========================================================
    # Configure cuDNN
    # ==========================================================

    cudnn.enabled = True

    if hasattr(config, "seed"):

        cudnn.deterministic = True
        cudnn.benchmark = False

    else:

        cudnn.benchmark = True

    # ==========================================================
    # Initialize Logger
    # ==========================================================

    log_directory = iteration_manager.get_log_directory()

    log_file = log_directory / "stage1.log"

    logger = logging.getLogger(f"Stage1_Iteration_{iteration}")

    logger.setLevel(getattr(logging, config.log_level))

    logger.handlers.clear()

    formatter = logging.Formatter(
        fmt="%(asctime)s | %(levelname)-8s | %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )

    file_handler = logging.FileHandler(log_file)

    file_handler.setFormatter(formatter)

    console_handler = logging.StreamHandler()

    console_handler.setFormatter(formatter)

    logger.addHandler(file_handler)

    logger.addHandler(console_handler)

    logger.propagate = False

    # ==========================================================
    # Initialize TensorBoard
    # ==========================================================

    writer = SummaryWriter(
        log_dir=str(iteration_manager.get_tensorboard_directory())
    )

    # ==========================================================
    # Initialize Timer
    # ==========================================================

    timer = pyutils.Timer(
        f"Stage-1 Iteration {iteration} Started"
    )

    # ==========================================================
    # Log Experiment Information
    # ==========================================================

    logger.info("=" * 80)
    logger.info("Stage-1 Classification Training")
    logger.info("=" * 80)

    logger.info(f"Iteration : {iteration}")
    logger.info(f"Device    : {device}")
    logger.info(f"Seed      : {getattr(config, 'seed', 'None')}")

    logger.info("")

    logger.info("Configuration")

    for key, value in sorted(vars(config).items()):
        logger.info(f"{key:30s}: {value}")

    logger.info("=" * 80)

    # ==========================================================
    # Create Runtime Context
    # ==========================================================

    runtime = RuntimeContext(
        device=device,
        writer=writer,
        logger=logger,
        timer=timer,
        iteration=iteration,
        start_time=datetime.now(),
    )

    return runtime
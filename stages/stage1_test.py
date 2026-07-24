"""
stages/stage1_test.py

Stage-1 classifier evaluation.

This stage evaluates the trained classifier on the testing dataset
and reports multi-label classification performance.
"""

from configs.model_config import get_curriculum_config

from stages.stage1.build_model import _build_test_model
from stages.stage1.run_inference import _run_inference
from stages.stage1.stage1_utils import (
    _create_test_result,
    _load_checkpoint,
)

from utils.logger import get_logger


def test_stage1(
    config,
):
    """
    Evaluate the trained Stage-1 classifier on the unseen test dataset.

    Parameters
    ----------
    config

    Returns
    -------
    Stage1TestResult
    """
    # ------------------------------------------------------------
    # Initialize Logger
    # ------------------------------------------------------------

    logger = get_logger(
        name="stage1_test",
        log_directory="logs/stage1",
        log_level=config.log_level,
    )
    
    logger.info("=" * 80)
    logger.info("Stage-1 Testing")
    logger.info("=" * 80)
    # ------------------------------------------------------------
    # Build Evaluation Model
    # ------------------------------------------------------------
    logger.info("Building evaluation model...")
    
    model = _build_test_model(
        config=config,
    )

    # ------------------------------------------------------------
    # Load Trained Checkpoint
    # ------------------------------------------------------------
    logger.info("Loading trained checkpoint...")

    checkpoint_path = _load_checkpoint(
        config=config,
        model=model,
        logger=logger,
    )

    # ------------------------------------------------------------
    # Run Inference
    # ------------------------------------------------------------
    logger.info("Running inference...")
    
    test_results = _run_inference(
        config=config,
        model=model,
        logger=logger,
    )

    # ------------------------------------------------------------
    # Package Results
    # ------------------------------------------------------------
    logger.info("Packaging test results...")

    result = _create_test_result(
        config=config,
        checkpoint_path=checkpoint_path,
        results=test_results,
        logger=logger,
    )
    logger.info("=" * 80)
    logger.info("Stage-1 Testing Completed")
    logger.info("=" * 80)

    return result


def main():

    cfg = get_curriculum_config()

    results = test_stage1(
        config=cfg,
    )

    print(results)


if __name__ == "__main__":
    main()
from stages.stage1.build_dataloader import _build_train_dataloader
from stages.stage1.build_model import _build_classifier
from stages.stage1.initialize import _initialize_runtime
from stages.stage1.stage1_utils import _build_optimizer, _create_stage1_result, _load_pretrained_weights, _load_stage1_checkpoint, _save_checkpoint
from stages.stage1.train_classifier import _train_classifier
from tool.GenDataset import _build_validation_dataloader

def train_stage1(
    config,
    iteration,
    iteration_manager,
):
    """
    Train the Stage-1 classification model for a single curriculum iteration.

    Parameters
    ----------
    config : Stage1Config
        Complete Stage-1 training configuration.

    iteration : int
        Current curriculum iteration.

    iteration_manager : IterationManager
        Provides all iteration-specific paths and resources.

    Returns
    -------
    Stage1Result
        Information about the completed Stage-1 training.
    """

    # ------------------------------------------------------------------
    # Phase 1 : Initialization
    # ------------------------------------------------------------------
    runtime = _initialize_runtime(
        config=config,
        iteration=iteration,
        iteration_manager=iteration_manager,
    )

    # ------------------------------------------------------------------
    # Phase 2 : Build Model
    # ------------------------------------------------------------------
    model = _build_classifier(
        config=config,
        runtime=runtime,
    )

    # ------------------------------------------------------------------
    # Phase 3 : Build Dataset
    # ------------------------------------------------------------------
    train_loader = _build_train_dataloader(
        config=config,
        iteration=iteration,
        iteration_manager=iteration_manager,
        runtime=runtime,
    )
    
    runtime.logger.info(f"Dataset size : {len(train_loader.dataset)}")
    runtime.logger.info(f"Number of batches : {len(train_loader)}")

    # ------------------------------------------------------------------
    # Phase 4 : Optimizer & Scheduler
    # ------------------------------------------------------------------
    optimizer = _build_optimizer(
        config=config,
        model=model,
        train_loader=train_loader,
    )


    # ------------------------------------------------------------------
    # Phase 5 : Load Pretrained Weights
    # ------------------------------------------------------------------
    if iteration == 0:
        # Start from ImageNet pretrained backbone
        _load_pretrained_weights(config=config, model=model,)
    else:
        # Continue from the best Stage-1 checkpoint
        _load_stage1_checkpoint(config=config, model=model, logger=runtime.logger)

    # ------------------------------------------------------------------
    # Phase 6 : Training
    # ------------------------------------------------------------------
    training_history = _train_classifier(
        config=config,
        iteration=iteration,
        runtime=runtime,
        model=model,
        train_loader=train_loader,
        optimizer=optimizer,
    )

    # ------------------------------------------------------------------
    # Phase 8 : Save Checkpoint
    # ------------------------------------------------------------------
        # checkpoint_path = _save_checkpoint(
        #     config=config,
        #     iteration_manager=iteration_manager,
        #     model=model,
        # )
    checkpoint_path = _save_checkpoint(
        config=config,
        training_history=training_history,
    )

    # ------------------------------------------------------------------
    # Phase 9 : Package Results
    # ------------------------------------------------------------------
    result = _create_stage1_result(
        config=config,
        iteration=iteration,
        iteration_manager=iteration_manager,
        checkpoint_path=checkpoint_path,
        training_history=training_history,
        runtime=runtime,
    )

    return result
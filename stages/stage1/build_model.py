import importlib

import torch


def _build_classifier(
    config,
    runtime,
):
    """
    Build the Stage-1 classification network.

    Responsibilities
    ----------------
    1. Dynamically import the requested classifier.
    2. Instantiate the network.
    3. Move the model to the runtime device.

    Parameters
    ----------
    config : Stage1Config
        Stage-1 configuration.

    runtime : RuntimeContext
        Runtime resources.

    Returns
    -------
    torch.nn.Module
        Classification model ready for weight initialization.
    """

    runtime.logger.info("-" * 80)
    runtime.logger.info("Building Stage-1 Classifier")
    runtime.logger.info("-" * 80)

    runtime.logger.info(f"Network : {config.stage1_network}")
    runtime.logger.info(f"Classes : {config.stage1_n_class}")

    # ------------------------------------------------------------
    # Dynamically import the requested network
    # ------------------------------------------------------------

    network_module = importlib.import_module(config.stage1_network)

    model = network_module.Net(
        config.stage1_init_gama,
        n_class=config.stage1_n_class,
    )

    # ------------------------------------------------------------
    # Move model to runtime device
    # ------------------------------------------------------------

    model = model.to(runtime.device)

    # ------------------------------------------------------------
    # Log model information
    # ------------------------------------------------------------

    total_parameters = sum(
        parameter.numel()
        for parameter in model.parameters()
    )

    trainable_parameters = sum(
        parameter.numel()
        for parameter in model.parameters()
        if parameter.requires_grad
    )

    runtime.logger.info(f"Total Parameters     : {total_parameters:,}")
    runtime.logger.info(f"Trainable Parameters : {trainable_parameters:,}")
    runtime.logger.info(f"Device               : {runtime.device}")

    runtime.logger.info("Classifier successfully initialized.\n")

    return model



def _build_test_model(config):
    """
    Build the Stage-1 inference model (Net_CAM).

    The training stage uses the classification network (Net),
    whereas the testing stage uses the corresponding CAM
    network (Net_CAM) for class activation map generation.

    Parameters
    ----------
    config : CurriculumConfig

    Returns
    -------
    torch.nn.Module
        Stage-1 inference model.
    """

    # ------------------------------------------------------------
    # Device
    # ------------------------------------------------------------

    device = torch.device(
        "cuda" if torch.cuda.is_available() else "cpu"
    )

    # ------------------------------------------------------------
    # Build Model
    # ------------------------------------------------------------

    network_module = importlib.import_module(
        config.stage1_network
    )

    model = network_module.Net_CAM(
        n_class=config.stage1_n_class,
    )

    # ------------------------------------------------------------
    # Move to Device
    # ------------------------------------------------------------

    model = model.to(device)

    return model
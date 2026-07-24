"""
curriculum/iteration_manager.py

Manages the filesystem layout of a curriculum experiment.
"""

from pathlib import Path


class IterationContext:
    """
    Represents a single curriculum iteration.
    """

    def __init__(self, output_root: Path, iteration: int):

        self.output_root = Path(output_root)

        self.iteration = iteration

        self.iteration_directory = self.output_root / f"iteration_{iteration}"

        self.checkpoint_directory = self.iteration_directory / "checkpoints"

        self.log_directory = self.iteration_directory / "logs"

        self.tensorboard_directory = self.iteration_directory / "tensorboard"

        self.pseudo_label_directory = self.iteration_directory / "pseudo_labels"

        self.pseudo_mask_directory = self.iteration_directory / "pseudo_masks"

        self._create_directories()

    # ------------------------------------------------------------------
    # Directory Creation
    # ------------------------------------------------------------------

    def _create_directories(self):

        directories = [

            self.iteration_directory,

            self.checkpoint_directory,

            self.log_directory,

            self.tensorboard_directory,

            self.pseudo_label_directory,

            self.pseudo_mask_directory,

        ]

        for directory in directories:
            directory.mkdir(
                parents=True,
                exist_ok=True,
            )

    # ------------------------------------------------------------------
    # Directory Getters
    # ------------------------------------------------------------------

    def get_iteration_directory(self):
        return self.iteration_directory

    def get_checkpoint_directory(self):
        return self.checkpoint_directory

    def get_log_directory(self):
        return self.log_directory

    def get_tensorboard_directory(self):
        return self.tensorboard_directory

    def get_pseudo_label_directory(self):
        return self.pseudo_label_directory

    def get_pseudo_mask_directory(self):
        return self.pseudo_mask_directory

    # ------------------------------------------------------------------
    # File Getters
    # ------------------------------------------------------------------

    def get_stage1_checkpoint_path(self):
        return self.checkpoint_directory / "classifier.pth"

    def get_stage2_checkpoint_path(self):
        return self.checkpoint_directory / "segmenter.pth"

    def get_pseudo_label_file(self):
        return self.pseudo_label_directory / "pseudo_labels.json"

    def get_pseudo_mask_directory_path(self):
        return self.pseudo_mask_directory
    
    def get_previous_pseudo_label_file(self):
        if self.iteration == 0:
            return None

        previous_dir = (
            self.output_root
            / f"iteration_{self.iteration - 1}"
            / "pseudo_labels"
        )

        return previous_dir / "pseudo_labels.json"


class IterationManager:
    """
    Factory for iteration-specific contexts.

    Example
    -------
    manager = IterationManager("outputs")

    iteration0 = manager.get_iteration(0)
    iteration1 = manager.get_iteration(1)
    """

    def __init__(self, output_root):

        self.output_root = Path(output_root)

        self.output_root.mkdir(
            parents=True,
            exist_ok=True,
        )

    def get_iteration(self, iteration: int) -> IterationContext:

        return IterationContext(
            output_root=self.output_root,
            iteration=iteration,
        )
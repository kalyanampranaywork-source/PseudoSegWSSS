"""
curriculum.py

Curriculum Learning Controller

This file orchestrates the complete weakly supervised semantic segmentation
pipeline. It does not implement any learning algorithms itself; instead it
calls the individual stages in the correct order.

Pipeline:

Iteration 0
------------
Stage1 Classification
        ↓
Pseudo Mask Generation
        ↓
Stage2 Segmentation
        ↓
Pseudo Label Generation

Iteration 1
------------
Stage1 Classification (GT + Pseudo Labels)
        ↓
Pseudo Mask Generation
        ↓
Stage2 Segmentation
        ↓
Pseudo Label Generation

...
"""

import os
import time
from pathlib import Path
from stages.generate_pseudo_mask import generate_pseudo_masks
from utils.logger import get_logger

from configs.model_config import get_curriculum_config

# ------------------------------------------------------------------
# Assume these modules already exist.
# We will implement them later.
# ------------------------------------------------------------------

from stages.stage1_train import train_stage1


from curriculum.iteration_manager import IterationManager
from curriculum.metrics_logger import MetricsLogger


# ==============================================================
# Curriculum Controller
# ==============================================================

class CurriculumController:

    def __init__(self, config):

        self.cfg = config

        self.manager = IterationManager(config.output_dir)

        self.logger = get_logger(
            name="curriculum",
            log_directory="logs/curriculum",
            log_level=config.log_level,
        )

        self.num_iterations = config.num_iterations


    # ----------------------------------------------------------
    # Run complete curriculum
    # ----------------------------------------------------------

    def run(self):

        self.logger.info("=" * 80)
        self.logger.info("Curriculum Weakly Supervised Semantic Segmentation")
        self.logger.info("=" * 80)
        
        self.logger.info("Experiment Configuration")
        

        total_start = time.time()

        for iteration in range(self.num_iterations):

            self.run_iteration(iteration)

        total_time = time.time() - total_start

    
        self.logger.info("=" * 80)
        self.logger.info("Curriculum Training Completed")
        self.logger.info(f"Total Runtime : {total_time:.2f} sec")
        self.logger.info("=" * 80)


    # ----------------------------------------------------------
    # One curriculum iteration
    # ----------------------------------------------------------

    def run_iteration(self, iteration):

        self.logger.info("")
        self.logger.info("=" * 80)
        self.logger.info(f"Curriculum Iteration {iteration}")
        self.logger.info("=" * 80)
        
        iteration_manager = self.manager.get_iteration(iteration)

        start = time.time()
        
        # ------------------------------------------------------
        # Stage 1
        # ------------------------------------------------------

        self.logger.info("[1/4] Stage-1 Classification Training")

        stage1_results = train_stage1(
            config=self.cfg,
            iteration=iteration,
            iteration_manager=iteration_manager,
        )
        
        self.logger.info("")
        self.logger.info("\n" + "=" * 90)
        self.logger.info("Stage-1 Training Summary")
        self.logger.info("=" * 90)

        self.logger.info(f"{'Metric':<35} {'Value'}")
        self.logger.info("-" * 90)

        self.logger.info(f"{'Iteration':<35} {stage1_results.iteration}")
        self.logger.info(f"{'Dataset':<35} {stage1_results.dataset}")
        self.logger.info(f"{'Network':<35} {stage1_results.network}")

        self.logger.info(f"{'Final Training Loss':<35} {stage1_results.training_history['loss']:.4f}")
        self.logger.info(f"{'Exact Match Accuracy':<35} {stage1_results.training_history['avg_ep_EM']:.4f}")
        self.logger.info(f"{'Classification Accuracy':<35} {stage1_results.training_history['avg_ep_acc']:.4f}")

        self.logger.info(f"{'Training Time (seconds)':<35} {stage1_results.runtime_seconds:.2f}")
        self.logger.info(f"{'Checkpoint':<35} {stage1_results.checkpoint_path}")

        self.logger.info("=" * 90)
        self.logger.info("")
        
        # ------------------------------------------------------
        # Generate pseudo masks
        # ------------------------------------------------------

        self.logger.info("[2/4] Pseudo Mask Generation")

        pseudo_mask_results = generate_pseudo_masks(
            config=self.cfg,
        )

        self.logger.info("")
        self.logger.info("=" * 90)
        self.logger.info("Pseudo Mask Generation Summary")
        self.logger.info("=" * 90)

        self.logger.info(f"{'Metric':<35} {'Value'}")
        self.logger.info("-" * 90)

        self.logger.info(
            f"{'Checkpoint Used':<35} "
            f"{pseudo_mask_results.checkpoint_path}"
        )

        self.logger.info(
            f"{'Dataset':<35} "
            f"{pseudo_mask_results.dataset}"
        )

        self.logger.info(
            f"{'Training Dataset':<35} "
            f"{pseudo_mask_results.data_root}"
        )

        self.logger.info(
            f"{'Feature Maps':<35} "
            f"{', '.join(pseudo_mask_results.output_paths.keys())}"
        )

        self.logger.info("")
        self.logger.info("Generated Pseudo Mask Directories")

        for feature_map, directory in pseudo_mask_results.output_paths.items():

            self.logger.info(
                f"  {feature_map:<10} : {directory}"
            )

        self.logger.info(
            f"{'Generation Time':<35} "
            f"{pseudo_mask_results.generation_time}"
        )

        self.logger.info("=" * 90)
        self.logger.info("")

       
        # ------------------------------------------------------
        # Stage 2
        # ------------------------------------------------------

        self.logger.info("[3/4] Stage-2 Segmentation Training")

       
        # ------------------------------------------------------
        # Generate pseudo image labels
        # ------------------------------------------------------

        self.logger.info("[4/4] Pseudo Label Generation")

     
        # ------------------------------------------------------
        # Save iteration information
        # ------------------------------------------------------

        elapsed = time.time() - start

        self.logger.info("")
        self.logger.info(f"Iteration {iteration} completed in {elapsed:.2f} sec")


# ==============================================================
# Main
# ==============================================================

def main():

    cfg = get_curriculum_config()

    controller = CurriculumController(cfg)

    controller.run()


if __name__ == "__main__":
    main()
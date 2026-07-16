from __future__ import annotations

import argparse
from pathlib import Path
from typing import Sequence


# ============================================================================
# BCSS Semantic Classes
# ============================================================================

BACKGROUND_CLASS = 0

BCSS_LABELS = {
    1: "tumor",
    2: "stroma",
    3: "lymphocyte",
    4: "necrosis",
}



def positive_int(value: str) -> int:
    """
    Ensure integer is positive.
    """
    ivalue = int(value)

    if ivalue <= 0:
        raise argparse.ArgumentTypeError(
            f"{ivalue} is not a positive integer."
        )

    return ivalue


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="HistoGraphWSL Pipeline")

    # --- Existing Dataset Options ---
    parser.add_argument("--dataset", type=str, default="luad", help="Dataset name.")
    parser.add_argument("--output_dir", type=str, default="outputs/")
    
    # New: Split dataset roots for train/test if required by your script
    parser.add_argument("--trainroot", type=str, default="datasets/LUAD-HistoSeg/train5/")
    parser.add_argument("--testroot", type=str, default="datasets/LUAD-HistoSeg/test/")
    parser.add_argument("--dataroot", default="datasets/LUAD-HistoSeg", type=str)
    parser.add_argument("--save_folder", type=str, default="checkpoints/")

    # --- Optimization & Training Options ---
    parser.add_argument("--batch_size", type=positive_int, default=20)
    parser.add_argument("--max_epoches", type=positive_int, default=20)
    parser.add_argument("--lr", type=float, default=0.01)
    parser.add_argument("--wt_dec", type=float, default=5e-4)
    parser.add_argument("--init_gama", type=float, default=1.0)
    parser.add_argument("--seed", type=int, default=42)

    # --- Network Options ---
    parser.add_argument("--network", type=str, default="network.resnet38_cls")
    parser.add_argument("--n_class", type=positive_int, default=4, help="Number of classes for segmentation. Ex: [Tumor, Stroma, Lymphocyte, Necrosis] = 4 classes") 
    
    # --- stage 1: intial training weights --- --- comment this line when you run other stages ---
    # parser.add_argument("--weights", type=str, default="init_weights/ilsvrc-cls_rna-a1_cls1000_ep-0001.pth")
    
    # --- stage 2: pseudo mask generation: stage 1 final weights ---  --- comment this line when you run other stages ---
    parser.add_argument("--weights", default='checkpoints/stage1_checkpoint_trained_on_luad.pth', type=str)

    # --- Runtime Environments ---
    parser.add_argument("--session_name", type=str, default="Stage 1")
    parser.add_argument("--env_name", type=str, default="PDA")
    parser.add_argument("--model_name", type=str, default="PDA")
    parser.add_argument("--num_workers", type=positive_int, default=2, help="Set to 2 for Colab environment stability.")
    parser.add_argument("--dry-run", action="store_true", default=False)
   
    # --- Logging ---
    parser.add_argument(
        "--log_level",
        default="INFO",
        choices=[
            "DEBUG",
            "INFO",
            "WARNING",
            "ERROR",
        ]
    )

    return parser


def create_output_dirs(output_root: Path) -> None:
    """
    Create project output folders.
    """

    folders = [
        "patches",
        "masks",
        "labels",
        "graphs",
        "metadata",
        "visualizations",
        "logs",
    ]

    for folder in folders:
        (output_root / folder).mkdir(
            parents=True,
            exist_ok=True,
        )


def get_config(argv: Sequence[str] | None = None) -> argparse.Namespace:
    """
    Parse and validate configuration.
    """

    parser = build_parser()

    args = parser.parse_args(argv)

    create_output_dirs(Path(args.output_dir))

    return args


"""
curriculum/metrics_logger.py

Utility for displaying training and testing metrics in a
consistent table format.
"""

from pathlib import Path


class MetricsLogger:
    """
    Utility class for formatting experiment metrics.
    """

    TABLE_WIDTH = 90

    @staticmethod
    def separator():
        print("=" * MetricsLogger.TABLE_WIDTH)

    @staticmethod
    def divider():
        print("-" * MetricsLogger.TABLE_WIDTH)

    @staticmethod
    def title(title):
        MetricsLogger.separator()
        print(title.center(MetricsLogger.TABLE_WIDTH))
        MetricsLogger.separator()

    @staticmethod
    def row(name, value):
        print(f"{name:<35} {value}")

    @staticmethod
    def stage1_summary(stage1_result):
        """
        Display Stage-1 training summary.
        """

        history = stage1_result.training_history

        MetricsLogger.title("Stage-1 Training Summary")

        MetricsLogger.row("Iteration", stage1_result.iteration)
        MetricsLogger.row("Dataset", stage1_result.dataset)
        MetricsLogger.row("Network", stage1_result.network)

        MetricsLogger.divider()

        MetricsLogger.row(
            "Final Training Loss",
            f"{history.get('loss'):.4f}",
        )

        MetricsLogger.row(
            "Exact Match Accuracy",
            f"{history.get('avg_ep_EM'):.4f}",
        )

        MetricsLogger.row(
            "Classification Accuracy",
            f"{history.get('avg_ep_acc'):.4f}",
        )

        MetricsLogger.divider()

        MetricsLogger.row(
            "Training Time (s)",
            f"{stage1_result.runtime_seconds:.2f}",
        )

        MetricsLogger.row(
            "Checkpoint",
            Path(stage1_result.checkpoint_path).name,
        )

        MetricsLogger.row(
            "Checkpoint Path",
            stage1_result.checkpoint_path,
        )

        MetricsLogger.separator()
        print()
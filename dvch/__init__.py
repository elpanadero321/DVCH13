"""DVCH: Dynamic Vacuum Coupling Hypothesis — reproducible background solver."""

from .background import BackgroundSolver, ConvergenceError, DVCHParams

__all__ = ["BackgroundSolver", "ConvergenceError", "DVCHParams"]
__version__ = "0.1.0"

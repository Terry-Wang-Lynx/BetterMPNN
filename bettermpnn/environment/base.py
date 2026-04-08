"""Environment abstraction: sequence -> score black box."""

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Optional


@dataclass
class EvalResult:
    """Result from evaluating a single sequence."""
    score: float
    structure_path: Optional[str] = None
    metrics: dict = field(default_factory=dict)


class Environment(ABC):
    """Abstract base class for sequence evaluation environments.

    An Environment takes a protein sequence and returns a scalar reward.
    Implementations can wrap any structure predictor (AlphaFold 3, Protenix,
    ESMFold, etc.) or even experimental assay data.
    """

    @abstractmethod
    def evaluate(self, sequence: str, step: int = 0, variant: int = 0) -> EvalResult:
        """Evaluate a single sequence.

        Args:
            sequence: Amino acid sequence string.
            step: Current training step (for output organization).
            variant: Variant index within the step.

        Returns:
            EvalResult with score and optional metadata.
        """
        ...

    def evaluate_batch(self, sequences: list[str], step: int = 0) -> list[EvalResult]:
        """Evaluate a batch of sequences. Override for parallel evaluation."""
        return [self.evaluate(seq, step, i) for i, seq in enumerate(sequences)]

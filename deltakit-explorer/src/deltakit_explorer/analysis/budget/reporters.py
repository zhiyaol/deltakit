from collections.abc import Sequence
from dataclasses import dataclass


def _indent(message: str, size: int, indent: str = "| ") -> str:
    indent *= size
    lines: list[str] = []
    for line in message.splitlines():
        lines.append((indent if line else "") + line)
    return "\n".join(lines)


def _value_and_stddev(value: float, stddev: float) -> str:
    return f"{value:.5g} ± {stddev:.3g}"


class BaseReporter:
    def __init__(self) -> None:
        self._warnings: list[str] = []

    def add_warning(self, warning_message: str) -> None:
        self._warnings.append(warning_message)

    @property
    def warnings(self) -> list[str]:
        return self._warnings

    def to_string(self) -> str:
        return "Warnings:\n  - " + "\n  - ".join(self.warnings) + "\n"


@dataclass
class LEPReporter(BaseReporter):
    nfails: int
    nshots: int

    @property
    def lep(self) -> float:
        if self.nshots > 0:
            return self.nfails / self.nshots
        return float("inf")

    @property
    def stddev(self) -> float:
        p = self.lep
        return p * (1 - p) / self.nshots

    def to_string(self) -> str:
        return _value_and_stddev(self.lep, self.stddev)


@dataclass
class LEPPRReporter(BaseReporter):
    num_rounds: list[int]
    leps: list[LEPReporter]
    leppr: float
    fit_stddev: float
    error_propagation_stddev: float

    @property
    def leppr_stddev(self) -> float:
        return self.fit_stddev * self.error_propagation_stddev

    def to_string(self) -> str:
        return (
            "LEPPR = "
            + _value_and_stddev(self.leppr, self.leppr_stddev)
            + f"([fit={self.fit_stddev:.3g}] * "
            + f"[propagation={self.error_propagation_stddev:.3g}])\n"
        ) + _indent("\n".join(rep.to_string() for rep in self.leps), 1)


@dataclass
class LambdaEstimationReporter(BaseReporter):
    distances: list[int]
    lepprs: list[LEPPRReporter]
    lambda_: float
    fit_stddev: float
    error_propagation_stddev: float

    @property
    def lambda_stddev(self) -> float:
        return self.fit_stddev * self.error_propagation_stddev

    def to_string(self) -> str:
        return (
            "Λ = "
            + _value_and_stddev(self.lambda_, self.lambda_stddev)
            + f"([fit={self.fit_stddev:.3g}] * "
            + f"[propagation={self.error_propagation_stddev:.3g}])\n"
        ) + _indent("\n".join(rep.to_string() for rep in self.lepprs), 1)


@dataclass
class LambdaReciprocalEstimationReporter(BaseReporter):
    lambda_: LambdaEstimationReporter
    lambda_reciprocal: float
    lambda_reciprocal_stddev: float
    x: Sequence[float]

    def to_string(self) -> str:
        return (
            "1/Λ("
            + ", ".join(f"{v:.3g}" for v in self.x)
            + ") = "
            + _value_and_stddev(self.lambda_reciprocal, self.lambda_reciprocal_stddev)
            + "\n"
        ) + _indent(self.lambda_.to_string(), 1)


@dataclass
class LambdaReciprocalDerivativeReporter(BaseReporter):
    xs: list[float]
    lambda_reciprocal: list[LambdaReciprocalEstimationReporter]
    value: float
    error: float

    def to_string(self) -> str:
        return (f"∇(1/Λ) = {_value_and_stddev(self.value, self.error)}\n") + _indent(
            "\n".join(rep.to_string() for rep in self.lambda_reciprocal), 1
        )

# (c) Copyright Riverlane 2020-2025.
"""Description of ``deltakit.circuit.noise_channels`` namespace here."""

from typing import Dict, Type, Union

from deltakit_circuit.noise_channels._abstract_noise_channels import (
    MultiProbabilityNoiseChannel,
    NoiseChannel,
    OneProbabilityNoiseChannel,
    OneQubitNoiseChannel,
    TwoQubitNoiseChannel,
)
from deltakit_circuit.noise_channels._correlated_noise import (
    CorrelatedError,
    ElseCorrelatedError,
    _CorrelatedNoise,
)
from deltakit_circuit.noise_channels._depolarising_noise import (
    Depolarise1,
    Depolarise2,
    _DepolarisingNoise,
)
from deltakit_circuit.noise_channels._leakage_noise import Leakage, Relax
from deltakit_circuit.noise_channels._pauli_noise import (
    PauliChannel1,
    PauliChannel2,
    PauliXError,
    PauliYError,
    PauliZError,
    _PauliNoise,
)

_UncorrelatedNoise = Union[_PauliNoise, _DepolarisingNoise]
_LeakageNoise = Union[Leakage, Relax]
_NoiseChannel = Union[_UncorrelatedNoise, _CorrelatedNoise, _LeakageNoise]
_OneQubitNoiseChannel = Union[
    PauliXError,
    PauliYError,
    PauliZError,
    PauliChannel1,
    Depolarise1,
    CorrelatedError,
    ElseCorrelatedError,
]

NOISE_CHANNEL_MAPPING: Dict[str, Type[_NoiseChannel]] = {
    CorrelatedError.stim_string: CorrelatedError,
    "E": CorrelatedError,
    ElseCorrelatedError.stim_string: ElseCorrelatedError,
    Depolarise1.stim_string: Depolarise1,
    Depolarise2.stim_string: Depolarise2,
    PauliChannel1.stim_string: PauliChannel1,
    PauliChannel2.stim_string: PauliChannel2,
    PauliXError.stim_string: PauliXError,
    PauliYError.stim_string: PauliYError,
    PauliZError.stim_string: PauliZError,
    Leakage.stim_string: Leakage,
    Relax.stim_string: Relax,
}

# Prevent import of non-public objects from this module.
del Dict, Type, Union

# List only public members in `__all__`.
__all__ = [s for s in dir() if not s.startswith("_")]

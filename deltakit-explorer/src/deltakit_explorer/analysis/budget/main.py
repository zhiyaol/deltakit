from __future__ import annotations

import pickle  # nosec B403

from deltakit_explorer.analysis.budget.gradient.estimation import (
    compute_ideal_rounds_for_noise_model_and_distance,
    get_lambda_reciprocal_gradient,
)
from deltakit_explorer.analysis.budget.noise import DeltakitNoise


def main() -> None:
    noise_model = DeltakitNoise.from_noise_parameters(
        t_1=100e-6,
        t_2=100e-6,
        time_1_qubit_gate=25e-9,
        time_2_qubit_gate=34e-9,
        time_measurement=500e-9,
        time_reset=160e-9,
        p_1_qubit_gate_error=1e-4,
        p_2_qubit_gate_error=1e-3,
        p_reset_error=5e-3,
        p_meas_qubit_error=5e-3,
        p_readout_flip=5e-3,
        name="deltakit",
    )

    # noise_model = SimplerNoise.from_noise_parameters(5e-3, 1e-2, name="simpler")
    distances = [5, 7, 9, 11]
    max_workers = 20
    num_shots = 5_000_000
    batch_size = 25_000

    print("Starting a simulation with the following parameters:")
    print(f"  noise_model: {noise_model.name}")
    print(f"  distances:   {distances}")
    print(f"  num_shots:   {num_shots}")
    print(f"  max_workers: {max_workers}")
    print(f"  batch_size:  {batch_size}")

    num_rounds_by_distances: dict[int, list[int]] = {}
    if not num_rounds_by_distances:
        print("Computing the optimal numbers of rounds per distances...")
        for d in distances:
            print(f"  Computing for distance {d}...")
            num_rounds_by_distances[d] = (
                compute_ideal_rounds_for_noise_model_and_distance(
                    noise_model,
                    d,
                    max_shots=500_000,
                    batch_size=10_000,
                    initial_round_number=4,
                    min_fails=50,
                    target_stddev=1e-6,
                    max_round_number=4096,
                )
            )
        print("Optimal distances:")
        print(num_rounds_by_distances)

    gradient, errors, reporters = get_lambda_reciprocal_gradient(
        noise_model.noise_parameters,
        type(noise_model),
        num_rounds_by_distances,
        absolute_maximum_xi_steps=0.45 * noise_model.noise_parameters,
        max_shots=num_shots,
        batch_size=batch_size,
        max_workers=max_workers,
        target_rse=1e-3,
    )

    for i, (g, e) in enumerate(zip(gradient, errors, strict=True)):
        print(f"∇(1/Λ)_{i} = {g:.5g>7} ± {e:.5g>7} ({noise_model.parameter_names[i]})")

    with open("./reporters.pkl", "wb") as f:
        pickle.dump(reporters, f)


if __name__ == "__main__":
    main()

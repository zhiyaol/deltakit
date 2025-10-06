import matplotlib.pyplot as plt
from typing import Sequence
from matplotlib.axes import Axes
from matplotlib.figure import Figure
import numpy as np
import numpy.typing as npt
from deltakit_core.constants import RIVERLANE_COLORS_LIST as colors
from deltakit_explorer.analysis import calculate_lep_and_lep_stddev, compute_logical_error_per_round, calculate_lambda_and_lambda_stddev, get_lambda_fit

def plot_leppr(
    num_failed_shots: npt.NDArray[np.int_] | Sequence[int],
    num_shots: npt.NDArray[np.int_] | Sequence[int],
    num_rounds: npt.NDArray[np.int_] | Sequence[int],
    force_include_single_round: bool = False,
    interpolation_points: int = 26,
    fig: Figure | None = None,
    ax: Axes | None = None,
    plot_error_band: bool = False,
) -> tuple[Figure, Axes]:
    """Plot the logical error probability per round data and the fitted curve.

    Args:
        num_failed_shots (npt.NDArray[np.int_] | Sequence[int]):
            a sequence of integers representing the number of failed shots for each of
            the number of rounds in ``num_rounds``. Should be the same length as
            ``num_rounds``.
        num_shots (npt.NDArray[np.int_] | Sequence[int]):
            a sequence of integers representing the total number of shots for each of
            the number of rounds in ``num_rounds``. Should be the same length as
            ``num_rounds``.
        num_rounds (npt.NDArray[np.int_] | Sequence[int]):
            a sequence of integers representing the number of rounds used to get the
            corresponding results in ``num_failed_shots`` and ``num_shots``.
        interpolation_points (int, optional):
            the number of points to use for plotting the fitted curve. Default is 26.
        fig (Figure | None, optional):
            a matplotlib Figure object to plot on. If None, a new figure will be created.
            Default is None.
        ax (Axes | None, optional):
            a matplotlib Axes object to plot on. If None, a new axes will be created.
            Default is None.
        plot_error_band (bool, optional):
            whether to plot the error band for the fitted curve. Default is False.

    Returns:
        tuple[Figure, Axes]: The matplotlib Figure and Axes objects containing the plot.

    Example
    fig, ax = plot_leppr(
            # num_failed_shots=[34, 651, 2356, 12000, 50000],
            # num_shots=[500000] * 5,
            # num_rounds=[2, 8, 32, 128, 512],
            num_failed_shots=[34, 151, 356],
            num_shots=[500000] * 3,
            num_rounds=[2, 4, 6],
            plot_error_band=True,
        )
    plt.show()
    """
    assert not (fig is None) ^ (ax is None)
    if fig is None and ax is None:
        fig, ax = plt.subplots()
    assert fig is not None and ax is not None

    res = compute_logical_error_per_round(
                num_failed_shots=num_failed_shots,
                num_shots=num_shots,
                num_rounds=num_rounds,
                force_include_single_round=force_include_single_round,
            )
    leppr, leppr_stddev = res.leppr, res.leppr_stddev
    spam, spam_stddev = res.spam_error, res.spam_error_stddev

    # plot logical error probabilities
    lep, lep_stddev = calculate_lep_and_lep_stddev(fails=num_failed_shots, shots=num_shots)
    ax.errorbar(num_rounds, lep, yerr=lep_stddev, fmt="o", color=colors[0])

    # plot fitted LEPPR curve
    rounds_interpolated = np.linspace(
        num_rounds[0], num_rounds[-1], interpolation_points,
        dtype=np.float64,
    )
    y_interpolated = [(1 - 2 * spam) * (1 - 2 * leppr) ** r for r in rounds_interpolated]

    # if an error probability is below 0, set to 0 for log scale plot performance
    y_interpolated = [0 if i < 0 else i for i in y_interpolated]
    LEP_interpolated = (1.0 - np.array(y_interpolated, dtype=np.float64)) * 0.5
    ax.plot(rounds_interpolated, LEP_interpolated,label=f"Fit, ε={leppr:.4f}" + r"$\pm$" + f"{leppr_stddev:.4f}", color=colors[0])

    # add error band to LEPPR curve
    if plot_error_band:
        LEP_interpolated_err = np.array([np.sqrt(((2*spam_stddev)/(1-2*spam_stddev))**2 + ((2*r*leppr_stddev)/(1-2*leppr_stddev))**2) for r in rounds_interpolated])
        ax.fill_between(rounds_interpolated, LEP_interpolated - LEP_interpolated_err, LEP_interpolated + LEP_interpolated_err, color=colors[0], alpha=0.2)

    ax.set_xlabel("Rounds")
    ax.set_ylabel("Logical error probability")
    ax.legend()
    #ax.set_yscale("log")

    return fig, ax

# next step: plot lambda
def plot_lambda(
    distances: npt.NDArray[np.int_] | Sequence[int],
    lep_per_round: npt.NDArray[np.float64] | Sequence[float],
    lep_stddev_per_round: npt.NDArray[np.float64] | Sequence[float],
    fig: Figure | None = None,
    ax: Axes | None = None,
) -> tuple[Figure, Axes]:
    """
    Args:
        distances (npt.NDArray[np.int\\_] | Sequence[int]): The distances of the code.
        lep_per_round (npt.NDArray[np.float64] | Sequence[float]):
            The logical error probabilities per round.
        lep_stddev_per_round (npt.NDArray[np.float64] | Sequence[float]):
            The standard deviation of the logical error probabilities per round.

    Returns:
        tuple[Figure, Axes]: The matplotlib Figure and Axes objects containing the plot.

    Example:
        fig, ax = plot_lambda(
            distances = [5, 7, 9],
            lep_per_round = [0.15, 0.1, 0.05],
            lep_stddev_per_round = [0.01, 0.008, 0.005],
        )
        ax.set_yscale("log")
        plt.show()
    """
    assert not (fig is None) ^ (ax is None)
    if fig is None and ax is None:
        fig, ax = plt.subplots()
    assert fig is not None and ax is not None

    lambda_val, lambda_val_stddev = calculate_lambda_and_lambda_stddev(
        distances=distances,
        lep_per_round=lep_per_round,
        lep_stddev_per_round=lep_stddev_per_round,
    )
    ax.errorbar(distances, lep_per_round, yerr=lep_stddev_per_round, fmt="o", color=colors[0])

    y_vals = get_lambda_fit(distances, lep_per_round, lep_stddev_per_round)
    ax.plot(distances,y_vals,label=f"Fit, λ={lambda_val:.4f}" + r"$\pm$" + f"{lambda_val_stddev:.4f}", color=colors[0])
    ax.set_xlabel("Distance")
    ax.set_ylabel("Logical error probability per round")
    ax.legend()
    return fig, ax
